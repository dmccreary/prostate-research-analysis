"""
src/stage8/eval_stage8_zero_shot.py - Stage 8: Zero-Shot Literature Screening Benchmark using MedCPT Cross-Encoder.

Evaluates out-of-the-box ncbi/MedCPT-Cross-Encoder without any fine-tuning on:
1. Validation Set (data/splits/stage5/val.csv, N=84: 45 Pos, 39 Neg) - for threshold calibration
2. Held-Out Test Set (data/splits/stage5/test.csv, N=105: 56 Pos, 49 Neg)
3. 2021 Prospective Cohort (data/2021 labeld.csv, N=117: 28 Pos, 89 Neg)

Tests two query formulations:
- PICO Natural Language Query (Stage 7 text)
- Search-style Query (PubMed user query style)
"""

import sys
import logging
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    precision_recall_curve,
    auc,
    confusion_matrix,
)

BASE_DIR = Path(__file__).resolve().parent.parent.parent
SPLITS_DIR = BASE_DIR / "data" / "splits" / "stage5"
RESULTS_DIR = BASE_DIR / "results" / "stage8"
REPORTS_DIR = BASE_DIR / "reports" / "stage8"
LOGS_DIR = BASE_DIR / "logs"

for d in [RESULTS_DIR, REPORTS_DIR, LOGS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOGS_DIR / "stage8_zero_shot.log", encoding="utf-8", mode="w"),
    ],
)
logger = logging.getLogger(__name__)

sys.path.insert(0, str(BASE_DIR / "src"))
from preprocessing import clean_clinical_text

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Execution Device: {device}")

# Model
MODEL_NAME = "ncbi/MedCPT-Cross-Encoder"
MAX_LENGTH = 512
BATCH_SIZE = 8

# Queries
QUERY_PICO_NATURAL = (
    "Screening criteria for localized prostate cancer: Primary curative treatment "
    "(radical prostatectomy, EBRT >= 72 Gy, brachytherapy, cryo, HIFU). "
    "Must be a comparative trial (exclude single-arm audits, post-treatment salvage, metastatic, or palliative). "
    "Reports oncologic survival (BRFS, OS, CSS) with follow-up >= 5 years."
)

QUERY_SEARCH_STYLE = (
    "localized prostate cancer primary curative treatments radical prostatectomy "
    "radiotherapy brachytherapy survival comparative study"
)

# 1. Dataset Class
class CrossEncoderEvalDataset(Dataset):
    def __init__(self, titles, abstracts, labels, tokenizer, query_text, max_length=MAX_LENGTH):
        self.labels = torch.tensor(labels, dtype=torch.float32)
        queries = [query_text] * len(titles)
        paper_texts = [f"{t}. {a}" if t else a for t, a in zip(titles, abstracts)]

        enc = tokenizer(
            text=queries,
            text_pair=paper_texts,
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        self.input_ids = enc["input_ids"]
        self.attention_mask = enc["attention_mask"]
        self.token_type_ids = enc.get("token_type_ids", torch.zeros_like(self.input_ids))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
            "token_type_ids": self.token_type_ids[idx],
            "label": self.labels[idx],
        }

# Metric calculation helper
def calc_metrics(y_true, y_prob, threshold, dataset_name, operating_point, query_name):
    y_pred = (y_prob >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    total = len(y_true)
    flagged = int(tp + fp)

    acc = float(accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    spec = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    f1 = float(f1_score(y_true, y_pred, zero_division=0))

    try:
        auroc = float(roc_auc_score(y_true, y_prob))
    except Exception:
        auroc = 0.5

    try:
        p_c, r_c, _ = precision_recall_curve(y_true, y_prob)
        pr_auc = float(auc(r_c, p_c))
    except Exception:
        pr_auc = float(np.mean(y_true))

    nns = round((flagged / tp), 2) if tp > 0 else float("inf")
    workload_pct = round((flagged / total) * 100.0, 2)
    reduction_pct = round(((total - flagged) / total) * 100.0, 2)

    return {
        "model": "MedCPT Cross-Encoder (Zero-Shot)",
        "query_formulation": query_name,
        "dataset": dataset_name,
        "operating_point": operating_point,
        "threshold": round(float(threshold), 3),
        "recall": round(rec * 100, 2),
        "specificity": round(spec * 100, 2),
        "precision": round(prec * 100, 2),
        "f1": round(f1, 4),
        "auroc": round(auroc, 4),
        "pr_auc": round(pr_auc, 4),
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
        "workload_pct": workload_pct,
        "reduction_pct": reduction_pct,
        "nns": nns,
    }

def find_threshold_for_recall(y_true, y_prob, target_recall):
    threshold_grid = np.linspace(0.001, 0.999, 500)
    valid_candidates = []
    for t in threshold_grid:
        y_pred = (y_prob >= t).astype(int)
        r = recall_score(y_true, y_pred, zero_division=0)
        if r >= target_recall:
            valid_candidates.append(t)
    return max(valid_candidates) if valid_candidates else 0.001

def run_inference(model, loader):
    model.eval()
    all_probs = []
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            token_type_ids = batch["token_type_ids"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
            )
            logits = outputs.logits.squeeze(-1)
            probs = torch.sigmoid(logits).cpu().numpy().tolist()
            if isinstance(probs, float):
                probs = [probs]
            all_probs.extend(probs)
    return np.array(all_probs)

def main():
    logger.info("Initializing Tokenizer and Pretrained Model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    model.to(device)
    logger.info("Model loaded successfully into device!")

    # Load Data
    val_df = pd.read_csv(SPLITS_DIR / "val.csv")
    test_df = pd.read_csv(SPLITS_DIR / "test.csv")
    df_2021 = pd.read_csv(BASE_DIR / "data" / "2021 labeld.csv")

    for df in [val_df, test_df, df_2021]:
        df["clean_title"] = [clean_clinical_text(str(t or "")) for t in df["title"]]
        df["clean_abstract"] = [clean_clinical_text(str(a or "")) for a in df["abstract"]]

    y_val = val_df["label"].to_numpy().astype(int)
    y_test = test_df["label"].to_numpy().astype(int)
    y_2021 = df_2021["label"].to_numpy().astype(int)

    logger.info(f"Loaded datasets: Val N={len(val_df)}, Test N={len(test_df)}, 2021 N={len(df_2021)}")

    queries = {
        "PICO Natural Language": QUERY_PICO_NATURAL,
        "Search Style": QUERY_SEARCH_STYLE,
    }

    all_records = []
    test_pred_df = test_df[["pmid", "title", "label"]].copy()
    cohort_2021_pred_df = df_2021[["pmid", "title", "label"]].copy()

    for q_name, q_text in queries.items():
        logger.info(f"\n{'='*70}\nEvaluating Query Formulation: [{q_name}]\n{'='*70}")

        val_ds = CrossEncoderEvalDataset(val_df["clean_title"], val_df["clean_abstract"], y_val, tokenizer, q_text)
        test_ds = CrossEncoderEvalDataset(test_df["clean_title"], test_df["clean_abstract"], y_test, tokenizer, q_text)
        cohort_ds = CrossEncoderEvalDataset(df_2021["clean_title"], df_2021["clean_abstract"], y_2021, tokenizer, q_text)

        val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
        test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)
        cohort_loader = DataLoader(cohort_ds, batch_size=BATCH_SIZE, shuffle=False)

        logger.info("Running inference on Validation set...")
        probs_val = run_inference(model, val_loader)

        logger.info("Running inference on Held-Out Test set...")
        probs_test = run_inference(model, test_loader)

        logger.info("Running inference on 2021 Prospective Cohort...")
        probs_2021 = run_inference(model, cohort_loader)

        # Store predictions
        col_suffix = "pico" if "PICO" in q_name else "search"
        test_pred_df[f"zero_shot_prob_{col_suffix}"] = probs_test
        cohort_2021_pred_df[f"zero_shot_prob_{col_suffix}"] = probs_2021

        # Determine thresholds
        # 1. Validation calibrated
        th_val_95 = find_threshold_for_recall(y_val, probs_val, 0.95)
        th_val_100 = find_threshold_for_recall(y_val, probs_val, 1.00)

        # 2. Empirical on Test
        th_test_95 = find_threshold_for_recall(y_test, probs_test, 0.95)
        th_test_100 = find_threshold_for_recall(y_test, probs_test, 1.00)

        # 3. Empirical on 2021
        th_2021_95 = find_threshold_for_recall(y_2021, probs_2021, 0.95)
        th_2021_100 = find_threshold_for_recall(y_2021, probs_2021, 1.00)

        logger.info(f"Thresholds for {q_name}: Val>=95%={th_val_95:.4f}, Val 100%={th_val_100:.4f}")

        # Metrics on Held-Out Test Set
        all_records.append(calc_metrics(y_test, probs_test, 0.50, "Held-Out Test Set", "Default (0.50)", q_name))
        all_records.append(calc_metrics(y_test, probs_test, th_val_95, "Held-Out Test Set", f"Val-Calibrated >=95% ({th_val_95:.3f})", q_name))
        all_records.append(calc_metrics(y_test, probs_test, th_val_100, "Held-Out Test Set", f"Val-Calibrated 100% ({th_val_100:.3f})", q_name))
        all_records.append(calc_metrics(y_test, probs_test, th_test_95, "Held-Out Test Set", f"Empirical >=95% ({th_test_95:.3f})", q_name))
        all_records.append(calc_metrics(y_test, probs_test, th_test_100, "Held-Out Test Set", f"Empirical 100% ({th_test_100:.3f})", q_name))

        # Metrics on 2021 Cohort
        all_records.append(calc_metrics(y_2021, probs_2021, 0.50, "2021 Prospective Cohort", "Default (0.50)", q_name))
        all_records.append(calc_metrics(y_2021, probs_2021, th_val_95, "2021 Prospective Cohort", f"Val-Calibrated >=95% ({th_val_95:.3f})", q_name))
        all_records.append(calc_metrics(y_2021, probs_2021, th_val_100, "2021 Prospective Cohort", f"Val-Calibrated 100% ({th_val_100:.3f})", q_name))
        all_records.append(calc_metrics(y_2021, probs_2021, th_2021_95, "2021 Prospective Cohort", f"Empirical >=95% ({th_2021_95:.3f})", q_name))
        all_records.append(calc_metrics(y_2021, probs_2021, th_2021_100, "2021 Prospective Cohort", f"Empirical 100% ({th_2021_100:.3f})", q_name))

    # Save CSVs
    summary_df = pd.DataFrame(all_records)
    summary_csv = RESULTS_DIR / "stage8_zero_shot_summary.csv"
    summary_df.to_csv(summary_csv, index=False)
    test_pred_df.to_csv(RESULTS_DIR / "stage8_zero_shot_test_predictions.csv", index=False)
    cohort_2021_pred_df.to_csv(RESULTS_DIR / "stage8_zero_shot_2021_predictions.csv", index=False)
    logger.info(f"Saved evaluation results to {summary_csv}")

    # Display Printout
    print("\n" + "=" * 115)
    print("STAGE 8 ZERO-SHOT MEDCPT CROSS-ENCODER BENCHMARK SUMMARY")
    print("=" * 115)
    for r in all_records:
        print(f"[{r['query_formulation'][:8]}] {r['dataset'][:17]:<18} | {r['operating_point']:<28} | Th: {r['threshold']:<5.3f} | Rec: {r['recall']:>5.1f}% | Spec: {r['specificity']:>5.1f}% | AUROC: {r['auroc']:<6.4f} | PR-AUC: {r['pr_auc']:<6.4f} | TP: {r['tp']} FP: {r['fp']} TN: {r['tn']} FN: {r['fn']}")
    print("=" * 115 + "\n")

if __name__ == "__main__":
    main()
