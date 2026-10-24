"""
src/stage7/eval_stage7_4_2021_and_test.py - Evaluate ASL Hybrid + 7 Rules (Unweighted)
on both:
1. Held-Out Test Set (data/splits/stage5/test.csv, N=105: 56 Pos, 49 Neg)
2. 2021 Labeled Cohort (data/2021 labeld.csv, N=117: 28 Pos, 89 Neg)

Outputs tables for:
- Default threshold (0.50)
- > 95% Recall threshold
- 100% Recall threshold
"""

import sys
from pathlib import Path
import logging
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    auc,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer

BASE_DIR = Path(__file__).resolve().parent.parent.parent
SPLITS_DIR = BASE_DIR / "data" / "splits" / "stage5"
MODELS_DIR = BASE_DIR / "models" / "stage7_4"
RESULTS_DIR = BASE_DIR / "results" / "stage7_4"

sys.path.insert(0, str(BASE_DIR / "src"))
from preprocessing import clean_clinical_text
from stage3.rule_engine import get_rule_feature_vector

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

# 1. Load Datasets
# A. Held-Out Test Set
test_df = pd.read_csv(SPLITS_DIR / "test.csv")
test_df["clean_title"] = [clean_clinical_text(str(t or "")) for t in test_df["title"]]
test_df["clean_abstract"] = [clean_clinical_text(str(a or "")) for a in test_df["abstract"]]
rule_vecs_test = []
for _, row in test_df.iterrows():
    pts = [t.strip() for t in str(row.get("publication_types", "")).split(";") if t.strip()]
    vec = get_rule_feature_vector(row["clean_title"], row["clean_abstract"], pts)
    rule_vecs_test.append(vec)
test_df["rule_vector"] = rule_vecs_test
y_test = test_df["label"].to_numpy().astype(int)
logger.info(f"Held-Out Test Set: N={len(test_df)} (Pos={y_test.sum()}, Neg={(y_test==0).sum()})")

# B. 2021 Prospective Cohort
df_2021 = pd.read_csv(BASE_DIR / "data" / "2021 labeld.csv")
df_2021["clean_title"] = [clean_clinical_text(str(t or "")) for t in df_2021["title"]]
df_2021["clean_abstract"] = [clean_clinical_text(str(a or "")) for a in df_2021["abstract"]]
rule_vecs_2021 = []
for _, row in df_2021.iterrows():
    pts = [t.strip() for t in str(row.get("publication_types", "")).split(";") if t.strip()]
    vec = get_rule_feature_vector(row["clean_title"], row["clean_abstract"], pts)
    rule_vecs_2021.append(vec)
df_2021["rule_vector"] = rule_vecs_2021
y_2021 = df_2021["label"].to_numpy().astype(int)
logger.info(f"2021 Prospective Cohort: N={len(df_2021)} (Pos={y_2021.sum()}, Neg={(y_2021==0).sum()})")

# 2. Dataset Definition
PRETRAINED_MODEL_NAME = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract"
MAX_LENGTH = 512
PICO_QUERY_NATURAL = (
    "Screening criteria for localized prostate cancer: Primary curative treatment "
    "(radical prostatectomy, EBRT >= 72 Gy, brachytherapy, cryo, HIFU). "
    "Must be a comparative trial (exclude single-arm audits, post-treatment salvage, metastatic, or palliative). "
    "Reports oncologic survival (BRFS, OS, CSS) with follow-up >= 5 years."
)

class PICODataset(Dataset):
    def __init__(self, titles, abstracts, labels, rule_vectors, tokenizer, pico_query=PICO_QUERY_NATURAL, max_length=MAX_LENGTH):
        self.labels = torch.tensor(labels, dtype=torch.float32)
        self.rule_vectors = torch.tensor(np.array(rule_vectors), dtype=torch.float32)
        pico_queries = [pico_query] * len(titles)
        paper_texts = [f"{t}. {a}" if t else a for t, a in zip(titles, abstracts)]

        enc = tokenizer(
            text=pico_queries,
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
            "rule_vector": self.rule_vectors[idx],
            "label": self.labels[idx],
        }

class FlexibleCrossEncoder(nn.Module):
    def __init__(self, pretrained_name=PRETRAINED_MODEL_NAME, rule_dim=7, dropout_rate=0.25):
        super().__init__()
        self.bert = AutoModel.from_pretrained(pretrained_name, local_files_only=True)
        self.dropout = nn.Dropout(dropout_rate)
        self.rule_dim = rule_dim
        hidden_size = self.bert.config.hidden_size  # 768
        total_dim = hidden_size + rule_dim          # 775

        self.classifier = nn.Sequential(
            nn.Linear(total_dim, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 1),
        )

    def forward(self, input_ids, attention_mask, token_type_ids, rule_vector=None):
        bert_out = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        cls_rep = self.dropout(bert_out.last_hidden_state[:, 0, :])
        if self.rule_dim > 0 and rule_vector is not None:
            combined = torch.cat([cls_rep, rule_vector], dim=1)
        else:
            combined = cls_rep
        return self.classifier(combined).squeeze(-1)

# 3. Model Inference
tokenizer = AutoTokenizer.from_pretrained(PRETRAINED_MODEL_NAME, local_files_only=True)

ds_test = PICODataset(test_df["clean_title"].tolist(), test_df["clean_abstract"].tolist(), y_test.tolist(), test_df["rule_vector"].tolist(), tokenizer)
loader_test = DataLoader(ds_test, batch_size=8, shuffle=False)

ds_2021 = PICODataset(df_2021["clean_title"].tolist(), df_2021["clean_abstract"].tolist(), y_2021.tolist(), df_2021["rule_vector"].tolist(), tokenizer)
loader_2021 = DataLoader(ds_2021, batch_size=8, shuffle=False)

model = FlexibleCrossEncoder(rule_dim=7).to(device)
checkpoint_path = MODELS_DIR / "model_7_4_hybrid_unweighted.pt"
logger.info(f"Loading checkpoint: {checkpoint_path}")
model.load_state_dict(torch.load(checkpoint_path, map_location=device))
model.eval()

def predict(loader):
    all_probs = []
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            token_type_ids = batch["token_type_ids"].to(device)
            rule_vector = batch["rule_vector"].to(device)
            logits = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, rule_vector=rule_vector)
            probs = torch.sigmoid(logits).cpu().numpy()
            all_probs.extend(probs.tolist())
    return np.array(all_probs, dtype=float)

logger.info("Running inference on Held-Out Test Set...")
probs_test = predict(loader_test)

logger.info("Running inference on 2021 Prospective Cohort...")
probs_2021 = predict(loader_2021)

# Save predictions
test_df["predicted_prob"] = probs_test
test_df.to_csv(RESULTS_DIR / "stage7_4_test_predictions_detailed.csv", index=False)

df_2021["predicted_prob"] = probs_2021
df_2021.to_csv(RESULTS_DIR / "stage7_4_2021_predictions_detailed.csv", index=False)

# 4. Evaluation Function
def calc_metrics(y_true, y_prob, threshold, dataset_name, operating_point):
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
        "dataset": dataset_name,
        "operating_point": operating_point,
        "threshold": round(float(threshold), 3),
        "recall": rec,
        "specificity": spec,
        "auroc": auroc,
        "pr_auc": pr_auc,
        "precision": prec,
        "f1_score": f1,
        "accuracy": acc,
        "true_positives": int(tp),
        "false_positives": int(fp),
        "true_negatives": int(tn),
        "false_negatives": int(fn),
        "workload_reduction_pct": reduction_pct,
        "nns": nns,
    }

def find_threshold_for_recall(y_true, y_prob, target_recall):
    threshold_grid = np.linspace(0.01, 0.99, 99)
    valid_candidates = []
    for t in threshold_grid:
        y_pred = (y_prob >= t).astype(int)
        r = recall_score(y_true, y_pred, zero_division=0)
        if r >= target_recall:
            valid_candidates.append(t)
    return max(valid_candidates) if valid_candidates else 0.01

# Find dataset-specific empirical thresholds
th_95_test_empirical = find_threshold_for_recall(y_test, probs_test, 0.95)
th_100_test_empirical = find_threshold_for_recall(y_test, probs_test, 1.00)

th_95_2021_empirical = find_threshold_for_recall(y_2021, probs_2021, 0.95)
th_100_2021_empirical = find_threshold_for_recall(y_2021, probs_2021, 1.00)

# Validation calibrated thresholds:
# Default = 0.50
# Val >= 95% = 0.82
# Val 100% = 0.46

# Compute all metrics
records = []

# Table 1: Default (0.50)
records.append(calc_metrics(y_test, probs_test, 0.50, "Held-Out Test Set", "Default (0.50)"))
records.append(calc_metrics(y_2021, probs_2021, 0.50, "2021 Labeled Cohort", "Default (0.50)"))

# Table 2: > 95% Recall
# A: Validation calibrated (0.82)
records.append(calc_metrics(y_test, probs_test, 0.82, "Held-Out Test Set", "Val-Calibrated >=95% (0.82)"))
records.append(calc_metrics(y_2021, probs_2021, 0.82, "2021 Labeled Cohort", "Val-Calibrated >=95% (0.82)"))
# B: Empirical on dataset
records.append(calc_metrics(y_test, probs_test, th_95_test_empirical, "Held-Out Test Set", f"Empirical >=95% ({th_95_test_empirical:.2f})"))
records.append(calc_metrics(y_2021, probs_2021, th_95_2021_empirical, "2021 Labeled Cohort", f"Empirical >=95% ({th_95_2021_empirical:.2f})"))

# Table 3: 100% Recall
# A: Validation calibrated (0.46)
records.append(calc_metrics(y_test, probs_test, 0.46, "Held-Out Test Set", "Val-Calibrated 100% (0.46)"))
records.append(calc_metrics(y_2021, probs_2021, 0.46, "2021 Labeled Cohort", "Val-Calibrated 100% (0.46)"))
# B: Empirical on dataset
records.append(calc_metrics(y_test, probs_test, th_100_test_empirical, "Held-Out Test Set", f"Empirical 100% ({th_100_test_empirical:.2f})"))
records.append(calc_metrics(y_2021, probs_2021, th_100_2021_empirical, "2021 Labeled Cohort", f"Empirical 100% ({th_100_2021_empirical:.2f})"))

all_df = pd.DataFrame(records)
all_df.to_csv(RESULTS_DIR / "stage7_4_external_and_test_eval_summary.csv", index=False)
logger.info("Saved all evaluation results to CSV.")

# Display Printout
print("\n" + "="*95)
print("ASL HYBRID + 7 RULES (UNWEIGHTED) EVALUATION SUMMARY")
print("="*95)
for r in records:
    print(f"{r['dataset']:<22} | {r['operating_point']:<32} | Thresh: {r['threshold']:<5.2f} | Rec: {r['recall']:<7.2%} | Spec: {r['specificity']:<7.2%} | AUROC: {r['auroc']:<6.4f} | TP: {r['true_positives']} FN: {r['false_negatives']} FP: {r['false_positives']} TN: {r['true_negatives']}")
print("="*95 + "\n")
