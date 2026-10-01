"""
evaluate_2021_dataset.py - Independent External Validation of Stage 4 PubMedBERT Models on 2021 labeld.csv.

Dataset: data/2021 labeld.csv (N = 117, 28 Positive, 89 Negative).
No model training or fine-tuning is performed.
Strictly pure inference at fixed threshold 0.50 for direct comparability with Stage 4 test sets.

Evaluates:
1. Stage 4 Original Model (Seed 42, Weighted)
2. Stage 4 Original Model (Seed 42, Unweighted)
3. Stage 4 Multi-Seed Model (Seed 101, Weighted)
4. Stage 4 Multi-Seed Model (Seed 123, Weighted)
5. Stage 4 Multi-Seed Model (Seed 456, Weighted)
6. Stage 4 Multi-Seed Model (Seed 789, Weighted)
7. Stage 4 Multi-Seed Model (Seed 2024, Weighted)
8. Multi-Seed Ensemble (Mean probability across the 5 independent seeds)
"""

import copy
import gc
import json
import logging
import os
from pathlib import Path
import shutil
import sys
import time
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
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
    roc_curve,
)
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

# Safe stdout on Windows
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"

BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(BASE_DIR / "src"))
from preprocessing import clean_clinical_text

DATASET_PATH = BASE_DIR / "data" / "2021 labeld.csv"
OUTPUT_DIR = BASE_DIR / "results" / "eval_2021_dataset"
REPORTS_DIR = BASE_DIR / "reports" / "eval_2021_dataset"
LOGS_DIR = BASE_DIR / "logs"

for d in [OUTPUT_DIR, REPORTS_DIR, LOGS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOGS_DIR / "eval_2021_dataset.log", encoding="utf-8", mode="w"),
    ],
)
logger = logging.getLogger(__name__)

PRETRAINED_MODEL_NAME = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract"
MAX_LENGTH = 384
BATCH_SIZE = 16


class DualInputPubMedBERT(nn.Module):
    """
    PubMedBERT for Title + Abstract dual-sequence input:
    [CLS] Title [SEP] Abstract [SEP]
    Architecture matching Stage 4 exactly:
    PubMedBERT -> CLS -> Dropout(0.25) -> Linear(768, 128) -> LayerNorm -> GELU -> Dropout(0.25) -> Linear(128, 1)
    """

    def __init__(
        self,
        pretrained_name: str = PRETRAINED_MODEL_NAME,
        dropout_rate: float = 0.25,
    ):
        super().__init__()
        self.bert = AutoModel.from_pretrained(pretrained_name, local_files_only=True)
        self.dropout = nn.Dropout(dropout_rate)
        hidden_size = self.bert.config.hidden_size  # 768

        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 1),
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        bert_out = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        cls_rep = bert_out.last_hidden_state[:, 0, :]
        cls_rep = self.dropout(cls_rep)
        logits = self.classifier(cls_rep).squeeze(-1)
        return logits


def load_model(checkpoint_path: Path, device: torch.device) -> DualInputPubMedBERT:
    model = DualInputPubMedBERT()
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def run_inference(
    model: nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    token_type_ids: torch.Tensor,
    batch_size: int = BATCH_SIZE,
    device: torch.device = torch.device("cuda"),
) -> np.ndarray:
    model.eval()
    all_probs = []
    n_samples = input_ids.shape[0]

    with torch.no_grad():
        for i in range(0, n_samples, batch_size):
            b_ids = input_ids[i : i + batch_size].to(device)
            b_mask = attention_mask[i : i + batch_size].to(device)
            b_type = token_type_ids[i : i + batch_size].to(device)

            logits = model(b_ids, b_mask, b_type)
            probs = torch.sigmoid(logits).cpu().numpy()
            all_probs.extend(probs.tolist())

    return np.array(all_probs, dtype=float)


def compute_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.50,
) -> Tuple[Dict, Dict]:
    y_pred = (y_prob >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    rec = recall_score(y_true, y_pred, zero_division=0)
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    auroc = roc_auc_score(y_true, y_prob)
    prec_arr, rec_arr, _ = precision_recall_curve(y_true, y_prob)
    pr_auc = auc(rec_arr, prec_arr)

    total = len(y_true)
    workload_reduction = (tn / total) * 100.0
    nns = (tp + fp) / tp if tp > 0 else float("inf")

    metrics = {
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
        "recall_sensitivity": round(float(rec), 4),
        "specificity": round(float(spec), 4),
        "accuracy": round(float(acc), 4),
        "precision": round(float(prec), 4),
        "f1_score": round(float(f1), 4),
        "auroc": round(float(auroc), 4),
        "pr_auc": round(float(pr_auc), 4),
        "workload_reduction_pct": round(float(workload_reduction), 2),
        "nns": round(float(nns), 2),
    }

    fpr, tpr, _ = roc_curve(y_true, y_prob)
    curve_data = {
        "fpr": fpr,
        "tpr": tpr,
        "rec_arr": rec_arr,
        "prec_arr": prec_arr,
        "auroc": auroc,
        "pr_auc": pr_auc,
    }

    return metrics, curve_data


def main():
    start_time = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"=== External Evaluation on 2021 labeld.csv ===")
    logger.info(f"Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    # 1. Load dataset
    df = pd.read_csv(DATASET_PATH)
    logger.info(f"Loaded dataset: {len(df)} samples (Pos: {(df['label'] == 1).sum()}, Neg: {(df['label'] == 0).sum()})")

    # 2. Tokenize
    logger.info("Tokenizing Title + Abstract text...")
    tokenizer = AutoTokenizer.from_pretrained(PRETRAINED_MODEL_NAME, local_files_only=True)
    titles = [clean_clinical_text(str(t or "")) for t in df["title"]]
    abstracts = [clean_clinical_text(str(a or "")) for a in df["abstract"]]
    y_true = df["label"].values.astype(int)

    enc = tokenizer(
        text=titles,
        text_pair=abstracts,
        max_length=MAX_LENGTH,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    input_ids = enc["input_ids"]
    attention_mask = enc["attention_mask"]
    token_type_ids = enc.get("token_type_ids", torch.zeros_like(input_ids))

    # 3. Model definitions
    models_to_eval = [
        ("Stage 4 Original (Seed 42, Weighted)", BASE_DIR / "models" / "stage4" / "pubmedbert_stage4_weighted.pt"),
        ("Stage 4 Original (Seed 42, Unweighted)", BASE_DIR / "models" / "stage4" / "pubmedbert_stage4_unweighted.pt"),
        ("Stage 4 Multi-Seed (Seed 101, Weighted)", BASE_DIR / "models" / "stage4_multi_seed" / "pubmedbert_weighted_seed_101.pt"),
        ("Stage 4 Multi-Seed (Seed 123, Weighted)", BASE_DIR / "models" / "stage4_multi_seed" / "pubmedbert_weighted_seed_123.pt"),
        ("Stage 4 Multi-Seed (Seed 456, Weighted)", BASE_DIR / "models" / "stage4_multi_seed" / "pubmedbert_weighted_seed_456.pt"),
        ("Stage 4 Multi-Seed (Seed 789, Weighted)", BASE_DIR / "models" / "stage4_multi_seed" / "pubmedbert_weighted_seed_789.pt"),
        ("Stage 4 Multi-Seed (Seed 2024, Weighted)", BASE_DIR / "models" / "stage4_multi_seed" / "pubmedbert_weighted_seed_2024.pt"),
    ]

    all_results = []
    all_curves = {}
    model_probs = {}

    for model_name, ckpt_path in models_to_eval:
        logger.info(f"Evaluating: {model_name}...")
        model = load_model(ckpt_path, device)
        y_prob = run_inference(model, input_ids, attention_mask, token_type_ids, device=device)
        model_probs[model_name] = y_prob

        metrics, curve = compute_metrics(y_true, y_prob, threshold=0.50)
        metrics["model_name"] = model_name
        metrics["checkpoint"] = ckpt_path.name
        all_results.append(metrics)
        all_curves[model_name] = curve

        logger.info(
            f"  -> Recall={metrics['recall_sensitivity']*100:.2f}%, Spec={metrics['specificity']*100:.2f}%, "
            f"Acc={metrics['accuracy']*100:.2f}%, F1={metrics['f1_score']:.4f}, AUROC={metrics['auroc']:.4f}, "
            f"PR-AUC={metrics['pr_auc']:.4f} (TP={metrics['tp']}, FP={metrics['fp']}, TN={metrics['tn']}, FN={metrics['fn']})"
        )

        # Save individual prediction CSV
        clean_name = model_name.replace(" ", "_").replace("(", "").replace(")", "").replace(",", "").lower()
        pred_df = df.copy()
        pred_df["predicted_probability"] = y_prob
        pred_df["predicted_label"] = (y_prob >= 0.50).astype(int)
        pred_df.to_csv(OUTPUT_DIR / f"predictions_{clean_name}.csv", index=False)

        del model
        torch.cuda.empty_cache()
        gc.collect()

    # 4. Multi-Seed Ensemble (Average of the 5 multi-seed models: 101, 123, 456, 789, 2024)
    multi_seed_names = [
        "Stage 4 Multi-Seed (Seed 101, Weighted)",
        "Stage 4 Multi-Seed (Seed 123, Weighted)",
        "Stage 4 Multi-Seed (Seed 456, Weighted)",
        "Stage 4 Multi-Seed (Seed 789, Weighted)",
        "Stage 4 Multi-Seed (Seed 2024, Weighted)",
    ]
    ensemble_prob = np.mean([model_probs[name] for name in multi_seed_names], axis=0)
    ens_metrics, ens_curve = compute_metrics(y_true, ensemble_prob, threshold=0.50)
    ens_metrics["model_name"] = "Multi-Seed 5-Model Ensemble (Mean Prob)"
    ens_metrics["checkpoint"] = "ensemble_5seeds"
    all_results.append(ens_metrics)
    all_curves["Multi-Seed 5-Model Ensemble"] = ens_curve
    model_probs["Multi-Seed 5-Model Ensemble"] = ensemble_prob

    ens_pred_df = df.copy()
    ens_pred_df["predicted_probability"] = ensemble_prob
    ens_pred_df["predicted_label"] = (ensemble_prob >= 0.50).astype(int)
    ens_pred_df.to_csv(OUTPUT_DIR / "predictions_multi_seed_ensemble.csv", index=False)

    logger.info(
        f"Ensemble (5 Seeds): Recall={ens_metrics['recall_sensitivity']*100:.2f}%, Spec={ens_metrics['specificity']*100:.2f}%, "
        f"Acc={ens_metrics['accuracy']*100:.2f}%, F1={ens_metrics['f1_score']:.4f}, AUROC={ens_metrics['auroc']:.4f}, "
        f"PR-AUC={ens_metrics['pr_auc']:.4f} (TP={ens_metrics['tp']}, FP={ens_metrics['fp']}, TN={ens_metrics['tn']}, FN={ens_metrics['fn']})"
    )

    # 5. Compute Statistics Across the 5 Multi-Seed Models
    res_df = pd.DataFrame(all_results)
    ms_df = res_df[res_df["model_name"].isin(multi_seed_names)]

    metric_cols = ["recall_sensitivity", "specificity", "accuracy", "precision", "f1_score", "auroc", "pr_auc", "workload_reduction_pct", "nns"]
    ms_means = ms_df[metric_cols].mean()
    ms_stds = ms_df[metric_cols].std(ddof=1)

    mean_row = {"model_name": "5-Seed Multi-Seed Mean", "checkpoint": "—"}
    std_row = {"model_name": "5-Seed Multi-Seed Std (±)", "checkpoint": "—"}
    for col in metric_cols:
        mean_row[col] = round(ms_means[col], 4)
        std_row[col] = round(ms_stds[col], 4)
    for c in ["tp", "fp", "tn", "fn"]:
        mean_row[c] = round(ms_df[c].mean(), 2)
        std_row[c] = round(ms_df[c].std(ddof=1), 2)

    final_summary_df = pd.concat([res_df, pd.DataFrame([mean_row]), pd.DataFrame([std_row])], ignore_index=True)
    final_summary_df.to_csv(OUTPUT_DIR / "eval_2021_summary_metrics.csv", index=False)
    logger.info("Saved eval_2021_summary_metrics.csv")

    # 6. Plot ROC and PR Curves
    colors = ["#1f77b4", "#aec7e8", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#000000"]

    # ROC Curves
    plt.figure(figsize=(9, 7), dpi=300)
    for i, (name, c) in enumerate(all_curves.items()):
        lw = 2.5 if "Ensemble" in name or "Seed 42, Weighted" in name else 1.5
        ls = "-" if "Ensemble" in name else "--"
        plt.plot(c["fpr"], c["tpr"], label=f"{name} (AUROC = {c['auroc']:.4f})", color=colors[i % len(colors)], linewidth=lw, linestyle=ls)
    plt.plot([0, 1], [0, 1], "k:", alpha=0.5, label="Random Chance (AUROC = 0.500)")
    plt.title(f"External Validation on 2021 Cohort (N = 117): ROC Curves", fontsize=12, fontweight="bold")
    plt.xlabel("False Positive Rate (1 - Specificity)", fontsize=11)
    plt.ylabel("True Positive Rate (Sensitivity / Recall)", fontsize=11)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(loc="lower right", fontsize=8)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "eval_2021_roc_curves.png")
    plt.close()

    # PR Curves
    plt.figure(figsize=(9, 7), dpi=300)
    for i, (name, c) in enumerate(all_curves.items()):
        lw = 2.5 if "Ensemble" in name or "Seed 42, Weighted" in name else 1.5
        ls = "-" if "Ensemble" in name else "--"
        plt.plot(c["rec_arr"], c["prec_arr"], label=f"{name} (PR-AUC = {c['pr_auc']:.4f})", color=colors[i % len(colors)], linewidth=lw, linestyle=ls)
    baseline_prev = 28.0 / 117.0
    plt.axhline(y=baseline_prev, color="black", linestyle=":", alpha=0.7, label=f"2021 Cohort Prevalence ({baseline_prev:.3f})")
    plt.title(f"External Validation on 2021 Cohort (N = 117): Precision-Recall Curves", fontsize=12, fontweight="bold")
    plt.xlabel("Recall (Sensitivity)", fontsize=11)
    plt.ylabel("Precision", fontsize=11)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(loc="lower left", fontsize=8)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "eval_2021_pr_curves.png")
    plt.close()

    # Copy plots to brain directory if exists
    brain_fig_dir = Path("C:/Users/mmahd/.gemini/antigravity/brain/97473cd2-23f6-466c-bee7-b37bda871005/figures")
    if brain_fig_dir.exists():
        shutil.copy(OUTPUT_DIR / "eval_2021_roc_curves.png", brain_fig_dir / "eval_2021_roc_curves.png")
        shutil.copy(OUTPUT_DIR / "eval_2021_pr_curves.png", brain_fig_dir / "eval_2021_pr_curves.png")

    # 7. Generate Comprehensive Markdown Report
    lines = []
    lines.append("# External Validation Report: Stage 4 PubMedBERT Models on `2021 labeld.csv`")
    lines.append(f"**Date:** {time.strftime('%Y-%m-%d %H:%M:%S')}  ")
    lines.append(f"**Dataset:** `data/2021 labeld.csv` (N = 117, 28 Positive [23.93%], 89 Negative [76.07%])  ")
    lines.append(f"**Architecture:** `DualInputPubMedBERT` (Title + Abstract ONLY, max_length = 384 tokens)  ")
    lines.append(f"**Evaluation Mode:** Pure inference (Zero training, zero fine-tuning on new data, fixed threshold = 0.50)  ")
    lines.append(f"**Compute Device:** {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}  ")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## 1. Master Performance Comparison Table (Threshold = 0.50)")
    lines.append("")
    lines.append("| Model Name | TP | FP | TN | FN | Recall | Specificity | Accuracy | Precision | F1-Score | AUROC | PR-AUC | Workload Red. % | NNS |")
    lines.append("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")
    for r in all_results:
        lines.append(
            f"| **{r['model_name']}** | {r['tp']} | {r['fp']} | {r['tn']} | {r['fn']} | "
            f"{r['recall_sensitivity']*100:.2f}% | {r['specificity']*100:.2f}% | {r['accuracy']*100:.2f}% | "
            f"{r['precision']*100:.2f}% | {r['f1_score']:.4f} | {r['auroc']:.4f} | {r['pr_auc']:.4f} | "
            f"{r['workload_reduction_pct']:.2f}% | {r['nns']:.2f} |"
        )
    lines.append(
        f"| **5-Seed Multi-Seed Mean** | {mean_row['tp']:.1f} | {mean_row['fp']:.1f} | {mean_row['tn']:.1f} | {mean_row['fn']:.1f} | "
        f"**{mean_row['recall_sensitivity']*100:.2f}%** | **{mean_row['specificity']*100:.2f}%** | **{mean_row['accuracy']*100:.2f}%** | "
        f"**{mean_row['precision']*100:.2f}%** | **{mean_row['f1_score']:.4f}** | **{mean_row['auroc']:.4f}** | **{mean_row['pr_auc']:.4f}** | "
        f"**{mean_row['workload_reduction_pct']:.2f}%** | **{mean_row['nns']:.2f}** |"
    )
    lines.append(
        f"| **5-Seed Multi-Seed Std (±)** | ±{std_row['tp']:.1f} | ±{std_row['fp']:.1f} | ±{std_row['tn']:.1f} | ±{std_row['fn']:.1f} | "
        f"**±{std_row['recall_sensitivity']*100:.2f}%** | **±{std_row['specificity']*100:.2f}%** | **±{std_row['accuracy']*100:.2f}%** | "
        f"**±{std_row['precision']*100:.2f}%** | **±{std_row['f1_score']:.4f}** | **±{std_row['auroc']:.4f}** | **±{std_row['pr_auc']:.4f}** | "
        f"**±{std_row['workload_reduction_pct']:.2f}%** | **±{std_row['nns']:.2f}** |"
    )
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## 2. Key Observations & Comparison with Stage 4 Test Cohort")
    lines.append("")
    lines.append("1. **Generalization to Unseen 2021 Data:** Evaluates completely unseen papers from a distinct prospective year group.")
    lines.append("2. **Comparison with Stage 4 Test Set:** Compare how the models hold up when tested outside their training cohort.")
    lines.append("")

    with open(REPORTS_DIR / "eval_2021_report.md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    logger.info("Report written successfully to reports/eval_2021_dataset/eval_2021_report.md")
    print("\n" + "=" * 90)
    print("EXTERNAL VALIDATION ON 2021 COHORT (N = 117)")
    print("=" * 90)
    for r in all_results:
        print(f"{r['model_name']:45s} | Rec: {r['recall_sensitivity']*100:6.2f}% | Spec: {r['specificity']*100:6.2f}% | F1: {r['f1_score']:.4f} | AUROC: {r['auroc']:.4f} | TP: {r['tp']:2d} | FN: {r['fn']:2d} | FP: {r['fp']:2d} | TN: {r['tn']:2d}")


if __name__ == "__main__":
    main()
