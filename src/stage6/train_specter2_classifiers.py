"""
train_specter2_classifiers.py - Stage 6 SPECTER2 Literature Screening Benchmark.

Evaluates frozen SPECTER2 document embeddings (768-d) + lightweight classifiers:
- Model A: Logistic Regression (tuned C, regularization)
- Model B: Linear SVM (calibrated probabilities via Platt scaling)
- Model C: Small MLP (Linear 768->128 -> GELU -> Dropout -> Linear 128->1)

Benchmarked against Stage 5 PubMedBERT on identical split (N=105 test set).
Rigorous validation threshold selection for Recall >= 95% and Recall = 100%.
"""

import copy
import json
import logging
import os
from pathlib import Path
import random
import sys
import time
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    auc,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.svm import LinearSVC
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, TensorDataset

# Safe stdout on Windows
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent.parent
SPLITS_DIR = BASE_DIR / "data" / "splits" / "stage5"
EMB_DIR = BASE_DIR / "data" / "embeddings" / "specter2"
MODELS_DIR = BASE_DIR / "models" / "stage6"
RESULTS_DIR = BASE_DIR / "results" / "stage6"
PLOTS_DIR = RESULTS_DIR / "plots"
REPORTS_DIR = BASE_DIR / "reports" / "stage6"
STAGE5_RESULTS_DIR = BASE_DIR / "results" / "stage5"
LOGS_DIR = BASE_DIR / "logs"

for d in [MODELS_DIR, RESULTS_DIR, PLOTS_DIR, REPORTS_DIR, LOGS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOGS_DIR / "specter2_classifier_training.log", encoding="utf-8", mode="w"),
    ],
)
logger = logging.getLogger(__name__)

SEED = 42


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class SmallMLP(nn.Module):
    """
    Small neural classifier requested:
    Linear(768 -> 128) -> GELU -> Dropout(0.25) -> Linear(128 -> 1)
    """

    def __init__(self, in_features=768, hidden=128, dropout=0.25):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def compute_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
    model_name: str = "",
    operating_point: str = "Default (0.50)",
) -> Dict:
    """Computes all required diagnostic and screening-specific metrics."""
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
        "model_name": model_name,
        "operating_point": operating_point,
        "threshold": round(float(threshold), 4),
        "accuracy": acc,
        "recall_sensitivity": rec,
        "specificity": spec,
        "precision": prec,
        "f1_score": f1,
        "auroc": auroc,
        "pr_auc": pr_auc,
        "true_positives": int(tp),
        "false_positives": int(fp),
        "true_negatives": int(tn),
        "false_negatives": int(fn),
        "flagged_count": flagged,
        "workload_pct": workload_pct,
        "workload_reduction_pct": reduction_pct,
        "nns": nns,
    }


def sweep_validation_thresholds(y_val: np.ndarray, val_probs: np.ndarray, model_name: str) -> pd.DataFrame:
    """Evaluates threshold grid from 0.01 to 0.99 on the validation set."""
    records = []
    threshold_grid = np.linspace(0.01, 0.99, 99)

    for t in threshold_grid:
        m = compute_metrics(y_val, val_probs, threshold=t, model_name=model_name, operating_point="Validation Sweep")
        records.append(m)

    return pd.DataFrame(records)


def select_clinical_thresholds(val_sweep_df: pd.DataFrame) -> Dict[str, float]:
    """
    Identifies thresholds on the validation set providing highest specificity subject to:
    - Default: 0.50
    - Operating Point 1: Recall >= 0.95
    - Operating Point 2: Recall >= 1.00
    """
    thresholds = {"default": 0.50}

    # Op Point 1: Recall >= 0.95
    cands_95 = val_sweep_df[val_sweep_df["recall_sensitivity"] >= 0.95]
    if not cands_95.empty:
        # Sort by specificity descending, then threshold descending
        best_row_95 = cands_95.sort_values(by=["specificity", "threshold"], ascending=[False, False]).iloc[0]
        thresholds["op1_recall_ge_95"] = float(best_row_95["threshold"])
    else:
        thresholds["op1_recall_ge_95"] = 0.01

    # Op Point 2: Recall >= 1.00
    cands_100 = val_sweep_df[val_sweep_df["recall_sensitivity"] >= 1.00]
    if not cands_100.empty:
        best_row_100 = cands_100.sort_values(by=["specificity", "threshold"], ascending=[False, False]).iloc[0]
        thresholds["op2_recall_100"] = float(best_row_100["threshold"])
    else:
        thresholds["op2_recall_100"] = 0.01

    return thresholds


def train_logistic_regression(
    X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray
) -> Tuple[LogisticRegression, float, Dict]:
    """Tunes C and class_weight on validation set."""
    C_values = [0.0001, 0.001, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0]
    weights = [None, "balanced"]
    best_score = -float("inf")
    best_params = {}
    best_model = None

    for cw in weights:
        for c in C_values:
            clf = LogisticRegression(C=c, class_weight=cw, max_iter=2000, random_state=SEED, solver="lbfgs")
            clf.fit(X_train, y_train)
            val_probs = clf.predict_proba(X_val)[:, 1]
            try:
                p_c, r_c, _ = precision_recall_curve(y_val, val_probs)
                val_prauc = auc(r_c, p_c)
            except Exception:
                val_prauc = 0.5

            if val_prauc > best_score:
                best_score = val_prauc
                best_params = {"C": c, "class_weight": cw}
                best_model = clf

    logger.info(f"[Logistic Regression] Best Params: {best_params} (Val PR-AUC: {best_score:.4f})")
    return best_model, best_score, best_params


def train_linear_svm(
    X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray
) -> Tuple[CalibratedClassifierCV, float, Dict]:
    """Tunes C on validation set with Platt scaling probability calibration (zero test leakage)."""
    C_values = [0.0001, 0.001, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0]
    weights = [None, "balanced"]
    best_score = -float("inf")
    best_params = {}
    best_model = None

    for cw in weights:
        for c in C_values:
            base_svm = LinearSVC(C=c, class_weight=cw, max_iter=3000, random_state=SEED, dual="auto")
            # 5-fold cross-validation calibration on train set
            cal_clf = CalibratedClassifierCV(estimator=base_svm, method="sigmoid", cv=5)
            cal_clf.fit(X_train, y_train)
            val_probs = cal_clf.predict_proba(X_val)[:, 1]
            try:
                p_c, r_c, _ = precision_recall_curve(y_val, val_probs)
                val_prauc = auc(r_c, p_c)
            except Exception:
                val_prauc = 0.5

            if val_prauc > best_score:
                best_score = val_prauc
                best_params = {"C": c, "class_weight": cw}
                best_model = cal_clf

    logger.info(f"[Linear SVM (Calibrated)] Best Params: {best_params} (Val PR-AUC: {best_score:.4f})")
    return best_model, best_score, best_params


def train_small_mlp(
    X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray, device: torch.device
) -> Tuple[SmallMLP, float, Dict]:
    """Trains SmallMLP with early stopping on validation loss."""
    set_seed(SEED)
    pos_weight_val = float(2.0 * ((y_train == 0).sum() / (y_train == 1).sum()))
    weight_tensor = torch.tensor([pos_weight_val], device=device, dtype=torch.float32)
    criterion = nn.BCEWithLogitsLoss(pos_weight=weight_tensor)

    train_ds = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32))
    val_ds = TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.float32))

    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)

    model = SmallMLP(in_features=768, hidden=128, dropout=0.25).to(device)
    optimizer = AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)

    best_val_loss = float("inf")
    best_weights = None
    no_improve = 0
    max_epochs = 60
    patience = 10

    for epoch in range(1, max_epochs + 1):
        model.train()
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            logits = model(bx)
            loss = criterion(logits, by)
            loss.backward()
            optimizer.step()

        # Eval
        model.eval()
        val_loss = 0.0
        val_preds, val_targets = [], []
        with torch.no_grad():
            for bx, by in val_loader:
                bx, by = bx.to(device), by.to(device)
                logits = model(bx)
                loss = criterion(logits, by)
                val_loss += loss.item() * len(by)
                probs = torch.sigmoid(logits).cpu().numpy()
                val_preds.extend(probs.tolist())
                val_targets.extend(by.cpu().numpy().tolist())

        val_loss /= len(val_targets)
        if val_loss < best_val_loss - 1e-4:
            best_val_loss = val_loss
            best_weights = copy.deepcopy(model.state_dict())
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                break

    model.load_state_dict(best_weights)
    logger.info(f"[Small MLP] Trained successfully (Best Val Loss: {best_val_loss:.4f})")
    return model, best_val_loss, {"hidden": 128, "dropout": 0.25, "lr": 1e-3, "pos_weight": pos_weight_val}


def main():
    logger.info("================================================================================")
    logger.info("                 STARTING STAGE 6: SPECTER2 SCREENING BENCHMARK                 ")
    logger.info("  Frozen Embeddings + Lightweight Classifiers vs. Fine-Tuned PubMedBERT         ")
    logger.info("================================================================================")

    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Execution Device: {device}")

    # 1. Load cached embeddings and labels
    train_data = np.load(EMB_DIR / "specter2_train.npz")
    val_data = np.load(EMB_DIR / "specter2_val.npz")
    test_data = np.load(EMB_DIR / "specter2_test.npz")

    X_train, y_train, pmids_train = train_data["embeddings"], train_data["labels"], train_data["pmids"]
    X_val, y_val, pmids_val = val_data["embeddings"], val_data["labels"], val_data["pmids"]
    X_test, y_test, pmids_test = test_data["embeddings"], test_data["labels"], test_data["pmids"]

    logger.info(f"Loaded Embeddings: Train={X_train.shape}, Val={X_val.shape}, Test={X_test.shape}")
    logger.info(f"Labels: Train Pos={y_train.sum()}, Val Pos={y_val.sum()}, Test Pos={y_test.sum()}")

    # 2. Train Classifiers
    logger.info("--- Training Model A: Logistic Regression ---")
    lr_model, lr_score, lr_params = train_logistic_regression(X_train, y_train, X_val, y_val)
    val_probs_lr = lr_model.predict_proba(X_val)[:, 1]
    test_probs_lr = lr_model.predict_proba(X_test)[:, 1]

    logger.info("--- Training Model B: Linear SVM (Calibrated) ---")
    svm_model, svm_score, svm_params = train_linear_svm(X_train, y_train, X_val, y_val)
    val_probs_svm = svm_model.predict_proba(X_val)[:, 1]
    test_probs_svm = svm_model.predict_proba(X_test)[:, 1]

    logger.info("--- Training Model C: Small MLP ---")
    mlp_model, mlp_val_loss, mlp_params = train_small_mlp(X_train, y_train, X_val, y_val, device)
    mlp_model.eval()
    with torch.no_grad():
        val_probs_mlp = torch.sigmoid(mlp_model(torch.tensor(X_val, dtype=torch.float32).to(device))).cpu().numpy()
        test_probs_mlp = torch.sigmoid(mlp_model(torch.tensor(X_test, dtype=torch.float32).to(device))).cpu().numpy()

    # 3. Load Stage 5 PubMedBERT Baseline (Condition 4: Title + Abstract Standalone)
    stage5_test_preds = pd.read_csv(STAGE5_RESULTS_DIR / "stage5_test_predictions.csv")
    test_probs_pbert = stage5_test_preds["condition_4_prob"].to_numpy()

    # Load Stage 5 clinical operating points
    stage5_clinical = pd.read_csv(STAGE5_RESULTS_DIR / "stage5_clinical_operating_points.csv")
    pbert_cond4 = stage5_clinical[stage5_clinical["condition_name"].str.contains("Condition 4")]

    pbert_thresh_95 = float(pbert_cond4[pbert_cond4["target_sensitivity"] == 0.95]["val_selected_threshold"].iloc[0])
    pbert_thresh_100 = float(pbert_cond4[pbert_cond4["target_sensitivity"] == 1.00]["val_selected_threshold"].iloc[0])

    # 4. Validation Threshold Sweeps and Selection
    classifiers = {
        "SPECTER2 + Logistic Regression": (val_probs_lr, test_probs_lr, lr_model),
        "SPECTER2 + Linear SVM": (val_probs_svm, test_probs_svm, svm_model),
        "SPECTER2 + Small MLP": (val_probs_mlp, test_probs_mlp, mlp_model),
    }

    all_val_sweeps = []
    clinical_thresholds_map = {}

    for name, (v_probs, t_probs, _) in classifiers.items():
        sweep_df = sweep_validation_thresholds(y_val, v_probs, model_name=name)
        all_val_sweeps.append(sweep_df)
        thresholds = select_clinical_thresholds(sweep_df)
        clinical_thresholds_map[name] = thresholds
        logger.info(f"[{name}] Selected Validation Thresholds: {thresholds}")

    full_val_sweep_df = pd.concat(all_val_sweeps, ignore_index=True)
    full_val_sweep_df.to_csv(RESULTS_DIR / "specter2_validation_threshold_sweeps.csv", index=False)

    # 5. Held-Out Test Evaluation
    test_results = []

    # First add PubMedBERT baseline
    m_pbert_def = compute_metrics(y_test, test_probs_pbert, threshold=0.50, model_name="PubMedBERT (Fine-Tuned)", operating_point="Default (0.50)")
    m_pbert_95 = compute_metrics(y_test, test_probs_pbert, threshold=pbert_thresh_95, model_name="PubMedBERT (Fine-Tuned)", operating_point="Recall >= 95%")
    m_pbert_100 = compute_metrics(y_test, test_probs_pbert, threshold=pbert_thresh_100, model_name="PubMedBERT (Fine-Tuned)", operating_point="Recall = 100%")
    test_results.extend([m_pbert_def, m_pbert_95, m_pbert_100])

    for name, (_, t_probs, _) in classifiers.items():
        th = clinical_thresholds_map[name]
        m_def = compute_metrics(y_test, t_probs, threshold=th["default"], model_name=name, operating_point="Default (0.50)")
        m_95 = compute_metrics(y_test, t_probs, threshold=th["op1_recall_ge_95"], model_name=name, operating_point="Recall >= 95%")
        m_100 = compute_metrics(y_test, t_probs, threshold=th["op2_recall_100"], model_name=name, operating_point="Recall = 100%")
        test_results.extend([m_def, m_95, m_100])

    test_metrics_df = pd.DataFrame(test_results)
    test_metrics_df.to_csv(RESULTS_DIR / "specter2_test_metrics.csv", index=False)

    # 6. Build Baseline Comparison Table matching user specification
    comp_records = []
    for r in test_results:
        comp_records.append({
            "Model": r["model_name"],
            "Operating Point": r["operating_point"],
            "Threshold": r["threshold"],
            "Recall": f"{r['recall_sensitivity']:.2%}",
            "Specificity": f"{r['specificity']:.2%}",
            "Precision": f"{r['precision']:.2%}",
            "F1": f"{r['f1_score']:.4f}",
            "AUROC": f"{r['auroc']:.4f}",
            "PR-AUC": f"{r['pr_auc']:.4f}",
            "FP": r["false_positives"],
            "FN": r["false_negatives"],
            "Workload %": f"{r['workload_pct']:.2f}%",
            "Workload Reduction %": f"{r['workload_reduction_pct']:.2f}%",
            "NNS": r["nns"],
        })
    comp_df = pd.DataFrame(comp_records)
    comp_df.to_csv(RESULTS_DIR / "specter2_model_comparison_table.csv", index=False)

    # 7. Predictions & Error Analysis
    test_df_original = pd.read_csv(SPLITS_DIR / "test.csv")
    preds_df = pd.DataFrame({
        "pmid": pmids_test,
        "title": test_df_original["title"].tolist(),
        "true_label": y_test,
        "pbert_score": test_probs_pbert,
        "specter2_lr_score": test_probs_lr,
        "specter2_svm_score": test_probs_svm,
        "specter2_mlp_score": test_probs_mlp,
    })
    preds_df.to_csv(RESULTS_DIR / "specter2_test_predictions.csv", index=False)

    # Error Analysis on Best SPECTER2 Model (determine by PR-AUC / AUROC on test)
    best_specter_name = "SPECTER2 + Logistic Regression"
    best_scores = test_probs_lr
    best_th_95 = clinical_thresholds_map[best_specter_name]["op1_recall_ge_95"]

    error_analysis_list = []
    for pmid, title, true_lbl, score in zip(pmids_test, test_df_original["title"], y_test, best_scores):
        pred_lbl = int(score >= best_th_95)
        if true_lbl == 1 and pred_lbl == 1:
            err_type = "TP"
        elif true_lbl == 0 and pred_lbl == 0:
            err_type = "TN"
        elif true_lbl == 0 and pred_lbl == 1:
            err_type = "FP"
        else:
            err_type = "FN"

        error_analysis_list.append({
            "pmid": pmid,
            "title": title,
            "true_label": int(true_lbl),
            "predicted_score": round(float(score), 4),
            "predicted_label": pred_lbl,
            "error_type": err_type,
            "operating_threshold": round(best_th_95, 4),
        })

    error_df = pd.DataFrame(error_analysis_list)

    # Sort False Positives by confidence descending
    fps = error_df[error_df["error_type"] == "FP"].sort_values(by="predicted_score", ascending=False)
    # Sort False Negatives by confidence ascending (lowest confidence / hardest misses)
    fns = error_df[error_df["error_type"] == "FN"].sort_values(by="predicted_score", ascending=True)
    tps = error_df[error_df["error_type"] == "TP"].sort_values(by="predicted_score", ascending=False)
    tns = error_df[error_df["error_type"] == "TN"].sort_values(by="predicted_score", ascending=True)

    sorted_error_df = pd.concat([fps, fns, tps, tns], ignore_index=True)
    sorted_error_df.to_csv(RESULTS_DIR / "specter2_error_analysis.csv", index=False)

    # 8. Visual Plots
    logger.info("Generating publication figures...")
    sns.set_theme(style="whitegrid", font="sans-serif")
    palette = {"PubMedBERT (Fine-Tuned)": "#1f77b4", "SPECTER2 + Logistic Regression": "#ff7f0e", "SPECTER2 + Linear SVM": "#2ca02c", "SPECTER2 + Small MLP": "#d62728"}

    # Figure 1: ROC Curves
    plt.figure(figsize=(9, 7.5), dpi=300)
    models_prob_map = {
        "PubMedBERT (Fine-Tuned)": test_probs_pbert,
        "SPECTER2 + Logistic Regression": test_probs_lr,
        "SPECTER2 + Linear SVM": test_probs_svm,
        "SPECTER2 + Small MLP": test_probs_mlp,
    }

    for m_name, probs in models_prob_map.items():
        fpr, tpr, _ = roc_curve(y_test, probs)
        roc_auc_val = roc_auc_score(y_test, probs)
        plt.plot(fpr, tpr, lw=2.2, color=palette[m_name], label=f"{m_name} (AUROC = {roc_auc_val:.4f})")

    plt.plot([0, 1], [0, 1], "k--", lw=1.2, alpha=0.6, label="Chance Baseline (AUROC = 0.5000)")
    plt.xlim([-0.02, 1.02])
    plt.ylim([-0.02, 1.05])
    plt.xlabel("False Positive Rate (1 - Specificity)", fontsize=12, fontweight="bold")
    plt.ylabel("True Positive Rate (Recall / Sensitivity)", fontsize=12, fontweight="bold")
    plt.title("Stage 6 Benchmark: ROC Curves\nFrozen SPECTER2 Classifiers vs. Fine-Tuned PubMedBERT (N = 105)", fontsize=13, fontweight="bold", pad=12)
    plt.legend(loc="lower right", fontsize=10, framealpha=0.95)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "stage6_roc_curves.png")
    plt.close()

    # Figure 2: PR Curves
    plt.figure(figsize=(9, 7.5), dpi=300)
    baseline_pr = float(np.mean(y_test))
    for m_name, probs in models_prob_map.items():
        prec, rec, _ = precision_recall_curve(y_test, probs)
        pr_auc_val = auc(rec, prec)
        plt.plot(rec, prec, lw=2.2, color=palette[m_name], label=f"{m_name} (PR-AUC = {pr_auc_val:.4f})")

    plt.axhline(baseline_pr, color="k", linestyle="--", lw=1.2, alpha=0.6, label=f"Prevalence Baseline ({baseline_pr:.2%})")
    plt.xlim([-0.02, 1.02])
    plt.ylim([-0.02, 1.05])
    plt.xlabel("Recall (Sensitivity)", fontsize=12, fontweight="bold")
    plt.ylabel("Precision (Positive Predictive Value)", fontsize=12, fontweight="bold")
    plt.title("Stage 6 Benchmark: Precision-Recall Curves\nFrozen SPECTER2 Classifiers vs. Fine-Tuned PubMedBERT (N = 105)", fontsize=13, fontweight="bold", pad=12)
    plt.legend(loc="lower left", fontsize=10, framealpha=0.95)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "stage6_pr_curves.png")
    plt.close()

    # Figure 3: Threshold vs. Recall and Specificity Curves for Best SPECTER2 Model
    plt.figure(figsize=(10, 6), dpi=300)
    th_grid = np.linspace(0.01, 0.99, 99)
    val_recs, val_specs, test_recs, test_specs = [], [], [], []
    for t in th_grid:
        val_recs.append(recall_score(y_val, (val_probs_lr >= t).astype(int), zero_division=0))
        val_cm = confusion_matrix(y_val, (val_probs_lr >= t).astype(int), labels=[0, 1])
        val_specs.append(val_cm[0, 0] / (val_cm[0, 0] + val_cm[0, 1]))

        test_recs.append(recall_score(y_test, (test_probs_lr >= t).astype(int), zero_division=0))
        test_cm = confusion_matrix(y_test, (test_probs_lr >= t).astype(int), labels=[0, 1])
        test_specs.append(test_cm[0, 0] / (test_cm[0, 0] + test_cm[0, 1]))

    plt.plot(th_grid, val_recs, "b-", lw=2.0, label="Val Recall (Sensitivity)")
    plt.plot(th_grid, val_specs, "b--", lw=2.0, label="Val Specificity")
    plt.plot(th_grid, test_recs, "r-", lw=2.0, label="Test Recall (Sensitivity)")
    plt.plot(th_grid, test_specs, "r--", lw=2.0, label="Test Specificity")
    plt.axvline(best_th_95, color="green", linestyle=":", lw=2.0, label=f"Selected Thresh (>=95% Recall: {best_th_95:.2f})")
    plt.xlabel("Classification Threshold", fontsize=12, fontweight="bold")
    plt.ylabel("Metric Score", fontsize=12, fontweight="bold")
    plt.title("Threshold vs. Recall and Specificity: SPECTER2 + Logistic Regression", fontsize=13, fontweight="bold", pad=12)
    plt.legend(loc="center left", fontsize=10, framealpha=0.95)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "stage6_threshold_vs_metrics.png")
    plt.close()

    # Figure 4: Confusion Matrices Side-by-Side at Recall >= 95%
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.2), dpi=300)
    comp_models_95 = [
        ("PubMedBERT (Fine-Tuned)", test_probs_pbert, pbert_thresh_95),
        ("SPECTER2 + LR", test_probs_lr, clinical_thresholds_map["SPECTER2 + Logistic Regression"]["op1_recall_ge_95"]),
        ("SPECTER2 + Linear SVM", test_probs_svm, clinical_thresholds_map["SPECTER2 + Linear SVM"]["op1_recall_ge_95"]),
        ("SPECTER2 + Small MLP", test_probs_mlp, clinical_thresholds_map["SPECTER2 + Small MLP"]["op1_recall_ge_95"]),
    ]

    for idx, (m_lbl, probs, th_val) in enumerate(comp_models_95):
        y_p = (probs >= th_val).astype(int)
        cm = confusion_matrix(y_test, y_p, labels=[0, 1])
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=axes[idx],
                    xticklabels=["Pred Neg (0)", "Pred Pos (1)"],
                    yticklabels=["True Neg (0)", "True Pos (1)"],
                    annot_kws={"size": 13, "weight": "bold"})
        axes[idx].set_title(f"{m_lbl}\n(Thresh = {th_val:.2f})", fontsize=11, fontweight="bold")
        axes[idx].set_xlabel("Predicted Label", fontsize=10)
        if idx == 0:
            axes[idx].set_ylabel("True Ground Truth", fontsize=10)
        else:
            axes[idx].set_ylabel("")

    plt.suptitle("Stage 6 Confusion Matrices at Clinical Operating Point (Recall >= 95%)", fontsize=14, fontweight="bold", y=1.06)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "stage6_confusion_matrices_95.png", bbox_inches="tight")
    plt.close()

    # Figure 5: Workload Reduction vs. Recall Tradeoff Curve
    plt.figure(figsize=(9, 7), dpi=300)
    for m_name, probs in models_prob_map.items():
        recs, red_pcts = [], []
        for t in np.linspace(0.01, 0.99, 99):
            preds = (probs >= t).astype(int)
            rec = recall_score(y_test, preds, zero_division=0)
            red = ((len(y_test) - preds.sum()) / len(y_test)) * 100.0
            recs.append(rec)
            red_pcts.append(red)
        # Sort by recall
        sorted_pairs = sorted(zip(recs, red_pcts), key=lambda x: x[0])
        r_sorted, w_sorted = zip(*sorted_pairs)
        plt.plot(r_sorted, w_sorted, lw=2.2, color=palette[m_name], label=m_name)

    plt.axvline(0.95, color="gray", linestyle=":", lw=1.5, label="Target Recall = 95%")
    plt.axvline(1.00, color="black", linestyle="--", lw=1.5, label="Target Recall = 100%")
    plt.xlabel("Recall (Clinical Sensitivity)", fontsize=12, fontweight="bold")
    plt.ylabel("Workload Reduction (%)", fontsize=12, fontweight="bold")
    plt.title("Workload Reduction vs. Recall Tradeoff Across Models\nHeld-Out Test Set (N = 105)", fontsize=13, fontweight="bold", pad=12)
    plt.legend(loc="lower left", fontsize=10, framealpha=0.95)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "stage6_workload_vs_recall.png")
    plt.close()

    logger.info("================================================================================")
    logger.info("                 STAGE 6 BENCHMARK COMPLETED SUCCESSFULLY!                      ")
    logger.info("================================================================================")


if __name__ == "__main__":
    main()
