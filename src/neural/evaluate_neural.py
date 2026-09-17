"""
evaluate_neural.py - Evaluation, confusion matrices, and publication-quality diagnostic curves for Stage 2.

Functions:
- compute_neural_metrics: Detailed classification metrics including TN, FP, FN, TP, AUROC, AP, sensitivity, specificity.
- plot_training_curves: Loss & validation PR-AUC curves per epoch.
- plot_neural_confusion_matrix: Annotated confusion matrix.
- plot_neural_roc_curve & plot_neural_pr_curve: Individual diagnostic curves.
- plot_combined_neural_roc & plot_combined_neural_pr: Comparative multi-model curves.
"""

import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    auc,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

logger = logging.getLogger(__name__)


def compute_neural_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    model_name: str,
    family: str,
    train_time_sec: float = 0.0,
    inference_latency_ms: float = 0.0,
    param_count: int = 0,
    threshold: float = 0.5,
) -> Dict[str, Any]:
    """Compute comprehensive test metrics for a neural model."""
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    y_pred = (y_prob >= threshold).astype(int)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    acc = float(accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    sens = rec
    spec = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    f1 = float(f1_score(y_true, y_pred, zero_division=0))

    try:
        auroc = float(roc_auc_score(y_true, y_prob))
        avg_prec = float(average_precision_score(y_true, y_prob))
    except Exception as e:
        logger.warning("Could not calculate AUROC/AP for %s: %s", model_name, e)
        auroc = float("nan")
        avg_prec = float("nan")

    return {
        "model_name": model_name,
        "family": family,
        "parameters": int(param_count),
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall_sensitivity": round(sens, 4),
        "specificity": round(spec, 4),
        "f1_score": round(f1, 4),
        "auroc": round(auroc, 4),
        "average_precision": round(avg_prec, 4),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
        "train_time_sec": round(float(train_time_sec), 3),
        "inference_latency_ms": round(float(inference_latency_ms), 2),
    }


def plot_training_curves(history_df: pd.DataFrame, model_display_name: str, output_path: str):
    """Plot training/validation loss and validation PR-AUC curves across epochs."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    epochs = history_df["epoch"]

    # Loss curve
    ax1.plot(epochs, history_df["train_loss"], label="Train Loss", color="#1f77b4", lw=2)
    ax1.plot(epochs, history_df["val_loss"], label="Val Loss", color="#ff7f0e", lw=2, linestyle="--")
    ax1.set_title(f"Loss Curves: {model_display_name}", fontsize=11, fontweight="bold")
    ax1.set_xlabel("Epoch", fontsize=10)
    ax1.set_ylabel("Weighted BCE Loss", fontsize=10)
    ax1.legend(loc="upper right")
    ax1.grid(True, alpha=0.3)

    # Validation PR-AUC and AUROC curves
    ax2.plot(epochs, history_df["val_ap"], label="Val PR-AUC (AP)", color="#2ca02c", lw=2)
    ax2.plot(epochs, history_df["val_auc"], label="Val AUROC", color="#d62728", lw=1.5, linestyle=":")
    ax2.plot(epochs, history_df["val_f1"], label="Val F1", color="#9467bd", lw=1.5, linestyle="-.")
    ax2.set_title(f"Validation Metrics: {model_display_name}", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Epoch", fontsize=10)
    ax2.set_ylabel("Metric Score", fontsize=10)
    ax2.set_ylim([0.0, 1.05])
    ax2.legend(loc="lower right")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_neural_confusion_matrix(cm: np.ndarray, model_display_name: str, output_path: str):
    """Save publication-quality annotated confusion matrix plot."""
    fig, ax = plt.subplots(figsize=(6, 5))
    cax = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Greens)
    ax.figure.colorbar(cax, ax=ax)

    classes = ("Not Relevant (0)", "Relevant (1)")
    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=classes,
        yticklabels=classes,
        title=f"Confusion Matrix: {model_display_name}",
        ylabel="True Class",
        xlabel="Predicted Class",
    )

    total = np.sum(cm)
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            val = cm[i, j]
            pct = (val / total) * 100
            ax.text(
                j,
                i,
                f"{val}\n({pct:.1f}%)",
                ha="center",
                va="center",
                color="white" if val > thresh else "black",
                fontsize=11,
                fontweight="bold",
            )

    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_combined_roc(curves_dict: Dict[str, Tuple[np.ndarray, np.ndarray, float]], output_path: str, title: str):
    """Plot multi-model ROC curves on a single canvas."""
    fig, ax = plt.subplots(figsize=(8.5, 6))
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22"]

    for idx, (name, (fpr, tpr, auroc)) in enumerate(curves_dict.items()):
        color = colors[idx % len(colors)]
        ax.plot(fpr, tpr, lw=2, color=color, label=f"{name} (AUC = {auroc:.3f})")

    ax.plot([0, 1], [0, 1], color="black", lw=1.5, linestyle="--", label="Random Chance (AUC = 0.500)")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate (1 - Specificity)", fontsize=11)
    ax.set_ylabel("True Positive Rate (Sensitivity)", fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_combined_pr(curves_dict: Dict[str, Tuple[np.ndarray, np.ndarray, float]], pos_prevalence: float, output_path: str, title: str):
    """Plot multi-model Precision-Recall curves on a single canvas."""
    fig, ax = plt.subplots(figsize=(8.5, 6))
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22"]

    for idx, (name, (rec, prec, ap)) in enumerate(curves_dict.items()):
        color = colors[idx % len(colors)]
        ax.plot(rec, prec, lw=2, color=color, label=f"{name} (AP = {ap:.3f})")

    ax.axhline(y=pos_prevalence, color="black", linestyle="--", lw=1.5, label=f"Baseline Prevalence ({pos_prevalence:.3f})")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("Recall (Sensitivity)", fontsize=11)
    ax.set_ylabel("Precision", fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.legend(loc="lower left", fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
