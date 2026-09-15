"""
evaluation.py - Comprehensive evaluation metrics and visualization generation.

Computes:
- Confusion Matrix (TN, FP, FN, TP) - numerical and annotated PNG plots
- Accuracy, Precision, Recall / Sensitivity, Specificity, F1-Score
- ROC Curve and AUROC (using continuous probabilities/calibrated decision scores)
- Precision-Recall Curve and Average Precision (PR-AUC)
- Training time and inference latency (ms/sample)
"""

import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend safe for scripts
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

# Safe standard output encoding on Windows terminals
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

logger = logging.getLogger(__name__)


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None,
    train_time_sec: float = 0.0,
    inference_time_ms_per_sample: float = 0.0,
) -> Dict[str, Any]:
    """
    Compute comprehensive clinical classification metrics.
    """
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    accuracy = float(accuracy_score(y_true, y_pred))
    precision = float(precision_score(y_true, y_pred, zero_division=0))
    recall = float(recall_score(y_true, y_pred, zero_division=0))
    sensitivity = recall  # By definition in medical screening
    specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    fpr = float(fp / (tn + fp)) if (tn + fp) > 0 else 0.0
    fnr = float(fn / (tp + fn)) if (tp + fn) > 0 else 0.0

    metrics = {
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
        "total_samples": int(len(y_true)),
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "f1_score": f1,
        "false_positive_rate": fpr,
        "false_negative_rate": fnr,
        "train_time_sec": float(train_time_sec),
        "inference_time_ms_per_sample": float(inference_time_ms_per_sample),
    }

    if y_prob is not None:
        try:
            auroc = float(roc_auc_score(y_true, y_prob))
            avg_precision = float(average_precision_score(y_true, y_prob))
        except Exception as e:
            logger.warning("Could not calculate AUROC/AP: %s", e)
            auroc = float("nan")
            avg_precision = float("nan")
    else:
        auroc = float("nan")
        avg_precision = float("nan")

    metrics["auroc"] = auroc
    metrics["average_precision"] = avg_precision

    return metrics


def plot_confusion_matrix(
    cm: np.ndarray,
    model_display_name: str,
    output_path: str,
    classes: Tuple[str, str] = ("Not Relevant (0)", "Relevant (1)"),
) -> None:
    """Save an annotated publication-quality confusion matrix plot."""
    fig, ax = plt.subplots(figsize=(6, 5))
    cax = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(cax, ax=ax)

    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=classes,
        yticklabels=classes,
        title=f"Confusion Matrix: {model_display_name}",
        ylabel="True Class",
        xlabel="Predicted Class",
    )

    # Annotate counts and percentages
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


def plot_single_roc_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    model_display_name: str,
    output_path: str,
) -> None:
    """Save an individual ROC curve."""
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (AUC = {roc_auc:.3f})")
    ax.plot([0, 1], [0, 1], color="navy", lw=1.5, linestyle="--", label="Random Chance (AUC = 0.500)")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate (1 - Specificity)", fontsize=11)
    ax.set_ylabel("True Positive Rate (Sensitivity)", fontsize=11)
    ax.set_title(f"ROC Curve: {model_display_name}", fontsize=12, fontweight="bold")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_single_pr_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    model_display_name: str,
    output_path: str,
) -> None:
    """Save an individual Precision-Recall curve."""
    prec, rec, _ = precision_recall_curve(y_true, y_prob)
    ap = average_precision_score(y_true, y_prob)
    pos_ratio = np.mean(y_true)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(rec, prec, color="forestgreen", lw=2, label=f"PR curve (AP = {ap:.3f})")
    ax.axhline(
        y=pos_ratio,
        color="crimson",
        linestyle="--",
        lw=1.5,
        label=f"Baseline Prevalance ({pos_ratio:.3f})",
    )
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("Recall (Sensitivity)", fontsize=11)
    ax.set_ylabel("Precision (PPV)", fontsize=11)
    ax.set_title(f"Precision-Recall Curve: {model_display_name}", fontsize=12, fontweight="bold")
    ax.legend(loc="lower left")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_combined_roc_curves(
    curves_data: Dict[str, Tuple[np.ndarray, np.ndarray, float]],
    output_path: str,
) -> None:
    """Plot combined ROC curves comparing all models on one figure."""
    fig, ax = plt.subplots(figsize=(8, 6))

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2"]
    for idx, (name, (fpr, tpr, auroc)) in enumerate(curves_data.items()):
        color = colors[idx % len(colors)]
        ax.plot(fpr, tpr, lw=2, color=color, label=f"{name} (AUC = {auroc:.3f})")

    ax.plot([0, 1], [0, 1], color="black", lw=1.5, linestyle="--", label="Random Chance (AUC = 0.500)")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate (1 - Specificity)", fontsize=12)
    ax.set_ylabel("True Positive Rate (Sensitivity)", fontsize=12)
    ax.set_title("Receiver Operating Characteristic (ROC) Comparison", fontsize=14, fontweight="bold")
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_combined_pr_curves(
    curves_data: Dict[str, Tuple[np.ndarray, np.ndarray, float]],
    pos_prevalence: float,
    output_path: str,
) -> None:
    """Plot combined Precision-Recall curves comparing all models on one figure."""
    fig, ax = plt.subplots(figsize=(8, 6))

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2"]
    for idx, (name, (rec, prec, ap)) in enumerate(curves_data.items()):
        color = colors[idx % len(colors)]
        ax.plot(rec, prec, lw=2, color=color, label=f"{name} (AP = {ap:.3f})")

    ax.axhline(
        y=pos_prevalence,
        color="black",
        linestyle="--",
        lw=1.5,
        label=f"Baseline Prevalence ({pos_prevalence:.3f})",
    )
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("Recall (Sensitivity)", fontsize=12)
    ax.set_ylabel("Precision", fontsize=12)
    ax.set_title("Precision-Recall (PR) Curve Comparison", fontsize=14, fontweight="bold")
    ax.legend(loc="lower left", fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
