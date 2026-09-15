"""
threshold_analysis.py - Clinical classification threshold analysis & screening workload optimization.

Investigates decision operating points for literature screening:
1. Evaluates fixed thresholds [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9] and fine sweep [0.02 - 0.98].
2. Computes: Accuracy, Precision, Recall/Sensitivity, Specificity, F1, FPR, FNR,
   Manual Review Workload (articles flagged), and Review Burden.
3. Finds specific clinical operating points targeting high sensitivity:
   - 90% Sensitivity
   - 95% Sensitivity
   - 99% Sensitivity
   - 100% Sensitivity (if achievable)
4. Quantifies the tradeoff: workload reduction vs. false positives and risk of missed relevant literature.
5. Saves CSV tables and publication-quality diagnostic plots.
"""

import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

# Safe standard output encoding on Windows terminals
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

logger = logging.getLogger(__name__)


def sweep_thresholds(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    thresholds: Optional[List[float]] = None,
    model_name: str = "Model",
) -> pd.DataFrame:
    """
    Evaluate performance metrics across a spectrum of decision thresholds.
    """
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    total_samples = len(y_true)
    total_positives = int(np.sum(y_true))
    total_negatives = total_samples - total_positives

    if thresholds is None:
        # Standard sweep + explicit decimals
        std_thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        fine_thresholds = np.linspace(0.02, 0.98, 49).round(3).tolist()
        thresholds = sorted(list(set(std_thresholds + fine_thresholds)))

    records = []
    for thresh in thresholds:
        y_pred = (y_prob >= thresh).astype(int)
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()

        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec = recall_score(y_true, y_pred, zero_division=0)
        sens = rec
        spec = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
        f1 = f1_score(y_true, y_pred, zero_division=0)
        fpr = float(fp / (tn + fp)) if (tn + fp) > 0 else 0.0
        fnr = float(fn / (tp + fn)) if (tp + fn) > 0 else 0.0

        flagged_for_review = tp + fp
        review_workload_pct = (flagged_for_review / total_samples) * 100.0
        workload_reduction_pct = ((total_samples - flagged_for_review) / total_samples) * 100.0

        records.append({
            "model": model_name,
            "threshold": float(thresh),
            "accuracy": float(acc),
            "precision": float(prec),
            "recall": float(rec),
            "sensitivity": float(sens),
            "specificity": float(spec),
            "f1_score": float(f1),
            "false_positive_rate": float(fpr),
            "false_negative_rate": float(fnr),
            "true_positives": int(tp),
            "false_positives": int(fp),
            "true_negatives": int(tn),
            "false_negatives": int(fn),
            "articles_flagged_for_review": int(flagged_for_review),
            "review_workload_pct": float(review_workload_pct),
            "workload_reduction_pct": float(workload_reduction_pct),
            "total_positives": total_positives,
            "total_negatives": total_negatives,
            "total_samples": total_samples,
        })

    return pd.DataFrame(records)


def find_high_sensitivity_operating_points(
    sweep_df: pd.DataFrame,
    target_sensitivities: Optional[List[float]] = None,
) -> pd.DataFrame:
    """
    Find optimal operating thresholds achieving >= target sensitivity.
    Chooses the highest threshold that satisfies the sensitivity requirement
    to maximize specificity and minimize unnecessary manual review (false positives).
    """
    if target_sensitivities is None:
        target_sensitivities = [0.90, 0.95, 0.99, 1.00]

    results = []
    for target in target_sensitivities:
        # Filter rows satisfying sensitivity requirement
        candidates = sweep_df[sweep_df["sensitivity"] >= target]
        if not candidates.empty:
            # Choose row with highest threshold (maximizes specificity, minimizes workload)
            best_row = candidates.sort_values(
                by=["threshold", "specificity", "f1_score"], ascending=[False, False, False]
            ).iloc[0].to_dict()
            best_row["target_sensitivity"] = target
            best_row["status"] = "ACHIEVED"
            results.append(best_row)
        else:
            # Fallback: take row with maximum sensitivity available
            best_row = sweep_df.sort_values(by="sensitivity", ascending=False).iloc[0].to_dict()
            best_row["target_sensitivity"] = target
            best_row["status"] = f"NOT_REACHABLE (Max: {best_row['sensitivity']:.3f})"
            results.append(best_row)

    return pd.DataFrame(results)


def plot_threshold_metrics(
    sweep_df: pd.DataFrame,
    model_display_name: str,
    output_path: str,
) -> None:
    """Plot Sensitivity, Specificity, Precision, and F1 across thresholds."""
    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(sweep_df["threshold"], sweep_df["sensitivity"], label="Sensitivity / Recall", color="#1f77b4", lw=2.5)
    ax.plot(sweep_df["threshold"], sweep_df["specificity"], label="Specificity", color="#2ca02c", lw=2.5)
    ax.plot(sweep_df["threshold"], sweep_df["precision"], label="Precision", color="#ff7f0e", lw=2)
    ax.plot(sweep_df["threshold"], sweep_df["f1_score"], label="F1-Score", color="#d62728", lw=2, linestyle="--")

    ax.axvline(x=0.5, color="gray", linestyle=":", alpha=0.7, label="Default Threshold (0.5)")

    ax.set_title(f"Performance Metrics vs. Decision Threshold: {model_display_name}", fontsize=12, fontweight="bold")
    ax.set_xlabel("Classification Decision Threshold", fontsize=11)
    ax.set_ylabel("Metric Value", fontsize=11)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.legend(loc="best", fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_sensitivity_workload_tradeoff(
    sweep_df: pd.DataFrame,
    model_display_name: str,
    output_path: str,
) -> None:
    """Plot Sensitivity vs. Screening Review Workload (% of articles flagged)."""
    fig, ax1 = plt.subplots(figsize=(8, 5))

    color_workload = "#ff7f0e"
    color_sens = "#1f77b4"

    ax1.set_xlabel("Classification Decision Threshold", fontsize=11)
    ax1.set_ylabel("Sensitivity (%)", color=color_sens, fontsize=11)
    line1 = ax1.plot(
        sweep_df["threshold"],
        sweep_df["sensitivity"] * 100,
        color=color_sens,
        lw=2.5,
        label="Sensitivity (%)",
    )
    ax1.tick_params(axis="y", labelcolor=color_sens)
    ax1.set_ylim([0, 105])

    ax2 = ax1.twinx()
    ax2.set_ylabel("Articles Flagged for Manual Review (%)", color=color_workload, fontsize=11)
    line2 = ax2.plot(
        sweep_df["threshold"],
        sweep_df["review_workload_pct"],
        color=color_workload,
        lw=2.5,
        linestyle="-.",
        label="Manual Screening Workload (%)",
    )
    ax2.tick_params(axis="y", labelcolor=color_workload)
    ax2.set_ylim([0, 105])

    # Combine legends
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc="center left", fontsize=10)

    plt.title(f"Clinical Screening Tradeoff: {model_display_name}", fontsize=12, fontweight="bold")
    ax1.grid(True, alpha=0.3)

    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
