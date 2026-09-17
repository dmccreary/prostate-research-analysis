"""
threshold_neural.py - Validation-based threshold selection and unbiased test evaluation for Stage 2.

Protocol:
1. Sweep thresholds [0.01 - 0.99] on the VALIDATION split.
2. Select the optimal threshold achieving target sensitivity (>= 90%, >= 95%, >= 99%, 100%)
   while maximizing specificity to minimize false positives.
3. Apply those exact frozen thresholds to the HELD-OUT TEST split.
4. Report validation selection metrics and final test metrics side-by-side to guarantee zero test leakage.
"""

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
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

logger = logging.getLogger(__name__)


def sweep_thresholds_array(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    thresholds: Optional[np.ndarray] = None,
) -> pd.DataFrame:
    """Evaluate performance metrics across decision thresholds [0.01 - 0.99]."""
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    total_samples = len(y_true)
    total_pos = int(np.sum(y_true))
    total_neg = total_samples - total_pos

    if thresholds is None:
        thresholds = np.linspace(0.01, 0.99, 99).round(3)

    records = []
    for thresh in thresholds:
        y_pred = (y_prob >= thresh).astype(int)
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()

        acc = float(accuracy_score(y_true, y_pred))
        prec = float(precision_score(y_true, y_pred, zero_division=0))
        rec = float(recall_score(y_true, y_pred, zero_division=0))
        sens = rec
        spec = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
        f1 = float(f1_score(y_true, y_pred, zero_division=0))

        flagged = int(tp + fp)
        workload_pct = (flagged / total_samples) * 100.0
        reduction_pct = ((total_samples - flagged) / total_samples) * 100.0

        records.append({
            "threshold": float(thresh),
            "accuracy": round(acc, 4),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "sensitivity": round(sens, 4),
            "specificity": round(spec, 4),
            "f1_score": round(f1, 4),
            "true_positives": int(tp),
            "false_positives": int(fp),
            "true_negatives": int(tn),
            "false_negatives": int(fn),
            "articles_flagged": flagged,
            "review_workload_pct": round(workload_pct, 2),
            "workload_reduction_pct": round(reduction_pct, 2),
            "total_positives": total_pos,
            "total_negatives": total_neg,
            "total_samples": total_samples,
        })

    return pd.DataFrame(records)


def select_validation_thresholds(
    y_val_true: np.ndarray,
    y_val_prob: np.ndarray,
    target_sensitivities: Optional[List[float]] = None,
) -> pd.DataFrame:
    """
    Select optimal operating thresholds on validation data for target sensitivities.
    Chooses the highest threshold that satisfies sensitivity >= target to minimize manual workload.
    """
    if target_sensitivities is None:
        target_sensitivities = [0.90, 0.95, 0.99, 1.00]

    val_sweep = sweep_thresholds_array(y_val_true, y_val_prob)
    selected = []

    for target in target_sensitivities:
        candidates = val_sweep[val_sweep["sensitivity"] >= target]
        if not candidates.empty:
            best_row = candidates.sort_values(
                by=["threshold", "specificity", "f1_score"], ascending=[False, False, False]
            ).iloc[0].to_dict()
            best_row["target_sensitivity"] = target
            best_row["val_status"] = "ACHIEVED"
            selected.append(best_row)
        else:
            best_row = val_sweep.sort_values(by="sensitivity", ascending=False).iloc[0].to_dict()
            best_row["target_sensitivity"] = target
            best_row["val_status"] = f"PARTIAL (Max: {best_row['sensitivity']:.3f})"
            selected.append(best_row)

    return pd.DataFrame(selected)


def apply_thresholds_to_test(
    val_selected_df: pd.DataFrame,
    y_test_true: np.ndarray,
    y_test_prob: np.ndarray,
    model_name: str,
) -> pd.DataFrame:
    """
    Apply validation-selected thresholds to the held-out test set.
    Produces an unbiased comparison table distinguishing validation selection from test results.
    """
    y_test_true = np.asarray(y_test_true).astype(int)
    y_test_prob = np.asarray(y_test_prob).astype(float)
    total_test = len(y_test_true)

    records = []
    for _, row in val_selected_df.iterrows():
        thresh = float(row["threshold"])
        target = float(row["target_sensitivity"])
        val_status = row["val_status"]

        y_pred = (y_test_prob >= thresh).astype(int)
        cm = confusion_matrix(y_test_true, y_pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()

        acc = float(accuracy_score(y_test_true, y_pred))
        prec = float(precision_score(y_test_true, y_pred, zero_division=0))
        rec = float(recall_score(y_test_true, y_pred, zero_division=0))
        sens = rec
        spec = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
        f1 = float(f1_score(y_test_true, y_pred, zero_division=0))

        flagged = int(tp + fp)
        workload_pct = (flagged / total_test) * 100.0
        reduction_pct = ((total_test - flagged) / total_test) * 100.0

        records.append({
            "model_name": model_name,
            "target_sensitivity": target,
            "val_selected_threshold": round(thresh, 3),
            "val_sensitivity": round(row["sensitivity"], 4),
            "val_specificity": round(row["specificity"], 4),
            "val_workload_pct": round(row["review_workload_pct"], 2),
            "test_achieved_sensitivity": round(sens, 4),
            "test_achieved_specificity": round(spec, 4),
            "test_achieved_precision": round(prec, 4),
            "test_achieved_f1": round(f1, 4),
            "test_flagged_articles": f"{flagged} / {total_test}",
            "test_workload_pct": round(workload_pct, 2),
            "test_workload_reduction_pct": round(reduction_pct, 2),
            "test_true_positives": int(tp),
            "test_false_positives": int(fp),
            "test_true_negatives": int(tn),
            "test_false_negatives": int(fn),
            "val_status": val_status,
        })

    return pd.DataFrame(records)


def plot_neural_threshold_tradeoff(sweep_df: pd.DataFrame, model_display_name: str, output_path: str):
    """Plot Sensitivity, Specificity, and Review Workload across thresholds."""
    fig, ax1 = plt.subplots(figsize=(8.5, 5))

    color_sens = "#1f77b4"
    color_spec = "#2ca02c"
    color_workload = "#ff7f0e"

    ax1.set_xlabel("Classification Decision Threshold", fontsize=11)
    ax1.set_ylabel("Metric Value (%)", fontsize=11)
    line1 = ax1.plot(sweep_df["threshold"], sweep_df["sensitivity"] * 100, label="Sensitivity / Recall (%)", color=color_sens, lw=2.5)
    line2 = ax1.plot(sweep_df["threshold"], sweep_df["specificity"] * 100, label="Specificity (%)", color=color_spec, lw=2.5)
    ax1.set_ylim([0, 105])
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    ax2.set_ylabel("Articles Flagged for Manual Review (%)", color=color_workload, fontsize=11)
    line3 = ax2.plot(sweep_df["threshold"], sweep_df["review_workload_pct"], label="Review Workload (%)", color=color_workload, lw=2, linestyle="-.")
    ax2.tick_params(axis="y", labelcolor=color_workload)
    ax2.set_ylim([0, 105])

    lines = line1 + line2 + line3
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc="center left", fontsize=10)

    plt.title(f"Clinical Screening Tradeoff: {model_display_name}", fontsize=12, fontweight="bold")
    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
