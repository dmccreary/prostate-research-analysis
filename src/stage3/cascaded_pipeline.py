"""
cascaded_pipeline.py - Two-stage cascaded screening and evaluation pipeline for Stage 3.

Implements:
1. Fast-Filter: Hard exclusion via Publication Types, Title keywords ('salvage'), and advanced disease acronyms.
2. Model Scoring: Downstream evaluation using trained models (PubMedBERT, SVM, Complement NB).
3. Full metrics computation: Confusion Matrix, AUROC, PR-AUC, Specificity, Workload Reduction %, and NNS (Number Needed to Screen).
4. Threshold sweep and calibration (validation-selected vs test-sweep).
"""

from typing import Callable, Dict, List, Optional, Tuple, Union
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


def run_cascaded_inference(
    df: pd.DataFrame,
    model_predict_fn: Callable[[pd.DataFrame], np.ndarray],
    use_pre_filter: bool = True,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Applies cascaded inference over a DataFrame.
    If use_pre_filter is True:
        Articles matching pre-filter are assigned prob = 0.0.
        Surviving articles are scored by model_predict_fn.
    Returns: (final_probs, is_filtered_mask, filter_reasons)
    """
    n_samples = len(df)
    final_probs = np.zeros(n_samples, dtype=float)
    is_filtered = np.zeros(n_samples, dtype=bool)
    reasons = [""] * n_samples

    if use_pre_filter:
        for i, (_, row) in enumerate(df.iterrows()):
            if row.get("is_pre_filtered", False):
                is_filtered[i] = True
                reasons[i] = row.get("filter_reason", "Pre-filtered")
                final_probs[i] = 0.0

        surviving_indices = np.where(~is_filtered)[0]
        if len(surviving_indices) > 0:
            surviving_df = df.iloc[surviving_indices].reset_index(drop=True)
            surviving_probs = model_predict_fn(surviving_df)
            final_probs[surviving_indices] = surviving_probs
    else:
        final_probs = model_predict_fn(df)

    return final_probs, is_filtered, reasons


def compute_screening_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
    model_name: str = "",
    pipeline_type: str = "Standalone",
) -> Dict[str, Union[str, float, int]]:
    """
    Computes standard screening and diagnostic metrics including NNS.
    """
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
        p_curve, r_curve, _ = precision_recall_curve(y_true, y_prob)
        pr_auc = float(auc(r_curve, p_curve))
    except Exception:
        pr_auc = float(np.mean(y_true))

    # Number Needed to Screen (NNS) = Total Flagged / True Positives
    # In systematic reviews, NNS represents how many flagged papers human reviewers must read to find 1 true eligible paper
    nns = round((flagged / tp), 2) if tp > 0 else float("inf")
    workload_pct = round((flagged / total) * 100.0, 2)
    reduction_pct = round(((total - flagged) / total) * 100.0, 2)

    return {
        "model_name": model_name,
        "pipeline_type": pipeline_type,
        "threshold": round(threshold, 3),
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall_sensitivity": round(rec, 4),
        "specificity": round(spec, 4),
        "f1_score": round(f1, 4),
        "auroc": round(auroc, 4),
        "pr_auc": round(pr_auc, 4),
        "true_positives": int(tp),
        "false_positives": int(fp),
        "true_negatives": int(tn),
        "false_negatives": int(fn),
        "flagged_count": flagged,
        "workload_pct": workload_pct,
        "workload_reduction_pct": reduction_pct,
        "nns": nns,
    }


def sweep_thresholds_for_targets(
    y_val_true: np.ndarray,
    y_val_prob: np.ndarray,
    y_test_true: np.ndarray,
    y_test_prob: np.ndarray,
    model_name: str,
    pipeline_type: str,
    target_sensitivities: List[float] = [0.90, 0.95, 0.99, 1.00],
) -> List[Dict]:
    """
    Selects threshold on validation set targeting sensitivity, then evaluates frozen threshold on test set.
    """
    val_thresholds = np.linspace(0.01, 0.99, 99)
    val_records = []

    for thresh in val_thresholds:
        val_pred = (y_val_prob >= thresh).astype(int)
        rec = recall_score(y_val_true, val_pred, zero_division=0)
        cm = confusion_matrix(y_val_true, val_pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        spec = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
        val_records.append({"thresh": thresh, "rec": rec, "spec": spec})

    val_df = pd.DataFrame(val_records)
    results = []

    for target in target_sensitivities:
        eligible = val_df[val_df["rec"] >= target]
        if not eligible.empty:
            # Pick highest threshold that meets target to maximize specificity
            best_val = eligible.sort_values(by=["thresh"], ascending=False).iloc[0]
            chosen_thresh = float(best_val["thresh"])
            val_rec = float(best_val["rec"])
            val_spec = float(best_val["spec"])
            status = "ACHIEVED"
        else:
            # Fallback to lowest threshold
            best_val = val_df.sort_values(by=["rec", "thresh"], ascending=[False, True]).iloc[0]
            chosen_thresh = float(best_val["thresh"])
            val_rec = float(best_val["rec"])
            val_spec = float(best_val["spec"])
            status = f"PARTIAL (Max: {round(val_rec, 3)})"

        # Evaluate on test set
        test_metrics = compute_screening_metrics(
            y_test_true, y_test_prob, threshold=chosen_thresh, model_name=model_name, pipeline_type=pipeline_type
        )
        test_metrics["target_sensitivity"] = target
        test_metrics["val_selected_thresh"] = round(chosen_thresh, 3)
        test_metrics["val_achieved_sensitivity"] = round(val_rec, 4)
        test_metrics["val_achieved_specificity"] = round(val_spec, 4)
        test_metrics["val_status"] = status
        results.append(test_metrics)

    return results
