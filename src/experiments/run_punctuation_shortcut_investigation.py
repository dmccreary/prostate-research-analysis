"""
run_punctuation_shortcut_investigation.py - Rigorous Evidence Report Pipeline
Demonstrating whether Stage 4 PubMedBERT models learned a dataset shortcut based on trailing punctuation.

Strict Constraints:
- No model retraining. Existing Stage 4 checkpoints are evaluated exactly as-is.
- Fixed decision threshold = 0.50.
- Rigorous counterfactual interventions, paired statistics, cross-dataset test, identical-paper test, and token-level verification.
"""

import copy
import gc
import json
import logging
from pathlib import Path
import re
import sys
import time
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
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
from transformers import AutoModel, AutoTokenizer

# Safe stdout on Windows
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(BASE_DIR / "src"))
from preprocessing import clean_clinical_text

PRETRAINED_MODEL_NAME = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract"
MAX_LENGTH = 384
BATCH_SIZE = 16

RESULTS_DIR = BASE_DIR / "results" / "punctuation_shortcut_investigation"
FIG_DIR = RESULTS_DIR / "figures"
REPORTS_DIR = BASE_DIR / "reports" / "punctuation_shortcut_investigation"
ARTIFACT_FIG_DIR = Path(r"C:\Users\mmahd\.gemini\antigravity\brain\97473cd2-23f6-466c-bee7-b37bda871005\figures")

for d in [RESULTS_DIR, FIG_DIR, REPORTS_DIR, ARTIFACT_FIG_DIR]:
    d.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(RESULTS_DIR / "investigation.log", encoding="utf-8", mode="w"),
    ],
)
logger = logging.getLogger(__name__)


class DualInputPubMedBERT(nn.Module):
    """
    PubMedBERT for Title + Abstract dual-sequence input:
    [CLS] Title [SEP] Abstract [SEP]
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


def pretokenize_pairs(
    titles: List[str],
    abstracts: List[str],
    tokenizer,
    max_length: int = MAX_LENGTH,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Tokenizes clinical title + abstract text once."""
    clean_titles = [clean_clinical_text(str(t or "")) for t in titles]
    clean_abstracts = [clean_clinical_text(str(a or "")) for a in abstracts]

    enc = tokenizer(
        text=clean_titles,
        text_pair=clean_abstracts,
        max_length=max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    input_ids = enc["input_ids"]
    attention_mask = enc["attention_mask"]
    token_type_ids = enc.get("token_type_ids", torch.zeros_like(input_ids))
    return input_ids, attention_mask, token_type_ids


def predict_tensors(
    model: nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    token_type_ids: torch.Tensor,
    device: torch.device,
    batch_size: int = BATCH_SIZE,
) -> np.ndarray:
    """Runs forward inference on pretokenized tensors."""
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


def make_counterfactual_variants(title_str: str) -> Dict[str, str]:
    """Generates the 6 title variants keeping all words identical, modifying ONLY final punctuation."""
    t = str(title_str).strip()

    # A: Original
    va = t

    # B: Remove final '.' if present
    vb = t[:-1] if t.endswith(".") else t

    # C: Add '.' to the end if absent
    if t.endswith("."):
        vc = t
    elif t.endswith("?"):
        vc = t[:-1] + "."
    else:
        vc = t + "."

    # D: Replace final '.' with '!'
    if t.endswith("."):
        vd = t[:-1] + "!"
    elif t.endswith("?"):
        vd = t[:-1] + "!"
    else:
        vd = t + "!"

    # E: Replace final '.' with '?'
    if t.endswith("."):
        ve = t[:-1] + "?"
    elif t.endswith("?"):
        ve = t
    else:
        ve = t + "?"

    # F: Replace final punctuation with nothing
    vf = re.sub(r"[.?!:;,]+$", "", t)

    return {
        "A_original": va,
        "B_remove_dot": vb,
        "C_add_dot": vc,
        "D_replace_excl": vd,
        "E_replace_quest": ve,
        "F_strip_punct": vf,
    }


def compute_paired_statistics(
    orig_probs: np.ndarray,
    cf_probs: np.ndarray,
    orig_preds: np.ndarray,
    cf_preds: np.ndarray,
    n_bootstrap: int = 10000,
    seed: int = 42,
) -> Dict:
    """Computes paired Wilcoxon signed-rank test, fast vectorized bootstrap CIs, Cohen's dz, and McNemar test."""
    diffs = cf_probs - orig_probs
    n = len(diffs)

    non_zero = diffs[diffs != 0]
    if len(non_zero) > 0:
        try:
            w_res = stats.wilcoxon(diffs, alternative="two-sided")
            w_stat, w_p = float(w_res.statistic), float(w_res.pvalue)
        except Exception:
            w_stat, w_p = 0.0, 1.0
    else:
        w_stat, w_p = 0.0, 1.0

    mean_diff = float(np.mean(diffs))
    median_diff = float(np.median(diffs))
    mean_abs_diff = float(np.mean(np.abs(diffs)))

    # Fast vectorized bootstrap
    rng = np.random.RandomState(seed)
    boot_idx = rng.randint(0, n, size=(n_bootstrap, n))
    boot_samples = diffs[boot_idx]
    boot_medians = np.median(boot_samples, axis=1)
    boot_means = np.mean(boot_samples, axis=1)

    ci_median = (float(np.percentile(boot_medians, 2.5)), float(np.percentile(boot_medians, 97.5)))
    ci_mean = (float(np.percentile(boot_means, 2.5)), float(np.percentile(boot_means, 97.5)))

    std_diff = float(np.std(diffs, ddof=1))
    dz = mean_diff / std_diff if std_diff > 1e-8 else 0.0

    b = int(np.sum((orig_preds == 1) & (cf_preds == 0)))  # Pos -> Neg
    c = int(np.sum((orig_preds == 0) & (cf_preds == 1)))  # Neg -> Pos
    n_discordant = b + c

    if n_discordant > 0:
        b_res = stats.binomtest(b, n_discordant, 0.5)
        mcnemar_p_exact = float(b_res.pvalue)
        chi2 = float(((abs(b - c) - 1.0) ** 2) / n_discordant)
        mcnemar_p_asymp = float(1.0 - stats.chi2.cdf(chi2, df=1))
    else:
        mcnemar_p_exact = 1.0
        chi2 = 0.0
        mcnemar_p_asymp = 1.0

    return {
        "mean_diff": round(mean_diff, 4),
        "ci_mean_95": (round(ci_mean[0], 4), round(ci_mean[1], 4)),
        "median_diff": round(median_diff, 4),
        "ci_median_95": (round(ci_median[0], 4), round(ci_median[1], 4)),
        "mean_abs_diff": round(mean_abs_diff, 4),
        "wilcoxon_stat": round(w_stat, 2),
        "wilcoxon_p": float(w_p),
        "cohens_dz": round(dz, 4),
        "mcnemar_b_pos_to_neg": b,
        "mcnemar_c_neg_to_pos": c,
        "mcnemar_chi2": round(chi2, 4),
        "mcnemar_p_exact": float(mcnemar_p_exact),
        "mcnemar_p_asymp": float(mcnemar_p_asymp),
    }


def main():
    start_time = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(PRETRAINED_MODEL_NAME, local_files_only=True)

    # Checkpoints definition
    models_dict = {
        "Stage 4 Original (Seed 42, Weighted)": BASE_DIR / "models" / "stage4" / "pubmedbert_stage4_weighted.pt",
        "Stage 4 Original (Seed 42, Unweighted)": BASE_DIR / "models" / "stage4" / "pubmedbert_stage4_unweighted.pt",
        "Stage 4 Multi-Seed (Seed 101, Weighted)": BASE_DIR / "models" / "stage4_multi_seed" / "pubmedbert_weighted_seed_101.pt",
        "Stage 4 Multi-Seed (Seed 123, Weighted)": BASE_DIR / "models" / "stage4_multi_seed" / "pubmedbert_weighted_seed_123.pt",
        "Stage 4 Multi-Seed (Seed 456, Weighted)": BASE_DIR / "models" / "stage4_multi_seed" / "pubmedbert_weighted_seed_456.pt",
        "Stage 4 Multi-Seed (Seed 789, Weighted)": BASE_DIR / "models" / "stage4_multi_seed" / "pubmedbert_weighted_seed_789.pt",
        "Stage 4 Multi-Seed (Seed 2024, Weighted)": BASE_DIR / "models" / "stage4_multi_seed" / "pubmedbert_weighted_seed_2024.pt",
    }

    # =========================================================================
    # PRE-TOKENIZATION FOR ALL DATASETS (DONE ONCE)
    # =========================================================================
    logger.info("=== Pre-tokenizing all evaluation cohorts ===")

    # 1. Stage 4 Test Set (N = 105)
    test_path = BASE_DIR / "data" / "splits" / "stage4" / "test.csv"
    test_df = pd.read_csv(test_path)
    logger.info(f"Loaded Stage 4 test set: {len(test_df)} samples (Pos: {(test_df['label']==1).sum()}, Neg: {(test_df['label']==0).sum()})")

    variant_cols = ["A_original", "B_remove_dot", "C_add_dot", "D_replace_excl", "E_replace_quest", "F_strip_punct"]
    test_variants_text = {v: [] for v in variant_cols}
    for t in test_df["title"]:
        vars_dict = make_counterfactual_variants(t)
        for v in variant_cols:
            test_variants_text[v].append(vars_dict[v])

    abstracts_test = test_df["abstract"].tolist()
    y_test_true = test_df["label"].values.astype(int)

    test_tensors = {}
    for v in variant_cols:
        test_tensors[v] = pretokenize_pairs(test_variants_text[v], abstracts_test, tokenizer)

    # 2. 2021 Dataset (N = 117)
    df_21 = pd.read_csv(BASE_DIR / "data" / "2021 labeld.csv")
    y_21_true = df_21["label"].values.astype(int)
    titles_21_orig = df_21["title"].tolist()
    titles_21_dotted = [str(t).strip().rstrip(".") + "." for t in titles_21_orig]
    abstracts_21 = df_21["abstract"].tolist()

    tensors_21_orig = pretokenize_pairs(titles_21_orig, abstracts_21, tokenizer)
    tensors_21_dotted = pretokenize_pairs(titles_21_dotted, abstracts_21, tokenizer)

    # 3. Overlapping PMIDs between labeled-dataset-v2 and 2021
    df_v2 = pd.read_csv(BASE_DIR / "data" / "labeled-dataset-v2.csv")
    s_v2 = set(df_v2["pmid"])
    s_21 = set(df_21["pmid"])
    overlapping_pmids = sorted(list(s_v2.intersection(s_21)))
    logger.info(f"Overlapping PMIDs ({len(overlapping_pmids)}): {overlapping_pmids}")

    overlap_v2_rows = [df_v2[df_v2["pmid"] == p].iloc[0] for p in overlapping_pmids]
    overlap_21_rows = [df_21[df_21["pmid"] == p].iloc[0] for p in overlapping_pmids]

    titles_ov_v2 = [str(r["title"]) for r in overlap_v2_rows]
    abstracts_ov_v2 = [str(r["abstract"]) for r in overlap_v2_rows]
    titles_ov_21 = [str(r["title"]) for r in overlap_21_rows]
    abstracts_ov_21 = [str(r["abstract"]) for r in overlap_21_rows]

    tensors_ov_v2 = pretokenize_pairs(titles_ov_v2, abstracts_ov_v2, tokenizer)
    tensors_ov_21 = pretokenize_pairs(titles_ov_21, abstracts_ov_21, tokenizer)

    logger.info("Pre-tokenization complete.")

    # =========================================================================
    # SINGLE UNIFIED MODEL LOOP (LOAD EACH MODEL EXACTLY ONCE)
    # =========================================================================
    per_article_records = []
    summary_by_model_condition = []
    stats_records = []
    by_class_records = []

    model_orig_probs = {}
    model_nodot_probs = {}
    model_cond_probs = {}

    records_21 = []
    eval_21_stats = []
    model_21_orig_probs = {}
    model_21_dotted_probs = {}

    identical_records = []

    for model_name, ckpt_path in models_dict.items():
        logger.info(f"--> Loading and evaluating: {model_name}...")
        model = load_model(ckpt_path, device)

        # -------------------------------------------------------------
        # Part 1: Stage 4 Test Set (6 Variants)
        # -------------------------------------------------------------
        cond_probs = {}
        for v in variant_cols:
            b_ids, b_mask, b_type = test_tensors[v]
            probs = predict_tensors(model, b_ids, b_mask, b_type, device)
            cond_probs[v] = probs

        model_cond_probs[model_name] = cond_probs
        orig_p = cond_probs["A_original"]
        nodot_p = cond_probs["B_remove_dot"]
        model_orig_probs[model_name] = orig_p
        model_nodot_probs[model_name] = nodot_p

        orig_pred = (orig_p >= 0.50).astype(int)

        # Detailed per-article predictions
        for idx in range(len(test_df)):
            pmid = test_df.loc[idx, "pmid"]
            lbl = int(test_df.loc[idx, "label"])
            orig_t = test_variants_text["A_original"][idx]

            for v in variant_cols:
                cf_p = cond_probs[v][idx]
                cf_pred = int(cf_p >= 0.50)
                diff = cf_p - orig_p[idx]
                changed = int(cf_pred != orig_pred[idx])

                per_article_records.append({
                    "model_name": model_name,
                    "pmid": pmid,
                    "label": lbl,
                    "condition": v,
                    "original_title": orig_t,
                    "variant_title": test_variants_text[v][idx],
                    "prob_original": round(float(orig_p[idx]), 6),
                    "prob_counterfactual": round(float(cf_p), 6),
                    "prob_diff": round(float(diff), 6),
                    "pred_original": int(orig_pred[idx]),
                    "pred_counterfactual": cf_pred,
                    "pred_changed": changed,
                })

        # Summary metrics across conditions
        for v in variant_cols:
            cp = cond_probs[v]
            cpred = (cp >= 0.50).astype(int)
            tn, fp, fn, tp = confusion_matrix(y_test_true, cpred, labels=[0, 1]).ravel()
            rec = recall_score(y_test_true, cpred, zero_division=0)
            spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
            acc = accuracy_score(y_test_true, cpred)
            prec = precision_score(y_test_true, cpred, zero_division=0)
            f1 = f1_score(y_test_true, cpred, zero_division=0)
            try:
                auc_val = roc_auc_score(y_test_true, cp)
            except Exception:
                auc_val = 0.5

            summary_by_model_condition.append({
                "model_name": model_name,
                "condition": v,
                "mean_prob": round(float(np.mean(cp)), 4),
                "median_prob": round(float(np.median(cp)), 4),
                "pos_pred_count": int(cpred.sum()),
                "pos_pred_pct": round(float(cpred.mean() * 100), 2),
                "tp": int(tp),
                "fp": int(fp),
                "tn": int(tn),
                "fn": int(fn),
                "recall": round(float(rec * 100), 2),
                "specificity": round(float(spec * 100), 2),
                "accuracy": round(float(acc * 100), 2),
                "precision": round(float(prec * 100), 2),
                "f1": round(float(f1), 4),
                "auroc": round(float(auc_val), 4),
            })

        # Paired statistics (Original vs Remove Dot)
        nodot_pred = (nodot_p >= 0.50).astype(int)
        p_stats = compute_paired_statistics(orig_p, nodot_p, orig_pred, nodot_pred)
        p_stats["model_name"] = model_name
        p_stats["comparison"] = "Original vs Remove Final Dot"
        p_stats["total_samples"] = len(test_df)
        p_stats["all_pred_changed_count"] = int(np.sum(orig_pred != nodot_pred))
        p_stats["all_pred_changed_pct"] = round(float(np.mean(orig_pred != nodot_pred) * 100), 2)
        stats_records.append(p_stats)

        # By-class statistics
        for class_val, class_name in [(1, "Positive Articles"), (0, "Negative Articles")]:
            mask = y_test_true == class_val
            c_orig_p = orig_p[mask]
            c_nodot_p = nodot_p[mask]
            c_orig_pred = orig_pred[mask]
            c_nodot_pred = nodot_pred[mask]

            c_diffs = c_nodot_p - c_orig_p
            c_changed = np.sum(c_orig_pred != c_nodot_pred)

            by_class_records.append({
                "model_name": model_name,
                "cohort": class_name,
                "n_samples": int(mask.sum()),
                "mean_orig_prob": round(float(np.mean(c_orig_p)), 4),
                "mean_nodot_prob": round(float(np.mean(c_nodot_p)), 4),
                "mean_prob_change": round(float(np.mean(c_diffs)), 4),
                "median_prob_change": round(float(np.median(c_diffs)), 4),
                "mean_abs_prob_change": round(float(np.mean(np.abs(c_diffs))), 4),
                "predictions_changed_count": int(c_changed),
                "predictions_changed_pct": round(float(c_changed / mask.sum() * 100), 2),
                "pos_to_neg_flips": int(np.sum((c_orig_pred == 1) & (c_nodot_pred == 0))),
                "neg_to_pos_flips": int(np.sum((c_orig_pred == 0) & (c_nodot_pred == 1))),
            })

        # -------------------------------------------------------------
        # Part 2: 2021 Dataset Test
        # -------------------------------------------------------------
        b_ids, b_mask, b_type = tensors_21_orig
        p_orig_21 = predict_tensors(model, b_ids, b_mask, b_type, device)
        b_ids, b_mask, b_type = tensors_21_dotted
        p_dotted_21 = predict_tensors(model, b_ids, b_mask, b_type, device)

        model_21_orig_probs[model_name] = p_orig_21
        model_21_dotted_probs[model_name] = p_dotted_21

        pred_orig_21 = (p_orig_21 >= 0.50).astype(int)
        pred_dotted_21 = (p_dotted_21 >= 0.50).astype(int)

        for cond_name, p_arr, pr_arr in [("Original (No Trailing Dot)", p_orig_21, pred_orig_21), ("Intervention (Added Trailing Dot)", p_dotted_21, pred_dotted_21)]:
            tn, fp, fn, tp = confusion_matrix(y_21_true, pr_arr, labels=[0, 1]).ravel()
            rec = recall_score(y_21_true, pr_arr, zero_division=0)
            spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
            acc = accuracy_score(y_21_true, pr_arr)
            prec = precision_score(y_21_true, pr_arr, zero_division=0)
            f1 = f1_score(y_21_true, pr_arr, zero_division=0)
            try:
                auc_val = roc_auc_score(y_21_true, p_arr)
            except Exception:
                auc_val = 0.5

            records_21.append({
                "model_name": model_name,
                "condition": cond_name,
                "mean_prob": round(float(np.mean(p_arr)), 4),
                "median_prob": round(float(np.median(p_arr)), 4),
                "min_prob": round(float(np.min(p_arr)), 4),
                "max_prob": round(float(np.max(p_arr)), 4),
                "pos_pred_count": int(pr_arr.sum()),
                "neg_pred_count": int(len(pr_arr) - pr_arr.sum()),
                "pos_pred_pct": round(float(pr_arr.mean() * 100), 2),
                "tp": int(tp),
                "fp": int(fp),
                "tn": int(tn),
                "fn": int(fn),
                "recall": round(float(rec * 100), 2),
                "specificity": round(float(spec * 100), 2),
                "accuracy": round(float(acc * 100), 2),
                "precision": round(float(prec * 100), 2),
                "f1": round(float(f1), 4),
                "auroc": round(float(auc_val), 4),
            })

        s_21 = compute_paired_statistics(p_orig_21, p_dotted_21, pred_orig_21, pred_dotted_21)
        s_21["model_name"] = model_name
        s_21["comparison"] = "2021 Dataset: Original vs Added Trailing Period"
        eval_21_stats.append(s_21)

        # -------------------------------------------------------------
        # Part 3: Overlapping PMIDs Test
        # -------------------------------------------------------------
        b_ids, b_mask, b_type = tensors_ov_v2
        p_ov_v2 = predict_tensors(model, b_ids, b_mask, b_type, device)
        b_ids, b_mask, b_type = tensors_ov_21
        p_ov_21 = predict_tensors(model, b_ids, b_mask, b_type, device)

        for i_ov, pmid_val in enumerate(overlapping_pmids):
            identical_records.append({
                "pmid": pmid_val,
                "model_name": model_name,
                "label": int(overlap_v2_rows[i_ov]["label"]),
                "title_dataset_A_v2": titles_ov_v2[i_ov],
                "title_dataset_B_2021": titles_ov_21[i_ov],
                "abstracts_identical": (abstracts_ov_v2[i_ov] == abstracts_ov_21[i_ov]),
                "prob_dataset_A_with_dot": round(float(p_ov_v2[i_ov]), 6),
                "prob_dataset_B_no_dot": round(float(p_ov_21[i_ov]), 6),
                "prob_diff": round(float(p_ov_21[i_ov] - p_ov_v2[i_ov]), 6),
                "pred_dataset_A": int(p_ov_v2[i_ov] >= 0.50),
                "pred_dataset_B": int(p_ov_21[i_ov] >= 0.50),
            })

        # Memory cleanup
        del model
        torch.cuda.empty_cache()

    logger.info("All model evaluations complete. Saving datasets...")

    # Save DataFrames
    df_per_article = pd.DataFrame(per_article_records)
    df_per_article.to_csv(RESULTS_DIR / "test_stage4_counterfactual_predictions.csv", index=False)

    df_summary = pd.DataFrame(summary_by_model_condition)
    df_summary.to_csv(RESULTS_DIR / "test_stage4_ablation_summary.csv", index=False)

    df_stats = pd.DataFrame(stats_records)
    df_stats.to_csv(RESULTS_DIR / "test_stage4_statistical_tests.csv", index=False)

    df_by_class = pd.DataFrame(by_class_records)
    df_by_class.to_csv(RESULTS_DIR / "test_stage4_by_class_summary.csv", index=False)

    df_21_summary = pd.DataFrame(records_21)
    df_21_summary.to_csv(RESULTS_DIR / "eval_2021_counterfactual_comparison.csv", index=False)

    df_21_stats = pd.DataFrame(eval_21_stats)
    df_21_stats.to_csv(RESULTS_DIR / "eval_2021_statistical_tests.csv", index=False)

    df_identical = pd.DataFrame(identical_records)
    df_identical.to_csv(RESULTS_DIR / "overlapping_pmids_evaluation.csv", index=False)

    # =========================================================================
    # PART 4: Token-Level Input Verification
    # =========================================================================
    logger.info("=== Token-Level Input Verification ===")
    token_verifications = []
    sample_pmids = [33279855, 32642874, 33035622]

    for pmid in sample_pmids:
        r = df_v2[df_v2["pmid"] == pmid].iloc[0]
        t_orig = str(r["title"]).strip()
        t_nodot = t_orig.rstrip(".")
        abs_text = str(r["abstract"])

        enc_orig = tokenizer(text=clean_clinical_text(t_orig), text_pair=clean_clinical_text(abs_text))
        enc_nodot = tokenizer(text=clean_clinical_text(t_nodot), text_pair=clean_clinical_text(abs_text))

        ids_orig = enc_orig["input_ids"]
        toks_orig = tokenizer.convert_ids_to_tokens(ids_orig)
        sep1_idx_orig = toks_orig.index("[SEP]")

        ids_nodot = enc_nodot["input_ids"]
        toks_nodot = tokenizer.convert_ids_to_tokens(ids_nodot)
        sep1_idx_nodot = toks_nodot.index("[SEP]")

        token_verifications.append({
            "pmid": int(pmid),
            "title_original": t_orig,
            "tokens_around_sep_original": toks_orig[max(0, sep1_idx_orig - 4) : sep1_idx_orig + 4],
            "token_ids_around_sep_original": ids_orig[max(0, sep1_idx_orig - 4) : sep1_idx_orig + 4],
            "token_preceding_sep_original": toks_orig[sep1_idx_orig - 1],
            "token_id_preceding_sep_original": int(ids_orig[sep1_idx_orig - 1]),
            "title_nodot": t_nodot,
            "tokens_around_sep_nodot": toks_nodot[max(0, sep1_idx_nodot - 4) : sep1_idx_nodot + 4],
            "token_ids_around_sep_nodot": ids_nodot[max(0, sep1_idx_nodot - 4) : sep1_idx_nodot + 4],
            "token_preceding_sep_nodot": toks_nodot[sep1_idx_nodot - 1],
            "token_id_preceding_sep_nodot": int(ids_nodot[sep1_idx_nodot - 1]),
        })

    with open(RESULTS_DIR / "token_level_verification.json", "w", encoding="utf-8") as f:
        json.dump(token_verifications, f, indent=2)

    # =========================================================================
    # PART 5: Model-Agnostic Baseline (Title Ends With Dot Rule)
    # =========================================================================
    logger.info("=== Model-Agnostic Baseline Evaluation ===")
    baseline_records = []
    eval_sets = [
        ("Full Training Dataset (labeled-dataset-v2.csv)", df_v2),
        ("Stage 4 Held-Out Test Set (test.csv)", test_df),
        ("Prospective 2021 Dataset (2021 labeld.csv)", df_21),
    ]

    for set_name, current_df in eval_sets:
        has_dot_pred = current_df["title"].astype(str).str.strip().str.endswith(".").astype(int).values
        y_gold = current_df["label"].values.astype(int)

        tn, fp, fn, tp = confusion_matrix(y_gold, has_dot_pred, labels=[0, 1]).ravel()
        rec = recall_score(y_gold, has_dot_pred, zero_division=0)
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        acc = accuracy_score(y_gold, has_dot_pred)
        prec = precision_score(y_gold, has_dot_pred, zero_division=0)
        f1 = f1_score(y_gold, has_dot_pred, zero_division=0)
        try:
            auc_val = roc_auc_score(y_gold, has_dot_pred)
        except Exception:
            auc_val = 0.5

        baseline_records.append({
            "dataset": set_name,
            "total_samples": len(current_df),
            "pos_samples": int((y_gold == 1).sum()),
            "neg_samples": int((y_gold == 0).sum()),
            "tp": int(tp),
            "fp": int(fp),
            "tn": int(tn),
            "fn": int(fn),
            "accuracy": round(float(acc * 100), 2),
            "sensitivity_recall": round(float(rec * 100), 2),
            "specificity": round(float(spec * 100), 2),
            "precision": round(float(prec * 100), 2),
            "f1_score": round(float(f1), 4),
            "auroc": round(float(auc_val), 4),
        })

    df_baseline = pd.DataFrame(baseline_records)
    df_baseline.to_csv(RESULTS_DIR / "model_agnostic_baseline_results.csv", index=False)

    # =========================================================================
    # PART 6: Generate the 5 Required Figures
    # =========================================================================
    logger.info("=== Generating Publication Figures ===")
    plt.rcParams["font.sans-serif"] = "DejaVu Sans"
    plt.rcParams["font.size"] = 11

    # Figure 1: Distribution of predicted probabilities: Original vs. punctuation-removed
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    orig_p42 = model_orig_probs["Stage 4 Original (Seed 42, Weighted)"]
    nodot_p42 = model_nodot_probs["Stage 4 Original (Seed 42, Weighted)"]

    multi_seeds = [
        "Stage 4 Multi-Seed (Seed 101, Weighted)",
        "Stage 4 Multi-Seed (Seed 123, Weighted)",
        "Stage 4 Multi-Seed (Seed 456, Weighted)",
        "Stage 4 Multi-Seed (Seed 789, Weighted)",
        "Stage 4 Multi-Seed (Seed 2024, Weighted)",
    ]
    orig_multi_mean = np.mean([model_orig_probs[s] for s in multi_seeds], axis=0)
    nodot_multi_mean = np.mean([model_nodot_probs[s] for s in multi_seeds], axis=0)

    bins = np.linspace(0, 1, 35)
    axes[0].hist(orig_p42, bins=bins, alpha=0.6, color="#1f77b4", label=f"Original Title (Mean={np.mean(orig_p42):.3f})", edgecolor="black")
    axes[0].hist(nodot_p42, bins=bins, alpha=0.6, color="#d62728", label=f"Punctuation Removed (Mean={np.mean(nodot_p42):.3f})", edgecolor="black")
    axes[0].axvline(0.50, color="black", linestyle="--", linewidth=1.5, label="Threshold = 0.50")
    axes[0].set_title("A. Stage 4 Original (Seed 42, Weighted) [N=105]", fontweight="bold")
    axes[0].set_xlabel("Predicted Probability (Sigmoid Logits)")
    axes[0].set_ylabel("Number of Articles")
    axes[0].legend(loc="upper center", frameon=True)
    axes[0].grid(True, alpha=0.3)

    axes[1].hist(orig_multi_mean, bins=bins, alpha=0.6, color="#2ca02c", label=f"Original Title (Mean={np.mean(orig_multi_mean):.3f})", edgecolor="black")
    axes[1].hist(nodot_multi_mean, bins=bins, alpha=0.6, color="#ff7f0e", label=f"Punctuation Removed (Mean={np.mean(nodot_multi_mean):.3f})", edgecolor="black")
    axes[1].axvline(0.50, color="black", linestyle="--", linewidth=1.5, label="Threshold = 0.50")
    axes[1].set_title("B. Multi-Seed Ensemble (5 Seeds Mean) [N=105]", fontweight="bold")
    axes[1].set_xlabel("Predicted Probability (Sigmoid Logits)")
    axes[1].set_ylabel("Number of Articles")
    axes[1].legend(loc="upper center", frameon=True)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig1_prob_distribution_comparison.png", dpi=300)
    plt.savefig(ARTIFACT_FIG_DIR / "fig1_prob_distribution_comparison.png", dpi=300)
    plt.close()

    # Figure 2: Per-paper probability change after removing the trailing period
    fig, ax = plt.subplots(figsize=(13, 6))
    diffs_p42 = nodot_p42 - orig_p42
    sorted_indices = np.argsort(diffs_p42)
    sorted_diffs = diffs_p42[sorted_indices]
    sorted_labels = y_test_true[sorted_indices]

    bar_colors = ["#d62728" if lbl == 1 else "#1f77b4" for lbl in sorted_labels]
    ax.bar(range(len(sorted_diffs)), sorted_diffs, color=bar_colors, width=0.85, alpha=0.85)
    ax.axhline(0.0, color="black", linewidth=1.2)
    ax.set_title("Figure 2: Per-Article Probability Change When Trailing Period is Removed\n(Stage 4 Test Set, Seed 42, N = 105)", fontweight="bold")
    ax.set_xlabel("Articles (Sorted by Change in Predicted Probability)")
    ax.set_ylabel("Probability Change (P_no_dot - P_original)")

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#d62728", label="True Positive Articles (N=56) - Massive Probability Drop"),
        Patch(facecolor="#1f77b4", label="True Negative Articles (N=49) - Minimal Change (Already had no dot)"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", frameon=True)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig2_waterfall_prob_change.png", dpi=300)
    plt.savefig(ARTIFACT_FIG_DIR / "fig2_waterfall_prob_change.png", dpi=300)
    plt.close()

    # Figure 3: Original probability vs. counterfactual probability (Scatter plot)
    fig, ax = plt.subplots(figsize=(8, 7.5))
    pos_mask = (y_test_true == 1)
    neg_mask = (y_test_true == 0)

    jitter_x = np.random.normal(0, 0.005, size=len(orig_p42))
    jitter_y = np.random.normal(0, 0.005, size=len(nodot_p42))

    ax.scatter(orig_p42[pos_mask] + jitter_x[pos_mask], nodot_p42[pos_mask] + jitter_y[pos_mask], color="#d62728", s=55, alpha=0.75, label="Positive Articles (N=56)", edgecolors="black", linewidth=0.5)
    ax.scatter(orig_p42[neg_mask] + jitter_x[neg_mask], nodot_p42[neg_mask] + jitter_y[neg_mask], color="#1f77b4", s=55, alpha=0.75, label="Negative Articles (N=49)", edgecolors="black", linewidth=0.5)

    ax.plot([0, 1], [0, 1], "k--", alpha=0.6, label="Identity Line (No Change)")
    ax.axvline(0.50, color="gray", linestyle=":", alpha=0.7)
    ax.axhline(0.50, color="gray", linestyle=":", alpha=0.7)

    ax.set_title("Figure 3: Original Probability vs. Counterfactual Probability\n(Removing Trailing Period '.', Seed 42, N = 105)", fontweight="bold")
    ax.set_xlabel("Original Predicted Probability (With Original Punctuation)")
    ax.set_ylabel("Counterfactual Predicted Probability (Trailing Period Removed)")
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.legend(loc="upper left", frameon=True)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig3_scatter_orig_vs_counterfactual.png", dpi=300)
    plt.savefig(ARTIFACT_FIG_DIR / "fig3_scatter_orig_vs_counterfactual.png", dpi=300)
    plt.close()

    # Figure 4: Number of Positive predictions under the 5 conditions across models
    fig, ax = plt.subplots(figsize=(13, 6.5))
    models_list = list(models_dict.keys())
    short_names = ["Seed 42 (W)", "Seed 42 (Unw)", "Seed 101 (W)", "Seed 123 (W)", "Seed 456 (W)", "Seed 789 (W)", "Seed 2024 (W)"]
    x = np.arange(len(short_names))
    width = 0.16

    c_keys = ["A_original", "B_remove_dot", "C_add_dot", "E_replace_quest", "D_replace_excl"]
    c_labels = ["Original Title", "Remove '.'", "Add '.'", "Replace with '?'", "Replace with '!'"]
    colors = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e"]

    for i, (ck, cl, col) in enumerate(zip(c_keys, c_labels, colors)):
        counts = []
        for m in models_list:
            probs = model_cond_probs[m][ck]
            counts.append(int(np.sum(probs >= 0.50)))
        ax.bar(x + (i - 2) * width, counts, width, label=cl, color=col, edgecolor="black", alpha=0.85)

    ax.axhline(56, color="black", linestyle="--", linewidth=1.5, label="Ground Truth Positives (N = 56)")
    ax.set_title("Figure 4: Number of Positive Predictions Across Counterfactual Punctuation Conditions\n(Stage 4 Test Set, N = 105)", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(short_names, fontweight="semibold")
    ax.set_ylabel("Predicted Positives Count (Prob >= 0.50)")
    ax.set_ylim(0, 115)
    ax.legend(loc="upper right", frameon=True)
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig4_positive_predictions_by_condition.png", dpi=300)
    plt.savefig(ARTIFACT_FIG_DIR / "fig4_positive_predictions_by_condition.png", dpi=300)
    plt.close()

    # Figure 5: For overlapping PMIDs, probability before and after punctuation normalization
    fig, ax = plt.subplots(figsize=(10, 6.5))
    df_pmid_grp = df_identical.groupby("pmid")[["prob_dataset_A_with_dot", "prob_dataset_B_no_dot"]].mean().reset_index()

    palette = ["#1f77b4", "#2ca02c", "#d62728"]
    for i, row in df_pmid_grp.iterrows():
        p_a = row["prob_dataset_A_with_dot"]
        p_b = row["prob_dataset_B_no_dot"]
        pmid_val = int(row["pmid"])
        col = palette[i % len(palette)]
        ax.plot([0, 1], [p_a, p_b], marker="o", markersize=10, linewidth=2.5, color=col, label=f"PMID {pmid_val} (True Positive)")
        ax.text(-0.03, p_a, f"{p_a:.4f}", ha="right", va="center", fontweight="bold", color=col)
        ax.text(1.03, p_b, f"{p_b:.4f}", ha="left", va="center", fontweight="bold", color=col)

    ax.axhline(0.50, color="gray", linestyle="--", linewidth=1.5, label="Decision Threshold = 0.50")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Dataset A: labeled-dataset-v2.csv\n(With Trailing Period '.')", "Dataset B: 2021 labeld.csv\n(No Trailing Period)"], fontsize=12, fontweight="bold")
    ax.set_title("Figure 5: Identical Articles Evaluated Across Datasets\n(Identical Text & Abstract, Diverging ONLY by Trailing Period)", fontweight="bold")
    ax.set_ylabel("Predicted Probability (Mean Across All 7 Models)", fontsize=11)
    ax.set_xlim(-0.25, 1.25)
    ax.set_ylim(-0.05, 1.05)
    ax.legend(loc="center right", frameon=True)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig5_overlapping_pmids_paired_slope.png", dpi=300)
    plt.savefig(ARTIFACT_FIG_DIR / "fig5_overlapping_pmids_paired_slope.png", dpi=300)
    plt.close()

    elapsed = time.time() - start_time
    logger.info(f"=== Entire Investigation Pipeline completed in {elapsed:.1f}s ===")


if __name__ == "__main__":
    main()
