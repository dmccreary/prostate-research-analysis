"""
src/stage9_1/train_stage9_1_cascade.py - Stage 9.1: Two-Stage Concept Bottleneck Screening Benchmark.

Implements the formal protocol specified in reports/stage9_1/stage9_1_design_and_protocol.md:
- Phase 0: Oracle Baseline (Direct Ground-Truth 25 features -> Stage 2 Classifiers)
- Phase 1: 5-Fold Stratified Cross-Validation for Stage 1 Criteria Extraction (Arms 1, 2, 3)
  * Generates Out-of-Fold (OOF) train matrix X_train_OOF (334 x 25)
  * 5-Model Ensemble inference on Val (84 x 25) and Test (105 x 25)
- Phase 2: Stage 2 Decision Modeling (MLP, Logistic Regression, Random Forest)
  * Sweeps thresholds strictly on Validation set
  * Evaluates strictly once on Held-Out Test set
"""

import argparse
import copy
import logging
from pathlib import Path
import random
import sys
import time
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
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
from sklearn.model_selection import StratifiedKFold
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup

# Safe stdout for Windows console
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / "data"
SPLITS_DIR = DATA_DIR / "splits" / "stage5"
MODELS_DIR = BASE_DIR / "models" / "stage9_1"
RESULTS_DIR = BASE_DIR / "results" / "stage9_1"
LOGS_DIR = BASE_DIR / "logs" / "stage9_1"

for d in [MODELS_DIR, RESULTS_DIR, LOGS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOGS_DIR / "stage9_1_cascade_training.log", encoding="utf-8", mode="a"),
    ],
)
logger = logging.getLogger(__name__)

sys.path.insert(0, str(BASE_DIR / "src"))
from preprocessing import clean_clinical_text

# Hyperparameters & Constants
SEED = 2025
N_SPLITS = 5
MAX_LENGTH = 512
BATCH_SIZE_TRAIN = 4
GRAD_ACCUM_STEPS = 2
BATCH_SIZE_EVAL = 8
LR_BACKBONE = 5e-6
LR_HEAD = 2e-5
WEIGHT_DECAY = 1e-4
MAX_GRAD_NORM = 1.0
EPOCHS = 8
PATIENCE = 4
DROPOUT_RATE = 0.25

# 25 Auxiliary Feature Column Names (Canonical Order)
AUX_FEATURE_COLS = [
    "risk_method_damico",
    "risk_method_nccn",
    "pretreatment_risk_stratification",
    "low_risk_group_status",
    "intermediate_risk_group_status",
    "high_risk_group_status",
    "retains_baseline_risk_groups",
    "radical_prostatectomy",
    "standard_ebrt",
    "hypofractionated_ebrt",
    "ultrahypofractionated_ebrt",
    "proton_therapy",
    "ldr_brachytherapy",
    "hdr_brachytherapy",
    "adt_combination",
    "cryotherapy",
    "hifu",
    "has_brfs",
    "has_os",
    "has_mfs",
    "has_css",
    "standard_ebrt_dose_status",
    "median_followup_status",
    "advanced_metastatic_cohort",
    "salvage_postfailure_cohort",
]

# Fixed Verbatim PICO Query for Arms 2 & 3
FIXED_PICO_QUERY = (
    "Find prostate cancer studies stratified by pretreatment low/intermediate/high risk (D’Amico, Zelefsky, or NCCN criteria) "
    "evaluating open/robotic radical prostatectomy; conventional external-beam radiotherapy ≥72 Gy; hypofractionated "
    "radiotherapy/SBRT; LDR/HDR brachytherapy; proton therapy; androgen-deprivation therapy combinations; cryotherapy; "
    "or HIFU. Retain all patients regardless of post-surgical pathological findings. Outcomes include long-term biochemical "
    "recurrence-free survival, overall survival, metastasis-free survival, or cancer-specific survival. When reported, "
    "consider risk-group-specific patient counts and median follow-up ≥5 years."
)


def set_all_seeds(seed: int = SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class AsymmetricLossMultiLabel(nn.Module):
    """
    Asymmetric Loss (ASL) for Multi-Label Clinical Feature Extraction.
    Suppresses gradients from easy negatives via margin shift.
    """

    def __init__(
        self,
        gamma_pos: float = 0.0,
        gamma_neg: float = 2.0,
        margin: float = 0.05,
        eps: float = 1e-7,
    ):
        super().__init__()
        self.gamma_pos = gamma_pos
        self.gamma_neg = gamma_neg
        self.margin = margin
        self.eps = eps

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        log_pos = F.logsigmoid(logits)
        pos_modulating = (1.0 - probs) ** self.gamma_pos if self.gamma_pos > 0 else 1.0
        loss_pos = - pos_modulating * log_pos

        p_m = torch.clamp(probs - self.margin, min=0.0, max=1.0)
        neg_modulating = p_m ** self.gamma_neg
        log_neg = torch.log(torch.clamp(1.0 - p_m, min=self.eps, max=1.0))
        loss_neg = - neg_modulating * log_neg

        total_loss = targets * loss_pos + (1.0 - targets) * loss_neg
        return total_loss.mean()


class CriteriaDataset(Dataset):
    """
    Dataset for Stage 1 Criteria Extraction (25 Auxiliary Targets only).
    """

    def __init__(
        self,
        abstracts: List[str],
        aux_labels: np.ndarray,
        tokenizer: AutoTokenizer,
        pico_query: str = None,
        max_length: int = MAX_LENGTH,
    ):
        self.aux_labels = torch.tensor(aux_labels, dtype=torch.float32)

        if pico_query is not None:
            queries = [pico_query] * len(abstracts)
            enc = tokenizer(
                text=queries,
                text_pair=abstracts,
                padding="max_length",
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
        else:
            enc = tokenizer(
                text=abstracts,
                padding="max_length",
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )

        self.input_ids = enc["input_ids"]
        self.attention_mask = enc["attention_mask"]
        self.token_type_ids = enc.get("token_type_ids", torch.zeros_like(self.input_ids))

    def __len__(self):
        return len(self.aux_labels)

    def __getitem__(self, idx):
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
            "token_type_ids": self.token_type_ids[idx],
            "aux_labels": self.aux_labels[idx],
        }


class CriteriaExtractor(nn.Module):
    """
    Stage 1 Model: Shared Transformer Encoder + 25 Parallel Linear Projection Heads.
    Pure concept bottleneck: NO primary label head.
    """

    def __init__(self, pretrained_name: str, num_features: int = 25, dropout_rate: float = DROPOUT_RATE):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(pretrained_name)
        hidden_size = self.encoder.config.hidden_size  # 768
        self.dropout = nn.Dropout(dropout_rate)

        # 25 independent linear heads
        self.heads = nn.ModuleList([
            nn.Linear(hidden_size, 1) for _ in range(num_features)
        ])

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        if token_type_ids is not None:
            outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        else:
            outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)

        cls_rep = self.dropout(outputs.last_hidden_state[:, 0, :])
        logits = torch.cat([head(cls_rep) for head in self.heads], dim=-1)  # [Batch, 25]
        return logits


class Stage2MLP(nn.Module):
    """
    Stage 2 Decision Model: Lightweight MLP mapping 25 clinical probabilities -> Primary Label.
    """

    def __init__(self, input_dim: int = 25, hidden_dim: int = 32, dropout_rate: float = 0.20):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def evaluate_extractor(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device):
    model.eval()
    total_loss_accum = 0.0
    all_true, all_prob = [], []

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            token_type_ids = batch["token_type_ids"].to(device)
            labels = batch["aux_labels"].to(device)

            logits = model(input_ids, attention_mask, token_type_ids)
            loss = criterion(logits, labels)

            total_loss_accum += loss.item() * len(labels)
            all_prob.append(torch.sigmoid(logits).cpu().numpy())
            all_true.append(labels.cpu().numpy())

    avg_loss = total_loss_accum / len(loader.dataset)
    y_true_np = np.vstack(all_true)
    y_prob_np = np.vstack(all_prob)
    return avg_loss, y_true_np, y_prob_np


def compute_primary_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
    y_pred = (y_prob >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    total = len(y_true)
    flagged = int(tp + fp)

    acc = accuracy_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred, zero_division=0)
    prec = precision_score(y_true, y_pred, zero_division=0)
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    f1 = f1_score(y_true, y_pred, zero_division=0)

    try:
        auroc = float(roc_auc_score(y_true, y_prob))
    except Exception:
        auroc = 0.5

    try:
        p_curve, r_curve, _ = precision_recall_curve(y_true, y_prob)
        pr_auc = float(auc(r_curve, p_curve))
    except Exception:
        pr_auc = float(np.mean(y_true))

    nns = round((flagged / tp), 2) if tp > 0 else float("inf")
    workload_pct = round((flagged / total) * 100.0, 2)
    reduction_pct = round(((total - flagged) / total) * 100.0, 2)

    return {
        "accuracy": float(acc),
        "recall": float(rec),
        "specificity": float(spec),
        "precision": float(prec),
        "f1_score": float(f1),
        "auroc": float(auroc),
        "pr_auc": float(pr_auc),
        "true_positives": int(tp),
        "false_positives": int(fp),
        "true_negatives": int(tn),
        "false_negatives": int(fn),
        "flagged_count": int(flagged),
        "workload_pct": float(workload_pct),
        "workload_reduction_pct": float(reduction_pct),
        "nns": float(nns),
    }


def compute_auxiliary_metrics(aux_true: np.ndarray, aux_prob: np.ndarray, feature_names: List[str], threshold: float = 0.5):
    records = []
    f1_list = []
    auroc_list = []

    for k, name in enumerate(feature_names):
        y_t = aux_true[:, k]
        y_p = aux_prob[:, k]
        y_pred = (y_p >= threshold).astype(int)

        acc = accuracy_score(y_t, y_pred)
        rec = recall_score(y_t, y_pred, zero_division=0)
        prec = precision_score(y_t, y_pred, zero_division=0)
        f1 = f1_score(y_t, y_pred, zero_division=0)
        f1_list.append(f1)

        pos_count = int(y_t.sum())
        neg_count = int((1 - y_t).sum())

        if pos_count > 0 and neg_count > 0:
            try:
                feat_auroc = float(roc_auc_score(y_t, y_p))
            except Exception:
                feat_auroc = 0.5
            try:
                p_c, r_c, _ = precision_recall_curve(y_t, y_p)
                feat_prauc = float(auc(r_c, p_c))
            except Exception:
                feat_prauc = float(np.mean(y_t))
            auroc_list.append(feat_auroc)
        else:
            feat_auroc = float("nan")
            feat_prauc = float("nan")

        records.append({
            "feature": name,
            "accuracy": round(float(acc), 4),
            "recall": round(float(rec), 4),
            "precision": round(float(prec), 4),
            "f1": round(float(f1), 4),
            "auroc": round(feat_auroc, 4) if not np.isnan(feat_auroc) else None,
            "pr_auc": round(feat_prauc, 4) if not np.isnan(feat_prauc) else None,
            "positive_count": pos_count,
            "total_count": len(y_t),
        })

    feat_df = pd.DataFrame(records)
    micro_f1 = f1_score(aux_true.ravel(), (aux_prob.ravel() >= threshold).astype(int), zero_division=0)
    macro_f1 = float(np.mean(f1_list))
    macro_auroc = float(np.mean(auroc_list)) if len(auroc_list) > 0 else 0.5

    summary = {
        "macro_f1": round(macro_f1, 4),
        "micro_f1": round(float(micro_f1), 4),
        "macro_auroc": round(macro_auroc, 4),
    }
    return feat_df, summary


def sweep_thresholds(y_true: np.ndarray, y_prob: np.ndarray, thresholds: np.ndarray = np.linspace(0.01, 0.99, 99)) -> pd.DataFrame:
    rows = []
    for t in thresholds:
        m = compute_primary_metrics(y_true, y_prob, threshold=float(t))
        m["threshold"] = round(float(t), 2)
        rows.append(m)
    return pd.DataFrame(rows)


# ==============================================================================
# PHASE 0: ORACLE CEILING TEST (Zero GPU Compute)
# ==============================================================================
def run_oracle_baseline(train_aux, train_y, val_aux, val_y, test_aux, test_y):
    logger.info("=" * 80)
    logger.info("STARTING PHASE 0: ORACLE CEILING TEST (Using Ground-Truth 25 Labels)")
    logger.info("=" * 80)

    results = []

    # 1. Oracle Logistic Regression
    lr = LogisticRegression(C=1.0, max_iter=1000, random_state=SEED)
    lr.fit(train_aux, train_y)
    val_probs_lr = lr.predict_proba(val_aux)[:, 1]
    test_probs_lr = lr.predict_proba(test_aux)[:, 1]

    val_sweep_lr = sweep_thresholds(val_y, val_probs_lr)
    c_100_lr = val_sweep_lr[val_sweep_lr["recall"] >= 1.0]
    t_100_lr = c_100_lr.sort_values(by=["specificity", "threshold"], ascending=[False, False]).iloc[0]["threshold"] if len(c_100_lr) > 0 else 0.1

    m_def = compute_primary_metrics(test_y, test_probs_lr, 0.50)
    m_def["model"] = "Oracle Logistic Regression"
    m_def["operating_point"] = "Default (0.50)"
    m_def["threshold"] = 0.50
    results.append(m_def)

    m_100 = compute_primary_metrics(test_y, test_probs_lr, t_100_lr)
    m_100["model"] = "Oracle Logistic Regression"
    m_100["operating_point"] = "Recall = 100%"
    m_100["threshold"] = t_100_lr
    results.append(m_100)

    # 2. Oracle Random Forest
    rf = RandomForestClassifier(n_estimators=100, max_depth=5, min_samples_leaf=4, random_state=SEED)
    rf.fit(train_aux, train_y)
    val_probs_rf = rf.predict_proba(val_aux)[:, 1]
    test_probs_rf = rf.predict_proba(test_aux)[:, 1]

    val_sweep_rf = sweep_thresholds(val_y, val_probs_rf)
    c_100_rf = val_sweep_rf[val_sweep_rf["recall"] >= 1.0]
    t_100_rf = c_100_rf.sort_values(by=["specificity", "threshold"], ascending=[False, False]).iloc[0]["threshold"] if len(c_100_rf) > 0 else 0.1

    m_def_rf = compute_primary_metrics(test_y, test_probs_rf, 0.50)
    m_def_rf["model"] = "Oracle Random Forest"
    m_def_rf["operating_point"] = "Default (0.50)"
    m_def_rf["threshold"] = 0.50
    results.append(m_def_rf)

    m_100_rf = compute_primary_metrics(test_y, test_probs_rf, t_100_rf)
    m_100_rf["model"] = "Oracle Random Forest"
    m_100_rf["operating_point"] = "Recall = 100%"
    m_100_rf["threshold"] = t_100_rf
    results.append(m_100_rf)

    # 3. Oracle MLP (PyTorch)
    set_all_seeds(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mlp = Stage2MLP(input_dim=25, hidden_dim=32, dropout_rate=0.2).to(device)
    opt = AdamW(mlp.parameters(), lr=1e-3, weight_decay=1e-3)
    crit = nn.BCEWithLogitsLoss()

    X_tr_t = torch.tensor(train_aux, dtype=torch.float32).to(device)
    y_tr_t = torch.tensor(train_y, dtype=torch.float32).to(device)
    X_val_t = torch.tensor(val_aux, dtype=torch.float32).to(device)
    X_te_t = torch.tensor(test_aux, dtype=torch.float32).to(device)

    best_loss = float("inf")
    best_weights = None
    for ep in range(150):
        mlp.train()
        opt.zero_grad()
        loss = crit(mlp(X_tr_t), y_tr_t)
        loss.backward()
        opt.step()

        mlp.eval()
        with torch.no_grad():
            v_loss = crit(mlp(X_val_t), torch.tensor(val_y, dtype=torch.float32).to(device)).item()
            if v_loss < best_loss:
                best_loss = v_loss
                best_weights = copy.deepcopy(mlp.state_dict())

    mlp.load_state_dict(best_weights)
    mlp.eval()
    with torch.no_grad():
        val_probs_mlp = torch.sigmoid(mlp(X_val_t)).cpu().numpy()
        test_probs_mlp = torch.sigmoid(mlp(X_te_t)).cpu().numpy()

    val_sweep_mlp = sweep_thresholds(val_y, val_probs_mlp)
    c_100_mlp = val_sweep_mlp[val_sweep_mlp["recall"] >= 1.0]
    t_100_mlp = c_100_mlp.sort_values(by=["specificity", "threshold"], ascending=[False, False]).iloc[0]["threshold"] if len(c_100_mlp) > 0 else 0.1

    m_def_mlp = compute_primary_metrics(test_y, test_probs_mlp, 0.50)
    m_def_mlp["model"] = "Oracle MLP"
    m_def_mlp["operating_point"] = "Default (0.50)"
    m_def_mlp["threshold"] = 0.50
    results.append(m_def_mlp)

    m_100_mlp = compute_primary_metrics(test_y, test_probs_mlp, t_100_mlp)
    m_100_mlp["model"] = "Oracle MLP"
    m_100_mlp["operating_point"] = "Recall = 100%"
    m_100_mlp["threshold"] = t_100_mlp
    results.append(m_100_mlp)

    oracle_df = pd.DataFrame(results)
    oracle_df.to_csv(RESULTS_DIR / "oracle_baseline_metrics.csv", index=False)
    logger.info(f"Oracle Ceiling Results Saved to {RESULTS_DIR / 'oracle_baseline_metrics.csv'}")
    logger.info(f"Oracle LR Test AUROC: {m_def['auroc']:.4f} | PR-AUC: {m_def['pr_auc']:.4f}")
    logger.info(f"Oracle MLP Test AUROC: {m_def_mlp['auroc']:.4f} | PR-AUC: {m_def_mlp['pr_auc']:.4f}")
    logger.info(f"Oracle RF Test AUROC: {m_def_rf['auroc']:.4f} | PR-AUC: {m_def_rf['pr_auc']:.4f}")
    return oracle_df


# ==============================================================================
# PHASE 1: 5-FOLD STAGE 1 TRAINING & OOF FEATURE EXTRACTION
# ==============================================================================
def train_and_extract_arm(arm_id: int, train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame):
    logger.info("=" * 80)
    logger.info(f"STARTING STAGE 9.1 CASCADE: ARM {arm_id} (5-FOLD OOF EXTRACTION)")
    logger.info("=" * 80)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if arm_id == 1:
        arm_name = "Arm 1: PubMedBERT Abstract-Only"
        pretrained_name = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract"
        pico_query = None
    elif arm_id == 2:
        arm_name = "Arm 2: PubMedBERT Abstract + PICO"
        pretrained_name = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract"
        pico_query = FIXED_PICO_QUERY
    elif arm_id == 3:
        arm_name = "Arm 3: MedCPT Abstract + PICO"
        pretrained_name = "ncbi/MedCPT-Cross-Encoder"
        pico_query = FIXED_PICO_QUERY
    else:
        raise ValueError(f"Invalid arm_id: {arm_id}")

    tokenizer = AutoTokenizer.from_pretrained(pretrained_name)

    # Clean text (Abstract ONLY)
    train_abstracts = [clean_clinical_text(str(a or "")) for a in train_df["abstract"]]
    val_abstracts = [clean_clinical_text(str(a or "")) for a in val_df["abstract"]]
    test_abstracts = [clean_clinical_text(str(a or "")) for a in test_df["abstract"]]

    train_aux = train_df[AUX_FEATURE_COLS].to_numpy().astype(np.float32)
    val_aux = val_df[AUX_FEATURE_COLS].to_numpy().astype(np.float32)
    test_aux = test_df[AUX_FEATURE_COLS].to_numpy().astype(np.float32)

    train_primary = train_df["label"].to_numpy().astype(int)
    val_primary = val_df["label"].to_numpy().astype(int)
    test_primary = test_df["label"].to_numpy().astype(int)

    # Stratified 5-Fold Split inside Train (N=334)
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    oof_predictions = np.zeros_like(train_aux)
    fold_models = []

    criterion = AsymmetricLossMultiLabel(gamma_pos=0.0, gamma_neg=2.0, margin=0.05)

    ds_val = CriteriaDataset(val_abstracts, val_aux, tokenizer, pico_query=pico_query)
    ds_test = CriteriaDataset(test_abstracts, test_aux, tokenizer, pico_query=pico_query)
    loader_val = DataLoader(ds_val, batch_size=BATCH_SIZE_EVAL, shuffle=False)
    loader_test = DataLoader(ds_test, batch_size=BATCH_SIZE_EVAL, shuffle=False)

    for fold, (tr_idx, val_fold_idx) in enumerate(skf.split(train_abstracts, train_primary), 1):
        logger.info(f"--- [{arm_name}] Fold {fold}/{N_SPLITS} (Train={len(tr_idx)}, OOF Val={len(val_fold_idx)}) ---")
        set_all_seeds(SEED + fold)

        tr_abs_fold = [train_abstracts[i] for i in tr_idx]
        tr_aux_fold = train_aux[tr_idx]

        val_abs_fold = [train_abstracts[i] for i in val_fold_idx]
        val_aux_fold = train_aux[val_fold_idx]

        ds_tr_fold = CriteriaDataset(tr_abs_fold, tr_aux_fold, tokenizer, pico_query=pico_query)
        ds_val_fold = CriteriaDataset(val_abs_fold, val_aux_fold, tokenizer, pico_query=pico_query)

        loader_tr_fold = DataLoader(ds_tr_fold, batch_size=BATCH_SIZE_TRAIN, shuffle=True)
        loader_val_fold = DataLoader(ds_val_fold, batch_size=BATCH_SIZE_EVAL, shuffle=False)

        model = CriteriaExtractor(pretrained_name=pretrained_name, num_features=len(AUX_FEATURE_COLS)).to(device)

        optimizer_grouped = [
            {"params": [p for n, p in model.encoder.named_parameters()], "lr": LR_BACKBONE, "weight_decay": WEIGHT_DECAY},
            {"params": [p for n, p in model.heads.named_parameters()], "lr": LR_HEAD, "weight_decay": WEIGHT_DECAY},
        ]
        optimizer = AdamW(optimizer_grouped)
        total_steps = (len(loader_tr_fold) // GRAD_ACCUM_STEPS) * EPOCHS
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(total_steps * 0.1), num_training_steps=total_steps)

        best_fold_loss = float("inf")
        best_fold_weights = None
        best_oof_probs = None
        patience_counter = 0

        for epoch in range(1, EPOCHS + 1):
            if epoch == 1:
                for p in model.encoder.parameters():
                    p.requires_grad = False
            elif epoch == 2:
                for p in model.encoder.parameters():
                    p.requires_grad = True

            model.train()
            train_loss_accum = 0.0
            optimizer.zero_grad()

            for step, batch in enumerate(loader_tr_fold):
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                token_type_ids = batch["token_type_ids"].to(device)
                labels = batch["aux_labels"].to(device)

                logits = model(input_ids, attention_mask, token_type_ids)
                loss = criterion(logits, labels) / GRAD_ACCUM_STEPS
                loss.backward()

                train_loss_accum += loss.item() * GRAD_ACCUM_STEPS * len(labels)

                if (step + 1) % GRAD_ACCUM_STEPS == 0 or (step + 1) == len(loader_tr_fold):
                    torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()

            # Fold validation evaluation
            v_loss, _, v_prob = evaluate_extractor(model, loader_val_fold, criterion, device)
            _, v_summary = compute_auxiliary_metrics(val_aux_fold, v_prob, AUX_FEATURE_COLS, threshold=0.5)

            logger.info(f"  Fold {fold} | Epoch {epoch:02d} | Train Loss: {train_loss_accum/len(ds_tr_fold):.4f} | "
                        f"Val Loss: {v_loss:.4f} | Aux Macro-F1: {v_summary['macro_f1']:.4f} | Macro-AUROC: {v_summary['macro_auroc']:.4f}")

            if v_loss < best_fold_loss:
                best_fold_loss = v_loss
                best_fold_weights = copy.deepcopy(model.state_dict())
                best_oof_probs = v_prob.copy()
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= PATIENCE:
                    logger.info(f"  Early stopping fold {fold} at epoch {epoch:02d}")
                    break

        oof_predictions[val_fold_idx] = best_oof_probs

        # Save fold model
        fold_ckpt_path = MODELS_DIR / f"arm_{arm_id}_fold_{fold}.pt"
        torch.save(best_fold_weights, fold_ckpt_path)
        logger.info(f"  Saved fold {fold} checkpoint to {fold_ckpt_path}")

        # Load best weights into memory for ensemble
        model.load_state_dict(best_fold_weights)
        fold_models.append(model)
        torch.cuda.empty_cache()

    # Save OOF train feature matrix
    oof_df = pd.DataFrame(oof_predictions, columns=AUX_FEATURE_COLS)
    oof_df["pmid"] = train_df["pmid"].values
    oof_df["label"] = train_primary
    oof_df.to_csv(RESULTS_DIR / f"arm_{arm_id}_oof_features.csv", index=False)

    # 5-Model Ensemble Inference on Val and Test
    logger.info(f"Running 5-Model Ensemble Inference on Val (N={len(val_df)}) and Test (N={len(test_df)})...")
    val_probs_ensemble = np.zeros((len(val_df), len(AUX_FEATURE_COLS)))
    test_probs_ensemble = np.zeros((len(test_df), len(AUX_FEATURE_COLS)))

    for m in fold_models:
        _, _, v_p = evaluate_extractor(m, loader_val, criterion, device)
        _, _, t_p = evaluate_extractor(m, loader_test, criterion, device)
        val_probs_ensemble += v_p / N_SPLITS
        test_probs_ensemble += t_p / N_SPLITS

    val_feat_df = pd.DataFrame(val_probs_ensemble, columns=AUX_FEATURE_COLS)
    val_feat_df["pmid"] = val_df["pmid"].values
    val_feat_df["label"] = val_primary
    val_feat_df.to_csv(RESULTS_DIR / f"arm_{arm_id}_val_features.csv", index=False)

    test_feat_df = pd.DataFrame(test_probs_ensemble, columns=AUX_FEATURE_COLS)
    test_feat_df["pmid"] = test_df["pmid"].values
    test_feat_df["label"] = test_primary
    test_feat_df.to_csv(RESULTS_DIR / f"arm_{arm_id}_test_features.csv", index=False)

    # Stage 1 Feature Extraction Quality on Test Set
    test_feat_eval_df, test_feat_summary = compute_auxiliary_metrics(test_aux, test_probs_ensemble, AUX_FEATURE_COLS, threshold=0.5)
    test_feat_eval_df["arm_id"] = arm_id
    test_feat_eval_df["arm_name"] = arm_name
    test_feat_eval_df.to_csv(RESULTS_DIR / f"arm_{arm_id}_stage1_test_feature_metrics.csv", index=False)
    logger.info(f"[{arm_name}] Stage 1 Test Aux Macro-F1: {test_feat_summary['macro_f1']:.4f} | Macro-AUROC: {test_feat_summary['macro_auroc']:.4f}")

    return {
        "arm_id": arm_id,
        "arm_name": arm_name,
        "X_train_oof": oof_predictions,
        "y_train": train_primary,
        "X_val": val_probs_ensemble,
        "y_val": val_primary,
        "X_test": test_probs_ensemble,
        "y_test": test_primary,
        "test_feat_summary": test_feat_summary,
    }


# ==============================================================================
# PHASE 2: STAGE 2 DECISION MODELING & BENCHMARKING
# ==============================================================================
def train_and_eval_stage2_classifiers(arm_data: Dict) -> pd.DataFrame:
    arm_id = arm_data["arm_id"]
    arm_name = arm_data["arm_name"]
    X_tr = arm_data["X_train_oof"]
    y_tr = arm_data["y_train"]
    X_va = arm_data["X_val"]
    y_va = arm_data["y_val"]
    X_te = arm_data["X_test"]
    y_te = arm_data["y_test"]

    logger.info("=" * 80)
    logger.info(f"PHASE 2: TRAINING STAGE 2 DECISION CLASSIFIERS FOR {arm_name}")
    logger.info("=" * 80)

    results = []

    # --------------------------------------------------------------------------
    # 1. Model A: Stage 2 MLP (PyTorch)
    # --------------------------------------------------------------------------
    set_all_seeds(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mlp = Stage2MLP(input_dim=25, hidden_dim=32, dropout_rate=0.20).to(device)
    opt = AdamW(mlp.parameters(), lr=1e-3, weight_decay=1e-3)
    crit = nn.BCEWithLogitsLoss()

    X_tr_t = torch.tensor(X_tr, dtype=torch.float32).to(device)
    y_tr_t = torch.tensor(y_tr, dtype=torch.float32).to(device)
    X_va_t = torch.tensor(X_va, dtype=torch.float32).to(device)
    X_te_t = torch.tensor(X_te, dtype=torch.float32).to(device)

    best_loss = float("inf")
    best_mlp_weights = None
    for ep in range(150):
        mlp.train()
        opt.zero_grad()
        loss = crit(mlp(X_tr_t), y_tr_t)
        loss.backward()
        opt.step()

        mlp.eval()
        with torch.no_grad():
            v_loss = crit(mlp(X_va_t), torch.tensor(y_va, dtype=torch.float32).to(device)).item()
            if v_loss < best_loss:
                best_loss = v_loss
                best_mlp_weights = copy.deepcopy(mlp.state_dict())

    mlp.load_state_dict(best_mlp_weights)
    torch.save(best_mlp_weights, MODELS_DIR / f"arm_{arm_id}_stage2_mlp.pt")

    mlp.eval()
    with torch.no_grad():
        va_prob_mlp = torch.sigmoid(mlp(X_va_t)).cpu().numpy()
        te_prob_mlp = torch.sigmoid(mlp(X_te_t)).cpu().numpy()

    va_sweep_mlp = sweep_thresholds(y_va, va_prob_mlp)

    # Threshold calibration on Val
    c_90 = va_sweep_mlp[va_sweep_mlp["recall"] >= 0.90]
    t_90_mlp = c_90.sort_values(by=["specificity", "threshold"], ascending=[False, False]).iloc[0]["threshold"] if len(c_90) > 0 else 0.50

    c_95 = va_sweep_mlp[va_sweep_mlp["recall"] >= 0.95]
    t_95_mlp = c_95.sort_values(by=["specificity", "threshold"], ascending=[False, False]).iloc[0]["threshold"] if len(c_95) > 0 else 0.50

    c_100 = va_sweep_mlp[va_sweep_mlp["recall"] >= 1.0]
    t_100_mlp = c_100.sort_values(by=["specificity", "threshold"], ascending=[False, False]).iloc[0]["threshold"] if len(c_100) > 0 else 0.10

    op_points = [
        ("Default (0.50)", 0.50),
        ("Recall >= 90%", t_90_mlp),
        ("Recall >= 95%", t_95_mlp),
        ("Recall = 100%", t_100_mlp),
    ]

    for op_name, t_val in op_points:
        m = compute_primary_metrics(y_te, te_prob_mlp, threshold=t_val)
        m["arm_id"] = arm_id
        m["arm_name"] = arm_name
        m["classifier"] = "Stage 2 MLP"
        m["operating_point"] = op_name
        m["threshold"] = round(float(t_val), 2)
        results.append(m)

    # --------------------------------------------------------------------------
    # 2. Model B: Logistic Regression (L2)
    # --------------------------------------------------------------------------
    lr = LogisticRegression(C=1.0, max_iter=1000, random_state=SEED)
    lr.fit(X_tr, y_tr)
    va_prob_lr = lr.predict_proba(X_va)[:, 1]
    te_prob_lr = lr.predict_proba(X_te)[:, 1]

    va_sweep_lr = sweep_thresholds(y_va, va_prob_lr)
    c_100_lr = va_sweep_lr[va_sweep_lr["recall"] >= 1.0]
    t_100_lr = c_100_lr.sort_values(by=["specificity", "threshold"], ascending=[False, False]).iloc[0]["threshold"] if len(c_100_lr) > 0 else 0.10

    m_lr_def = compute_primary_metrics(y_te, te_prob_lr, 0.50)
    m_lr_def["arm_id"] = arm_id
    m_lr_def["arm_name"] = arm_name
    m_lr_def["classifier"] = "Stage 2 Logistic Regression"
    m_lr_def["operating_point"] = "Default (0.50)"
    m_lr_def["threshold"] = 0.50
    results.append(m_lr_def)

    m_lr_100 = compute_primary_metrics(y_te, te_prob_lr, t_100_lr)
    m_lr_100["arm_id"] = arm_id
    m_lr_100["arm_name"] = arm_name
    m_lr_100["classifier"] = "Stage 2 Logistic Regression"
    m_lr_100["operating_point"] = "Recall = 100%"
    m_lr_100["threshold"] = round(float(t_100_lr), 2)
    results.append(m_lr_100)

    # --------------------------------------------------------------------------
    # 3. Model C: Random Forest
    # --------------------------------------------------------------------------
    rf = RandomForestClassifier(n_estimators=100, max_depth=5, min_samples_leaf=4, random_state=SEED)
    rf.fit(X_tr, y_tr)
    va_prob_rf = rf.predict_proba(X_va)[:, 1]
    te_prob_rf = rf.predict_proba(X_te)[:, 1]

    va_sweep_rf = sweep_thresholds(y_va, va_prob_rf)
    c_100_rf = va_sweep_rf[va_sweep_rf["recall"] >= 1.0]
    t_100_rf = c_100_rf.sort_values(by=["specificity", "threshold"], ascending=[False, False]).iloc[0]["threshold"] if len(c_100_rf) > 0 else 0.10

    m_rf_def = compute_primary_metrics(y_te, te_prob_rf, 0.50)
    m_rf_def["arm_id"] = arm_id
    m_rf_def["arm_name"] = arm_name
    m_rf_def["classifier"] = "Stage 2 Random Forest"
    m_rf_def["operating_point"] = "Default (0.50)"
    m_rf_def["threshold"] = 0.50
    results.append(m_rf_def)

    m_rf_100 = compute_primary_metrics(y_te, te_prob_rf, t_100_rf)
    m_rf_100["arm_id"] = arm_id
    m_rf_100["arm_name"] = arm_name
    m_rf_100["classifier"] = "Stage 2 Random Forest"
    m_rf_100["operating_point"] = "Recall = 100%"
    m_rf_100["threshold"] = round(float(t_100_rf), 2)
    results.append(m_rf_100)

    arm_test_df = pd.DataFrame(results)
    arm_test_df.to_csv(RESULTS_DIR / f"arm_{arm_id}_decision_test_metrics.csv", index=False)

    logger.info(f"[{arm_name}] Test MLP AUROC: {results[0]['auroc']:.4f} | PR-AUC: {results[0]['pr_auc']:.4f}")
    logger.info(f"[{arm_name}] Test MLP Spec @ 100% Target: {results[3]['specificity']:.4f} | FN: {results[3]['false_negatives']}")
    logger.info(f"[{arm_name}] Test LR AUROC: {m_lr_def['auroc']:.4f} | Test RF AUROC: {m_rf_def['auroc']:.4f}")

    return arm_test_df


# ==============================================================================
# MAIN ORCHESTRATOR
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(description="Stage 9.1 Concept Bottleneck Cascade Benchmark")
    parser.add_argument("--arm", type=str, default="all", choices=["1", "2", "3", "all", "oracle"], help="Arms to run")
    args = parser.parse_args()

    # Load data
    aux25_df = pd.read_csv(DATA_DIR / "labeled-dataset-v2-aux25.csv")
    train_split = pd.read_csv(SPLITS_DIR / "train.csv")
    val_split = pd.read_csv(SPLITS_DIR / "val.csv")
    test_split = pd.read_csv(SPLITS_DIR / "test.csv")

    train_df = train_split[["pmid"]].merge(aux25_df, on="pmid", how="inner")
    val_df = val_split[["pmid"]].merge(aux25_df, on="pmid", how="inner")
    test_df = test_split[["pmid"]].merge(aux25_df, on="pmid", how="inner")

    train_aux_gt = train_df[AUX_FEATURE_COLS].to_numpy().astype(np.float32)
    val_aux_gt = val_df[AUX_FEATURE_COLS].to_numpy().astype(np.float32)
    test_aux_gt = test_df[AUX_FEATURE_COLS].to_numpy().astype(np.float32)

    train_y = train_df["label"].to_numpy().astype(int)
    val_y = val_df["label"].to_numpy().astype(int)
    test_y = test_df["label"].to_numpy().astype(int)

    # 1. Run Phase 0 (Oracle Ceiling)
    oracle_df = run_oracle_baseline(train_aux_gt, train_y, val_aux_gt, val_y, test_aux_gt, test_y)

    if args.arm == "oracle":
        logger.info("Oracle baseline finished. Exiting as requested.")
        return

    arms_to_run = [1, 2, 3] if args.arm == "all" else [int(args.arm)]
    all_decision_metrics = []

    for arm in arms_to_run:
        arm_data = train_and_extract_arm(arm, train_df, val_df, test_df)
        decision_metrics_df = train_and_eval_stage2_classifiers(arm_data)
        all_decision_metrics.append(decision_metrics_df)

    comb_decision_df = pd.concat(all_decision_metrics, ignore_index=True)
    comb_decision_df.to_csv(RESULTS_DIR / "stage9_1_test_metrics.csv", index=False)

    logger.info("=" * 80)
    logger.info(f"STAGE 9.1 CASCADE BENCHMARK COMPLETED SUCCESSFULLY! Results saved to {RESULTS_DIR}")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
