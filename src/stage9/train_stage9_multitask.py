"""
src/stage9/train_stage9_multitask.py - Stage 9: Multitask Parallel-Head Screening Benchmark.

Implements the formal protocol specified in reports/stage9/stage9_design_and_protocol.md:
Three Multitask Experiments:
- Arm 1: PubMedBERT, Abstract ONLY (no query) + 26 heads (1 primary + 25 auxiliary)
- Arm 2: PubMedBERT, Abstract + Fixed PICO Query + 26 heads
- Arm 3: MedCPT Cross-Encoder, Abstract + Fixed PICO Query + 26 heads

Dataset: data/labeled-dataset-v2-aux25.csv (N=523)
Zero-leakage split matching Stage 5:
- Sub-Train: N=334 (180 Pos, 154 Neg)
- Validation: N=84 (45 Pos, 39 Neg)
- Held-Out Test: N=105 (56 Pos, 49 Neg)
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
MODELS_DIR = BASE_DIR / "models" / "stage9"
RESULTS_DIR = BASE_DIR / "results" / "stage9"
LOGS_DIR = BASE_DIR / "logs" / "stage9"

for d in [MODELS_DIR, RESULTS_DIR, LOGS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOGS_DIR / "stage9_multitask_training.log", encoding="utf-8", mode="a"),
    ],
)
logger = logging.getLogger(__name__)

sys.path.insert(0, str(BASE_DIR / "src"))
from preprocessing import clean_clinical_text

# Hyperparameters & Constants
SEED = 2025
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

# 25 Auxiliary Feature Column Names (in exact canonical order)
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


class AsymmetricLossWithCostWeighting(nn.Module):
    """
    Asymmetric Loss (ASL) with Margin Shift.
    L_+ = - pos_weight * (1 - p)^gamma_pos * log(p)
    L_- = - (p_m)^gamma_neg * log(1 - p_m + eps), where p_m = max(p - margin, 0)
    """

    def __init__(
        self,
        gamma_pos: float = 0.0,
        gamma_neg: float = 2.0,
        margin: float = 0.05,
        pos_weight: float = 1.0,
        eps: float = 1e-7,
    ):
        super().__init__()
        self.gamma_pos = gamma_pos
        self.gamma_neg = gamma_neg
        self.margin = margin
        self.pos_weight = pos_weight
        self.eps = eps

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        log_pos = F.logsigmoid(logits)
        pos_modulating = (1.0 - probs) ** self.gamma_pos if self.gamma_pos > 0 else 1.0
        loss_pos = - self.pos_weight * pos_modulating * log_pos

        p_m = torch.clamp(probs - self.margin, min=0.0, max=1.0)
        neg_modulating = p_m ** self.gamma_neg
        log_neg = torch.log(torch.clamp(1.0 - p_m, min=self.eps, max=1.0))
        loss_neg = - neg_modulating * log_neg

        total_loss = targets * loss_pos + (1.0 - targets) * loss_neg
        return total_loss.mean()


class MultitaskDataset(Dataset):
    """
    Multitask Dataset supporting:
    - Single sequence (abstract only) for Arm 1
    - Dual sequence (PICO query + abstract) for Arms 2 & 3
    """

    def __init__(
        self,
        abstracts: List[str],
        primary_labels: List[int],
        aux_labels: np.ndarray,
        tokenizer: AutoTokenizer,
        pico_query: str = None,
        max_length: int = MAX_LENGTH,
    ):
        self.primary_labels = torch.tensor(primary_labels, dtype=torch.float32)
        self.aux_labels = torch.tensor(aux_labels, dtype=torch.float32)

        if pico_query is not None:
            # Cross-encoder: Query (Segment A) + Abstract (Segment B)
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
            # Single sequence: Abstract only
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
        return len(self.primary_labels)

    def __getitem__(self, idx):
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
            "token_type_ids": self.token_type_ids[idx],
            "primary_label": self.primary_labels[idx],
            "aux_labels": self.aux_labels[idx],
        }


class MultitaskClinicalModel(nn.Module):
    """
    Shared Transformer Encoder + 26 Parallel Linear Heads:
    - 1 Primary screening head: Linear(768, 1)
    - 25 Auxiliary clinical feature heads: 25 x Linear(768, 1)
    """

    def __init__(self, pretrained_name: str, num_aux_features: int = 25, dropout_rate: float = DROPOUT_RATE):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(pretrained_name)
        hidden_size = self.encoder.config.hidden_size  # 768
        self.dropout = nn.Dropout(dropout_rate)

        # Primary screening head
        self.primary_head = nn.Linear(hidden_size, 1)

        # 25 auxiliary clinical feature heads
        self.aux_heads = nn.ModuleList([
            nn.Linear(hidden_size, 1) for _ in range(num_aux_features)
        ])

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        if token_type_ids is not None:
            outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        else:
            outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)

        cls_rep = self.dropout(outputs.last_hidden_state[:, 0, :])
        primary_logit = self.primary_head(cls_rep).squeeze(-1)
        aux_logits = torch.cat([head(cls_rep) for head in self.aux_heads], dim=-1)  # [B, 25]
        return primary_logit, aux_logits


def evaluate_loader(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    aux_loss_weight: float = 0.04,
) -> Tuple[float, float, float, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    total_loss_accum = 0.0
    primary_loss_accum = 0.0
    aux_loss_accum = 0.0

    all_primary_true, all_primary_prob = [], []
    all_aux_true, all_aux_prob = [], []

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            token_type_ids = batch["token_type_ids"].to(device)
            primary_labels = batch["primary_label"].to(device)
            aux_labels = batch["aux_labels"].to(device)

            p_logits, a_logits = model(input_ids, attention_mask, token_type_ids)

            p_loss = criterion(p_logits, primary_labels)
            a_loss = 0.0
            for k in range(a_logits.shape[1]):
                a_loss += criterion(a_logits[:, k], aux_labels[:, k])

            batch_total_loss = p_loss + aux_loss_weight * a_loss

            batch_n = len(primary_labels)
            total_loss_accum += batch_total_loss.item() * batch_n
            primary_loss_accum += p_loss.item() * batch_n
            aux_loss_accum += a_loss.item() * batch_n

            all_primary_prob.extend(torch.sigmoid(p_logits).cpu().numpy().tolist())
            all_primary_true.extend(primary_labels.cpu().numpy().tolist())

            all_aux_prob.append(torch.sigmoid(a_logits).cpu().numpy())
            all_aux_true.append(aux_labels.cpu().numpy())

    dataset_n = len(loader.dataset)
    avg_total_loss = total_loss_accum / dataset_n
    avg_primary_loss = primary_loss_accum / dataset_n
    avg_aux_loss = aux_loss_accum / dataset_n

    primary_true_np = np.array(all_primary_true)
    primary_prob_np = np.array(all_primary_prob)
    aux_true_np = np.vstack(all_aux_true)
    aux_prob_np = np.vstack(all_aux_prob)

    return avg_total_loss, avg_primary_loss, avg_aux_loss, primary_true_np, primary_prob_np, aux_true_np, aux_prob_np


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


def compute_auxiliary_metrics(aux_true: np.ndarray, aux_prob: np.ndarray, feature_names: List[str], threshold: float = 0.5) -> Tuple[pd.DataFrame, Dict[str, float]]:
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


def run_experiment_arm(arm_id: int):
    logger.info("=" * 80)
    logger.info(f"STARTING STAGE 9 MULTITASK EXPERIMENT: ARM {arm_id}")
    logger.info("=" * 80)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Execution Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    if arm_id == 1:
        arm_name = "Arm 1: PubMedBERT Abstract-Only (Multitask)"
        pretrained_name = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract"
        pico_query = None
    elif arm_id == 2:
        arm_name = "Arm 2: PubMedBERT Abstract + PICO (Multitask)"
        pretrained_name = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract"
        pico_query = FIXED_PICO_QUERY
    elif arm_id == 3:
        arm_name = "Arm 3: MedCPT Abstract + PICO (Multitask)"
        pretrained_name = "ncbi/MedCPT-Cross-Encoder"
        pico_query = FIXED_PICO_QUERY
    else:
        raise ValueError(f"Invalid arm_id: {arm_id}. Expected 1, 2, or 3.")

    logger.info(f"Model Backbone: {pretrained_name}")
    logger.info(f"PICO Query: {'None (Abstract Only)' if pico_query is None else 'Fixed Verbatim 73-word Query'}")

    set_all_seeds(SEED)

    aux25_df = pd.read_csv(DATA_DIR / "labeled-dataset-v2-aux25.csv")
    train_split = pd.read_csv(SPLITS_DIR / "train.csv")
    val_split = pd.read_csv(SPLITS_DIR / "val.csv")
    test_split = pd.read_csv(SPLITS_DIR / "test.csv")

    train_df = train_split[["pmid"]].merge(aux25_df, on="pmid", how="inner")
    val_df = val_split[["pmid"]].merge(aux25_df, on="pmid", how="inner")
    test_df = test_split[["pmid"]].merge(aux25_df, on="pmid", how="inner")

    logger.info(f"Loaded Splits: Train={len(train_df)} (Pos={int(train_df['label'].sum())}, Neg={len(train_df)-int(train_df['label'].sum())}), "
                f"Val={len(val_df)} (Pos={int(val_df['label'].sum())}, Neg={len(val_df)-int(val_df['label'].sum())}), "
                f"Test={len(test_df)} (Pos={int(test_df['label'].sum())}, Neg={len(test_df)-int(test_df['label'].sum())})")

    # Clean text (Abstract ONLY)
    train_abstracts = [clean_clinical_text(str(a or "")) for a in train_df["abstract"]]
    val_abstracts = [clean_clinical_text(str(a or "")) for a in val_df["abstract"]]
    test_abstracts = [clean_clinical_text(str(a or "")) for a in test_df["abstract"]]

    train_primary = train_df["label"].tolist()
    val_primary = val_df["label"].tolist()
    test_primary = test_df["label"].tolist()

    train_aux = train_df[AUX_FEATURE_COLS].to_numpy().astype(np.float32)
    val_aux = val_df[AUX_FEATURE_COLS].to_numpy().astype(np.float32)
    test_aux = test_df[AUX_FEATURE_COLS].to_numpy().astype(np.float32)

    tokenizer = AutoTokenizer.from_pretrained(pretrained_name)

    ds_train = MultitaskDataset(train_abstracts, train_primary, train_aux, tokenizer, pico_query=pico_query)
    ds_val = MultitaskDataset(val_abstracts, val_primary, val_aux, tokenizer, pico_query=pico_query)
    ds_test = MultitaskDataset(test_abstracts, test_primary, test_aux, tokenizer, pico_query=pico_query)

    loader_train = DataLoader(ds_train, batch_size=BATCH_SIZE_TRAIN, shuffle=True)
    loader_val = DataLoader(ds_val, batch_size=BATCH_SIZE_EVAL, shuffle=False)
    loader_test = DataLoader(ds_test, batch_size=BATCH_SIZE_EVAL, shuffle=False)

    model = MultitaskClinicalModel(pretrained_name=pretrained_name, num_aux_features=len(AUX_FEATURE_COLS)).to(device)

    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.encoder.named_parameters()],
            "lr": LR_BACKBONE,
            "weight_decay": WEIGHT_DECAY,
        },
        {
            "params": [p for n, p in model.primary_head.named_parameters()] + [p for n, p in model.aux_heads.named_parameters()],
            "lr": LR_HEAD,
            "weight_decay": WEIGHT_DECAY,
        },
    ]

    optimizer = AdamW(optimizer_grouped_parameters)

    total_opt_steps = (len(loader_train) // GRAD_ACCUM_STEPS) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_opt_steps * 0.1),
        num_training_steps=total_opt_steps,
    )

    criterion = AsymmetricLossWithCostWeighting(gamma_pos=0.0, gamma_neg=2.0, margin=0.05, pos_weight=1.0)
    aux_loss_weight = 0.04

    best_val_loss = float("inf")
    best_epoch = 0
    best_model_weights = None
    best_val_primary_probs = None
    best_val_aux_probs = None
    history_records = []
    patience_counter = 0

    start_train_time = time.time()

    for epoch in range(1, EPOCHS + 1):
        epoch_start = time.time()

        if epoch == 1:
            logger.info("  [Warmup Stage 1] Freezing BERT backbone for Epoch 1 (Head-only adaptation)...")
            for p in model.encoder.parameters():
                p.requires_grad = False
        elif epoch == 2:
            logger.info(f"  [Stage 2] Unfreezing all BERT layers with conservative LR ({LR_BACKBONE})...")
            for p in model.encoder.parameters():
                p.requires_grad = True

        model.train()
        train_loss_accum = 0.0
        train_p_loss_accum = 0.0
        train_a_loss_accum = 0.0
        optimizer.zero_grad()

        for step, batch in enumerate(loader_train):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            token_type_ids = batch["token_type_ids"].to(device)
            primary_labels = batch["primary_label"].to(device)
            aux_labels = batch["aux_labels"].to(device)

            p_logits, a_logits = model(input_ids, attention_mask, token_type_ids)

            p_loss = criterion(p_logits, primary_labels)
            a_loss = 0.0
            for k in range(a_logits.shape[1]):
                a_loss += criterion(a_logits[:, k], aux_labels[:, k])

            batch_total_loss = (p_loss + aux_loss_weight * a_loss) / GRAD_ACCUM_STEPS
            batch_total_loss.backward()

            train_loss_accum += (p_loss + aux_loss_weight * a_loss).item() * len(primary_labels)
            train_p_loss_accum += p_loss.item() * len(primary_labels)
            train_a_loss_accum += a_loss.item() * len(primary_labels)

            if (step + 1) % GRAD_ACCUM_STEPS == 0 or (step + 1) == len(loader_train):
                torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

        train_total_loss = train_loss_accum / len(ds_train)
        train_p_loss = train_p_loss_accum / len(ds_train)
        train_a_loss = train_a_loss_accum / len(ds_train)

        val_total_loss, val_p_loss, val_a_loss, val_p_true, val_p_prob, val_a_true, val_a_prob = evaluate_loader(
            model, loader_val, criterion, device, aux_loss_weight=aux_loss_weight
        )

        val_primary_metrics = compute_primary_metrics(val_p_true, val_p_prob, threshold=0.50)
        _, val_aux_summary = compute_auxiliary_metrics(val_a_true, val_a_prob, AUX_FEATURE_COLS, threshold=0.50)

        epoch_duration = time.time() - epoch_start

        logger.info(
            f"[{arm_name}] Epoch {epoch:02d}/{EPOCHS:02d} | "
            f"Train Loss: {train_total_loss:.4f} (P:{train_p_loss:.4f}, A:{train_a_loss:.4f}) | "
            f"Val Loss: {val_total_loss:.4f} (P:{val_p_loss:.4f}, A:{val_a_loss:.4f}) | "
            f"Val Recall: {val_primary_metrics['recall']:.4f} | Val Spec: {val_primary_metrics['specificity']:.4f} | "
            f"Val AUROC: {val_primary_metrics['auroc']:.4f} | Val PR-AUC: {val_primary_metrics['pr_auc']:.4f} | "
            f"Aux Macro-F1: {val_aux_summary['macro_f1']:.4f} | Time: {epoch_duration:.1f}s"
        )

        history_records.append({
            "arm_id": arm_id,
            "arm_name": arm_name,
            "epoch": epoch,
            "train_total_loss": round(train_total_loss, 4),
            "train_primary_loss": round(train_p_loss, 4),
            "train_aux_loss": round(train_a_loss, 4),
            "val_total_loss": round(val_total_loss, 4),
            "val_primary_loss": round(val_p_loss, 4),
            "val_aux_loss": round(val_a_loss, 4),
            "val_recall": round(val_primary_metrics["recall"], 4),
            "val_specificity": round(val_primary_metrics["specificity"], 4),
            "val_precision": round(val_primary_metrics["precision"], 4),
            "val_f1": round(val_primary_metrics["f1_score"], 4),
            "val_auroc": round(val_primary_metrics["auroc"], 4),
            "val_pr_auc": round(val_primary_metrics["pr_auc"], 4),
            "val_aux_macro_f1": val_aux_summary["macro_f1"],
            "val_aux_micro_f1": val_aux_summary["micro_f1"],
            "val_aux_macro_auroc": val_aux_summary["macro_auroc"],
            "epoch_duration_s": round(epoch_duration, 1),
        })

        if val_total_loss < best_val_loss:
            best_val_loss = val_total_loss
            best_epoch = epoch
            best_model_weights = copy.deepcopy(model.state_dict())
            best_val_primary_probs = val_p_prob.copy()
            best_val_aux_probs = val_a_prob.copy()
            patience_counter = 0
            logger.info(f"  --> Saved new best checkpoint at epoch {epoch:02d} (Val Total Loss: {val_total_loss:.4f})")
        else:
            patience_counter += 1
            logger.info(f"  No improvement for {patience_counter} consecutive epoch(s).")
            if patience_counter >= PATIENCE:
                logger.info(f"Early stopping triggered at epoch {epoch:02d}.")
                break

    total_train_time = time.time() - start_train_time
    logger.info(f"Training completed in {total_train_time:.1f}s. Best Epoch: {best_epoch:02d} (Val Loss: {best_val_loss:.4f})")

    checkpoint_path = MODELS_DIR / f"arm_{arm_id}_best.pt"
    torch.save(best_model_weights, checkpoint_path)
    logger.info(f"Saved checkpoint to {checkpoint_path}")

    model.load_state_dict(best_model_weights)

    val_sweeps_df = sweep_thresholds(val_primary, best_val_primary_probs)
    val_sweeps_df["arm_id"] = arm_id
    val_sweeps_df["arm_name"] = arm_name

    t_default = 0.50

    cand_90 = val_sweeps_df[val_sweeps_df["recall"] >= 0.90]
    t_90 = cand_90.sort_values(by=["specificity", "threshold"], ascending=[False, False]).iloc[0]["threshold"] if len(cand_90) > 0 else 0.50

    cand_95 = val_sweeps_df[val_sweeps_df["recall"] >= 0.95]
    t_95 = cand_95.sort_values(by=["specificity", "threshold"], ascending=[False, False]).iloc[0]["threshold"] if len(cand_95) > 0 else 0.50

    cand_100 = val_sweeps_df[val_sweeps_df["recall"] >= 1.00]
    t_100 = cand_100.sort_values(by=["specificity", "threshold"], ascending=[False, False]).iloc[0]["threshold"] if len(cand_100) > 0 else 0.10

    operating_thresholds = [
        ("Default (0.50)", float(t_default)),
        ("Recall >= 90%", float(t_90)),
        ("Recall >= 95%", float(t_95)),
        ("Recall = 100%", float(t_100)),
    ]

    val_op_records = []
    for op_name, t_val in operating_thresholds:
        m = compute_primary_metrics(val_primary, best_val_primary_probs, threshold=t_val)
        m["arm_id"] = arm_id
        m["arm_name"] = arm_name
        m["operating_point"] = op_name
        m["threshold"] = round(t_val, 2)
        val_op_records.append(m)

    val_metrics_df = pd.DataFrame(val_op_records)

    val_aux_feat_df, val_aux_summary = compute_auxiliary_metrics(val_aux, best_val_aux_probs, AUX_FEATURE_COLS, threshold=0.50)
    val_aux_feat_df["arm_id"] = arm_id
    val_aux_feat_df["arm_name"] = arm_name

    logger.info("Evaluating Best Checkpoint on Held-Out Test Set (N=105)...")
    test_total_loss, test_p_loss, test_a_loss, test_p_true, test_p_prob, test_a_true, test_a_prob = evaluate_loader(
        model, loader_test, criterion, device, aux_loss_weight=aux_loss_weight
    )

    test_op_records = []
    for op_name, t_val in operating_thresholds:
        m = compute_primary_metrics(test_p_true, test_p_prob, threshold=t_val)
        m["arm_id"] = arm_id
        m["arm_name"] = arm_name
        m["operating_point"] = op_name
        m["threshold"] = round(t_val, 2)
        test_op_records.append(m)

    test_metrics_df = pd.DataFrame(test_op_records)

    test_aux_feat_df, test_aux_summary = compute_auxiliary_metrics(test_aux, test_a_prob, AUX_FEATURE_COLS, threshold=0.50)
    test_aux_feat_df["arm_id"] = arm_id
    test_aux_feat_df["arm_name"] = arm_name

    test_preds_df = pd.DataFrame({
        "pmid": test_df["pmid"],
        "arm_id": arm_id,
        "arm_name": arm_name,
        "primary_true": test_p_true,
        "primary_prob": np.round(test_p_prob, 5),
    })
    for k, col in enumerate(AUX_FEATURE_COLS):
        test_preds_df[f"aux_true_{col}"] = test_a_true[:, k]
        test_preds_df[f"aux_prob_{col}"] = np.round(test_a_prob[:, k], 5)

    logger.info(f"[{arm_name}] Test AUROC: {test_metrics_df.iloc[0]['auroc']:.4f} | Test PR-AUC: {test_metrics_df.iloc[0]['pr_auc']:.4f}")
    logger.info(f"[{arm_name}] Test Recall @ 100% Target: {test_metrics_df.iloc[3]['recall']:.4f} | Test Spec @ 100% Target: {test_metrics_df.iloc[3]['specificity']:.4f}")
    logger.info(f"[{arm_name}] Test Aux Macro-F1: {test_aux_summary['macro_f1']:.4f} | Test Aux Micro-F1: {test_aux_summary['micro_f1']:.4f}")

    # Immediately write arm-specific files to results directory
    arm_history_df = pd.DataFrame(history_records)
    arm_history_df.to_csv(RESULTS_DIR / f"arm_{arm_id}_training_history.csv", index=False)
    val_sweeps_df.to_csv(RESULTS_DIR / f"arm_{arm_id}_val_threshold_sweeps.csv", index=False)
    val_metrics_df.to_csv(RESULTS_DIR / f"arm_{arm_id}_val_metrics.csv", index=False)
    val_aux_feat_df.to_csv(RESULTS_DIR / f"arm_{arm_id}_val_aux_metrics.csv", index=False)
    test_metrics_df.to_csv(RESULTS_DIR / f"arm_{arm_id}_test_metrics.csv", index=False)
    test_aux_feat_df.to_csv(RESULTS_DIR / f"arm_{arm_id}_test_aux_metrics.csv", index=False)
    test_preds_df.to_csv(RESULTS_DIR / f"arm_{arm_id}_test_predictions.csv", index=False)

    return {
        "arm_id": arm_id,
        "history_df": arm_history_df,
        "val_sweeps_df": val_sweeps_df,
        "val_metrics_df": val_metrics_df,
        "val_aux_df": val_aux_feat_df,
        "val_aux_summary": val_aux_summary,
        "test_metrics_df": test_metrics_df,
        "test_aux_df": test_aux_feat_df,
        "test_aux_summary": test_aux_summary,
        "test_preds_df": test_preds_df,
    }


def main():
    parser = argparse.ArgumentParser(description="Stage 9 Multitask Experiments")
    parser.add_argument("--arm", type=str, default="all", choices=["1", "2", "3", "all"], help="Experiment arm to run")
    args = parser.parse_args()

    arms_to_run = [1, 2, 3] if args.arm == "all" else [int(args.arm)]

    all_histories = []
    all_val_sweeps = []
    all_val_metrics = []
    all_val_aux = []
    all_test_metrics = []
    all_test_aux = []
    all_test_preds = []

    for arm in arms_to_run:
        res = run_experiment_arm(arm)
        all_histories.append(res["history_df"])
        all_val_sweeps.append(res["val_sweeps_df"])
        all_val_metrics.append(res["val_metrics_df"])
        all_val_aux.append(res["val_aux_df"])
        all_test_metrics.append(res["test_metrics_df"])
        all_test_aux.append(res["test_aux_df"])
        all_test_preds.append(res["test_preds_df"])

    # Aggregate and save master results
    comb_history = pd.concat(all_histories, ignore_index=True)
    comb_val_sweeps = pd.concat(all_val_sweeps, ignore_index=True)
    comb_val_metrics = pd.concat(all_val_metrics, ignore_index=True)
    comb_val_aux = pd.concat(all_val_aux, ignore_index=True)
    comb_test_metrics = pd.concat(all_test_metrics, ignore_index=True)
    comb_test_aux = pd.concat(all_test_aux, ignore_index=True)
    comb_test_preds = pd.concat(all_test_preds, ignore_index=True)

    comb_history.to_csv(RESULTS_DIR / "stage9_training_histories.csv", index=False)
    comb_val_sweeps.to_csv(RESULTS_DIR / "stage9_validation_threshold_sweeps.csv", index=False)
    comb_val_metrics.to_csv(RESULTS_DIR / "stage9_validation_metrics.csv", index=False)
    comb_val_aux.to_csv(RESULTS_DIR / "stage9_validation_auxiliary_metrics.csv", index=False)
    comb_test_metrics.to_csv(RESULTS_DIR / "stage9_test_metrics.csv", index=False)
    comb_test_aux.to_csv(RESULTS_DIR / "stage9_test_auxiliary_metrics.csv", index=False)
    comb_test_preds.to_csv(RESULTS_DIR / "stage9_test_predictions.csv", index=False)

    logger.info("=" * 80)
    logger.info(f"STAGE 9 EXPERIMENTS COMPLETED SUCCESSFULLY! Results saved to {RESULTS_DIR}")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
