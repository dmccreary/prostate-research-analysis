"""
train_pubmedbert_512.py - Stage 5.5: PubMedBERT 512-Token Extension Benchmark.

Evaluates performance gains from extending the input sequence length from 384 to 512 tokens
on the existing Stage 5 dataset splits (API-refetched labeled-dataset-v2.csv, seed 2025).

Models Trained:
1. Model A (512 tokens, Unweighted BCE)
2. Model B (512 tokens, Cost-Sensitive Weighted BCE, pos_weight=1.7222)

Compared against:
- Stage 5 Baseline (384 tokens, Title + Abstract, Weighted BCE, Condition 4)

Hardware: NVIDIA GeForce GTX 1050 Ti (4GB VRAM)
- Physical BATCH_SIZE_TRAIN = 4
- GRADIENT_ACCUMULATION_STEPS = 2 (effective batch size = 8)
- BATCH_SIZE_EVAL = 8
- Sequence Length = 512 tokens
"""

import copy
import gc
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
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup

# Safe Windows stdout
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Force offline mode for HuggingFace cache loading
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"

# Directory paths
BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(BASE_DIR / "src"))
from preprocessing import clean_clinical_text

DATA_DIR = BASE_DIR / "data"
SPLITS_DIR = DATA_DIR / "splits" / "stage5"
MODELS_DIR = BASE_DIR / "models" / "stage5_512"
RESULTS_DIR = BASE_DIR / "results" / "stage5_512"
PLOTS_DIR = RESULTS_DIR / "plots"
REPORTS_DIR = BASE_DIR / "reports" / "stage5_512"
LOGS_DIR = BASE_DIR / "logs"
STAGE5_RESULTS_DIR = BASE_DIR / "results" / "stage5"

for d in [MODELS_DIR, RESULTS_DIR, PLOTS_DIR, REPORTS_DIR, LOGS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOGS_DIR / "pubmedbert_512_experiments.log", encoding="utf-8", mode="w"),
    ],
)
logger = logging.getLogger(__name__)

# Constants
PRETRAINED_MODEL_NAME = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract"
SEED = 2025
MAX_LENGTH = 512
BATCH_SIZE_TRAIN = 4
GRAD_ACCUM_STEPS = 2
BATCH_SIZE_EVAL = 8
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 1e-4
EPOCHS = 8
PATIENCE = 4
DROPOUT_RATE = 0.25
MAX_GRAD_NORM = 1.0


def set_all_seeds(seed: int = SEED):
    """Ensures strict reproducibility across all operations."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class PubMedBERT512Dataset(Dataset):
    """Dual-sequence Title + Abstract Dataset tokenized to 512 tokens."""

    def __init__(
        self,
        titles: List[str],
        abstracts: List[str],
        labels: List[int],
        tokenizer: AutoTokenizer,
        max_length: int = MAX_LENGTH,
    ):
        self.labels = torch.tensor(labels, dtype=torch.float32)
        enc = tokenizer(
            titles,
            abstracts,
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        self.input_ids = enc["input_ids"]
        self.attention_mask = enc["attention_mask"]
        self.token_type_ids = enc.get("token_type_ids", torch.zeros_like(self.input_ids))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
            "token_type_ids": self.token_type_ids[idx],
            "label": self.labels[idx],
        }


class FlexiblePubMedBERT(nn.Module):
    """
    Identical architecture to Stage 5 baseline (rule_dim=0):
    [CLS] (768) -> Linear(768, 128) -> LayerNorm(128) -> GELU() -> Dropout(0.25) -> Linear(128, 1)
    """

    def __init__(
        self,
        pretrained_name: str = PRETRAINED_MODEL_NAME,
        dropout_rate: float = DROPOUT_RATE,
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


def evaluate_loader(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, np.ndarray, np.ndarray]:
    """Evaluates model over DataLoader, returning (mean_loss, y_true, y_prob)."""
    model.eval()
    total_loss = 0.0
    all_true, all_prob = [], []

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            token_type_ids = batch.get("token_type_ids")
            if token_type_ids is not None:
                token_type_ids = token_type_ids.to(device)
            labels = batch["label"].to(device)

            logits = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
            )
            loss = criterion(logits, labels)
            total_loss += loss.item() * len(labels)

            probs = torch.sigmoid(logits).cpu().numpy()
            all_prob.extend(probs.tolist())
            all_true.extend(labels.cpu().numpy().tolist())

    mean_loss = total_loss / len(all_true) if len(all_true) > 0 else 0.0
    return mean_loss, np.array(all_true, dtype=int), np.array(all_prob, dtype=float)


def compute_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
    model_name: str = "",
    operating_point: str = "Default (0.50)",
) -> Dict:
    """Computes full set of screening and clinical classification metrics."""
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
        "threshold": round(float(threshold), 3),
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


def sweep_validation_thresholds(
    y_val: np.ndarray,
    val_probs: np.ndarray,
    target_sensitivities: List[float] = [0.90, 0.95, 1.00],
) -> Dict[float, float]:
    """
    Selects clinical operating thresholds strictly on validation set.
    For each target sensitivity (0.90, 0.95, 1.00), finds the highest threshold
    that achieves val_recall >= target, maximizing specificity.
    """
    threshold_grid = np.linspace(0.01, 0.99, 99)
    selected_thresholds = {}

    for target in target_sensitivities:
        valid_candidates = []
        for t in threshold_grid:
            val_preds = (val_probs >= t).astype(int)
            val_rec = recall_score(y_val, val_preds, zero_division=0)
            if val_rec >= target:
                valid_candidates.append(t)

        if valid_candidates:
            best_t = max(valid_candidates)
        else:
            best_t = 0.01

        selected_thresholds[target] = float(best_t)

    return selected_thresholds


def train_pubmedbert_512(
    model_name: str,
    pos_weight_val: Optional[float],
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    save_path: Path,
) -> Tuple[FlexiblePubMedBERT, pd.DataFrame]:
    """
    Trains PubMedBERT at 512 tokens with gradient accumulation to fit within 4GB VRAM.
    Early stopping on validation loss (patience=4).
    """
    set_all_seeds(SEED)
    model = FlexiblePubMedBERT().to(device)

    if pos_weight_val is not None:
        weight_tensor = torch.tensor([pos_weight_val], device=device, dtype=torch.float32)
        criterion = nn.BCEWithLogitsLoss(pos_weight=weight_tensor)
        logger.info(f"[{model_name}] Loss: Cost-Sensitive BCEWithLogitsLoss (pos_weight={pos_weight_val:.4f})")
    else:
        criterion = nn.BCEWithLogitsLoss()
        logger.info(f"[{model_name}] Loss: Standard Unweighted BCEWithLogitsLoss")

    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": WEIGHT_DECAY,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=LEARNING_RATE)

    # Effective optimizer steps per epoch accounts for gradient accumulation
    effective_steps_per_epoch = (len(train_loader) + GRAD_ACCUM_STEPS - 1) // GRAD_ACCUM_STEPS
    total_steps = effective_steps_per_epoch * EPOCHS
    warmup_steps = int(total_steps * 0.1)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )

    best_val_loss = float("inf")
    best_weights = None
    no_improve_epochs = 0
    best_epoch = 0
    history = []

    logger.info(
        f"[{model_name}] Starting training: {EPOCHS} epochs, batch_size={BATCH_SIZE_TRAIN}, "
        f"grad_accum={GRAD_ACCUM_STEPS} (effective batch=8), max_len={MAX_LENGTH}, lr={LEARNING_RATE}..."
    )

    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss = 0.0
        train_samples = 0
        optimizer.zero_grad()

        for step, batch in enumerate(train_loader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            token_type_ids = batch.get("token_type_ids")
            if token_type_ids is not None:
                token_type_ids = token_type_ids.to(device)
            labels = batch["label"].to(device)

            logits = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
            )
            loss = criterion(logits, labels)

            # Scale loss for gradient accumulation
            (loss / GRAD_ACCUM_STEPS).backward()

            if (step + 1) % GRAD_ACCUM_STEPS == 0 or (step + 1) == len(train_loader):
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=MAX_GRAD_NORM)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            train_loss += loss.item() * len(labels)
            train_samples += len(labels)

        train_mean_loss = train_loss / train_samples
        val_mean_loss, val_y_true, val_y_prob = evaluate_loader(model, val_loader, criterion, device)

        try:
            val_auroc = roc_auc_score(val_y_true, val_y_prob)
            val_p_curve, val_r_curve, _ = precision_recall_curve(val_y_true, val_y_prob)
            val_prauc = auc(val_r_curve, val_p_curve)
        except Exception:
            val_auroc, val_prauc = 0.5, 0.5

        val_f1 = f1_score(val_y_true, (val_y_prob >= 0.5).astype(int), zero_division=0)
        val_rec = recall_score(val_y_true, (val_y_prob >= 0.5).astype(int), zero_division=0)

        history.append({
            "model_name": model_name,
            "epoch": epoch,
            "train_loss": train_mean_loss,
            "val_loss": val_mean_loss,
            "val_auroc": val_auroc,
            "val_prauc": val_prauc,
            "val_f1": val_f1,
            "val_recall": val_rec,
        })

        logger.info(
            f"[{model_name}] Epoch {epoch}/{EPOCHS} - "
            f"Train Loss: {train_mean_loss:.4f} | "
            f"Val Loss: {val_mean_loss:.4f} | "
            f"Val AUROC: {val_auroc:.4f} | "
            f"Val PR-AUC: {val_prauc:.4f} | "
            f"Val F1@0.5: {val_f1:.4f}"
        )

        # Early stopping on validation loss
        if val_mean_loss < best_val_loss:
            best_val_loss = val_mean_loss
            best_epoch = epoch
            best_weights = copy.deepcopy(model.state_dict())
            no_improve_epochs = 0
            torch.save(best_weights, save_path)
            logger.info(f"[{model_name}] New best checkpoint saved at epoch {epoch} (Val Loss: {val_mean_loss:.4f})")
        else:
            no_improve_epochs += 1
            if no_improve_epochs >= PATIENCE:
                logger.info(f"[{model_name}] Early stopping triggered at epoch {epoch} (best epoch: {best_epoch})")
                break

    # Restore best checkpoint
    if best_weights is not None:
        model.load_state_dict(best_weights)
    elif save_path.exists():
        model.load_state_dict(torch.load(save_path, map_location=device))

    return model, pd.DataFrame(history)


def main():
    logger.info("================================================================================")
    logger.info("    STARTING STAGE 5.5: PubMedBERT 512-TOKEN EXTENSION BENCHMARK                ")
    logger.info("================================================================================")

    set_all_seeds(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Execution Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    # 1. Load exact Stage 5 splits
    train_path = SPLITS_DIR / "train.csv"
    val_path = SPLITS_DIR / "val.csv"
    test_path = SPLITS_DIR / "test.csv"

    if not train_path.exists() or not val_path.exists() or not test_path.exists():
        raise FileNotFoundError(f"Missing Stage 5 splits in {SPLITS_DIR}.")

    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    test_df = pd.read_csv(test_path)

    logger.info(f"Loaded Splits: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")

    # 2. Text Preprocessing
    train_titles = [clean_clinical_text(str(t or "")) for t in train_df["title"]]
    train_abstracts = [clean_clinical_text(str(a or "")) for a in train_df["abstract"]]
    y_train = train_df["label"].to_numpy().astype(int)

    val_titles = [clean_clinical_text(str(t or "")) for t in val_df["title"]]
    val_abstracts = [clean_clinical_text(str(a or "")) for a in val_df["abstract"]]
    y_val = val_df["label"].to_numpy().astype(int)

    test_titles = [clean_clinical_text(str(t or "")) for t in test_df["title"]]
    test_abstracts = [clean_clinical_text(str(a or "")) for a in test_df["abstract"]]
    y_test = test_df["label"].to_numpy().astype(int)

    # Class balance and pos_weight calculation (identical to Stage 5: 2.0 * neg / pos)
    n_pos = int((y_train == 1).sum())
    n_neg = int((y_train == 0).sum())
    pos_weight = 2.0 * (n_neg / n_pos)  # 2.0 * 155 / 180 = 1.7222
    logger.info(f"Train Counts: Pos={n_pos}, Neg={n_neg} -> Cost-Sensitive pos_weight={pos_weight:.4f}")

    # 3. Tokenizer & Datasets (512 tokens)
    logger.info(f"Loading local tokenizer for {PRETRAINED_MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(PRETRAINED_MODEL_NAME, local_files_only=True)

    ds_train = PubMedBERT512Dataset(train_titles, train_abstracts, y_train, tokenizer, max_length=MAX_LENGTH)
    ds_val = PubMedBERT512Dataset(val_titles, val_abstracts, y_val, tokenizer, max_length=MAX_LENGTH)
    ds_test = PubMedBERT512Dataset(test_titles, test_abstracts, y_test, tokenizer, max_length=MAX_LENGTH)

    loader_train = DataLoader(ds_train, batch_size=BATCH_SIZE_TRAIN, shuffle=True)
    loader_val = DataLoader(ds_val, batch_size=BATCH_SIZE_EVAL, shuffle=False)
    loader_test = DataLoader(ds_test, batch_size=BATCH_SIZE_EVAL, shuffle=False)

    dummy_criterion = nn.BCEWithLogitsLoss()
    all_histories = []

    # -------------------------------------------------------------------------
    # Experiment 1: Model A (PubMedBERT 512 tokens, Unweighted BCE)
    # -------------------------------------------------------------------------
    logger.info("--------------------------------------------------------------------------------")
    logger.info("TRAINING MODEL A: PubMedBERT (512 tokens, Unweighted BCE)")
    logger.info("--------------------------------------------------------------------------------")
    save_path_unweighted = MODELS_DIR / "pubmedbert_512_unweighted.pt"

    model_unw, hist_unw = train_pubmedbert_512(
        model_name="PubMedBERT 512 (Unweighted)",
        pos_weight_val=None,
        train_loader=loader_train,
        val_loader=loader_val,
        device=device,
        save_path=save_path_unweighted,
    )
    all_histories.append(hist_unw)

    _, _, val_probs_unw = evaluate_loader(model_unw, loader_val, dummy_criterion, device)
    _, _, test_probs_unw = evaluate_loader(model_unw, loader_test, dummy_criterion, device)

    del model_unw
    torch.cuda.empty_cache()
    gc.collect()

    # -------------------------------------------------------------------------
    # Experiment 2: Model B (PubMedBERT 512 tokens, Weighted BCE pos_weight=1.7222)
    # -------------------------------------------------------------------------
    logger.info("--------------------------------------------------------------------------------")
    logger.info("TRAINING MODEL B: PubMedBERT (512 tokens, Weighted BCE, pos_weight=1.7222)")
    logger.info("--------------------------------------------------------------------------------")
    save_path_weighted = MODELS_DIR / "pubmedbert_512_weighted.pt"

    model_w, hist_w = train_pubmedbert_512(
        model_name="PubMedBERT 512 (Weighted)",
        pos_weight_val=pos_weight,
        train_loader=loader_train,
        val_loader=loader_val,
        device=device,
        save_path=save_path_weighted,
    )
    all_histories.append(hist_w)

    _, _, val_probs_w = evaluate_loader(model_w, loader_val, dummy_criterion, device)
    _, _, test_probs_w = evaluate_loader(model_w, loader_test, dummy_criterion, device)

    del model_w
    torch.cuda.empty_cache()
    gc.collect()

    # Save histories
    full_history_df = pd.concat(all_histories, ignore_index=True)
    full_history_df.to_csv(RESULTS_DIR / "pubmedbert_512_training_histories.csv", index=False)

    # -------------------------------------------------------------------------
    # Load Stage 5 Baseline (384 tokens, Title + Abstract, Weighted)
    # -------------------------------------------------------------------------
    stage5_preds_path = STAGE5_RESULTS_DIR / "stage5_test_predictions.csv"
    if not stage5_preds_path.exists():
        raise FileNotFoundError(f"Missing Stage 5 predictions at {stage5_preds_path}")
    stage5_preds_df = pd.read_csv(stage5_preds_path)
    test_probs_384 = stage5_preds_df["condition_4_prob"].to_numpy()

    # Also compute validation probabilities for Stage 5 Baseline to get exact validation operating thresholds
    stage5_ckpt_path = BASE_DIR / "models" / "stage5" / "pubmedbert_title_abstract_weighted.pt"
    ds_val_384 = PubMedBERT512Dataset(val_titles, val_abstracts, y_val, tokenizer, max_length=384)
    loader_val_384 = DataLoader(ds_val_384, batch_size=BATCH_SIZE_EVAL, shuffle=False)

    model_384 = FlexiblePubMedBERT().to(device)
    model_384.load_state_dict(torch.load(stage5_ckpt_path, map_location=device))
    _, _, val_probs_384 = evaluate_loader(model_384, loader_val_384, dummy_criterion, device)
    del model_384
    torch.cuda.empty_cache()
    gc.collect()

    # -------------------------------------------------------------------------
    # Clinical Operating Threshold Selection (Validation Set 99-point sweep)
    # -------------------------------------------------------------------------
    models_dict = {
        "PubMedBERT 384 (Stage 5 Baseline, Weighted)": (val_probs_384, test_probs_384),
        "PubMedBERT 512 (Model A, Unweighted)": (val_probs_unw, test_probs_unw),
        "PubMedBERT 512 (Model B, Weighted)": (val_probs_w, test_probs_w),
    }

    all_test_metrics = []
    clinical_operating_records = []

    for m_name, (v_probs, t_probs) in models_dict.items():
        # 1. Default Threshold 0.50
        def_metrics = compute_metrics(y_test, t_probs, threshold=0.50, model_name=m_name, operating_point="Default (0.50)")
        all_test_metrics.append(def_metrics)

        # 2. Sweep Validation Thresholds for >=90%, >=95%, 100% Recall
        val_thresholds = sweep_validation_thresholds(y_val, v_probs, target_sensitivities=[0.90, 0.95, 1.00])
        logger.info(f"[{m_name}] Validation-Selected Thresholds: {val_thresholds}")

        for target_rec, sel_t in val_thresholds.items():
            op_label = f"Recall >= {int(target_rec*100)}%" if target_rec < 1.00 else "Recall = 100%"
            op_metrics = compute_metrics(y_test, t_probs, threshold=sel_t, model_name=m_name, operating_point=op_label)
            op_metrics["target_sensitivity"] = target_rec
            op_metrics["val_selected_threshold"] = sel_t
            all_test_metrics.append(op_metrics)
            clinical_operating_records.append(op_metrics)

    test_metrics_df = pd.DataFrame(all_test_metrics)
    test_metrics_df.to_csv(RESULTS_DIR / "pubmedbert_512_test_metrics.csv", index=False)

    clinical_df = pd.DataFrame(clinical_operating_records)
    clinical_df.to_csv(RESULTS_DIR / "pubmedbert_512_clinical_operating_points.csv", index=False)

    # -------------------------------------------------------------------------
    # Save Test Predictions CSV
    # -------------------------------------------------------------------------
    pmids_test = [str(p).strip().replace(".0", "") for p in test_df["pmid"]]
    predictions_df = pd.DataFrame({
        "pmid": pmids_test,
        "title": test_df["title"].tolist(),
        "true_label": y_test,
        "pubmedbert_384_weighted_prob": test_probs_384,
        "pubmedbert_512_unweighted_prob": test_probs_unw,
        "pubmedbert_512_weighted_prob": test_probs_w,
        "pubmedbert_384_weighted_pred_0.5": (test_probs_384 >= 0.50).astype(int),
        "pubmedbert_512_unweighted_pred_0.5": (test_probs_unw >= 0.50).astype(int),
        "pubmedbert_512_weighted_pred_0.5": (test_probs_w >= 0.50).astype(int),
    })
    predictions_df.to_csv(RESULTS_DIR / "pubmedbert_512_test_predictions.csv", index=False)

    # -------------------------------------------------------------------------
    # Markdown Comparison Table (Formatted for Review)
    # -------------------------------------------------------------------------
    comp_records = []
    for r in all_test_metrics:
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
            "TP": r["true_positives"],
            "FP": r["false_positives"],
            "TN": r["true_negatives"],
            "FN": r["false_negatives"],
            "Workload %": f"{r['workload_pct']:.2f}%",
            "Workload Reduction %": f"{r['workload_reduction_pct']:.2f}%",
            "NNS": r["nns"],
        })
    comparison_table_df = pd.DataFrame(comp_records)
    comparison_table_df.to_csv(RESULTS_DIR / "pubmedbert_512_vs_384_comparison.csv", index=False)

    # -------------------------------------------------------------------------
    # Visual Plots
    # -------------------------------------------------------------------------
    logger.info("Generating publication figures...")
    sns.set_theme(style="whitegrid", font="sans-serif")
    palette = {
        "PubMedBERT 384 (Stage 5 Baseline, Weighted)": "#1f77b4",
        "PubMedBERT 512 (Model A, Unweighted)": "#2ca02c",
        "PubMedBERT 512 (Model B, Weighted)": "#d62728",
    }

    # Figure 1: ROC Curves
    plt.figure(figsize=(8.5, 7), dpi=300)
    for m_name, (_, probs) in models_dict.items():
        fpr, tpr, _ = roc_curve(y_test, probs)
        roc_auc_val = roc_auc_score(y_test, probs)
        plt.plot(fpr, tpr, lw=2.4, color=palette[m_name], label=f"{m_name} (AUROC = {roc_auc_val:.4f})")
    plt.plot([0, 1], [0, 1], "k--", lw=1.2, alpha=0.6, label="Chance Baseline (AUROC = 0.5000)")
    plt.xlim([-0.02, 1.02])
    plt.ylim([-0.02, 1.05])
    plt.xlabel("False Positive Rate (1 - Specificity)", fontsize=12, fontweight="bold")
    plt.ylabel("True Positive Rate (Recall / Sensitivity)", fontsize=12, fontweight="bold")
    plt.title("ROC Curves: 512-Token PubMedBERT vs. 384-Token Baseline\nHeld-Out Test Set (N = 105, 56 Pos / 49 Neg)", fontsize=13, fontweight="bold", pad=12)
    plt.legend(loc="lower right", fontsize=9.5, framealpha=0.95)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "pubmedbert_512_roc_curves.png")
    plt.close()

    # Figure 2: PR Curves
    plt.figure(figsize=(8.5, 7), dpi=300)
    prevalence = float(np.mean(y_test))
    for m_name, (_, probs) in models_dict.items():
        prec, rec, _ = precision_recall_curve(y_test, probs)
        pr_auc_val = auc(rec, prec)
        plt.plot(rec, prec, lw=2.4, color=palette[m_name], label=f"{m_name} (PR-AUC = {pr_auc_val:.4f})")
    plt.axhline(prevalence, color="k", linestyle="--", lw=1.2, alpha=0.6, label=f"Prevalence Baseline ({prevalence:.2%})")
    plt.xlim([-0.02, 1.02])
    plt.ylim([-0.02, 1.05])
    plt.xlabel("Recall (Sensitivity)", fontsize=12, fontweight="bold")
    plt.ylabel("Precision (Positive Predictive Value)", fontsize=12, fontweight="bold")
    plt.title("PR Curves: 512-Token PubMedBERT vs. 384-Token Baseline\nHeld-Out Test Set (N = 105, 56 Pos / 49 Neg)", fontsize=13, fontweight="bold", pad=12)
    plt.legend(loc="lower left", fontsize=9.5, framealpha=0.95)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "pubmedbert_512_pr_curves.png")
    plt.close()

    # Figure 3: Side-by-Side Confusion Matrices at Default Threshold 0.50
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5), dpi=300)
    for idx, (m_name, (_, probs)) in enumerate(models_dict.items()):
        y_p = (probs >= 0.50).astype(int)
        cm = confusion_matrix(y_test, y_p, labels=[0, 1])
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=axes[idx],
                    xticklabels=["Pred Neg (0)", "Pred Pos (1)"],
                    yticklabels=["True Neg (0)", "True Pos (1)"],
                    annot_kws={"size": 13, "weight": "bold"})
        short_name = m_name.replace(" (Stage 5 Baseline, Weighted)", "\n(384 tokens, Weighted)").replace(" (Model A, Unweighted)", "\n(512 tokens, Unweighted)").replace(" (Model B, Weighted)", "\n(512 tokens, Weighted)")
        axes[idx].set_title(f"{short_name}\n(Threshold = 0.50)", fontsize=11, fontweight="bold")
        axes[idx].set_xlabel("Predicted Label", fontsize=10)
        if idx == 0:
            axes[idx].set_ylabel("True Ground Truth", fontsize=10)
        else:
            axes[idx].set_ylabel("")

    plt.suptitle("Confusion Matrices at Default Operating Point (Threshold = 0.50)", fontsize=14, fontweight="bold", y=1.05)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "pubmedbert_512_confusion_matrices.png", bbox_inches="tight")
    plt.close()

    # Figure 4: Workload Reduction vs. Recall Tradeoff Curve
    plt.figure(figsize=(8.5, 7), dpi=300)
    for m_name, (_, probs) in models_dict.items():
        recs, red_pcts = [], []
        for t in np.linspace(0.01, 0.99, 99):
            preds = (probs >= t).astype(int)
            rec = recall_score(y_test, preds, zero_division=0)
            red = ((len(y_test) - preds.sum()) / len(y_test)) * 100.0
            recs.append(rec)
            red_pcts.append(red)
        sorted_pairs = sorted(zip(recs, red_pcts), key=lambda x: x[0])
        r_sorted, w_sorted = zip(*sorted_pairs)
        plt.plot(r_sorted, w_sorted, lw=2.4, color=palette[m_name], label=m_name)

    plt.axvline(0.90, color="lightgray", linestyle=":", lw=1.2, label="Recall = 90%")
    plt.axvline(0.95, color="gray", linestyle=":", lw=1.5, label="Recall = 95%")
    plt.axvline(1.00, color="black", linestyle="--", lw=1.5, label="Recall = 100%")
    plt.xlabel("Recall (Clinical Sensitivity)", fontsize=12, fontweight="bold")
    plt.ylabel("Workload Reduction (%)", fontsize=12, fontweight="bold")
    plt.title("Workload Reduction vs. Recall Tradeoff\nHeld-Out Test Set (N = 105)", fontsize=13, fontweight="bold", pad=12)
    plt.legend(loc="lower left", fontsize=9.0, framealpha=0.95)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "pubmedbert_512_workload_vs_recall.png")
    plt.close()

    # Figure 5: Token Length Distribution
    full_df = pd.read_csv(DATA_DIR / "labeled-dataset-v2.csv")
    full_titles = [clean_clinical_text(str(t or "")) for t in full_df["title"]]
    full_abstracts = [clean_clinical_text(str(a or "")) for a in full_df["abstract"]]
    token_lengths = [len(tokenizer.encode(t, a)) for t, a in zip(full_titles, full_abstracts)]

    plt.figure(figsize=(9, 5.5), dpi=300)
    plt.hist(token_lengths, bins=40, color="#3498db", edgecolor="black", alpha=0.75)
    plt.axvline(384, color="#e74c3c", linestyle="--", lw=2.2, label="Stage 5 Baseline Max (384 tokens, 63.9% truncated)")
    plt.axvline(512, color="#2ecc71", linestyle="-", lw=2.2, label="New Max Length (512 tokens, 14.5% truncated)")
    plt.xlabel("Total Token Count (Title + Abstract)", fontsize=12, fontweight="bold")
    plt.ylabel("Number of Papers", fontsize=12, fontweight="bold")
    plt.title("Token Length Distribution Across Clinical Dataset (N = 524)", fontsize=13, fontweight="bold", pad=12)
    plt.legend(loc="upper right", fontsize=10, framealpha=0.95)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "token_length_distribution.png")
    plt.close()

    logger.info("================================================================================")
    logger.info("   STAGE 5.5 (512 TOKENS) BENCHMARK COMPLETED SUCCESSFULLY!                    ")
    logger.info("================================================================================")


if __name__ == "__main__":
    main()
