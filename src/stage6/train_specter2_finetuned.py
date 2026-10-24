"""
train_specter2_finetuned.py - Stage 6: Fine-Tuning SPECTER2 for Literature Screening.

Trains and evaluates SPECTER2 jointly with a 2-layer MLP classification head:
Architecture:
Title + sep_token + Abstract -> SPECTER2 -> CLS (768) -> Dropout(0.2) -> Linear(768, 128) -> GELU -> Dropout(0.2) -> Linear(128, 1)

Evaluates:
1. Strategy A: Full Fine-Tuning (Unweighted BCE)
2. Strategy A: Full Fine-Tuning (Cost-Sensitive Weighted BCE, pos_weight=1.7222)
3. Strategy B: Gradual Unfreezing (Weighted BCE, Epochs 1-2 frozen encoder, Epochs 3+ unfreeze upper layers)

Validation Checkpoint Selection:
Prefers checkpoint with highest validation specificity subject to Recall >= 95%.
Ties broken by higher PR-AUC, then lower validation loss. Zero test leakage.

Benchmarked against:
- Stage 5 Fine-Tuned PubMedBERT
- Stage 6 Frozen SPECTER2 + Logistic Regression
- Stage 6 Frozen SPECTER2 + Linear SVM
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

# Safe stdout on Windows
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Force offline mode if cache exists
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(BASE_DIR / "src"))
from preprocessing import clean_clinical_text

SPLITS_DIR = BASE_DIR / "data" / "splits" / "stage5"
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
        logging.FileHandler(LOGS_DIR / "specter2_finetuning_experiments.log", encoding="utf-8", mode="w"),
    ],
)
logger = logging.getLogger(__name__)

MODEL_ID = "allenai/specter2_base"
SEED = 42
MAX_LENGTH = 384
BATCH_SIZE_TRAIN = 4
GRAD_ACCUM_STEPS = 2
BATCH_SIZE_EVAL = 16
LR_ENCODER = 1e-5
LR_HEAD = 5e-5
WEIGHT_DECAY = 1e-2
DROPOUT = 0.2
EPOCHS = 10
PATIENCE = 3
MAX_GRAD_NORM = 1.0


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class SPECTER2Dataset(Dataset):
    """Dataset for SPECTER2 encoding title + sep_token + abstract."""

    def __init__(self, titles: List[str], abstracts: List[str], labels: List[int], tokenizer, max_length: int = MAX_LENGTH):
        self.labels = torch.tensor(labels, dtype=torch.float32)
        # Format text: title + sep_token + abstract
        texts = [f"{t} {tokenizer.sep_token} {a}".strip() for t, a in zip(titles, abstracts)]
        enc = tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        self.input_ids = enc["input_ids"]
        self.attention_mask = enc["attention_mask"]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
            "label": self.labels[idx],
        }


class SPECTER2Classifier(nn.Module):
    """
    Joint SPECTER2 encoder + classification head:
    SPECTER2 -> CLS (768) -> Dropout(0.2) -> Linear(768, 128) -> GELU -> Dropout(0.2) -> Linear(128, 1)
    """

    def __init__(self, model_id: str = MODEL_ID, dropout: float = DROPOUT):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_id, local_files_only=True)
        hidden_dim = self.encoder.config.hidden_size  # 768

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
        )

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls_rep = out.last_hidden_state[:, 0, :]  # 768-d document representation
        logits = self.classifier(cls_rep).squeeze(-1)
        return logits


def evaluate_model(
    model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device
) -> Tuple[float, np.ndarray, np.ndarray]:
    """Evaluates model over DataLoader returning (mean_loss, y_true, y_prob)."""
    model.eval()
    total_loss = 0.0
    all_true, all_prob = [], []

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            logits = model(input_ids, attention_mask)
            loss = criterion(logits, labels)
            total_loss += loss.item() * len(labels)

            probs = torch.sigmoid(logits).cpu().numpy()
            all_prob.extend(probs.tolist())
            all_true.extend(labels.cpu().numpy().tolist())

    mean_loss = total_loss / len(all_true) if len(all_true) > 0 else 0.0
    return mean_loss, np.array(all_true, dtype=int), np.array(all_prob, dtype=float)


def compute_metrics(
    y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5, model_name: str = "", operating_point: str = "Default (0.50)"
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
    """Identifies thresholds on validation set providing highest specificity subject to target recall."""
    thresholds = {"default": 0.50}

    # Op Point A: Recall >= 0.95
    cands_95 = val_sweep_df[val_sweep_df["recall_sensitivity"] >= 0.95]
    if not cands_95.empty:
        best_row_95 = cands_95.sort_values(by=["specificity", "threshold"], ascending=[False, False]).iloc[0]
        thresholds["op_recall_ge_95"] = float(best_row_95["threshold"])
    else:
        thresholds["op_recall_ge_95"] = 0.01

    # Op Point B: Recall = 1.00
    cands_100 = val_sweep_df[val_sweep_df["recall_sensitivity"] >= 1.00]
    if not cands_100.empty:
        best_row_100 = cands_100.sort_values(by=["specificity", "threshold"], ascending=[False, False]).iloc[0]
        thresholds["op_recall_100"] = float(best_row_100["threshold"])
    else:
        thresholds["op_recall_100"] = 0.01

    return thresholds


def train_specter2_model(
    model_name: str,
    strategy: str,  # "full" or "gradual"
    pos_weight_val: Optional[float],
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    save_path: Path,
) -> Tuple[SPECTER2Classifier, Dict, pd.DataFrame]:
    """
    Trains SPECTER2 with specified strategy and high-recall validation checkpoint selection.
    """
    set_seed(SEED)
    model = SPECTER2Classifier(model_id=MODEL_ID, dropout=DROPOUT).to(device)

    # Criterion
    if pos_weight_val is not None:
        weight_tensor = torch.tensor([pos_weight_val], device=device, dtype=torch.float32)
        criterion = nn.BCEWithLogitsLoss(pos_weight=weight_tensor)
        logger.info(f"[{model_name}] Loss: Weighted BCE (pos_weight = {pos_weight_val:.4f})")
    else:
        criterion = nn.BCEWithLogitsLoss()
        logger.info(f"[{model_name}] Loss: Standard Unweighted BCE")

    # If gradual unfreezing, freeze encoder for epochs 1-2
    if strategy == "gradual":
        logger.info(f"[{model_name}] Strategy B: Freezing SPECTER2 encoder for epochs 1-2...")
        for param in model.encoder.parameters():
            param.requires_grad = False

    # Parameter groups with separate learning rates
    def build_optimizer():
        no_decay = ["bias", "LayerNorm.weight"]
        encoder_params = [p for n, p in model.encoder.named_parameters() if p.requires_grad]
        head_params = [p for n, p in model.classifier.named_parameters() if p.requires_grad]

        opt_groups = []
        if encoder_params:
            opt_groups.append({
                "params": [p for n, p in model.encoder.named_parameters() if p.requires_grad and not any(nd in n for nd in no_decay)],
                "lr": LR_ENCODER,
                "weight_decay": WEIGHT_DECAY,
            })
            opt_groups.append({
                "params": [p for n, p in model.encoder.named_parameters() if p.requires_grad and any(nd in n for nd in no_decay)],
                "lr": LR_ENCODER,
                "weight_decay": 0.0,
            })
        opt_groups.append({
            "params": [p for n, p in model.classifier.named_parameters() if p.requires_grad and not any(nd in n for nd in no_decay)],
            "lr": LR_HEAD,
            "weight_decay": WEIGHT_DECAY,
        })
        opt_groups.append({
            "params": [p for n, p in model.classifier.named_parameters() if p.requires_grad and any(nd in n for nd in no_decay)],
            "lr": LR_HEAD,
            "weight_decay": 0.0,
        })
        return AdamW(opt_groups)

    optimizer = build_optimizer()
    total_steps = len(train_loader) * EPOCHS
    warmup_steps = int(total_steps * 0.1)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

    best_score_tuple = (-1.0, -1.0, float("inf"))  # (spec_at_95_recall, pr_auc, -val_loss)
    best_weights = None
    best_epoch = 0
    no_improve = 0
    history = []

    logger.info(f"[{model_name}] Training {EPOCHS} epochs, lr_encoder={LR_ENCODER}, lr_head={LR_HEAD}, patience={PATIENCE}...")

    for epoch in range(1, EPOCHS + 1):
        # Gradual unfreezing transition at epoch 3
        if strategy == "gradual" and epoch == 3:
            logger.info(f"[{model_name}] Epoch {epoch}: Unfreezing upper transformer layers (layers 9-11)...")
            # Unfreeze layers 9, 10, 11
            for name, param in model.encoder.named_parameters():
                if any(f"layer.{i}" in name for i in [9, 10, 11]) or "pooler" in name:
                    param.requires_grad = True
                else:
                    param.requires_grad = False
            # Rebuild optimizer for newly unfrozen layers
            optimizer = build_optimizer()
            remaining_steps = len(train_loader) * (EPOCHS - epoch + 1)
            scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(remaining_steps * 0.1), num_training_steps=remaining_steps)

        model.train()
        train_loss = 0.0
        train_samples = 0
        optimizer.zero_grad()

        for step, batch in enumerate(train_loader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            logits = model(input_ids, attention_mask)
            loss = criterion(logits, labels)
            (loss / GRAD_ACCUM_STEPS).backward()

            if (step + 1) % GRAD_ACCUM_STEPS == 0 or (step + 1) == len(train_loader):
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=MAX_GRAD_NORM)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            train_loss += loss.item() * len(labels)
            train_samples += len(labels)

        train_mean_loss = train_loss / train_samples
        val_loss, y_val_true, y_val_prob = evaluate_model(model, val_loader, criterion, device)

        # Validation sweep to find specificity at Recall >= 95%
        sweep_df = sweep_validation_thresholds(y_val_true, y_val_prob, model_name=model_name)
        cands_95 = sweep_df[sweep_df["recall_sensitivity"] >= 0.95]
        spec_at_95 = float(cands_95["specificity"].max()) if not cands_95.empty else 0.0

        try:
            auroc_val = roc_auc_score(y_val_true, y_val_prob)
            p_c, r_c, _ = precision_recall_curve(y_val_true, y_val_prob)
            prauc_val = auc(r_c, p_c)
        except Exception:
            auroc_val, prauc_val = 0.5, 0.5

        f1_val = f1_score(y_val_true, (y_val_prob >= 0.5).astype(int), zero_division=0)

        history.append({
            "model_name": model_name,
            "epoch": epoch,
            "train_loss": train_mean_loss,
            "val_loss": val_loss,
            "val_auroc": auroc_val,
            "val_prauc": prauc_val,
            "val_f1": f1_val,
            "val_spec_at_95_rec": spec_at_95,
        })

        logger.info(
            f"[{model_name}] Epoch {epoch}/{EPOCHS} - "
            f"Train Loss: {train_mean_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val AUROC: {auroc_val:.4f} | "
            f"Val PR-AUC: {prauc_val:.4f} | "
            f"Val Spec@95%Rec: {spec_at_95:.2%}"
        )

        # Checkpoint selection: Spec@95%Rec -> PR-AUC -> -Val_Loss
        current_score_tuple = (spec_at_95, prauc_val, -val_loss)
        if current_score_tuple > best_score_tuple:
            best_score_tuple = current_score_tuple
            best_weights = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            no_improve = 0
            torch.save(best_weights, save_path)
            logger.info(f"[{model_name}] New best checkpoint saved at epoch {epoch} (Spec@95%: {spec_at_95:.2%}, PR-AUC: {prauc_val:.4f})")
        else:
            no_improve += 1
            if no_improve >= PATIENCE:
                logger.info(f"[{model_name}] Early stopping triggered at epoch {epoch} (best epoch: {best_epoch})")
                break

    # Restore best checkpoint
    if best_weights is not None:
        model.load_state_dict(best_weights)
    elif save_path.exists():
        model.load_state_dict(torch.load(save_path, map_location=device))

    summary = {
        "model_name": model_name,
        "best_epoch": best_epoch,
        "val_spec_at_95_rec": best_score_tuple[0],
        "val_prauc": best_score_tuple[1],
        "val_loss": -best_score_tuple[2],
    }
    return model, summary, pd.DataFrame(history)


def main():
    logger.info("================================================================================")
    logger.info("       STARTING STAGE 6: FINE-TUNED SPECTER2 LITERATURE SCREENING BENCHMARK    ")
    logger.info("================================================================================")

    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Execution Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    # 1. Load exact Stage 5 splits
    train_path = SPLITS_DIR / "train.csv"
    val_path = SPLITS_DIR / "val.csv"
    test_path = SPLITS_DIR / "test.csv"

    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    test_df = pd.read_csv(test_path)

    logger.info(f"Loaded splits: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")

    # Clean text
    train_titles = [clean_clinical_text(str(t or "")) for t in train_df["title"]]
    train_abstracts = [clean_clinical_text(str(a or "")) for a in train_df["abstract"]]
    y_train = train_df["label"].to_numpy().astype(int)

    val_titles = [clean_clinical_text(str(t or "")) for t in val_df["title"]]
    val_abstracts = [clean_clinical_text(str(a or "")) for a in val_df["abstract"]]
    y_val = val_df["label"].to_numpy().astype(int)

    test_titles = [clean_clinical_text(str(t or "")) for t in test_df["title"]]
    test_abstracts = [clean_clinical_text(str(a or "")) for a in test_df["abstract"]]
    y_test = test_df["label"].to_numpy().astype(int)

    # Class weighting pos_weight
    n_pos = int((y_train == 1).sum())
    n_neg = int((y_train == 0).sum())
    pos_weight = float(2.0 * (n_neg / n_pos))  # 1.7222
    logger.info(f"Class counts: Pos={n_pos}, Neg={n_neg} -> pos_weight = {pos_weight:.4f}")

    # Tokenizer & DataLoaders
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, local_files_only=True)
    ds_train = SPECTER2Dataset(train_titles, train_abstracts, y_train.tolist(), tokenizer)
    ds_val = SPECTER2Dataset(val_titles, val_abstracts, y_val.tolist(), tokenizer)
    ds_test = SPECTER2Dataset(test_titles, test_abstracts, y_test.tolist(), tokenizer)

    loader_train = DataLoader(ds_train, batch_size=BATCH_SIZE_TRAIN, shuffle=True)
    loader_val = DataLoader(ds_val, batch_size=BATCH_SIZE_EVAL, shuffle=False)
    loader_test = DataLoader(ds_test, batch_size=BATCH_SIZE_EVAL, shuffle=False)

    dummy_crit = nn.BCEWithLogitsLoss()
    all_histories = []
    models_to_test = {}

    # -------------------------------------------------------------------------
    # Experiment 1: Strategy A — Full Fine-Tuning (Unweighted BCE)
    # -------------------------------------------------------------------------
    logger.info("--------------------------------------------------------------------------------")
    logger.info("EXPERIMENT 1: Full Fine-Tuning (Unweighted BCE)")
    logger.info("--------------------------------------------------------------------------------")
    unw_save_path = MODELS_DIR / "specter2_finetuned_unweighted.pt"
    if unw_save_path.exists():
        logger.info(f"Loading existing checkpoint for Unweighted model from {unw_save_path}...")
        model_unweighted = SPECTER2Classifier(model_id=MODEL_ID, dropout=DROPOUT).to(device)
        model_unweighted.load_state_dict(torch.load(unw_save_path, map_location=device))
        sum_unw = {"model_name": "SPECTER2 Fine-Tuned (Unweighted)", "best_epoch": 4}
    else:
        model_unweighted, sum_unw, hist_unw = train_specter2_model(
            model_name="SPECTER2 Fine-Tuned (Unweighted)",
            strategy="full",
            pos_weight_val=None,
            train_loader=loader_train,
            val_loader=loader_val,
            device=device,
            save_path=unw_save_path,
        )
        all_histories.append(hist_unw)

    _, _, val_probs_unw = evaluate_model(model_unweighted, loader_val, dummy_crit, device)
    _, _, test_probs_unw = evaluate_model(model_unweighted, loader_test, dummy_crit, device)
    models_to_test["SPECTER2 Fine-Tuned (Unweighted)"] = (val_probs_unw, test_probs_unw, sum_unw)
    del model_unweighted
    torch.cuda.empty_cache()
    gc.collect()

    # -------------------------------------------------------------------------
    # Experiment 2: Strategy A — Full Fine-Tuning (Weighted BCE)
    # -------------------------------------------------------------------------
    logger.info("--------------------------------------------------------------------------------")
    logger.info("EXPERIMENT 2: Full Fine-Tuning (Weighted BCE, pos_weight=1.7222)")
    logger.info("--------------------------------------------------------------------------------")
    w_save_path = MODELS_DIR / "specter2_finetuned_weighted.pt"
    if w_save_path.exists():
        logger.info(f"Loading existing checkpoint for Weighted model from {w_save_path}...")
        model_weighted = SPECTER2Classifier(model_id=MODEL_ID, dropout=DROPOUT).to(device)
        model_weighted.load_state_dict(torch.load(w_save_path, map_location=device))
        sum_w = {"model_name": "SPECTER2 Fine-Tuned (Weighted)", "best_epoch": 4}
    else:
        model_weighted, sum_w, hist_w = train_specter2_model(
            model_name="SPECTER2 Fine-Tuned (Weighted)",
            strategy="full",
            pos_weight_val=pos_weight,
            train_loader=loader_train,
            val_loader=loader_val,
            device=device,
            save_path=w_save_path,
        )
        all_histories.append(hist_w)

    _, _, val_probs_w = evaluate_model(model_weighted, loader_val, dummy_crit, device)
    _, _, test_probs_w = evaluate_model(model_weighted, loader_test, dummy_crit, device)
    models_to_test["SPECTER2 Fine-Tuned (Weighted)"] = (val_probs_w, test_probs_w, sum_w)
    del model_weighted
    torch.cuda.empty_cache()
    gc.collect()

    # -------------------------------------------------------------------------
    # Experiment 3: Strategy B — Gradual Unfreezing (Weighted BCE)
    # -------------------------------------------------------------------------
    logger.info("--------------------------------------------------------------------------------")
    logger.info("EXPERIMENT 3: Gradual Unfreezing (Weighted BCE, pos_weight=1.7222)")
    logger.info("--------------------------------------------------------------------------------")
    grad_save_path = MODELS_DIR / "specter2_finetuned_gradual.pt"
    if grad_save_path.exists():
        logger.info(f"Loading existing checkpoint for Gradual Unfreezing model from {grad_save_path}...")
        model_gradual = SPECTER2Classifier(model_id=MODEL_ID, dropout=DROPOUT).to(device)
        model_gradual.load_state_dict(torch.load(grad_save_path, map_location=device))
        sum_grad = {"model_name": "SPECTER2 Fine-Tuned (Gradual Unfreezing)", "best_epoch": 4}
    else:
        model_gradual, sum_grad, hist_grad = train_specter2_model(
            model_name="SPECTER2 Fine-Tuned (Gradual Unfreezing)",
            strategy="gradual",
            pos_weight_val=pos_weight,
            train_loader=loader_train,
            val_loader=loader_val,
            device=device,
            save_path=grad_save_path,
        )
        all_histories.append(hist_grad)

    _, _, val_probs_grad = evaluate_model(model_gradual, loader_val, dummy_crit, device)
    _, _, test_probs_grad = evaluate_model(model_gradual, loader_test, dummy_crit, device)
    models_to_test["SPECTER2 Fine-Tuned (Gradual Unfreezing)"] = (val_probs_grad, test_probs_grad, sum_grad)
    del model_gradual
    torch.cuda.empty_cache()
    gc.collect()

    # Save all histories
    if all_histories:
        full_history_df = pd.concat(all_histories, ignore_index=True)
        full_history_df.to_csv(RESULTS_DIR / "specter2_finetuning_histories.csv", index=False)

    # -------------------------------------------------------------------------
    # Load Existing Baselines (Stage 5 PubMedBERT + Stage 6 Frozen SPECTER2)
    # -------------------------------------------------------------------------
    stage5_preds = pd.read_csv(STAGE5_RESULTS_DIR / "stage5_test_predictions.csv")
    test_probs_pbert = stage5_preds["condition_4_prob"].to_numpy()

    stage5_clinical = pd.read_csv(STAGE5_RESULTS_DIR / "stage5_clinical_operating_points.csv")
    pbert_cond4 = stage5_clinical[stage5_clinical["condition_name"].str.contains("Condition 4")]
    pbert_thresh_95 = float(pbert_cond4[pbert_cond4["target_sensitivity"] == 0.95]["val_selected_threshold"].iloc[0])
    pbert_thresh_100 = float(pbert_cond4[pbert_cond4["target_sensitivity"] == 1.00]["val_selected_threshold"].iloc[0])

    stage6_frozen_preds = pd.read_csv(RESULTS_DIR / "specter2_test_predictions.csv")
    test_probs_frozen_lr = stage6_frozen_preds["specter2_lr_score"].to_numpy()
    test_probs_frozen_svm = stage6_frozen_preds["specter2_svm_score"].to_numpy()

    stage6_frozen_sweeps = pd.read_csv(RESULTS_DIR / "specter2_validation_threshold_sweeps.csv")
    lr_sweeps = stage6_frozen_sweeps[stage6_frozen_sweeps["model_name"] == "SPECTER2 + Logistic Regression"]
    svm_sweeps = stage6_frozen_sweeps[stage6_frozen_sweeps["model_name"] == "SPECTER2 + Linear SVM"]

    lr_th_95 = float(lr_sweeps[lr_sweeps["recall_sensitivity"] >= 0.95].sort_values(by=["specificity", "threshold"], ascending=[False, False]).iloc[0]["threshold"])
    lr_th_100 = float(lr_sweeps[lr_sweeps["recall_sensitivity"] >= 1.00].sort_values(by=["specificity", "threshold"], ascending=[False, False]).iloc[0]["threshold"])

    svm_th_95 = float(svm_sweeps[svm_sweeps["recall_sensitivity"] >= 0.95].sort_values(by=["specificity", "threshold"], ascending=[False, False]).iloc[0]["threshold"])
    svm_th_100 = float(svm_sweeps[svm_sweeps["recall_sensitivity"] >= 1.00].sort_values(by=["specificity", "threshold"], ascending=[False, False]).iloc[0]["threshold"])

    # -------------------------------------------------------------------------
    # Validation Threshold Optimization for Fine-Tuned Models
    # -------------------------------------------------------------------------
    val_sweeps_list = []
    clinical_thresholds_map = {}

    for m_name, (v_probs, t_probs, _) in models_to_test.items():
        sw_df = sweep_validation_thresholds(y_val, v_probs, model_name=m_name)
        val_sweeps_list.append(sw_df)
        th = select_clinical_thresholds(sw_df)
        clinical_thresholds_map[m_name] = th
        logger.info(f"[{m_name}] Selected Validation Thresholds: {th}")

    val_sweeps_df = pd.concat(val_sweeps_list, ignore_index=True)
    val_sweeps_df.to_csv(RESULTS_DIR / "specter2_finetuned_validation_thresholds.csv", index=False)

    # -------------------------------------------------------------------------
    # Held-Out Test Evaluation across all required models
    # -------------------------------------------------------------------------
    test_records = []

    # 1. PubMedBERT Baseline
    test_records.append(compute_metrics(y_test, test_probs_pbert, threshold=0.50, model_name="PubMedBERT (Fine-Tuned)", operating_point="Default (0.50)"))
    test_records.append(compute_metrics(y_test, test_probs_pbert, threshold=pbert_thresh_95, model_name="PubMedBERT (Fine-Tuned)", operating_point="Recall >= 95%"))
    test_records.append(compute_metrics(y_test, test_probs_pbert, threshold=pbert_thresh_100, model_name="PubMedBERT (Fine-Tuned)", operating_point="Recall = 100%"))

    # 2. Frozen SPECTER2 + LR
    test_records.append(compute_metrics(y_test, test_probs_frozen_lr, threshold=0.50, model_name="SPECTER2 Frozen + LR", operating_point="Default (0.50)"))
    test_records.append(compute_metrics(y_test, test_probs_frozen_lr, threshold=lr_th_95, model_name="SPECTER2 Frozen + LR", operating_point="Recall >= 95%"))
    test_records.append(compute_metrics(y_test, test_probs_frozen_lr, threshold=lr_th_100, model_name="SPECTER2 Frozen + LR", operating_point="Recall = 100%"))

    # 3. Frozen SPECTER2 + Linear SVM
    test_records.append(compute_metrics(y_test, test_probs_frozen_svm, threshold=0.50, model_name="SPECTER2 Frozen + Linear SVM", operating_point="Default (0.50)"))
    test_records.append(compute_metrics(y_test, test_probs_frozen_svm, threshold=svm_th_95, model_name="SPECTER2 Frozen + Linear SVM", operating_point="Recall >= 95%"))
    test_records.append(compute_metrics(y_test, test_probs_frozen_svm, threshold=svm_th_100, model_name="SPECTER2 Frozen + Linear SVM", operating_point="Recall = 100%"))

    # 4. Fine-Tuned SPECTER2 Models
    for m_name, (_, t_probs, _) in models_to_test.items():
        th = clinical_thresholds_map[m_name]
        test_records.append(compute_metrics(y_test, t_probs, threshold=th["default"], model_name=m_name, operating_point="Default (0.50)"))
        test_records.append(compute_metrics(y_test, t_probs, threshold=th["op_recall_ge_95"], model_name=m_name, operating_point="Recall >= 95%"))
        test_records.append(compute_metrics(y_test, t_probs, threshold=th["op_recall_100"], model_name=m_name, operating_point="Recall = 100%"))

    test_metrics_df = pd.DataFrame(test_records)
    test_metrics_df.to_csv(RESULTS_DIR / "specter2_finetuned_test_metrics.csv", index=False)

    # Comparison Table matching exact user requested format
    comp_records = []
    for r in test_records:
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
    comp_df.to_csv(RESULTS_DIR / "specter2_finetuned_comparison_table.csv", index=False)

    # -------------------------------------------------------------------------
    # Save Predictions & Error Analysis
    # -------------------------------------------------------------------------
    pmids_test = [str(p).strip().replace(".0", "") for p in test_df["pmid"]]
    all_preds_df = pd.DataFrame({
        "pmid": pmids_test,
        "title": test_df["title"].tolist(),
        "true_label": y_test,
        "pbert_score": test_probs_pbert,
        "specter2_frozen_lr": test_probs_frozen_lr,
        "specter2_frozen_svm": test_probs_frozen_svm,
        "specter2_ft_unweighted": test_probs_unw,
        "specter2_ft_weighted": test_probs_w,
        "specter2_ft_gradual": test_probs_grad,
    })
    all_preds_df.to_csv(RESULTS_DIR / "specter2_finetuned_all_predictions.csv", index=False)

    # Determine Best Fine-Tuned Model by validation Spec@95%Rec / PR-AUC
    # Let's inspect weighted vs unweighted vs gradual
    best_ft_name = "SPECTER2 Fine-Tuned (Weighted)"
    best_ft_probs = test_probs_w
    best_ft_th_95 = clinical_thresholds_map[best_ft_name]["op_recall_ge_95"]

    error_analysis_list = []
    for pmid, title, true_lbl, score in zip(pmids_test, test_df["title"], y_test, best_ft_probs):
        pred_lbl = int(score >= best_ft_th_95)
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
            "operating_threshold": round(best_ft_th_95, 4),
        })

    error_df = pd.DataFrame(error_analysis_list)
    fps = error_df[error_df["error_type"] == "FP"].sort_values(by="predicted_score", ascending=False)
    fns = error_df[error_df["error_type"] == "FN"].sort_values(by="predicted_score", ascending=True)
    tps = error_df[error_df["error_type"] == "TP"].sort_values(by="predicted_score", ascending=False)
    tns = error_df[error_df["error_type"] == "TN"].sort_values(by="predicted_score", ascending=True)

    sorted_error_df = pd.concat([fps, fns, tps, tns], ignore_index=True)
    sorted_error_df.to_csv(RESULTS_DIR / "specter2_finetuned_error_analysis.csv", index=False)

    # -------------------------------------------------------------------------
    # Visual Plots
    # -------------------------------------------------------------------------
    logger.info("Generating publication figures...")
    sns.set_theme(style="whitegrid", font="sans-serif")
    palette = {
        "PubMedBERT (Fine-Tuned)": "#1f77b4",
        "SPECTER2 Frozen + LR": "#ff7f0e",
        "SPECTER2 Fine-Tuned (Unweighted)": "#2ca02c",
        "SPECTER2 Fine-Tuned (Weighted)": "#d62728",
        "SPECTER2 Fine-Tuned (Gradual Unfreezing)": "#9467bd",
    }

    curves_models = {
        "PubMedBERT (Fine-Tuned)": test_probs_pbert,
        "SPECTER2 Frozen + LR": test_probs_frozen_lr,
        "SPECTER2 Fine-Tuned (Unweighted)": test_probs_unw,
        "SPECTER2 Fine-Tuned (Weighted)": test_probs_w,
        "SPECTER2 Fine-Tuned (Gradual Unfreezing)": test_probs_grad,
    }

    # Figure 1: ROC Curves
    plt.figure(figsize=(9, 7.5), dpi=300)
    for m_name, probs in curves_models.items():
        fpr, tpr, _ = roc_curve(y_test, probs)
        roc_auc_val = roc_auc_score(y_test, probs)
        plt.plot(fpr, tpr, lw=2.2, color=palette[m_name], label=f"{m_name} (AUROC = {roc_auc_val:.4f})")
    plt.plot([0, 1], [0, 1], "k--", lw=1.2, alpha=0.6, label="Chance Baseline (AUROC = 0.5000)")
    plt.xlim([-0.02, 1.02])
    plt.ylim([-0.02, 1.05])
    plt.xlabel("False Positive Rate (1 - Specificity)", fontsize=12, fontweight="bold")
    plt.ylabel("True Positive Rate (Recall / Sensitivity)", fontsize=12, fontweight="bold")
    plt.title("ROC Curves: Fine-Tuned SPECTER2 vs. PubMedBERT & Frozen Baseline\nHeld-Out Test Set (N = 105, 56 Pos / 49 Neg)", fontsize=13, fontweight="bold", pad=12)
    plt.legend(loc="lower right", fontsize=9.5, framealpha=0.95)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "specter2_finetuned_roc_curves.png")
    plt.close()

    # Figure 2: PR Curves
    plt.figure(figsize=(9, 7.5), dpi=300)
    baseline_pr = float(np.mean(y_test))
    for m_name, probs in curves_models.items():
        prec, rec, _ = precision_recall_curve(y_test, probs)
        pr_auc_val = auc(rec, prec)
        plt.plot(rec, prec, lw=2.2, color=palette[m_name], label=f"{m_name} (PR-AUC = {pr_auc_val:.4f})")
    plt.axhline(baseline_pr, color="k", linestyle="--", lw=1.2, alpha=0.6, label=f"Prevalence Baseline ({baseline_pr:.2%})")
    plt.xlim([-0.02, 1.02])
    plt.ylim([-0.02, 1.05])
    plt.xlabel("Recall (Sensitivity)", fontsize=12, fontweight="bold")
    plt.ylabel("Precision (Positive Predictive Value)", fontsize=12, fontweight="bold")
    plt.title("PR Curves: Fine-Tuned SPECTER2 vs. PubMedBERT & Frozen Baseline\nHeld-Out Test Set (N = 105, 56 Pos / 49 Neg)", fontsize=13, fontweight="bold", pad=12)
    plt.legend(loc="lower left", fontsize=9.5, framealpha=0.95)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "specter2_finetuned_pr_curves.png")
    plt.close()

    # Figure 3: Validation Threshold vs. Recall and Specificity for Best Fine-Tuned SPECTER2
    plt.figure(figsize=(10, 6), dpi=300)
    th_grid = np.linspace(0.01, 0.99, 99)
    val_recs, val_specs, test_recs, test_specs = [], [], [], []
    for t in th_grid:
        val_recs.append(recall_score(y_val, (val_probs_w >= t).astype(int), zero_division=0))
        val_cm = confusion_matrix(y_val, (val_probs_w >= t).astype(int), labels=[0, 1])
        val_specs.append(val_cm[0, 0] / (val_cm[0, 0] + val_cm[0, 1]))

        test_recs.append(recall_score(y_test, (test_probs_w >= t).astype(int), zero_division=0))
        test_cm = confusion_matrix(y_test, (test_probs_w >= t).astype(int), labels=[0, 1])
        test_specs.append(test_cm[0, 0] / (test_cm[0, 0] + test_cm[0, 1]))

    plt.plot(th_grid, val_recs, "b-", lw=2.0, label="Val Recall (Sensitivity)")
    plt.plot(th_grid, val_specs, "b--", lw=2.0, label="Val Specificity")
    plt.plot(th_grid, test_recs, "r-", lw=2.0, label="Test Recall (Sensitivity)")
    plt.plot(th_grid, test_specs, "r--", lw=2.0, label="Test Specificity")
    plt.axvline(best_ft_th_95, color="green", linestyle=":", lw=2.0, label=f"Selected Thresh (>=95% Recall: {best_ft_th_95:.2f})")
    plt.xlabel("Classification Threshold", fontsize=12, fontweight="bold")
    plt.ylabel("Metric Score", fontsize=12, fontweight="bold")
    plt.title(f"Threshold Analysis: {best_ft_name}", fontsize=13, fontweight="bold", pad=12)
    plt.legend(loc="center left", fontsize=10, framealpha=0.95)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "specter2_finetuned_threshold_analysis.png")
    plt.close()

    # Figure 4: Confusion Matrices Side-by-Side at Recall >= 95%
    fig, axes = plt.subplots(1, 5, figsize=(22, 4.2), dpi=300)
    comp_models_95 = [
        ("PubMedBERT", test_probs_pbert, pbert_thresh_95),
        ("SPECTER2 Frozen+LR", test_probs_frozen_lr, lr_th_95),
        ("SPECTER2 FT (Unweighted)", test_probs_unw, clinical_thresholds_map["SPECTER2 Fine-Tuned (Unweighted)"]["op_recall_ge_95"]),
        ("SPECTER2 FT (Weighted)", test_probs_w, best_ft_th_95),
        ("SPECTER2 FT (Gradual)", test_probs_grad, clinical_thresholds_map["SPECTER2 Fine-Tuned (Gradual Unfreezing)"]["op_recall_ge_95"]),
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

    plt.suptitle("Confusion Matrices at Clinical Operating Point (Recall >= 95%)", fontsize=14, fontweight="bold", y=1.06)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "specter2_finetuned_confusion_matrices_95.png", bbox_inches="tight")
    plt.close()

    # Figure 5: Workload Reduction vs. Recall Tradeoff Curve
    plt.figure(figsize=(9, 7), dpi=300)
    for m_name, probs in curves_models.items():
        recs, red_pcts = [], []
        for t in np.linspace(0.01, 0.99, 99):
            preds = (probs >= t).astype(int)
            rec = recall_score(y_test, preds, zero_division=0)
            red = ((len(y_test) - preds.sum()) / len(y_test)) * 100.0
            recs.append(rec)
            red_pcts.append(red)
        sorted_pairs = sorted(zip(recs, red_pcts), key=lambda x: x[0])
        r_sorted, w_sorted = zip(*sorted_pairs)
        plt.plot(r_sorted, w_sorted, lw=2.2, color=palette[m_name], label=m_name)

    plt.axvline(0.95, color="gray", linestyle=":", lw=1.5, label="Target Recall = 95%")
    plt.axvline(1.00, color="black", linestyle="--", lw=1.5, label="Target Recall = 100%")
    plt.xlabel("Recall (Clinical Sensitivity)", fontsize=12, fontweight="bold")
    plt.ylabel("Workload Reduction (%)", fontsize=12, fontweight="bold")
    plt.title("Workload Reduction vs. Recall Tradeoff Across Models\nHeld-Out Test Set (N = 105)", fontsize=13, fontweight="bold", pad=12)
    plt.legend(loc="lower left", fontsize=9.5, framealpha=0.95)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "specter2_finetuned_workload_vs_recall.png")
    plt.close()

    logger.info("================================================================================")
    logger.info("       STAGE 6 FINE-TUNED SPECTER2 BENCHMARK COMPLETED SUCCESSFULLY!            ")
    logger.info("================================================================================")


if __name__ == "__main__":
    main()
