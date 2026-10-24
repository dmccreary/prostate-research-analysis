"""
train_pico_cross_encoder.py - Stage 7: PICO Cross-Encoder Literature Screening.

Trains and evaluates a PICO-conditioned Cross-Encoder on PubMedBERT:
- Sequence length: 512 tokens
- Segment A: 73-token PICO Eligibility Criteria Query (token_type_ids = 0)
- Segment B: Title + Abstract of the clinical paper (token_type_ids = 1)
- Loss: Cost-Sensitive Weighted BCE (pos_weight = 1.7222)
- Architecture: FlexiblePubMedBERT [CLS] (768) -> Linear(768, 128) -> LayerNorm(128) -> GELU() -> Dropout(0.25) -> Linear(128, 1)
- Optimization: AdamW (lr=2e-5, weight_decay=1e-4), linear warmup (10% steps), batch_size=4, grad_accum=2 (effective 8), 8 epochs, patience=4
- Benchmarked against:
  * Stage 5.5 PubMedBERT 512-token Weighted (Title vs Abstract baseline)
  * Stage 5 PubMedBERT 384-token Weighted (Condition 4 baseline)
  * Stage 6 SPECTER2 Fine-Tuned (Unweighted & Weighted)
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

# Force offline mode for local HuggingFace cache loading
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"

# Directory paths
BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(BASE_DIR / "src"))
from preprocessing import clean_clinical_text

DATA_DIR = BASE_DIR / "data"
SPLITS_DIR = DATA_DIR / "splits" / "stage5"
MODELS_DIR = BASE_DIR / "models" / "stage7"
RESULTS_DIR = BASE_DIR / "results" / "stage7"
PLOTS_DIR = RESULTS_DIR / "plots"
REPORTS_DIR = BASE_DIR / "reports" / "stage7"
LOGS_DIR = BASE_DIR / "logs"

STAGE5_RESULTS_DIR = BASE_DIR / "results" / "stage5"
STAGE5_512_RESULTS_DIR = BASE_DIR / "results" / "stage5_512"
STAGE6_RESULTS_DIR = BASE_DIR / "results" / "stage6"

for d in [MODELS_DIR, RESULTS_DIR, PLOTS_DIR, REPORTS_DIR, LOGS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOGS_DIR / "stage7_pico_cross_encoder.log", encoding="utf-8", mode="w"),
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

# 73-token PICO Criteria Query (Segment A)
PICO_QUERY = (
    "Screening criteria for localized prostate cancer: Primary curative treatment "
    "(radical prostatectomy, EBRT >= 72 Gy, brachytherapy, cryo, HIFU). "
    "Must be a comparative trial (exclude single-arm audits, post-treatment salvage, metastatic, or palliative). "
    "Reports oncologic survival (BRFS, OS, CSS) with follow-up >= 5 years."
)


def set_all_seeds(seed: int = SEED):
    """Ensures strict determinism across all random number generators."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class PICOCrossEncoderDataset(Dataset):
    """
    Cross-Encoder Dataset:
    - Segment A (token_type_ids = 0): PICO Criteria Query
    - Segment B (token_type_ids = 1): Title + Abstract of the candidate paper
    """

    def __init__(
        self,
        titles: List[str],
        abstracts: List[str],
        labels: List[int],
        tokenizer: AutoTokenizer,
        pico_query: str = PICO_QUERY,
        max_length: int = MAX_LENGTH,
    ):
        self.labels = torch.tensor(labels, dtype=torch.float32)
        pico_queries = [pico_query] * len(titles)
        paper_texts = [f"{t}. {a}" if t else a for t, a in zip(titles, abstracts)]

        enc = tokenizer(
            text=pico_queries,
            text_pair=paper_texts,
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
    PubMedBERT Backbone + 2-layer MLP classification head:
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
    """Computes screening and diagnostic metrics."""
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
    Selects clinical operating thresholds strictly on validation data.
    Finds the highest threshold achieving validation recall >= target.
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


def main():
    logger.info("================================================================================")
    logger.info("       STARTING STAGE 7: PICO CROSS-ENCODER SCREENING EXPERIMENT                ")
    logger.info("================================================================================")

    set_all_seeds(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Execution Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    # 1. Load exact Stage 5 splits
    train_path = SPLITS_DIR / "train.csv"
    val_path = SPLITS_DIR / "val.csv"
    test_path = SPLITS_DIR / "test.csv"

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

    # Calculate exact cost-sensitive pos_weight (identical to Stage 5 & 5.5: 2.0 * neg / pos)
    n_pos = int((y_train == 1).sum())
    n_neg = int((y_train == 0).sum())
    pos_weight = 2.0 * (n_neg / n_pos)  # 2.0 * 155 / 180 = 1.7222
    logger.info(f"Train Counts: Pos={n_pos}, Neg={n_neg} -> pos_weight={pos_weight:.4f}")

    # 3. Load Tokenizer & Build PICO Datasets
    logger.info(f"Loading local tokenizer for {PRETRAINED_MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(PRETRAINED_MODEL_NAME, local_files_only=True)

    pico_tokens_count = len(tokenizer.encode(PICO_QUERY, add_special_tokens=False))
    logger.info(f"PICO Query Length: {pico_tokens_count} tokens.")
    logger.info(f"Remaining budget for Title + Abstract: {MAX_LENGTH - (pico_tokens_count + 3)} tokens.")

    ds_train = PICOCrossEncoderDataset(train_titles, train_abstracts, y_train, tokenizer, pico_query=PICO_QUERY, max_length=MAX_LENGTH)
    ds_val = PICOCrossEncoderDataset(val_titles, val_abstracts, y_val, tokenizer, pico_query=PICO_QUERY, max_length=MAX_LENGTH)
    ds_test = PICOCrossEncoderDataset(test_titles, test_abstracts, y_test, tokenizer, pico_query=PICO_QUERY, max_length=MAX_LENGTH)

    loader_train = DataLoader(ds_train, batch_size=BATCH_SIZE_TRAIN, shuffle=True)
    loader_val = DataLoader(ds_val, batch_size=BATCH_SIZE_EVAL, shuffle=False)
    loader_test = DataLoader(ds_test, batch_size=BATCH_SIZE_EVAL, shuffle=False)

    # 4. Initialize Model & Training Configuration
    model = FlexiblePubMedBERT().to(device)
    weight_tensor = torch.tensor([pos_weight], device=device, dtype=torch.float32)
    criterion = nn.BCEWithLogitsLoss(pos_weight=weight_tensor)

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

    effective_steps_per_epoch = (len(loader_train) + GRAD_ACCUM_STEPS - 1) // GRAD_ACCUM_STEPS
    total_steps = effective_steps_per_epoch * EPOCHS
    warmup_steps = int(total_steps * 0.1)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )

    save_path = MODELS_DIR / "pico_cross_encoder_weighted.pt"
    best_val_loss = float("inf")
    best_weights = None
    no_improve_epochs = 0
    best_epoch = 0
    history = []

    logger.info(
        f"Starting PICO Cross-Encoder training: {EPOCHS} epochs, batch_size={BATCH_SIZE_TRAIN}, "
        f"grad_accum={GRAD_ACCUM_STEPS} (effective 8), max_len={MAX_LENGTH}, lr={LEARNING_RATE}, pos_weight={pos_weight:.4f}..."
    )

    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss = 0.0
        train_samples = 0
        optimizer.zero_grad()

        for step, batch in enumerate(loader_train):
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

            if (step + 1) % GRAD_ACCUM_STEPS == 0 or (step + 1) == len(loader_train):
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=MAX_GRAD_NORM)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            train_loss += loss.item() * len(labels)
            train_samples += len(labels)

        train_mean_loss = train_loss / train_samples
        val_mean_loss, val_y_true, val_y_prob = evaluate_loader(model, loader_val, criterion, device)

        try:
            val_auroc = roc_auc_score(val_y_true, val_y_prob)
            val_p_curve, val_r_curve, _ = precision_recall_curve(val_y_true, val_y_prob)
            val_prauc = auc(val_r_curve, val_p_curve)
        except Exception:
            val_auroc, val_prauc = 0.5, 0.5

        val_f1 = f1_score(val_y_true, (val_y_prob >= 0.5).astype(int), zero_division=0)
        val_rec = recall_score(val_y_true, (val_y_prob >= 0.5).astype(int), zero_division=0)

        history.append({
            "model_name": "PICO Cross-Encoder (Weighted)",
            "epoch": epoch,
            "train_loss": train_mean_loss,
            "val_loss": val_mean_loss,
            "val_auroc": val_auroc,
            "val_prauc": val_prauc,
            "val_f1": val_f1,
            "val_recall": val_rec,
        })

        logger.info(
            f"[PICO Cross-Encoder] Epoch {epoch}/{EPOCHS} - "
            f"Train Loss: {train_mean_loss:.4f} | "
            f"Val Loss: {val_mean_loss:.4f} | "
            f"Val AUROC: {val_auroc:.4f} | "
            f"Val PR-AUC: {val_prauc:.4f} | "
            f"Val F1@0.5: {val_f1:.4f} | Val Recall@0.5: {val_rec:.2%}"
        )

        # Early stopping on validation loss
        if val_mean_loss < best_val_loss:
            best_val_loss = val_mean_loss
            best_epoch = epoch
            best_weights = copy.deepcopy(model.state_dict())
            no_improve_epochs = 0
            torch.save(best_weights, save_path)
            logger.info(f"[PICO Cross-Encoder] New best checkpoint saved at epoch {epoch} (Val Loss: {val_mean_loss:.4f})")
        else:
            no_improve_epochs += 1
            if no_improve_epochs >= PATIENCE:
                logger.info(f"[PICO Cross-Encoder] Early stopping triggered at epoch {epoch} (best epoch: {best_epoch})")
                break

    # Restore best checkpoint
    if best_weights is not None:
        model.load_state_dict(best_weights)
    elif save_path.exists():
        model.load_state_dict(torch.load(save_path, map_location=device))

    # Save training history
    history_df = pd.DataFrame(history)
    history_df.to_csv(RESULTS_DIR / "pico_cross_encoder_history.csv", index=False)

    # 5. Evaluate Validation & Test Probabilities
    dummy_crit = nn.BCEWithLogitsLoss()
    _, _, val_probs_pico = evaluate_loader(model, loader_val, dummy_crit, device)
    _, _, test_probs_pico = evaluate_loader(model, loader_test, dummy_crit, device)

    del model
    torch.cuda.empty_cache()
    gc.collect()

    # -------------------------------------------------------------------------
    # 6. Load Existing Project Baselines for Direct Comparison
    # -------------------------------------------------------------------------
    # Baseline 1: Stage 5.5 PubMedBERT 512-token Weighted
    stage5_512_preds = pd.read_csv(STAGE5_512_RESULTS_DIR / "pubmedbert_512_test_predictions.csv")
    test_probs_pbert_512 = stage5_512_preds["pubmedbert_512_weighted_prob"].to_numpy()

    # Baseline 2: Stage 5 PubMedBERT 384-token Weighted (Condition 4)
    stage5_preds = pd.read_csv(STAGE5_RESULTS_DIR / "stage5_test_predictions.csv")
    test_probs_pbert_384 = stage5_preds["condition_4_prob"].to_numpy()

    # Baseline 3: Stage 6 SPECTER2 Fine-Tuned (Unweighted & Weighted)
    stage6_preds = pd.read_csv(STAGE6_RESULTS_DIR / "specter2_finetuned_all_predictions.csv")
    test_probs_specter2_unw = stage6_preds["specter2_ft_unweighted"].to_numpy()
    test_probs_specter2_w = stage6_preds["specter2_ft_weighted"].to_numpy()

    # Get validation probabilities for PubMedBERT 512 Weighted to run fair threshold sweeps
    # We can load the Stage 5.5 validation thresholds from its clinical operating points CSV:
    stage5_512_clinical = pd.read_csv(STAGE5_512_RESULTS_DIR / "pubmedbert_512_clinical_operating_points.csv")
    pbert_512_w_rows = stage5_512_clinical[stage5_512_clinical["model_name"] == "PubMedBERT 512 (Model B, Weighted)"]
    th_512_w_90 = float(pbert_512_w_rows[pbert_512_w_rows["target_sensitivity"] == 0.90]["val_selected_threshold"].iloc[0])
    th_512_w_95 = float(pbert_512_w_rows[pbert_512_w_rows["target_sensitivity"] == 0.95]["val_selected_threshold"].iloc[0])
    th_512_w_100 = float(pbert_512_w_rows[pbert_512_w_rows["target_sensitivity"] == 1.00]["val_selected_threshold"].iloc[0])

    # Validation thresholds for Stage 5 384-token Baseline
    stage5_clinical = pd.read_csv(STAGE5_RESULTS_DIR / "stage5_clinical_operating_points.csv")
    cond4_rows = stage5_clinical[stage5_clinical["condition_name"].str.contains("Condition 4")]
    th_384_w_90 = float(cond4_rows[cond4_rows["target_sensitivity"] == 0.90]["val_selected_threshold"].iloc[0])
    th_384_w_95 = float(cond4_rows[cond4_rows["target_sensitivity"] == 0.95]["val_selected_threshold"].iloc[0])
    th_384_w_100 = float(cond4_rows[cond4_rows["target_sensitivity"] == 1.00]["val_selected_threshold"].iloc[0])

    # -------------------------------------------------------------------------
    # 7. Sweep Validation Thresholds for PICO Cross-Encoder
    # -------------------------------------------------------------------------
    pico_val_thresholds = sweep_validation_thresholds(y_val, val_probs_pico, target_sensitivities=[0.90, 0.95, 1.00])
    logger.info(f"PICO Cross-Encoder Validation-Selected Thresholds: {pico_val_thresholds}")

    # -------------------------------------------------------------------------
    # 8. Compute Test Metrics Across All Operating Points
    # -------------------------------------------------------------------------
    all_records = []

    # A. PICO Cross-Encoder (Stage 7)
    all_records.append(compute_metrics(y_test, test_probs_pico, threshold=0.50, model_name="PICO Cross-Encoder (PubMedBERT 512, Weighted)", operating_point="Default (0.50)"))
    all_records.append(compute_metrics(y_test, test_probs_pico, threshold=pico_val_thresholds[0.90], model_name="PICO Cross-Encoder (PubMedBERT 512, Weighted)", operating_point="Recall >= 90%"))
    all_records.append(compute_metrics(y_test, test_probs_pico, threshold=pico_val_thresholds[0.95], model_name="PICO Cross-Encoder (PubMedBERT 512, Weighted)", operating_point="Recall >= 95%"))
    all_records.append(compute_metrics(y_test, test_probs_pico, threshold=pico_val_thresholds[1.00], model_name="PICO Cross-Encoder (PubMedBERT 512, Weighted)", operating_point="Recall = 100%"))

    # B. PubMedBERT 512 Weighted Baseline (Stage 5.5)
    all_records.append(compute_metrics(y_test, test_probs_pbert_512, threshold=0.50, model_name="PubMedBERT 512 (Title+Abstract, Weighted)", operating_point="Default (0.50)"))
    all_records.append(compute_metrics(y_test, test_probs_pbert_512, threshold=th_512_w_90, model_name="PubMedBERT 512 (Title+Abstract, Weighted)", operating_point="Recall >= 90%"))
    all_records.append(compute_metrics(y_test, test_probs_pbert_512, threshold=th_512_w_95, model_name="PubMedBERT 512 (Title+Abstract, Weighted)", operating_point="Recall >= 95%"))
    all_records.append(compute_metrics(y_test, test_probs_pbert_512, threshold=th_512_w_100, model_name="PubMedBERT 512 (Title+Abstract, Weighted)", operating_point="Recall = 100%"))

    # C. PubMedBERT 384 Weighted Baseline (Stage 5 Condition 4)
    all_records.append(compute_metrics(y_test, test_probs_pbert_384, threshold=0.50, model_name="PubMedBERT 384 (Stage 5 Baseline, Weighted)", operating_point="Default (0.50)"))
    all_records.append(compute_metrics(y_test, test_probs_pbert_384, threshold=th_384_w_90, model_name="PubMedBERT 384 (Stage 5 Baseline, Weighted)", operating_point="Recall >= 90%"))
    all_records.append(compute_metrics(y_test, test_probs_pbert_384, threshold=th_384_w_95, model_name="PubMedBERT 384 (Stage 5 Baseline, Weighted)", operating_point="Recall >= 95%"))
    all_records.append(compute_metrics(y_test, test_probs_pbert_384, threshold=th_384_w_100, model_name="PubMedBERT 384 (Stage 5 Baseline, Weighted)", operating_point="Recall = 100%"))

    test_metrics_df = pd.DataFrame(all_records)
    test_metrics_df.to_csv(RESULTS_DIR / "pico_cross_encoder_test_metrics.csv", index=False)

    # Formatted Comparison Table
    comp_records = []
    for r in all_records:
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
    comp_df = pd.DataFrame(comp_records)
    comp_df.to_csv(RESULTS_DIR / "stage7_model_comparison.csv", index=False)

    # -------------------------------------------------------------------------
    # 9. Save Predictions & Detailed Error Audit
    # -------------------------------------------------------------------------
    pmids_test = [str(p).strip().replace(".0", "") for p in test_df["pmid"]]
    predictions_df = pd.DataFrame({
        "pmid": pmids_test,
        "title": test_df["title"].tolist(),
        "true_label": y_test,
        "pico_cross_encoder_prob": test_probs_pico,
        "pubmedbert_512_weighted_prob": test_probs_pbert_512,
        "pubmedbert_384_weighted_prob": test_probs_pbert_384,
        "specter2_ft_weighted_prob": test_probs_specter2_w,
        "pico_pred_0.50": (test_probs_pico >= 0.50).astype(int),
        "pico_pred_95rec": (test_probs_pico >= pico_val_thresholds[0.95]).astype(int),
    })
    predictions_df.to_csv(RESULTS_DIR / "pico_cross_encoder_test_predictions.csv", index=False)

    # Detailed Error Audit at 0.50
    error_list = []
    for pmid, title, true_lbl, prob in zip(pmids_test, test_df["title"], y_test, test_probs_pico):
        pred_lbl = int(prob >= 0.50)
        if true_lbl == 1 and pred_lbl == 1:
            cat = "TP"
        elif true_lbl == 0 and pred_lbl == 0:
            cat = "TN"
        elif true_lbl == 1 and pred_lbl == 0:
            cat = "FN"
        else:
            cat = "FP"
        error_list.append({
            "pmid": pmid,
            "title": title,
            "true_label": int(true_lbl),
            "pico_prob": round(float(prob), 4),
            "predicted_label": pred_lbl,
            "category": cat,
        })
    error_df = pd.DataFrame(error_list)
    error_df.to_csv(RESULTS_DIR / "pico_cross_encoder_error_audit.csv", index=False)

    # -------------------------------------------------------------------------
    # 10. Publication Figures
    # -------------------------------------------------------------------------
    logger.info("Generating publication figures...")
    sns.set_theme(style="whitegrid", font="sans-serif")

    plot_models = {
        "PICO Cross-Encoder (512 tokens, Weighted)": (test_probs_pico, "#d62728"),
        "PubMedBERT 512 Baseline (Weighted)": (test_probs_pbert_512, "#1f77b4"),
        "PubMedBERT 384 Baseline (Weighted)": (test_probs_pbert_384, "#7f7f7f"),
        "SPECTER2 Fine-Tuned (Weighted)": (test_probs_specter2_w, "#2ca02c"),
    }

    # Figure 1: ROC Curves
    plt.figure(figsize=(8.5, 7), dpi=300)
    for m_lbl, (probs, color) in plot_models.items():
        fpr, tpr, _ = roc_curve(y_test, probs)
        roc_val = roc_auc_score(y_test, probs)
        plt.plot(fpr, tpr, lw=2.4, color=color, label=f"{m_lbl} (AUROC = {roc_val:.4f})")
    plt.plot([0, 1], [0, 1], "k--", lw=1.2, alpha=0.6, label="Chance Baseline (AUROC = 0.5000)")
    plt.xlim([-0.02, 1.02])
    plt.ylim([-0.02, 1.05])
    plt.xlabel("False Positive Rate (1 - Specificity)", fontsize=12, fontweight="bold")
    plt.ylabel("True Positive Rate (Recall / Sensitivity)", fontsize=12, fontweight="bold")
    plt.title("ROC Curves: Stage 7 PICO Cross-Encoder vs. Previous Baselines\nHeld-Out Test Set (N = 105, 56 Pos / 49 Neg)", fontsize=13, fontweight="bold", pad=12)
    plt.legend(loc="lower right", fontsize=9.5, framealpha=0.95)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "pico_cross_encoder_roc_curves.png")
    plt.close()

    # Figure 2: PR Curves
    plt.figure(figsize=(8.5, 7), dpi=300)
    prev = float(np.mean(y_test))
    for m_lbl, (probs, color) in plot_models.items():
        prec, rec, _ = precision_recall_curve(y_test, probs)
        prauc_val = auc(rec, prec)
        plt.plot(rec, prec, lw=2.4, color=color, label=f"{m_lbl} (PR-AUC = {prauc_val:.4f})")
    plt.axhline(prev, color="k", linestyle="--", lw=1.2, alpha=0.6, label=f"Prevalence Baseline ({prev:.2%})")
    plt.xlim([-0.02, 1.02])
    plt.ylim([-0.02, 1.05])
    plt.xlabel("Recall (Sensitivity)", fontsize=12, fontweight="bold")
    plt.ylabel("Precision (Positive Predictive Value)", fontsize=12, fontweight="bold")
    plt.title("PR Curves: Stage 7 PICO Cross-Encoder vs. Previous Baselines\nHeld-Out Test Set (N = 105, 56 Pos / 49 Neg)", fontsize=13, fontweight="bold", pad=12)
    plt.legend(loc="lower left", fontsize=9.5, framealpha=0.95)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "pico_cross_encoder_pr_curves.png")
    plt.close()

    # Figure 3: Side-by-Side Confusion Matrices at Default Threshold 0.50
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5), dpi=300)
    cm_models = [
        ("PubMedBERT 384 Baseline", test_probs_pbert_384, 0.50),
        ("PubMedBERT 512 Baseline", test_probs_pbert_512, 0.50),
        ("PICO Cross-Encoder (512)", test_probs_pico, 0.50),
    ]
    for idx, (m_lbl, probs, th_val) in enumerate(cm_models):
        y_p = (probs >= th_val).astype(int)
        cm = confusion_matrix(y_test, y_p, labels=[0, 1])
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=axes[idx],
                    xticklabels=["Pred Neg (0)", "Pred Pos (1)"],
                    yticklabels=["True Neg (0)", "True Pos (1)"],
                    annot_kws={"size": 13, "weight": "bold"})
        axes[idx].set_title(f"{m_lbl}\n(Threshold = {th_val:.2f})", fontsize=11, fontweight="bold")
        axes[idx].set_xlabel("Predicted Label", fontsize=10)
        if idx == 0:
            axes[idx].set_ylabel("True Ground Truth", fontsize=10)
        else:
            axes[idx].set_ylabel("")

    plt.suptitle("Confusion Matrices at Default Operating Point (Threshold = 0.50)", fontsize=14, fontweight="bold", y=1.05)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "pico_cross_encoder_confusion_matrices.png", bbox_inches="tight")
    plt.close()

    # Figure 4: Workload Reduction vs. Recall Tradeoff Curve
    plt.figure(figsize=(8.5, 7), dpi=300)
    for m_lbl, (probs, color) in plot_models.items():
        recs, red_pcts = [], []
        for t in np.linspace(0.01, 0.99, 99):
            preds = (probs >= t).astype(int)
            rec = recall_score(y_test, preds, zero_division=0)
            red = ((len(y_test) - preds.sum()) / len(y_test)) * 100.0
            recs.append(rec)
            red_pcts.append(red)
        sorted_pairs = sorted(zip(recs, red_pcts), key=lambda x: x[0])
        r_sorted, w_sorted = zip(*sorted_pairs)
        plt.plot(r_sorted, w_sorted, lw=2.4, color=color, label=m_lbl)

    plt.axvline(0.90, color="lightgray", linestyle=":", lw=1.2, label="Target Recall = 90%")
    plt.axvline(0.95, color="gray", linestyle=":", lw=1.5, label="Target Recall = 95%")
    plt.axvline(1.00, color="black", linestyle="--", lw=1.5, label="Target Recall = 100%")
    plt.xlabel("Recall (Clinical Sensitivity)", fontsize=12, fontweight="bold")
    plt.ylabel("Workload Reduction (%)", fontsize=12, fontweight="bold")
    plt.title("Workload Reduction vs. Recall Tradeoff Curve\nHeld-Out Test Set (N = 105)", fontsize=13, fontweight="bold", pad=12)
    plt.legend(loc="lower left", fontsize=9.0, framealpha=0.95)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "pico_cross_encoder_workload_vs_recall.png")
    plt.close()

    logger.info("================================================================================")
    logger.info("       STAGE 7 PICO CROSS-ENCODER EXPERIMENT COMPLETED SUCCESSFULLY!            ")
    logger.info("================================================================================")


if __name__ == "__main__":
    main()
