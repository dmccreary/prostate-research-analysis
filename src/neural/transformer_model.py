"""
transformer_model.py - Biomedical Transformer (PubMedBERT) fine-tuning and evaluation.

Model: microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract
Hardware-conscious configuration:
- GPU: GTX 1050 Ti (4 GB VRAM)
- Sequence length: max_length=384 (covers >95% of abstracts without padding bloat)
- Batch size: 8 with gradient accumulation steps=2 (effective batch size 16)
- Optimizer: AdamW (lr=2e-5, weight_decay=0.01) with linear warmup
- Early stopping based on validation Average Precision (PR-AUC)
"""

import copy
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from preprocessing import clean_clinical_text

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

logger = logging.getLogger(__name__)

DEFAULT_MODEL_NAME = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract"


class TransformerAbstractDataset(Dataset):
    """
    Tokenizes raw clinical abstract text using a HuggingFace tokenizer.
    """

    def __init__(self, texts: List[str], labels: List[int], tokenizer, max_len: int = 384):
        self.labels = labels
        clean_texts = [clean_clinical_text(t) for t in texts]
        self.encodings = tokenizer(
            clean_texts,
            truncation=True,
            max_length=max_len,
            padding="max_length",
            return_tensors="pt",
        )

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {
            "input_ids": self.encodings["input_ids"][idx],
            "attention_mask": self.encodings["attention_mask"][idx],
            "label": torch.tensor(self.labels[idx], dtype=torch.float32),
        }
        return item


def count_transformer_parameters(model: nn.Module) -> int:
    """Return total number of trainable parameters in the transformer."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train_transformer_model(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    model_name: str = DEFAULT_MODEL_NAME,
    output_dir: str = "models/neural",
    history_dir: str = "results/neural/training_history",
    max_len: int = 384,
    batch_size: int = 8,
    grad_accum_steps: int = 2,
    epochs: int = 8,
    lr: float = 2e-5,
    weight_decay: float = 0.01,
    pos_weight: float = 2.03,
    patience: int = 4,
    device: Optional[torch.device] = None,
) -> Tuple[nn.Module, Any, Dict[str, Any], pd.DataFrame, DataLoader]:
    """
    Fine-tune PubMedBERT for binary abstract classification.
    """
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    logger.info("Initializing Tokenizer and Model: %s on %s", model_name, device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # num_labels=1 outputs raw binary logit
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=1)
    model = model.to(device)

    train_ds = TransformerAbstractDataset(train_df["abstract"].tolist(), train_df["label"].tolist(), tokenizer, max_len)
    val_ds = TransformerAbstractDataset(val_df["abstract"].tolist(), val_df["label"].tolist(), tokenizer, max_len)
    test_ds = TransformerAbstractDataset(test_df["abstract"].tolist(), test_df["label"].tolist(), tokenizer, max_len)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size * 2, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size * 2, shuffle=False)

    pos_weight_tensor = torch.tensor([pos_weight], device=device, dtype=torch.float32)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    total_steps = (len(train_loader) // grad_accum_steps) * epochs
    warmup_steps = int(0.1 * total_steps)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

    best_state = copy.deepcopy(model.state_dict())
    best_val_ap = 0.0
    best_epoch = 0
    patience_counter = 0

    history_records = []
    t0_start = time.time()

    logger.info(
        "Beginning PubMedBERT fine-tuning: Train=%d, Val=%d, Epochs=%d, BatchSize=%d (Accum=%d, Effective=%d)",
        len(train_ds),
        len(val_ds),
        epochs,
        batch_size,
        grad_accum_steps,
        batch_size * grad_accum_steps,
    )

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        total_samples = 0
        optimizer.zero_grad()

        for step, batch in enumerate(train_loader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits.squeeze(-1)

            loss = criterion(logits, labels)
            loss_scaled = loss / grad_accum_steps
            loss_scaled.backward()

            train_loss += loss.item() * len(labels)
            total_samples += len(labels)

            if (step + 1) % grad_accum_steps == 0 or (step + 1) == len(train_loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

        avg_train_loss = train_loss / total_samples

        # Validation
        model.eval()
        val_loss = 0.0
        val_samples = 0
        all_val_y = []
        all_val_probs = []

        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["label"].to(device)

                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits.squeeze(-1)
                loss = criterion(logits, labels)
                probs = torch.sigmoid(logits)

                val_loss += loss.item() * len(labels)
                val_samples += len(labels)
                all_val_y.extend(labels.cpu().numpy().tolist())
                all_val_probs.extend(probs.cpu().numpy().tolist())

        avg_val_loss = val_loss / val_samples
        val_y = np.array(all_val_y)
        val_probs = np.array(all_val_probs)
        val_preds = (val_probs >= 0.5).astype(int)

        try:
            val_ap = float(average_precision_score(val_y, val_probs))
            val_auc = float(roc_auc_score(val_y, val_probs))
        except Exception:
            val_ap = 0.0
            val_auc = 0.5
        val_f1 = float(f1_score(val_y, val_preds, zero_division=0))

        improved = val_ap > best_val_ap + 1e-4
        if improved:
            best_val_ap = val_ap
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1

        history_records.append({
            "epoch": epoch,
            "train_loss": round(avg_train_loss, 4),
            "val_loss": round(avg_val_loss, 4),
            "val_ap": round(val_ap, 4),
            "val_auc": round(val_auc, 4),
            "val_f1": round(val_f1, 4),
            "lr": optimizer.param_groups[0]["lr"],
        })

        logger.info(
            "Transformer Epoch %02d/%d: Train Loss=%.4f, Val Loss=%.4f, Val AP=%.4f, Val AUC=%.4f, Val F1=%.4f %s",
            epoch,
            epochs,
            avg_train_loss,
            avg_val_loss,
            val_ap,
            val_auc,
            val_f1,
            "(BEST)" if improved else "",
        )

        if patience_counter >= patience:
            logger.info("Early stopping triggered for PubMedBERT at epoch %d", epoch)
            break

    total_train_time = time.time() - t0_start

    # Load best model weights
    model.load_state_dict(best_state)

    # Save checkpoint
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    save_path = out_dir / "biomedical_transformer.pt"
    torch.save({
        "model_name": "biomedical_transformer",
        "base_model": model_name,
        "state_dict": best_state,
        "best_epoch": best_epoch,
        "best_val_ap": best_val_ap,
        "train_time_sec": total_train_time,
    }, save_path)
    logger.info("Saved PubMedBERT checkpoint to %s", save_path)

    # Save history
    hist_dir = Path(history_dir)
    hist_dir.mkdir(parents=True, exist_ok=True)
    history_df = pd.DataFrame(history_records)
    history_df.to_csv(hist_dir / "biomedical_transformer_history.csv", index=False)

    summary = {
        "model_name": "Biomedical Transformer (PubMedBERT)",
        "base_model": model_name,
        "best_epoch": best_epoch,
        "best_val_ap": best_val_ap,
        "total_epochs": len(history_records),
        "train_time_sec": total_train_time,
        "device": str(device),
        "params": count_transformer_parameters(model),
        "max_len": max_len,
    }

    return model, tokenizer, summary, history_df, test_loader


@torch.no_grad()
def evaluate_transformer_loader(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Evaluate PubMedBERT on test loader and return true labels, probabilities, and latency."""
    model.eval()
    all_y = []
    all_probs = []

    t0 = time.time()
    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"]

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits.squeeze(-1)
        probs = torch.sigmoid(logits)

        all_y.extend(labels.cpu().numpy().tolist())
        all_probs.extend(probs.cpu().numpy().tolist())

    total_time = time.time() - t0
    latency_ms = (total_time / len(all_y)) * 1000.0 if len(all_y) > 0 else 0.0

    return np.array(all_y), np.array(all_probs), latency_ms
