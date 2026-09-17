"""
trainer.py - PyTorch training, early stopping, and validation tracking for Stage 2.

Key features:
1. Weighted Binary Cross-Entropy (BCEWithLogitsLoss) to address ~2:1 class imbalance.
2. Metric tracking per epoch (Loss, AUROC, Average Precision / PR-AUC, F1-Score).
3. Early stopping based on validation Average Precision (PR-AUC) with configurable patience.
4. Model checkpointing to models/neural/<model_name>.pt.
5. Export of training curves to results/neural/training_history/<model_name>_history.csv.
6. Precise measurement of training time and inference latency (ms/sample).
"""

import copy
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from torch.utils.data import DataLoader

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

logger = logging.getLogger(__name__)


class EarlyStopping:
    """Early stops the training if validation metric doesn't improve after a given patience."""

    def __init__(self, patience: int = 7, mode: str = "max", delta: float = 1e-4):
        self.patience = patience
        self.mode = mode
        self.delta = delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_epoch = 0

    def step(self, score: float, epoch: int) -> bool:
        if self.best_score is None:
            self.best_score = score
            self.best_epoch = epoch
            return True

        if self.mode == "max":
            improved = score > (self.best_score + self.delta)
        else:
            improved = score < (self.best_score - self.delta)

        if improved:
            self.best_score = score
            self.best_epoch = epoch
            self.counter = 0
            return True
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
            return False


def get_device() -> torch.device:
    """Return GPU if available, else CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    return torch.device("cpu")


def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    is_tfidf: bool = False,
) -> float:
    """Train one epoch and return average training loss."""
    model.train()
    total_loss = 0.0
    total_samples = 0

    for batch in train_loader:
        optimizer.zero_grad()

        if is_tfidf:
            x, y = batch
            x, y = x.to(device), y.to(device)
            logits = model(x)
        else:
            x, lengths, y = batch
            x, lengths, y = x.to(device), lengths.to(device), y.to(device)
            logits = model(x, lengths)

        loss = criterion(logits, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        total_loss += loss.item() * len(y)
        total_samples += len(y)

    return total_loss / total_samples if total_samples > 0 else 0.0


@torch.no_grad()
def evaluate_loader(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    is_tfidf: bool = False,
) -> Tuple[float, np.ndarray, np.ndarray]:
    """Evaluate model and return loss, true labels, and predicted probabilities."""
    model.eval()
    total_loss = 0.0
    total_samples = 0
    all_y = []
    all_probs = []

    for batch in loader:
        if is_tfidf:
            x, y = batch
            x, y = x.to(device), y.to(device)
            logits = model(x)
        else:
            x, lengths, y = batch
            x, lengths, y = x.to(device), lengths.to(device), y.to(device)
            logits = model(x, lengths)

        loss = criterion(logits, y)
        probs = torch.sigmoid(logits)

        total_loss += loss.item() * len(y)
        total_samples += len(y)
        all_y.extend(y.cpu().numpy().tolist())
        all_probs.extend(probs.cpu().numpy().tolist())

    avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
    return avg_loss, np.array(all_y), np.array(all_probs)


def train_neural_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    model_name: str,
    output_dir: str = "models/neural",
    history_dir: str = "results/neural/training_history",
    epochs: int = 50,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    pos_weight: float = 2.03,
    patience: int = 7,
    is_tfidf: bool = False,
    device: Optional[torch.device] = None,
) -> Tuple[nn.Module, Dict[str, Any], pd.DataFrame]:
    """
    Main training routine for PyTorch neural models.
    """
    if device is None:
        device = get_device()

    model = model.to(device)
    pos_weight_tensor = torch.tensor([pos_weight], device=device, dtype=torch.float32)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=3
    )
    early_stopping = EarlyStopping(patience=patience, mode="max")

    best_state = copy.deepcopy(model.state_dict())
    best_val_ap = 0.0

    history_records = []
    t0_start = time.time()

    logger.info(
        "Starting training for %s on device=%s (max_epochs=%d, lr=%.4f, pos_weight=%.2f)",
        model_name,
        device,
        epochs,
        lr,
        pos_weight,
    )

    for epoch in range(1, epochs + 1):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device, is_tfidf=is_tfidf)
        val_loss, y_val_true, y_val_prob = evaluate_loader(
            model, val_loader, criterion, device, is_tfidf=is_tfidf
        )

        val_preds = (y_val_prob >= 0.5).astype(int)
        try:
            val_ap = float(average_precision_score(y_val_true, y_val_prob))
            val_auc = float(roc_auc_score(y_val_true, y_val_prob))
        except Exception:
            val_ap = 0.0
            val_auc = 0.5
        val_f1 = float(f1_score(y_val_true, val_preds, zero_division=0))

        scheduler.step(val_ap)

        history_records.append({
            "epoch": epoch,
            "train_loss": round(train_loss, 4),
            "val_loss": round(val_loss, 4),
            "val_ap": round(val_ap, 4),
            "val_auc": round(val_auc, 4),
            "val_f1": round(val_f1, 4),
            "lr": optimizer.param_groups[0]["lr"],
        })

        improved = early_stopping.step(val_ap, epoch)
        if improved:
            best_state = copy.deepcopy(model.state_dict())
            best_val_ap = val_ap

        if epoch % 5 == 0 or improved or epoch == 1:
            logger.info(
                "Epoch %02d/%d: Train Loss=%.4f, Val Loss=%.4f, Val AP=%.4f, Val AUC=%.4f, Val F1=%.4f %s",
                epoch,
                epochs,
                train_loss,
                val_loss,
                val_ap,
                val_auc,
                val_f1,
                "(BEST)" if improved else "",
            )

        if early_stopping.early_stop:
            logger.info(
                "Early stopping triggered at epoch %d. Best epoch was %d with Val AP=%.4f",
                epoch,
                early_stopping.best_epoch,
                early_stopping.best_score,
            )
            break

    total_train_time = time.time() - t0_start

    # Load best weights
    model.load_state_dict(best_state)

    # Save model checkpoint
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = out_dir / f"{model_name}.pt"
    torch.save({
        "model_name": model_name,
        "state_dict": best_state,
        "best_epoch": early_stopping.best_epoch,
        "best_val_ap": best_val_ap,
        "train_time_sec": total_train_time,
    }, checkpoint_path)
    logger.info("Saved best checkpoint to %s", checkpoint_path)

    # Save training history
    hist_dir = Path(history_dir)
    hist_dir.mkdir(parents=True, exist_ok=True)
    history_df = pd.DataFrame(history_records)
    history_df.to_csv(hist_dir / f"{model_name}_history.csv", index=False)

    train_summary = {
        "model_name": model_name,
        "best_epoch": early_stopping.best_epoch,
        "best_val_ap": best_val_ap,
        "total_epochs": len(history_records),
        "train_time_sec": total_train_time,
        "device": str(device),
    }

    return model, train_summary, history_df


@torch.no_grad()
def measure_inference_latency(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device,
    is_tfidf: bool = False,
    num_runs: int = 3,
) -> float:
    """Measure inference latency in milliseconds per abstract."""
    model.eval()
    total_samples = 0
    total_time = 0.0

    # Warmup
    for batch in test_loader:
        if is_tfidf:
            x, _ = batch
            _ = model(x.to(device))
        else:
            x, lengths, _ = batch
            _ = model(x.to(device), lengths.to(device))
        break

    for _ in range(num_runs):
        t0 = time.time()
        for batch in test_loader:
            if is_tfidf:
                x, y = batch
                _ = model(x.to(device))
            else:
                x, lengths, y = batch
                _ = model(x.to(device), lengths.to(device))
            total_samples += len(y)
        total_time += time.time() - t0

    return (total_time / total_samples) * 1000.0 if total_samples > 0 else 0.0
