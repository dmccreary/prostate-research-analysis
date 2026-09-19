"""
transformer_stage3.py - PubMedBERT Dual-Input (Title + Abstract) fine-tuning with class weighting.

Supports:
1. Dual-sequence input: [CLS] Title [SEP] Abstract [SEP] with segment type IDs.
2. Class-weighted BCE loss (pos_weight = 2.03) to penalize false negatives.
3. Early stopping on validation loss/AUROC with best model checkpointing.
4. Inference latency measurement and probability generation.
"""

import copy
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import (
    AutoModel,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

logger = logging.getLogger(__name__)

PRETRAINED_MODEL_NAME = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract"


class DualInputPubMedBERT(nn.Module):
    """
    PubMedBERT for dual-input sequence classification.
    Optionally incorporates dense rule features concatenated to [CLS] embedding.
    """

    def __init__(
        self,
        pretrained_name: str = PRETRAINED_MODEL_NAME,
        rule_dim: int = 0,
        dropout_rate: float = 0.25,
    ):
        super().__init__()
        self.bert = AutoModel.from_pretrained(pretrained_name)
        self.dropout = nn.Dropout(dropout_rate)
        self.rule_dim = rule_dim

        hidden_size = self.bert.config.hidden_size  # 768
        total_dim = hidden_size + rule_dim

        self.classifier = nn.Sequential(
            nn.Linear(total_dim, 128),
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
        rule_vector: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        bert_out = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        # Use [CLS] representation (pooled or first token)
        cls_rep = bert_out.last_hidden_state[:, 0, :]
        cls_rep = self.dropout(cls_rep)

        if self.rule_dim > 0 and rule_vector is not None:
            combined = torch.cat([cls_rep, rule_vector], dim=1)
        else:
            combined = cls_rep

        logits = self.classifier(combined).squeeze(-1)
        return logits


def evaluate_transformer(
    model: nn.Module,
    data_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    Evaluates transformer on DataLoader.
    Returns: (mean_loss, y_true, y_prob)
    """
    model.eval()
    total_loss = 0.0
    all_true, all_prob = [], []

    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            token_type_ids = batch.get("token_type_ids")
            if token_type_ids is not None:
                token_type_ids = token_type_ids.to(device)
            rule_vec = batch.get("rule_vector")
            if rule_vec is not None:
                rule_vec = rule_vec.to(device)
            labels = batch["label"].to(device)

            logits = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                rule_vector=rule_vec,
            )
            loss = criterion(logits, labels)
            total_loss += loss.item() * len(labels)

            probs = torch.sigmoid(logits).cpu().numpy()
            all_prob.extend(probs.tolist())
            all_true.extend(labels.cpu().numpy().tolist())

    mean_loss = total_loss / len(all_true) if len(all_true) > 0 else 0.0
    return mean_loss, np.array(all_true, dtype=int), np.array(all_prob, dtype=float)


def train_dual_pubmedbert(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    output_dir: Path,
    epochs: int = 8,
    lr: float = 2e-5,
    weight_decay: float = 1e-4,
    pos_weight: float = 2.03,
    patience: int = 4,
) -> Tuple[nn.Module, Dict, pd.DataFrame]:
    """
    Trains DualInputPubMedBERT with AdamW, linear warmup, class-weighted BCE, and early stopping.
    """
    model = model.to(device)
    output_dir.mkdir(parents=True, exist_ok=True)
    best_model_path = output_dir / "pubmedbert_stage3_best.pt"

    weight_tensor = torch.tensor([pos_weight], device=device, dtype=torch.float32)
    criterion = nn.BCEWithLogitsLoss(pos_weight=weight_tensor)

    # Optimizer with differential weight decay
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=lr)

    total_steps = len(train_loader) * epochs
    warmup_steps = int(total_steps * 0.1)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

    best_val_loss = float("inf")
    best_val_auc = 0.0
    best_epoch = 0
    best_weights = None
    no_improve_epochs = 0
    history = []

    start_time = time.time()
    logger.info("Starting Stage 3 Dual-Input PubMedBERT Training (%d epochs, pos_weight=%.2f)...", epochs, pos_weight)

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        train_samples = 0

        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            token_type_ids = batch.get("token_type_ids")
            if token_type_ids is not None:
                token_type_ids = token_type_ids.to(device)
            rule_vec = batch.get("rule_vector")
            if rule_vec is not None:
                rule_vec = rule_vec.to(device)
            labels = batch["label"].to(device)

            optimizer.zero_grad()
            logits = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                rule_vector=rule_vec,
            )
            loss = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            train_loss += loss.item() * len(labels)
            train_samples += len(labels)

        mean_train_loss = train_loss / train_samples
        val_loss, y_val_true, y_val_prob = evaluate_transformer(model, val_loader, criterion, device)
        val_auc = roc_auc_score(y_val_true, y_val_prob) if len(set(y_val_true)) > 1 else 0.5
        val_f1 = f1_score(y_val_true, (y_val_prob >= 0.5).astype(int), zero_division=0)

        history.append({
            "epoch": epoch,
            "train_loss": round(mean_train_loss, 4),
            "val_loss": round(val_loss, 4),
            "val_auroc": round(val_auc, 4),
            "val_f1": round(val_f1, 4),
        })

        logger.info(
            "Epoch %d/%d - Train Loss: %.4f | Val Loss: %.4f | Val AUROC: %.4f | Val F1: %.4f",
            epoch, epochs, mean_train_loss, val_loss, val_auc, val_f1
        )

        # Early stopping checkpoint on val_loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_auc = val_auc
            best_epoch = epoch
            best_weights = copy.deepcopy(model.state_dict())
            torch.save(best_weights, best_model_path)
            no_improve_epochs = 0
            logger.info("  -> Saved new best checkpoint at epoch %d (Val Loss: %.4f, AUROC: %.4f)", epoch, val_loss, val_auc)
        else:
            no_improve_epochs += 1
            if no_improve_epochs >= patience:
                logger.info("Early stopping triggered after %d epochs without improvement.", patience)
                break

    training_time = time.time() - start_time
    if best_weights is not None:
        model.load_state_dict(best_weights)
        logger.info("Restored best model weights from epoch %d.", best_epoch)

    history_df = pd.DataFrame(history)
    summary = {
        "train_time_sec": round(training_time, 2),
        "best_epoch": best_epoch,
        "best_val_loss": round(best_val_loss, 4),
        "best_val_auroc": round(best_val_auc, 4),
        "model_path": str(best_model_path),
    }
    return model, summary, history_df
