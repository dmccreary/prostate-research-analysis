"""
src/stage8/train_stage8_2_1_asl_medcpt.py - Stage 8.2.1: Fine-Tuning MedCPT Cross-Encoder with Asymmetric Loss (ASL).

Evaluates the Standalone MedCPT Cross-Encoder (rule_dim = 0) with one loss configuration:
1. Model 8.2.1: ASL Standalone, Unweighted (pos_weight = 1.0)

Differences from Stage 8.2:
- Changes only the criteria text from the detailed Stage 8.2 eligibility wording to the supplied query-style description suited to MedCPT.
- Preserves the unweighted ASL configuration (pos_weight = 1.0), model architecture, tokenizer settings, training hyperparameters, data splits, evaluation procedure, and zero-leakage policy.
- Uses a maximum of 8 epochs and the Stage 7.4 Early Stopping policy: strictly lowest Validation Loss selects the checkpoint and triggers patience counting.
- Sets patience = 4 consecutive epochs without a lower Validation Loss.
- AUROC remains a reported validation metric but does not control checkpoint selection or Early Stopping.
- Uses separate Stage 8.2.1 output directories and filenames so Stage 8.2 and other experiment results are not overwritten.

Key Architectural & Optimization Specifications:
- Backbone: ncbi/MedCPT-Cross-Encoder (Initialized from 255M PubMed search logs)
- Sequence Budget: 512 tokens max length
- Input Pair: Segment A = Query-style eligibility criteria tailored to MedCPT (new wording in Stage 8.2.1), Segment B = Title + Abstract
- Anti-Catastrophic Forgetting Strategy:
  * Two-Stage Gradual Unfreezing:
    - Epoch 1: Freeze BERT backbone (requires_grad = False), train classifier head only (LR = 2e-5)
    - Epoch 2+: Unfreeze full model with differential LR (BERT = 5e-6, Classifier = 2e-5)
  * Linear Warmup over 10% of training steps
  * Conservative weight decay = 1e-4, max gradient clipping = 1.0
- Loss Function: Asymmetric Loss (ASL: gamma_neg = 2.0, gamma_pos = 0.0, margin = 0.05)
- Strict Zero-Leakage:
  * Trained ONLY on Sub-Train set (train.csv, N=335: 180 Pos, 155 Neg)
  * Evaluated STRICTLY on Validation set (val.csv, N=84: 45 Pos, 39 Neg)
  * Held-Out Test Set (N=105) and 2021 Cohort (N=117) remain completely UNTOUCHED
"""

import copy
import gc
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
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

# Safe stdout on Windows
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent.parent
SPLITS_DIR = BASE_DIR / "data" / "splits" / "stage5"
MODELS_DIR = BASE_DIR / "models" / "stage8_2_1"
RESULTS_DIR = BASE_DIR / "results" / "stage8_2_1"
LOGS_DIR = BASE_DIR / "logs"

for d in [MODELS_DIR, RESULTS_DIR, LOGS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOGS_DIR / "stage8_2_1_medcpt_finetuning.log", encoding="utf-8", mode="w"),
    ],
)
logger = logging.getLogger(__name__)

sys.path.insert(0, str(BASE_DIR / "src"))
from preprocessing import clean_clinical_text

# Constants
PRETRAINED_MODEL_NAME = "ncbi/MedCPT-Cross-Encoder"
SEED = 2025
MAX_LENGTH = 512
BATCH_SIZE_TRAIN = 4
GRAD_ACCUM_STEPS = 2
BATCH_SIZE_EVAL = 8
LR_BACKBONE = 5e-6
LR_HEAD = 2e-5
WEIGHT_DECAY = 1e-4
EPOCHS = 8
PATIENCE = 4
MAX_GRAD_NORM = 1.0

# Query-style eligibility criteria tailored to MedCPT Cross-Encoder retrieval/reranking
PICO_QUERY_MEDCPT = (
    "Studies of prostate cancer patients stratified into low-, intermediate-, and high-risk groups "
    "using D’Amico, Zelefsky, or NCCN criteria, evaluating radical prostatectomy, external beam "
    "radiotherapy, brachytherapy, proton therapy, androgen deprivation therapy combinations, "
    "cryotherapy, or HIFU. Studies report long-term biochemical recurrence-free, overall, "
    "metastasis-free, or cancer-specific survival, retain all patients regardless of pathological "
    "findings, use modern radiation doses, include adequate risk-group-specific cohorts, and have "
    "at least five years’ median follow-up."
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
    Asymmetric Loss (ASL) with Positive Cost-Sensitive Weighting for High-Recall Screening.
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

        # Positive loss (logsigmoid prevents log(0) underflow)
        log_pos = F.logsigmoid(logits)
        pos_modulating = (1.0 - probs) ** self.gamma_pos if self.gamma_pos > 0 else 1.0
        loss_pos = - self.pos_weight * pos_modulating * log_pos

        # Negative loss with margin truncation
        p_m = torch.clamp(probs - self.margin, min=0.0, max=1.0)
        neg_modulating = p_m ** self.gamma_neg
        log_neg = torch.log(torch.clamp(1.0 - p_m, min=self.eps, max=1.0))
        loss_neg = - neg_modulating * log_neg

        total_loss = targets * loss_pos + (1.0 - targets) * loss_neg
        return total_loss.mean()


class PICODataset(Dataset):
    def __init__(
        self,
        titles: List[str],
        abstracts: List[str],
        labels: List[int],
        tokenizer: AutoTokenizer,
        pico_query: str = PICO_QUERY_MEDCPT,
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


def evaluate_loader(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, np.ndarray, np.ndarray]:
    model.eval()
    total_loss = 0.0
    all_true, all_prob = [], []

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            token_type_ids = batch["token_type_ids"].to(device)
            labels = batch["label"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
            )
            logits = outputs.logits.squeeze(-1)
            loss = criterion(logits, labels)

            total_loss += loss.item() * len(labels)
            probs = torch.sigmoid(logits).cpu().numpy().tolist()
            if isinstance(probs, float):
                probs = [probs]
            all_prob.extend(probs)
            all_true.extend(labels.cpu().numpy().tolist())

    avg_loss = total_loss / len(loader.dataset)
    return avg_loss, np.array(all_true), np.array(all_prob)


def sweep_thresholds(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    thresholds: np.ndarray = np.linspace(0.01, 0.99, 99),
) -> pd.DataFrame:
    rows = []
    total = len(y_true)
    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        flagged = int(tp + fp)

        acc = accuracy_score(y_true, y_pred)
        rec = recall_score(y_true, y_pred, zero_division=0)
        prec = precision_score(y_true, y_pred, zero_division=0)
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        f1 = f1_score(y_true, y_pred, zero_division=0)
        nns = round((flagged / tp), 2) if tp > 0 else float("inf")
        workload_pct = round((flagged / total) * 100.0, 2)
        reduction_pct = round(((total - flagged) / total) * 100.0, 2)

        rows.append({
            "threshold": round(float(t), 2),
            "accuracy": float(acc),
            "recall": float(rec),
            "specificity": float(spec),
            "precision": float(prec),
            "f1_score": float(f1),
            "true_positives": int(tp),
            "false_positives": int(fp),
            "true_negatives": int(tn),
            "false_negatives": int(fn),
            "flagged_count": flagged,
            "workload_pct": workload_pct,
            "workload_reduction_pct": reduction_pct,
            "nns": nns,
        })
    return pd.DataFrame(rows)


def extract_operating_point_metrics(
    df_sweep: pd.DataFrame,
    y_true: np.ndarray,
    y_prob: np.ndarray,
    model_name: str,
    model_id: str,
) -> List[Dict]:
    try:
        auroc = float(roc_auc_score(y_true, y_prob))
    except Exception:
        auroc = 0.5

    try:
        p_c, r_c, _ = precision_recall_curve(y_true, y_prob)
        pr_auc = float(auc(r_c, p_c))
    except Exception:
        pr_auc = float(np.mean(y_true))

    operating_points = []

    # 1. Default (0.50)
    row_default = df_sweep.loc[(df_sweep["threshold"] - 0.50).abs().idxmin()].to_dict()
    row_default["operating_point"] = "Default (0.50)"
    operating_points.append(row_default)

    # 2. Recall >= 90%
    cands_90 = df_sweep[df_sweep["recall"] >= 0.90]
    if not cands_90.empty:
        idx = cands_90["specificity"].idxmax()
        row_90 = df_sweep.loc[idx].to_dict()
    else:
        row_90 = df_sweep.loc[df_sweep["recall"].idxmax()].to_dict()
    row_90["operating_point"] = "Recall >= 90%"
    operating_points.append(row_90)

    # 3. Recall >= 95%
    cands_95 = df_sweep[df_sweep["recall"] >= 0.95]
    if not cands_95.empty:
        idx = cands_95["specificity"].idxmax()
        row_95 = df_sweep.loc[idx].to_dict()
    else:
        row_95 = df_sweep.loc[df_sweep["recall"].idxmax()].to_dict()
    row_95["operating_point"] = "Recall >= 95%"
    operating_points.append(row_95)

    # 4. Recall = 100%
    cands_100 = df_sweep[df_sweep["recall"] >= 1.00]
    if not cands_100.empty:
        idx = cands_100["specificity"].idxmax()
        row_100 = df_sweep.loc[idx].to_dict()
    else:
        row_100 = df_sweep.loc[df_sweep["recall"].idxmax()].to_dict()
    row_100["operating_point"] = "Recall = 100%"
    operating_points.append(row_100)

    res = []
    for r in operating_points:
        res.append({
            "model_name": model_name,
            "model_id": model_id,
            "operating_point": r["operating_point"],
            "threshold": r["threshold"],
            "accuracy": r["accuracy"],
            "recall": r["recall"],
            "specificity": r["specificity"],
            "precision": r["precision"],
            "f1_score": r["f1_score"],
            "auroc": auroc,
            "pr_auc": pr_auc,
            "true_positives": r["true_positives"],
            "false_positives": r["false_positives"],
            "true_negatives": r["true_negatives"],
            "false_negatives": r["false_negatives"],
            "flagged_count": r["flagged_count"],
            "workload_pct": r["workload_pct"],
            "workload_reduction_pct": r["workload_reduction_pct"],
            "nns": r["nns"],
        })
    return res


def train_single_model(
    model_id: str,
    model_name: str,
    pos_weight: float,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
) -> Tuple[nn.Module, Dict]:
    logger.info("=" * 80)
    logger.info(f"TRAINING [{model_id}]: {model_name} (pos_weight={pos_weight:.4f})")
    logger.info("=" * 80)

    set_all_seeds(SEED)
    model = AutoModelForSequenceClassification.from_pretrained(PRETRAINED_MODEL_NAME)
    model.to(device)

    criterion = AsymmetricLossWithCostWeighting(
        gamma_pos=0.0,
        gamma_neg=2.0,
        margin=0.05,
        pos_weight=pos_weight,
    )

    # Differential Optimizer Setup:
    # Separate parameter groups for backbone and classifier head
    # No decay on bias and LayerNorm
    no_decay = ["bias", "LayerNorm.weight", "layer_norm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p for n, p in model.bert.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            "lr": LR_BACKBONE,
            "weight_decay": WEIGHT_DECAY,
        },
        {
            "params": [
                p for n, p in model.bert.named_parameters()
                if any(nd in n for nd in no_decay)
            ],
            "lr": LR_BACKBONE,
            "weight_decay": 0.0,
        },
        {
            "params": [p for n, p in model.classifier.named_parameters()],
            "lr": LR_HEAD,
            "weight_decay": WEIGHT_DECAY,
        },
    ]

    optimizer = AdamW(optimizer_grouped_parameters)

    total_update_steps = (len(train_loader) // GRAD_ACCUM_STEPS) * EPOCHS
    warmup_steps = int(0.10 * total_update_steps)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_update_steps,
    )

    best_val_loss = float("inf")
    best_model_weights = None
    best_epoch = -1
    no_improve_epochs = 0
    history = []

    start_time = time.time()

    for epoch in range(1, EPOCHS + 1):
        # Two-Stage Gradual Unfreezing:
        # In Epoch 1, freeze BERT backbone completely to stabilize classifier head
        if epoch == 1:
            logger.info("  [Warmup Stage 1] Freezing BERT backbone for Epoch 1 (Head-only adaptation)...")
            for p in model.bert.parameters():
                p.requires_grad = False
        elif epoch == 2:
            logger.info("  [Stage 2] Unfreezing all BERT layers with conservative LR (5e-6)...")
            for p in model.bert.parameters():
                p.requires_grad = True

        model.train()
        train_loss = 0.0
        optimizer.zero_grad()

        for step, batch in enumerate(train_loader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            token_type_ids = batch["token_type_ids"].to(device)
            labels = batch["label"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
            )
            logits = outputs.logits.squeeze(-1)
            loss = criterion(logits, labels)
            loss_scaled = loss / GRAD_ACCUM_STEPS
            loss_scaled.backward()

            train_loss += loss.item() * len(labels)

            if (step + 1) % GRAD_ACCUM_STEPS == 0 or (step + 1) == len(train_loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

        avg_train_loss = train_loss / len(train_loader.dataset)

        # Validation evaluation
        val_loss, y_val_true, y_val_prob = evaluate_loader(model, val_loader, criterion, device)
        y_val_pred_def = (y_val_prob >= 0.50).astype(int)
        val_rec = recall_score(y_val_true, y_val_pred_def, zero_division=0)
        val_spec = (
            confusion_matrix(y_val_true, y_val_pred_def, labels=[0, 1]).ravel()[0]
            / (y_val_true == 0).sum()
        )
        try:
            val_auroc = roc_auc_score(y_val_true, y_val_prob)
        except Exception:
            val_auroc = 0.5

        history.append({
            "epoch": epoch,
            "train_loss": avg_train_loss,
            "val_loss": val_loss,
            "val_recall_def": val_rec,
            "val_spec_def": val_spec,
            "val_auroc": val_auroc,
        })

        logger.info(
            f"[{model_id}] Epoch {epoch:02d}/{EPOCHS:02d} | "
            f"Train Loss: {avg_train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Recall: {val_rec:.4f} | "
            f"Val Spec: {val_spec:.4f} | "
            f"Val AUROC: {val_auroc:.4f}"
        )

        # Selection and Early Stopping criterion: strictly lowest Validation Loss (same policy as Stage 7.4 and Stage 8.2)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            best_model_weights = copy.deepcopy(model.state_dict())
            no_improve_epochs = 0
            logger.info(f"  --> Saved new best checkpoint at epoch {epoch:02d} (Val Loss: {val_loss:.4f})")
        else:
            no_improve_epochs += 1
            if no_improve_epochs >= PATIENCE:
                logger.info(f"[{model_id}] Early stopping triggered at epoch {epoch:02d} after {PATIENCE} consecutive epochs without lower Val Loss.")
                break

    train_time = time.time() - start_time
    logger.info(f"[{model_id}] Training finished in {train_time:.1f}s. Best Epoch: {best_epoch:02d} (Val Loss: {best_val_loss:.4f})")

    # Load best checkpoint
    model.load_state_dict(best_model_weights)
    checkpoint_path = MODELS_DIR / f"{model_id}_best.pt"
    torch.save(model.state_dict(), checkpoint_path)
    logger.info(f"[{model_id}] Checkpoint written to {checkpoint_path}")

    # Final evaluation on validation set
    _, y_val_true, y_val_prob_best = evaluate_loader(model, val_loader, criterion, device)
    df_sweep = sweep_thresholds(y_val_true, y_val_prob_best)
    op_metrics = extract_operating_point_metrics(df_sweep, y_val_true, y_val_prob_best, model_name, model_id)

    return model, {
        "model_id": model_id,
        "model_name": model_name,
        "best_epoch": best_epoch,
        "train_time_sec": train_time,
        "history": history,
        "sweep_df": df_sweep,
        "op_metrics": op_metrics,
        "val_probs": y_val_prob_best,
        "val_labels": y_val_true,
    }


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Execution Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    # Load Datasets
    train_df = pd.read_csv(SPLITS_DIR / "train.csv")
    val_df = pd.read_csv(SPLITS_DIR / "val.csv")

    for df in [train_df, val_df]:
        df["clean_title"] = [clean_clinical_text(str(t or "")) for t in df["title"]]
        df["clean_abstract"] = [clean_clinical_text(str(a or "")) for a in df["abstract"]]

    y_train = train_df["label"].to_numpy().astype(int)
    y_val = val_df["label"].to_numpy().astype(int)

    logger.info(f"Loaded Splits: Train={len(train_df)} (Pos={y_train.sum()}, Neg={(y_train==0).sum()}), Val={len(val_df)} (Pos={y_val.sum()}, Neg={(y_val==0).sum()})")
    logger.info("Held-out Test set and 2021 prospective cohort are STRICTLY EXCLUDED.")

    tokenizer = AutoTokenizer.from_pretrained(PRETRAINED_MODEL_NAME)

    train_ds = PICODataset(train_df["clean_title"].tolist(), train_df["clean_abstract"].tolist(), y_train.tolist(), tokenizer)
    val_ds = PICODataset(val_df["clean_title"].tolist(), val_df["clean_abstract"].tolist(), y_val.tolist(), tokenizer)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE_TRAIN, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE_EVAL, shuffle=False)

    configs = [
        {
            "model_id": "model_8_2_1_standalone_unweighted",
            "model_name": "MedCPT Standalone ASL (Unweighted)",
            "pos_weight": 1.0,
        },
    ]

    all_op_metrics = []
    all_sweeps = []
    all_histories = []
    val_predictions_df = val_df[["pmid", "title", "label"]].copy()

    for cfg in configs:
        _, res = train_single_model(
            model_id=cfg["model_id"],
            model_name=cfg["model_name"],
            pos_weight=cfg["pos_weight"],
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
        )

        all_op_metrics.extend(res["op_metrics"])

        # Collect sweep
        df_sw = res["sweep_df"].copy()
        df_sw["model_id"] = cfg["model_id"]
        df_sw["model_name"] = cfg["model_name"]
        all_sweeps.append(df_sw)

        # Collect history
        df_hist = pd.DataFrame(res["history"])
        df_hist["model_id"] = cfg["model_id"]
        df_hist["model_name"] = cfg["model_name"]
        all_histories.append(df_hist)

        # Save predictions
        val_predictions_df[f"prob_{cfg['model_id']}"] = res["val_probs"]

        # Clean memory
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Save CSV outputs
    df_metrics = pd.DataFrame(all_op_metrics)
    df_metrics.to_csv(RESULTS_DIR / "stage8_2_1_validation_metrics.csv", index=False)

    df_sweeps_all = pd.concat(all_sweeps, ignore_index=True)
    df_sweeps_all.to_csv(RESULTS_DIR / "stage8_2_1_validation_threshold_sweeps.csv", index=False)

    df_hist_all = pd.concat(all_histories, ignore_index=True)
    df_hist_all.to_csv(RESULTS_DIR / "stage8_2_1_training_histories.csv", index=False)

    val_predictions_df.to_csv(RESULTS_DIR / "stage8_2_1_validation_predictions.csv", index=False)

    logger.info(f"All Stage 8.2.1 results saved to {RESULTS_DIR}")

    # Display Printout
    print("\n" + "=" * 115)
    print("STAGE 8.2.1 MEDCPT CROSS-ENCODER VALIDATION BENCHMARK SUMMARY (N = 84)")
    print("=" * 115)
    for m in all_op_metrics:
        print(
            f"{m['model_name']:<35} | {m['operating_point']:<18} | "
            f"Thresh: {m['threshold']:<5.2f} | "
            f"Rec: {m['recall']*100:>5.1f}% | "
            f"Spec: {m['specificity']*100:>5.1f}% | "
            f"AUROC: {m['auroc']:<6.4f} | "
            f"PR-AUC: {m['pr_auc']:<6.4f} | "
            f"F1: {m['f1_score']:<6.4f} | "
            f"TP: {m['true_positives']:>2} FP: {m['false_positives']:>2} TN: {m['true_negatives']:>2} FN: {m['false_negatives']:>2} | "
            f"Workload Red: {m['workload_reduction_pct']:>5.1f}%"
        )
    print("=" * 115 + "\n")


if __name__ == "__main__":
    main()
