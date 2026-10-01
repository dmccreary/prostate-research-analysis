"""
run_stage4_multi_seed.py - Robustness Evaluation of Weighted PubMedBERT on labeled-dataset-v2.csv.

Evaluates Stage 4 Cost-Sensitive Weighted PubMedBERT across all 5 independent random splits:
Seeds: [101, 123, 456, 789, 2024]

Dataset: labeled-dataset-v2.csv (N = 524, 281 positive, 243 negative).
Split specifications per seed (64% / 16% / 20% stratified):
- Sub-Train: 335 (180 positive, 155 negative)
- Validation: 84 (45 positive, 39 negative)
- Held-Out Test: 105 (56 positive, 49 negative)

Model evaluated:
- Weighted PubMedBERT (pos_weight = 1.7222, cost-sensitive clinical loss)
"""

import copy
import gc
import json
import logging
import os
from pathlib import Path
import random
import shutil
import sys
import time
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
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
    roc_curve,
)
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup

# Safe stdout on Windows
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Force offline mode for fast HuggingFace cache loading
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"

# Add src to path
BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(BASE_DIR / "src"))
from preprocessing import clean_clinical_text

# Directories
DATA_DIR = BASE_DIR / "data"
SPLITS_DIR = DATA_DIR / "splits" / "stage4_multi_seed"
MODELS_DIR = BASE_DIR / "models" / "stage4_multi_seed"
RESULTS_DIR = BASE_DIR / "results" / "stage4_multi_seed"
REPORTS_DIR = BASE_DIR / "reports" / "stage4_multi_seed"
LOGS_DIR = BASE_DIR / "logs"

for d in [MODELS_DIR, RESULTS_DIR, REPORTS_DIR, LOGS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOGS_DIR / "stage4_multi_seed_experiments.log", encoding="utf-8", mode="a"),
    ],
)
logger = logging.getLogger(__name__)

PRETRAINED_MODEL_NAME = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract"
SEEDS = [101, 123, 456, 789, 2024]
MAX_LENGTH = 384
BATCH_SIZE_TRAIN = 8
BATCH_SIZE_EVAL = 16
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 1e-4
EPOCHS = 8
PATIENCE = 4
DROPOUT_RATE = 0.25


def set_all_seeds(seed: int):
    """Ensure strict determinism."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class DualInputPubMedBERT(nn.Module):
    """
    PubMedBERT for Title + Abstract dual-sequence input:
    [CLS] Title [SEP] Abstract [SEP]
    Architecture:
    PubMedBERT -> CLS -> Dropout(0.25) -> Linear(768, 128) -> LayerNorm -> GELU -> Dropout(0.25) -> Linear(128, 1)
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


def evaluate_model(
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
            input_ids, attention_mask, token_type_ids, labels = [b.to(device) for b in batch]

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


def pretokenize_df(
    df: pd.DataFrame,
    tokenizer,
    max_length: int = MAX_LENGTH,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pretokenizes dataframe text into tensors for ultra-fast GPU training."""
    titles = [clean_clinical_text(str(t or "")) for t in df["title"]]
    abstracts = [clean_clinical_text(str(a or "")) for a in df["abstract"]]
    labels = df["label"].values.astype(float)

    enc = tokenizer(
        text=titles,
        text_pair=abstracts,
        max_length=max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )

    input_ids = enc["input_ids"]
    attention_mask = enc["attention_mask"]
    token_type_ids = enc.get("token_type_ids", torch.zeros_like(input_ids))
    labels_tensor = torch.tensor(labels, dtype=torch.float32)

    return input_ids, attention_mask, token_type_ids, labels_tensor


def create_dataloader(
    tensors: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    batch_size: int,
    shuffle: bool = False,
) -> DataLoader:
    ds = TensorDataset(*tensors)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


def get_existing_run(seed: int, model_type: str, test_df: pd.DataFrame) -> Optional[Tuple[Dict, Dict]]:
    """Checks if a completed run exists and loads its metrics and predictions."""
    pred_path = RESULTS_DIR / f"predictions_{model_type}_seed_{seed}.csv"
    history_path = RESULTS_DIR / f"training_history_{model_type}_seed_{seed}.csv"
    model_path = MODELS_DIR / f"pubmedbert_{model_type}_seed_{seed}.pt"

    if not (pred_path.exists() and history_path.exists() and model_path.exists()):
        return None

    pred_df = pd.read_csv(pred_path)
    if len(pred_df) != len(test_df):
        return None

    history_df = pd.read_csv(history_path)
    best_epoch = int(history_df.loc[history_df["val_loss"].idxmin(), "epoch"]) if "val_loss" in history_df.columns else 8

    y_test_true = pred_df["true_label"].values.astype(int)
    y_test_prob = pred_df["prob"].values.astype(float)
    y_test_pred = pred_df["pred_label"].values.astype(int)

    tn, fp, fn, tp = confusion_matrix(y_test_true, y_test_pred).ravel()
    rec = recall_score(y_test_true, y_test_pred, zero_division=0)
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    acc = accuracy_score(y_test_true, y_test_pred)
    prec = precision_score(y_test_true, y_test_pred, zero_division=0)
    f1 = f1_score(y_test_true, y_test_pred, zero_division=0)
    auroc = roc_auc_score(y_test_true, y_test_prob)
    prec_arr, rec_arr, _ = precision_recall_curve(y_test_true, y_test_prob)
    pr_auc = auc(rec_arr, prec_arr)

    total_test = len(y_test_true)
    workload_reduction = (tn / total_test) * 100.0
    nns = (tp + fp) / tp if tp > 0 else float("inf")

    fp_papers = pred_df[(pred_df["true_label"] == 0) & (pred_df["pred_label"] == 1)][["pmid", "title", "prob"]].to_dict("records")
    fn_papers = pred_df[(pred_df["true_label"] == 1) & (pred_df["pred_label"] == 0)][["pmid", "title", "prob"]].to_dict("records")

    metrics_result = {
        "model_type": model_type,
        "seed": seed,
        "train_size": 335,
        "val_size": 84,
        "test_size": len(test_df),
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
        "recall_sensitivity": round(float(rec), 4),
        "specificity": round(float(spec), 4),
        "accuracy": round(float(acc), 4),
        "precision": round(float(prec), 4),
        "f1_score": round(float(f1), 4),
        "auroc": round(float(auroc), 4),
        "pr_auc": round(float(pr_auc), 4),
        "workload_reduction_pct": round(float(workload_reduction), 2),
        "nns": round(float(nns), 2),
        "best_epoch": int(best_epoch),
        "train_time_sec": 0.0,
    }

    curve_data = {
        "fpr": roc_curve(y_test_true, y_test_prob)[0],
        "tpr": roc_curve(y_test_true, y_test_prob)[1],
        "rec_arr": rec_arr,
        "prec_arr": prec_arr,
        "auroc": auroc,
        "pr_auc": pr_auc,
        "fp_papers": fp_papers,
        "fn_papers": fn_papers,
    }

    logger.info(
        f"[Reused Existing Run] Seed {seed} ({model_type}): Recall={rec*100:.2f}%, Spec={spec*100:.2f}%, "
        f"Acc={acc*100:.2f}%, F1={f1:.4f}, AUROC={auroc:.4f}, PR-AUC={pr_auc:.4f} (TP={tp}, FP={fp}, TN={tn}, FN={fn})"
    )
    return metrics_result, curve_data


def train_single_run(
    seed: int,
    model_type: str,  # "weighted"
    train_tensors,
    val_tensors,
    test_tensors,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    device: torch.device,
) -> Tuple[Dict, Dict]:
    # Check if already completed
    existing = get_existing_run(seed, model_type, test_df)
    if existing is not None:
        return existing

    set_all_seeds(seed)

    loader_train = create_dataloader(train_tensors, batch_size=BATCH_SIZE_TRAIN, shuffle=True)
    loader_val = create_dataloader(val_tensors, batch_size=BATCH_SIZE_EVAL, shuffle=False)
    loader_test = create_dataloader(test_tensors, batch_size=BATCH_SIZE_EVAL, shuffle=False)

    model = DualInputPubMedBERT(pretrained_name=PRETRAINED_MODEL_NAME, dropout_rate=DROPOUT_RATE).to(device)

    y_train = train_df["label"].values.astype(int)
    pos_count = (y_train == 1).sum()
    neg_count = (y_train == 0).sum()

    # Cost-sensitive: (C_FN / C_FP) * (N_neg / N_pos) = 2.0 * (155 / 180) = 1.7222
    pos_weight_val = 2.0 * (neg_count / pos_count) if pos_count > 0 else 1.0
    weight_tensor = torch.tensor([pos_weight_val], device=device, dtype=torch.float32)
    criterion = nn.BCEWithLogitsLoss(pos_weight=weight_tensor)
    tag = f"Weighted (pos_weight={pos_weight_val:.4f})"

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

    total_steps = len(loader_train) * EPOCHS
    warmup_steps = int(total_steps * 0.1)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )

    best_val_loss = float("inf")
    best_val_auc = 0.0
    best_epoch = 0
    best_weights = None
    no_improve_epochs = 0
    history = []

    model_save_path = MODELS_DIR / f"pubmedbert_{model_type}_seed_{seed}.pt"

    start_time = time.time()
    logger.info(f"--- [Seed {seed} | {tag}] Training Started ---")

    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss = 0.0
        train_samples = 0

        for batch in loader_train:
            input_ids, attention_mask, token_type_ids, labels = [b.to(device) for b in batch]

            optimizer.zero_grad()
            logits = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
            )
            loss = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            train_loss += loss.item() * len(labels)
            train_samples += len(labels)

        mean_train_loss = train_loss / train_samples
        val_loss, y_v_true, y_v_prob = evaluate_model(model, loader_val, criterion, device)
        val_auc = roc_auc_score(y_v_true, y_v_prob) if len(set(y_v_true)) > 1 else 0.5
        val_f1 = f1_score(y_v_true, (y_v_prob >= 0.5).astype(int), zero_division=0)

        history.append({
            "epoch": epoch,
            "train_loss": round(mean_train_loss, 4),
            "val_loss": round(val_loss, 4),
            "val_auroc": round(val_auc, 4),
            "val_f1": round(val_f1, 4),
        })

        logger.info(
            f"Seed {seed} ({model_type}) | Epoch {epoch}/{EPOCHS} - Train: {mean_train_loss:.4f} | "
            f"Val: {val_loss:.4f} | Val AUC: {val_auc:.4f} | Val F1: {val_f1:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_auc = val_auc
            best_epoch = epoch
            best_weights = copy.deepcopy(model.state_dict())
            torch.save(best_weights, model_save_path)
            no_improve_epochs = 0
        else:
            no_improve_epochs += 1
            if no_improve_epochs >= PATIENCE:
                logger.info(f"Early stopping triggered for Seed {seed} ({model_type}) at epoch {epoch}")
                break

    training_time = time.time() - start_time

    # Restore best weights
    if best_weights is not None:
        model.load_state_dict(best_weights)

    # Evaluate on Held-Out Test Set
    test_loss, y_test_true, y_test_prob = evaluate_model(model, loader_test, criterion, device)
    y_test_pred = (y_test_prob >= 0.5).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_test_true, y_test_pred).ravel()
    rec = recall_score(y_test_true, y_test_pred, zero_division=0)
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    acc = accuracy_score(y_test_true, y_test_pred)
    prec = precision_score(y_test_true, y_test_pred, zero_division=0)
    f1 = f1_score(y_test_true, y_test_pred, zero_division=0)
    auroc = roc_auc_score(y_test_true, y_test_prob)
    prec_arr, rec_arr, _ = precision_recall_curve(y_test_true, y_test_prob)
    pr_auc = auc(rec_arr, prec_arr)

    # Screening metrics
    total_test = len(y_test_true)
    workload_reduction = (tn / total_test) * 100.0
    nns = (tp + fp) / tp if tp > 0 else float("inf")

    # Track misclassified articles
    pred_df = test_df.copy()
    pred_df["true_label"] = y_test_true
    pred_df["prob"] = y_test_prob
    pred_df["pred_label"] = y_test_pred

    fp_papers = pred_df[(pred_df["true_label"] == 0) & (pred_df["pred_label"] == 1)][["pmid", "title", "prob"]].to_dict("records")
    fn_papers = pred_df[(pred_df["true_label"] == 1) & (pred_df["pred_label"] == 0)][["pmid", "title", "prob"]].to_dict("records")

    # Save predictions and training history
    pred_df.to_csv(RESULTS_DIR / f"predictions_{model_type}_seed_{seed}.csv", index=False)
    pd.DataFrame(history).to_csv(RESULTS_DIR / f"training_history_{model_type}_seed_{seed}.csv", index=False)

    metrics_result = {
        "model_type": model_type,
        "seed": seed,
        "train_size": len(train_df),
        "val_size": len(val_df),
        "test_size": len(test_df),
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
        "recall_sensitivity": round(float(rec), 4),
        "specificity": round(float(spec), 4),
        "accuracy": round(float(acc), 4),
        "precision": round(float(prec), 4),
        "f1_score": round(float(f1), 4),
        "auroc": round(float(auroc), 4),
        "pr_auc": round(float(pr_auc), 4),
        "workload_reduction_pct": round(float(workload_reduction), 2),
        "nns": round(float(nns), 2),
        "best_epoch": int(best_epoch),
        "train_time_sec": round(float(training_time), 1),
    }

    curve_data = {
        "fpr": roc_curve(y_test_true, y_test_prob)[0],
        "tpr": roc_curve(y_test_true, y_test_prob)[1],
        "rec_arr": rec_arr,
        "prec_arr": prec_arr,
        "auroc": auroc,
        "pr_auc": pr_auc,
        "fp_papers": fp_papers,
        "fn_papers": fn_papers,
    }

    logger.info(
        f"Result Seed {seed} ({model_type}): Recall={rec*100:.2f}%, Spec={spec*100:.2f}%, "
        f"Acc={acc*100:.2f}%, F1={f1:.4f}, AUROC={auroc:.4f}, PR-AUC={pr_auc:.4f} (TP={tp}, FP={fp}, TN={tn}, FN={fn})"
    )

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    return metrics_result, curve_data


def run_all_experiments():
    start_all = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"=== Stage 4 5-Seed Robustness Evaluation (Weighted PubMedBERT) ===")
    logger.info(f"Compute Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    logger.info(f"Seeds: {SEEDS}")

    # Load tokenizer once
    logger.info("Loading PubMedBERT Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(PRETRAINED_MODEL_NAME, local_files_only=True)

    # Pre-tokenize splits for all 5 seeds upfront
    logger.info("Pre-tokenizing datasets for all 5 seeds...")
    splits_data = {}
    for seed in SEEDS:
        seed_dir = SPLITS_DIR / f"split_seed_{seed}"
        train_df = pd.read_csv(seed_dir / "train.csv")
        val_df = pd.read_csv(seed_dir / "val.csv")
        test_df = pd.read_csv(seed_dir / "test.csv")

        train_tensors = pretokenize_df(train_df, tokenizer)
        val_tensors = pretokenize_df(val_df, tokenizer)
        test_tensors = pretokenize_df(test_df, tokenizer)

        splits_data[seed] = {
            "train_df": train_df,
            "val_df": val_df,
            "test_df": test_df,
            "train_tensors": train_tensors,
            "val_tensors": val_tensors,
            "test_tensors": test_tensors,
        }
    logger.info("All splits pre-tokenized successfully!")

    results = []
    curves = {}

    for seed in SEEDS:
        data = splits_data[seed]
        res, curve = train_single_run(
            seed=seed,
            model_type="weighted",
            train_tensors=data["train_tensors"],
            val_tensors=data["val_tensors"],
            test_tensors=data["test_tensors"],
            train_df=data["train_df"],
            val_df=data["val_df"],
            test_df=data["test_df"],
            device=device,
        )
        results.append(res)
        curves[seed] = curve

    total_time = time.time() - start_all
    logger.info(f"All 5 seeds evaluated/completed in {total_time/60:.2f} minutes!")

    # Summary calculations
    metric_cols = ["recall_sensitivity", "specificity", "accuracy", "precision", "f1_score", "auroc", "pr_auc", "workload_reduction_pct", "nns"]
    df_res = pd.DataFrame(results)
    means = df_res[metric_cols].mean()
    stds = df_res[metric_cols].std(ddof=1)

    mean_row = {"model_type": "weighted", "seed": "Mean", "train_size": 335, "val_size": 84, "test_size": 105}
    std_row = {"model_type": "weighted", "seed": "Std", "train_size": "—", "val_size": "—", "test_size": "—"}

    for col in metric_cols:
        mean_row[col] = round(means[col], 4)
        std_row[col] = round(stds[col], 4)

    for c in ["tp", "fp", "tn", "fn", "best_epoch", "train_time_sec"]:
        mean_row[c] = round(df_res[c].mean(), 2)
        std_row[c] = round(df_res[c].std(ddof=1), 2)

    full_df = pd.concat([df_res, pd.DataFrame([mean_row]), pd.DataFrame([std_row])], ignore_index=True)
    full_df.to_csv(RESULTS_DIR / "stage4_5seed_weighted_summary.csv", index=False)
    logger.info("Saved stage4_5seed_weighted_summary.csv")

    # Plot ROC and PR Curves for Weighted Model
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
    
    # ROC Curves
    plt.figure(figsize=(8, 6), dpi=300)
    for i, seed in enumerate(SEEDS):
        c = curves[seed]
        plt.plot(c["fpr"], c["tpr"], label=f"Seed {seed} (AUROC = {c['auroc']:.4f})", color=colors[i], linewidth=2)
    plt.plot([0, 1], [0, 1], "k--", alpha=0.6, label="Random Chance (AUROC = 0.500)")
    plt.title(f"Stage 4 Weighted PubMedBERT 5-Seed ROC Curves\n(Mean AUROC = {mean_row['auroc']:.4f} ± {std_row['auroc']:.4f})", fontsize=12, fontweight="bold")
    plt.xlabel("False Positive Rate (1 - Specificity)", fontsize=11)
    plt.ylabel("True Positive Rate (Sensitivity / Recall)", fontsize=11)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(loc="lower right", fontsize=9)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "stage4_5seed_weighted_roc_curves.png")
    plt.close()

    # PR Curves
    plt.figure(figsize=(8, 6), dpi=300)
    for i, seed in enumerate(SEEDS):
        c = curves[seed]
        plt.plot(c["rec_arr"], c["prec_arr"], label=f"Seed {seed} (PR-AUC = {c['pr_auc']:.4f})", color=colors[i], linewidth=2)
    baseline_prev = 56.0 / 105.0
    plt.axhline(y=baseline_prev, color="black", linestyle=":", alpha=0.7, label=f"Test Prevalence ({baseline_prev:.3f})")
    plt.title(f"Stage 4 Weighted PubMedBERT 5-Seed PR Curves\n(Mean PR-AUC = {mean_row['pr_auc']:.4f} ± {std_row['pr_auc']:.4f})", fontsize=12, fontweight="bold")
    plt.xlabel("Recall (Sensitivity)", fontsize=11)
    plt.ylabel("Precision", fontsize=11)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(loc="lower left", fontsize=9)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "stage4_5seed_weighted_pr_curves.png")
    plt.close()

    # Copy plots to brain directory if exists
    brain_fig_dir = Path("C:/Users/mmahd/.gemini/antigravity/brain/97473cd2-23f6-466c-bee7-b37bda871005/figures")
    if brain_fig_dir.exists():
        shutil.copy(RESULTS_DIR / "stage4_5seed_weighted_roc_curves.png", brain_fig_dir / "stage4_5seed_weighted_roc_curves.png")
        shutil.copy(RESULTS_DIR / "stage4_5seed_weighted_pr_curves.png", brain_fig_dir / "stage4_5seed_weighted_pr_curves.png")

    # Generate Markdown Report
    lines = []
    lines.append("# Stage 4 Robustness Evaluation: Cost-Sensitive Weighted PubMedBERT (5 Seeds)")
    lines.append(f"**Date:** {time.strftime('%Y-%m-%d %H:%M:%S')}  ")
    lines.append(f"**Dataset:** `data/labeled-dataset-v2.csv` (N = 524, 281 Pos, 243 Neg)  ")
    lines.append(f"**Architecture:** `DualInputPubMedBERT` (`microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract` + 2-layer MLP head: 768 -> 128 -> 1)  ")
    lines.append(f"**Loss Function:** Cost-Sensitive BCEWithLogitsLoss (pos_weight = 1.7222)  ")
    lines.append(f"**Input Modality:** Strictly Title + Abstract ONLY (`[CLS] Title [SEP] Abstract [SEP]`)  ")
    lines.append(f"**Hardware Device:** {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}  ")
    lines.append(f"**Total Execution Time:** {total_time/60:.2f} minutes across all 5 seeds.  ")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## 1. Weighted PubMedBERT Results Across All 5 Random Seeds")
    lines.append("")
    lines.append("| Split | Seed | Train $N$ | Val $N$ | Test $N$ | TP | FP | TN | FN | Recall | Specificity | Accuracy | Precision | F1-Score | AUROC | PR-AUC | Workload Red. % | NNS |")
    lines.append("| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")
    for r in results:
        lines.append(
            f"| Split | {r['seed']} | {r['train_size']} | {r['val_size']} | {r['test_size']} | "
            f"{r['tp']} | {r['fp']} | {r['tn']} | {r['fn']} | "
            f"{r['recall_sensitivity']*100:.2f}% | {r['specificity']*100:.2f}% | {r['accuracy']*100:.2f}% | "
            f"{r['precision']*100:.2f}% | {r['f1_score']:.4f} | {r['auroc']:.4f} | {r['pr_auc']:.4f} | "
            f"{r['workload_reduction_pct']:.2f}% | {r['nns']:.2f} |"
        )
    lines.append(
        f"| **Mean** | — | 335 | 84 | 105 | "
        f"{mean_row['tp']:.1f} | {mean_row['fp']:.1f} | {mean_row['tn']:.1f} | {mean_row['fn']:.1f} | "
        f"**{mean_row['recall_sensitivity']*100:.2f}%** | **{mean_row['specificity']*100:.2f}%** | **{mean_row['accuracy']*100:.2f}%** | "
        f"**{mean_row['precision']*100:.2f}%** | **{mean_row['f1_score']:.4f}** | **{mean_row['auroc']:.4f}** | **{mean_row['pr_auc']:.4f}** | "
        f"**{mean_row['workload_reduction_pct']:.2f}%** | **{mean_row['nns']:.2f}** |"
    )
    lines.append(
        f"| **Std (±)** | — | — | — | — | "
        f"±{std_row['tp']:.1f} | ±{std_row['fp']:.1f} | ±{std_row['tn']:.1f} | ±{std_row['fn']:.1f} | "
        f"**±{std_row['recall_sensitivity']*100:.2f}%** | **±{std_row['specificity']*100:.2f}%** | **±{std_row['accuracy']*100:.2f}%** | "
        f"**±{std_row['precision']*100:.2f}%** | **±{std_row['f1_score']:.4f}** | **±{std_row['auroc']:.4f}** | **±{std_row['pr_auc']:.4f}** | "
        f"**±{std_row['workload_reduction_pct']:.2f}%** | **±{std_row['nns']:.2f}** |"
    )
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## 2. Misclassification and Error Analysis Across All 5 Seeds")
    lines.append("")
    for seed in SEEDS:
        c = curves[seed]
        lines.append(f"### Seed {seed}")
        if len(c["fn_papers"]) == 0 and len(c["fp_papers"]) == 0:
            lines.append("- **Perfect Classification:** 0 False Positives, 0 False Negatives (100% Sensitivity, 100% Specificity).")
        else:
            if len(c["fn_papers"]) > 0:
                lines.append(f"- **False Negatives ({len(c['fn_papers'])} missed):**")
                for p in c["fn_papers"]:
                    lines.append(f"  - PMID {p['pmid']}: *{p['title']}* (prob = {p['prob']:.4f})")
            else:
                lines.append("- **False Negatives:** 0 missed papers (100% Recall).")
            if len(c["fp_papers"]) > 0:
                lines.append(f"- **False Positives ({len(c['fp_papers'])} flagged):**")
                for p in c["fp_papers"]:
                    lines.append(f"  - PMID {p['pmid']}: *{p['title']}* (prob = {p['prob']:.4f})")
            else:
                lines.append("- **False Positives:** 0 false alarms (100% Specificity).")
        lines.append("")

    with open(REPORTS_DIR / "stage4_5seed_weighted_report.md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    logger.info("Report written successfully to reports/stage4_multi_seed/stage4_5seed_weighted_report.md")


if __name__ == "__main__":
    run_all_experiments()
