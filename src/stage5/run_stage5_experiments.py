"""
run_stage5_experiments.py - Master Training and Evaluation Pipeline for Stage 5.

Evaluates 5 distinct model conditions on the newly split data/labeled-dataset-v2.csv (API-refetched):
1. PubMedBERT abstract only, weighted, using deterministic filtering (Cascaded)
2. PubMedBERT abstract only, weighted, without using deterministic filtering (Standalone)
3. PubMedBERT abstract + title, weighted, using deterministic filtering (Cascaded)
4. PubMedBERT abstract + title, weighted, without using deterministic filtering (Standalone)
5. PubMedBERT abstract + title, weighted, using 7 Clinical Indicators vector (Hybrid)

Hardware: NVIDIA GeForce GTX 1050 Ti
Loss: Cost-Sensitive BCEWithLogitsLoss(pos_weight=1.7222)
Split: Seed 2025 (Sub-train N=335, Val N=84, Test N=105)
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

# Force offline mode for fast HuggingFace cache loading
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"

# Add src to path
BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(BASE_DIR / "src"))
from preprocessing import clean_clinical_text
from stage3.rule_engine import evaluate_pre_filter, get_rule_feature_vector

# Directories
DATA_DIR = BASE_DIR / "data"
SPLITS_DIR = DATA_DIR / "splits" / "stage5"
MODELS_DIR = BASE_DIR / "models" / "stage5"
RESULTS_DIR = BASE_DIR / "results" / "stage5"
PLOTS_DIR = RESULTS_DIR / "plots"
REPORTS_DIR = BASE_DIR / "reports" / "stage5"
LOGS_DIR = BASE_DIR / "logs"

for d in [MODELS_DIR, RESULTS_DIR, PLOTS_DIR, REPORTS_DIR, LOGS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOGS_DIR / "stage5_experiments.log", encoding="utf-8", mode="w"),
    ],
)
logger = logging.getLogger(__name__)

PRETRAINED_MODEL_NAME = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract"
SEED = 2025
MAX_LENGTH = 384
BATCH_SIZE_TRAIN = 8
BATCH_SIZE_EVAL = 16
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 1e-4
EPOCHS = 8
PATIENCE = 4
DROPOUT_RATE = 0.25


def set_all_seeds(seed: int = SEED):
    """Ensure strict determinism."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class Stage5Dataset(Dataset):
    """Universal Dataset for Abstract-Only, Title+Abstract, and Hybrid Rules."""

    def __init__(
        self,
        titles: List[str],
        abstracts: List[str],
        labels: List[int],
        rule_vectors: Optional[List[np.ndarray]],
        tokenizer: AutoTokenizer,
        mode: str = "title_abstract",  # "abstract_only" or "title_abstract"
        max_length: int = MAX_LENGTH,
    ):
        self.labels = torch.tensor(labels, dtype=torch.float32)
        self.rule_vectors = rule_vectors
        self.mode = mode

        if mode == "abstract_only":
            enc = tokenizer(
                abstracts,
                padding="max_length",
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
        else:
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
        item = {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
            "token_type_ids": self.token_type_ids[idx],
            "label": self.labels[idx],
        }
        if self.rule_vectors is not None:
            item["rule_vector"] = torch.tensor(self.rule_vectors[idx], dtype=torch.float32)
        return item


class FlexiblePubMedBERT(nn.Module):
    """
    PubMedBERT supporting:
    - rule_dim = 0 (Pure text: Abstract-Only or Title+Abstract)
    - rule_dim = 7 (Hybrid text + 7 Clinical Indicators vector)
    """

    def __init__(
        self,
        pretrained_name: str = PRETRAINED_MODEL_NAME,
        rule_dim: int = 0,
        dropout_rate: float = DROPOUT_RATE,
    ):
        super().__init__()
        self.bert = AutoModel.from_pretrained(pretrained_name, local_files_only=True)
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
        cls_rep = bert_out.last_hidden_state[:, 0, :]
        cls_rep = self.dropout(cls_rep)

        if self.rule_dim > 0 and rule_vector is not None:
            combined = torch.cat([cls_rep, rule_vector], dim=1)
        else:
            combined = cls_rep

        logits = self.classifier(combined).squeeze(-1)
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

            rule_vector = batch.get("rule_vector")
            if rule_vector is not None:
                rule_vector = rule_vector.to(device)

            logits = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                rule_vector=rule_vector,
            )
            loss = criterion(logits, labels)
            total_loss += loss.item() * len(labels)

            probs = torch.sigmoid(logits).cpu().numpy()
            all_prob.extend(probs.tolist())
            all_true.extend(labels.cpu().numpy().tolist())

    mean_loss = total_loss / len(all_true) if len(all_true) > 0 else 0.0
    return mean_loss, np.array(all_true, dtype=int), np.array(all_prob, dtype=float)


def train_model(
    model_name: str,
    rule_dim: int,
    pos_weight_val: float,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    save_path: Path,
) -> Tuple[FlexiblePubMedBERT, pd.DataFrame]:
    """Trains model with early stopping on validation loss."""
    set_all_seeds(SEED)
    model = FlexiblePubMedBERT(rule_dim=rule_dim).to(device)

    weight_tensor = torch.tensor([pos_weight_val], device=device, dtype=torch.float32)
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

    total_steps = len(train_loader) * EPOCHS
    warmup_steps = int(total_steps * 0.1)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )

    best_val_loss = float("inf")
    best_weights = None
    no_improve_epochs = 0
    history = []

    logger.info(f"[{model_name}] Starting training: {EPOCHS} epochs, pos_weight={pos_weight_val:.4f}, rule_dim={rule_dim}...")

    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss = 0.0
        train_samples = 0

        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            token_type_ids = batch.get("token_type_ids")
            if token_type_ids is not None:
                token_type_ids = token_type_ids.to(device)
            labels = batch["label"].to(device)

            rule_vector = batch.get("rule_vector")
            if rule_vector is not None:
                rule_vector = rule_vector.to(device)

            optimizer.zero_grad()
            logits = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                rule_vector=rule_vector,
            )
            loss = criterion(logits, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

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

        history.append({
            "model_name": model_name,
            "epoch": epoch,
            "train_loss": train_mean_loss,
            "val_loss": val_mean_loss,
            "val_auroc": val_auroc,
            "val_prauc": val_prauc,
            "val_f1": val_f1,
        })

        logger.info(
            f"[{model_name}] Epoch {epoch}/{EPOCHS} - "
            f"Train Loss: {train_mean_loss:.4f} | "
            f"Val Loss: {val_mean_loss:.4f} | "
            f"Val AUROC: {val_auroc:.4f} | "
            f"Val PR-AUC: {val_prauc:.4f} | "
            f"Val F1: {val_f1:.4f}"
        )

        if val_mean_loss < best_val_loss - 1e-4:
            best_val_loss = val_mean_loss
            best_weights = copy.deepcopy(model.state_dict())
            no_improve_epochs = 0
            torch.save(best_weights, save_path)
            logger.info(f"[{model_name}] Checkpoint saved at epoch {epoch} (Val Loss: {val_mean_loss:.4f})")
        else:
            no_improve_epochs += 1
            if no_improve_epochs >= PATIENCE:
                logger.info(f"[{model_name}] Early stopping triggered at epoch {epoch}")
                break

    # Restore best checkpoint
    if best_weights is not None:
        model.load_state_dict(best_weights)
    elif save_path.exists():
        model.load_state_dict(torch.load(save_path, map_location=device))

    return model, pd.DataFrame(history)


def compute_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
    condition_name: str = "",
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
        "condition_name": condition_name,
        "threshold": threshold,
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


def sweep_thresholds(
    y_val: np.ndarray,
    val_probs: np.ndarray,
    y_test: np.ndarray,
    test_probs: np.ndarray,
    condition_name: str,
    target_sensitivities: List[float] = [0.90, 0.95, 1.00],
) -> List[Dict]:
    """Sweeps thresholds on validation set to find operating points and tests them on test set."""
    results = []
    threshold_grid = np.linspace(0.01, 0.99, 99)

    for target in target_sensitivities:
        best_t = 0.50
        found = False
        valid_candidates = []

        for t in threshold_grid:
            val_preds = (val_probs >= t).astype(int)
            val_rec = recall_score(y_val, val_preds, zero_division=0)
            if val_rec >= target:
                valid_candidates.append(t)

        if valid_candidates:
            # Pick highest threshold that still meets target recall
            best_t = max(valid_candidates)
            found = True
        else:
            best_t = 0.01

        m = compute_metrics(y_test, test_probs, threshold=best_t, condition_name=condition_name)
        m["target_sensitivity"] = target
        m["val_selected_threshold"] = round(best_t, 3)
        results.append(m)

    return results


def main():
    logger.info("================================================================================")
    logger.info("                 STARTING STAGE 5 MASTER BENCHMARK PIPELINE                     ")
    logger.info("  Retraining on Updated API-Refetched V2 Dataset Across All 5 Conditions        ")
    logger.info("================================================================================")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Execution Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    # 1. Load Stage 5 Splits
    train_path = SPLITS_DIR / "train.csv"
    val_path = SPLITS_DIR / "val.csv"
    test_path = SPLITS_DIR / "test.csv"

    if not train_path.exists() or not val_path.exists() or not test_path.exists():
        raise FileNotFoundError(f"Missing Stage 5 splits in {SPLITS_DIR}. Please run split_stage5.py first.")

    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    test_df = pd.read_csv(test_path)

    logger.info(f"Loaded Splits: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")

    # 2. Enrich text & clinical rule vectors
    def enrich_data(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["clean_title"] = [clean_clinical_text(str(t or "")) for t in df["title"]]
        df["clean_abstract"] = [clean_clinical_text(str(a or "")) for a in df["abstract"]]

        pub_types_list, is_ex_list, reasons_list, rule_vecs = [], [], [], []
        for _, row in df.iterrows():
            pts = [t.strip() for t in str(row.get("publication_types", "")).split(";") if t.strip()]
            pub_types_list.append(pts)
            is_ex, reason = evaluate_pre_filter(row["clean_title"], row["clean_abstract"], pts)
            is_ex_list.append(is_ex)
            reasons_list.append(reason)
            vec = get_rule_feature_vector(row["clean_title"], row["clean_abstract"], pts)
            rule_vecs.append(vec)

        df["pub_types_parsed"] = pub_types_list
        df["is_pre_filtered"] = is_ex_list
        df["filter_reason"] = reasons_list
        df["rule_vector"] = rule_vecs
        return df

    logger.info("Extracting clean text, publication types, and 7 Clinical Indicator vectors...")
    train_df = enrich_data(train_df)
    val_df = enrich_data(val_df)
    test_df = enrich_data(test_df)

    y_train = train_df["label"].to_numpy().astype(int)
    y_val = val_df["label"].to_numpy().astype(int)
    y_test = test_df["label"].to_numpy().astype(int)

    n_pos = int((y_train == 1).sum())
    n_neg = int((y_train == 0).sum())
    pos_weight = float(2.0 * (n_neg / n_pos))
    logger.info(f"Sub-train distribution: Pos={n_pos}, Neg={n_neg} -> pos_weight = {pos_weight:.4f}")

    # 3. Tokenizer & DataLoaders
    tokenizer = AutoTokenizer.from_pretrained(PRETRAINED_MODEL_NAME, local_files_only=True)

    # A. Abstract-Only datasets
    ds_train_abs = Stage5Dataset(train_df["clean_title"].tolist(), train_df["clean_abstract"].tolist(), y_train.tolist(), None, tokenizer, mode="abstract_only")
    ds_val_abs = Stage5Dataset(val_df["clean_title"].tolist(), val_df["clean_abstract"].tolist(), y_val.tolist(), None, tokenizer, mode="abstract_only")
    ds_test_abs = Stage5Dataset(test_df["clean_title"].tolist(), test_df["clean_abstract"].tolist(), y_test.tolist(), None, tokenizer, mode="abstract_only")

    # B. Title + Abstract datasets
    ds_train_ta = Stage5Dataset(train_df["clean_title"].tolist(), train_df["clean_abstract"].tolist(), y_train.tolist(), None, tokenizer, mode="title_abstract")
    ds_val_ta = Stage5Dataset(val_df["clean_title"].tolist(), val_df["clean_abstract"].tolist(), y_val.tolist(), None, tokenizer, mode="title_abstract")
    ds_test_ta = Stage5Dataset(test_df["clean_title"].tolist(), test_df["clean_abstract"].tolist(), y_test.tolist(), None, tokenizer, mode="title_abstract")

    # C. Title + Abstract + 7 Clinical Indicators datasets
    ds_train_hybrid = Stage5Dataset(train_df["clean_title"].tolist(), train_df["clean_abstract"].tolist(), y_train.tolist(), train_df["rule_vector"].tolist(), tokenizer, mode="title_abstract")
    ds_val_hybrid = Stage5Dataset(val_df["clean_title"].tolist(), val_df["clean_abstract"].tolist(), y_val.tolist(), val_df["rule_vector"].tolist(), tokenizer, mode="title_abstract")
    ds_test_hybrid = Stage5Dataset(test_df["clean_title"].tolist(), test_df["clean_abstract"].tolist(), y_test.tolist(), test_df["rule_vector"].tolist(), tokenizer, mode="title_abstract")

    loader_train_abs = DataLoader(ds_train_abs, batch_size=BATCH_SIZE_TRAIN, shuffle=True)
    loader_val_abs = DataLoader(ds_val_abs, batch_size=BATCH_SIZE_EVAL, shuffle=False)
    loader_test_abs = DataLoader(ds_test_abs, batch_size=BATCH_SIZE_EVAL, shuffle=False)

    loader_train_ta = DataLoader(ds_train_ta, batch_size=BATCH_SIZE_TRAIN, shuffle=True)
    loader_val_ta = DataLoader(ds_val_ta, batch_size=BATCH_SIZE_EVAL, shuffle=False)
    loader_test_ta = DataLoader(ds_test_ta, batch_size=BATCH_SIZE_EVAL, shuffle=False)

    loader_train_hybrid = DataLoader(ds_train_hybrid, batch_size=BATCH_SIZE_TRAIN, shuffle=True)
    loader_val_hybrid = DataLoader(ds_val_hybrid, batch_size=BATCH_SIZE_EVAL, shuffle=False)
    loader_test_hybrid = DataLoader(ds_test_hybrid, batch_size=BATCH_SIZE_EVAL, shuffle=False)

    # 4. Train the 3 underlying neural networks
    all_histories = []
    dummy_crit = nn.BCEWithLogitsLoss()

    # Network 1: Abstract-Only
    logger.info("--------------------------------------------------------------------------------")
    logger.info("TRAINING ARCHITECTURE 1: PubMedBERT (Abstract Only, Weighted)")
    logger.info("--------------------------------------------------------------------------------")
    model_abs, hist_abs = train_model(
        model_name="PubMedBERT (Abstract Only)",
        rule_dim=0,
        pos_weight_val=pos_weight,
        train_loader=loader_train_abs,
        val_loader=loader_val_abs,
        device=device,
        save_path=MODELS_DIR / "pubmedbert_abstract_only_weighted.pt",
    )
    all_histories.append(hist_abs)

    # Network 2: Title + Abstract
    logger.info("--------------------------------------------------------------------------------")
    logger.info("TRAINING ARCHITECTURE 2: PubMedBERT (Title + Abstract, Weighted)")
    logger.info("--------------------------------------------------------------------------------")
    model_ta, hist_ta = train_model(
        model_name="PubMedBERT (Title + Abstract)",
        rule_dim=0,
        pos_weight_val=pos_weight,
        train_loader=loader_train_ta,
        val_loader=loader_val_ta,
        device=device,
        save_path=MODELS_DIR / "pubmedbert_title_abstract_weighted.pt",
    )
    all_histories.append(hist_ta)

    # Network 3: Hybrid (Title + Abstract + 7 Clinical Indicators Vector)
    logger.info("--------------------------------------------------------------------------------")
    logger.info("TRAINING ARCHITECTURE 3: PubMedBERT (Title + Abstract + 7 Clinical Indicators, Weighted)")
    logger.info("--------------------------------------------------------------------------------")
    model_hybrid, hist_hybrid = train_model(
        model_name="PubMedBERT (Title + Abstract + 7 Clinical Indicators)",
        rule_dim=7,
        pos_weight_val=pos_weight,
        train_loader=loader_train_hybrid,
        val_loader=loader_val_hybrid,
        device=device,
        save_path=MODELS_DIR / "pubmedbert_hybrid_rules_weighted.pt",
    )
    all_histories.append(hist_hybrid)

    # Save training histories
    full_history_df = pd.concat(all_histories, ignore_index=True)
    full_history_df.to_csv(RESULTS_DIR / "stage5_training_histories.csv", index=False)

    # 5. Extract Model Raw Probabilities on Val and Test
    logger.info("Generating raw model probabilities...")
    _, _, val_probs_abs = evaluate_loader(model_abs, loader_val_abs, dummy_crit, device)
    _, _, test_probs_abs = evaluate_loader(model_abs, loader_test_abs, dummy_crit, device)

    _, _, val_probs_ta = evaluate_loader(model_ta, loader_val_ta, dummy_crit, device)
    _, _, test_probs_ta = evaluate_loader(model_ta, loader_test_ta, dummy_crit, device)

    _, _, val_probs_hybrid = evaluate_loader(model_hybrid, loader_val_hybrid, dummy_crit, device)
    _, _, test_probs_hybrid = evaluate_loader(model_hybrid, loader_test_hybrid, dummy_crit, device)

    # 6. Apply Deterministic Filtering for Cascaded Conditions
    val_is_filtered = val_df["is_pre_filtered"].to_numpy().astype(bool)
    test_is_filtered = test_df["is_pre_filtered"].to_numpy().astype(bool)

    logger.info(f"Test Set Pre-Filtered: {test_is_filtered.sum()} / {len(test_df)} articles")

    # Cascaded versions: If filtered -> 0.0, else model prob
    val_probs_abs_cascaded = np.where(val_is_filtered, 0.0, val_probs_abs)
    test_probs_abs_cascaded = np.where(test_is_filtered, 0.0, test_probs_abs)

    val_probs_ta_cascaded = np.where(val_is_filtered, 0.0, val_probs_ta)
    test_probs_ta_cascaded = np.where(test_is_filtered, 0.0, test_probs_ta)

    val_probs_hybrid_cascaded = np.where(val_is_filtered, 0.0, val_probs_hybrid)
    test_probs_hybrid_cascaded = np.where(test_is_filtered, 0.0, test_probs_hybrid)

    # 7. Define the 5 Evaluated Experimental Conditions
    conditions = [
        (
            "Condition 1: Abstract Only, Weighted, WITH Deterministic Filtering (Cascaded)",
            val_probs_abs_cascaded,
            test_probs_abs_cascaded,
            "Cascaded",
            "Abstract Only",
        ),
        (
            "Condition 2: Abstract Only, Weighted, WITHOUT Deterministic Filtering (Standalone)",
            val_probs_abs,
            test_probs_abs,
            "Standalone",
            "Abstract Only",
        ),
        (
            "Condition 3: Abstract + Title, Weighted, WITH Deterministic Filtering (Cascaded)",
            val_probs_ta_cascaded,
            test_probs_ta_cascaded,
            "Cascaded",
            "Title + Abstract",
        ),
        (
            "Condition 4: Abstract + Title, Weighted, WITHOUT Deterministic Filtering (Standalone)",
            val_probs_ta,
            test_probs_ta,
            "Standalone",
            "Title + Abstract",
        ),
        (
            "Condition 5: Abstract + Title, Weighted, WITH 7 Clinical Indicators Vector (Hybrid)",
            val_probs_hybrid,
            test_probs_hybrid,
            "Hybrid",
            "Title + Abstract + 7 Rules",
        ),
    ]

    # Compute default 0.50 threshold metrics
    all_default_metrics = []
    clinical_operating_points = []
    predictions_dict = {
        "pmid": test_df["pmid"].tolist(),
        "title": test_df["title"].tolist(),
        "label": y_test.tolist(),
        "is_pre_filtered": test_is_filtered.tolist(),
        "filter_reason": test_df["filter_reason"].tolist(),
    }

    roc_curves_data = {}
    pr_curves_data = {}

    for cond_name, v_probs, t_probs, pipe_type, input_type in conditions:
        m = compute_metrics(y_test, t_probs, threshold=0.5, condition_name=cond_name)
        m["pipeline_type"] = pipe_type
        m["input_modality"] = input_type
        all_default_metrics.append(m)

        # Swept clinical points (>=90%, >=95%, 100%)
        swept = sweep_thresholds(y_val, v_probs, y_test, t_probs, condition_name=cond_name)
        for s in swept:
            s["pipeline_type"] = pipe_type
            s["input_modality"] = input_type
            clinical_operating_points.append(s)

        # Store predictions
        col_prefix = cond_name.split(":")[0].replace(" ", "_").lower()
        predictions_dict[f"{col_prefix}_prob"] = t_probs.tolist()
        predictions_dict[f"{col_prefix}_pred_0.5"] = (t_probs >= 0.5).astype(int).tolist()

        # Curves
        fpr, tpr, _ = roc_curve(y_test, t_probs)
        prec, rec, _ = precision_recall_curve(y_test, t_probs)
        roc_curves_data[cond_name] = (fpr, tpr, m["auroc"])
        pr_curves_data[cond_name] = (rec, prec, m["pr_auc"])

    # Save Metrics CSV
    metrics_df = pd.DataFrame(all_default_metrics)
    metrics_df.to_csv(RESULTS_DIR / "stage5_test_metrics.csv", index=False)

    preds_df = pd.DataFrame(predictions_dict)
    preds_df.to_csv(RESULTS_DIR / "stage5_test_predictions.csv", index=False)

    clinical_df = pd.DataFrame(clinical_operating_points)
    clinical_df.to_csv(RESULTS_DIR / "stage5_clinical_operating_points.csv", index=False)

    logger.info("Saved metrics and predictions to results/stage5/")

    # 8. Generate Publication Plots
    logger.info("Generating publication figures...")
    sns.set_theme(style="whitegrid", font="sans-serif")
    palette = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]

    # Figure 1: ROC Curves
    plt.figure(figsize=(10, 8), dpi=300)
    for idx, (cond_name, (fpr, tpr, auroc_val)) in enumerate(roc_curves_data.items()):
        short_name = cond_name.split(":")[0] + ": " + cond_name.split(":")[1].split("(")[0].strip()
        plt.plot(fpr, tpr, lw=2.2, color=palette[idx], label=f"{short_name} (AUROC = {auroc_val:.4f})")
    plt.plot([0, 1], [0, 1], "k--", lw=1.2, alpha=0.6, label="Chance Baseline (AUROC = 0.5000)")
    plt.xlim([-0.02, 1.02])
    plt.ylim([-0.02, 1.05])
    plt.xlabel("False Positive Rate (1 - Specificity)", fontsize=12, fontweight="bold")
    plt.ylabel("True Positive Rate (Recall / Sensitivity)", fontsize=12, fontweight="bold")
    plt.title("Stage 5 Benchmark: Receiver Operating Characteristic (ROC) Curves\nHeld-Out Test Set (N = 105, 56 Pos / 49 Neg)", fontsize=14, fontweight="bold", pad=12)
    plt.legend(loc="lower right", fontsize=10, framealpha=0.95)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "stage5_roc_curves.png")
    plt.close()

    # Figure 2: Precision-Recall Curves
    plt.figure(figsize=(10, 8), dpi=300)
    baseline_pr = float(np.mean(y_test))
    for idx, (cond_name, (rec, prec, prauc_val)) in enumerate(pr_curves_data.items()):
        short_name = cond_name.split(":")[0] + ": " + cond_name.split(":")[1].split("(")[0].strip()
        plt.plot(rec, prec, lw=2.2, color=palette[idx], label=f"{short_name} (PR-AUC = {prauc_val:.4f})")
    plt.axhline(baseline_pr, color="k", linestyle="--", lw=1.2, alpha=0.6, label=f"Prevalence Baseline ({baseline_pr:.2%})")
    plt.xlim([-0.02, 1.02])
    plt.ylim([-0.02, 1.05])
    plt.xlabel("Recall (Sensitivity)", fontsize=12, fontweight="bold")
    plt.ylabel("Precision (Positive Predictive Value)", fontsize=12, fontweight="bold")
    plt.title("Stage 5 Benchmark: Precision-Recall (PR) Curves\nHeld-Out Test Set (N = 105, 56 Pos / 49 Neg)", fontsize=14, fontweight="bold", pad=12)
    plt.legend(loc="lower left", fontsize=10, framealpha=0.95)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "stage5_pr_curves.png")
    plt.close()

    # Figure 3: Confusion Matrices side by side
    fig, axes = plt.subplots(1, 5, figsize=(22, 4.5), dpi=300)
    for idx, (cond_name, _, t_probs, _, _) in enumerate(conditions):
        y_pred = (t_probs >= 0.5).astype(int)
        cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
        short_label = cond_name.split(":")[0] + "\n" + cond_name.split("(")[-1].replace(")", "")
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=axes[idx],
                    xticklabels=["Pred Neg (0)", "Pred Pos (1)"],
                    yticklabels=["True Neg (0)", "True Pos (1)"],
                    annot_kws={"size": 13, "weight": "bold"})
        axes[idx].set_title(short_label, fontsize=11, fontweight="bold")
        axes[idx].set_xlabel("Predicted Label", fontsize=10)
        if idx == 0:
            axes[idx].set_ylabel("True Ground Truth", fontsize=10)
        else:
            axes[idx].set_ylabel("")
    plt.suptitle("Stage 5 Held-Out Test Confusion Matrices (Threshold = 0.50)", fontsize=15, fontweight="bold", y=1.05)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "stage5_confusion_matrices.png", bbox_inches="tight")
    plt.close()

    # Figure 4: Predicted Probability Distributions
    plt.figure(figsize=(14, 7), dpi=300)
    prob_plot_data = []
    for idx, (cond_name, _, t_probs, _, _) in enumerate(conditions):
        c_short = cond_name.split(":")[0]
        for prob_val, true_lbl in zip(t_probs, y_test):
            prob_plot_data.append({
                "Condition": c_short,
                "Probability": prob_val,
                "True Label": "Eligible (Pos)" if true_lbl == 1 else "Ineligible (Neg)",
            })
    prob_df = pd.DataFrame(prob_plot_data)
    sns.boxplot(x="Condition", y="Probability", hue="True Label", data=prob_df, palette={"Eligible (Pos)": "#2ca02c", "Ineligible (Neg)": "#d62728"}, width=0.6)
    plt.title("Stage 5 Benchmark: Predicted Probability Distributions by True Class\nHeld-Out Test Set (N = 105)", fontsize=14, fontweight="bold", pad=12)
    plt.xlabel("Model Condition", fontsize=12, fontweight="bold")
    plt.ylabel("Predicted Probability p̂", fontsize=12, fontweight="bold")
    plt.axhline(0.5, color="grey", linestyle=":", lw=1.2, label="Default Threshold (0.50)")
    plt.legend(loc="upper right", framealpha=0.95)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "stage5_prob_distributions.png")
    plt.close()

    # Figure 5: Sensitivity vs Specificity Trade-off
    plt.figure(figsize=(10, 6), dpi=300)
    cond_labels = [c[0].split(":")[0] for c in conditions]
    sens_vals = [m["recall_sensitivity"] * 100 for m in all_default_metrics]
    spec_vals = [m["specificity"] * 100 for m in all_default_metrics]
    x_pos = np.arange(len(cond_labels))
    bar_width = 0.35
    plt.bar(x_pos - bar_width/2, sens_vals, width=bar_width, label="Recall / Sensitivity (%)", color="#1f77b4")
    plt.bar(x_pos + bar_width/2, spec_vals, width=bar_width, label="Specificity (%)", color="#ff7f0e")
    plt.xticks(x_pos, cond_labels, fontsize=11, fontweight="bold")
    plt.ylabel("Metric Score (%)", fontsize=12, fontweight="bold")
    plt.ylim([0, 115])
    for i in range(len(cond_labels)):
        plt.text(x_pos[i] - bar_width/2, sens_vals[i] + 2, f"{sens_vals[i]:.1f}%", ha="center", fontsize=9, fontweight="bold")
        plt.text(x_pos[i] + bar_width/2, spec_vals[i] + 2, f"{spec_vals[i]:.1f}%", ha="center", fontsize=9, fontweight="bold")
    plt.title("Sensitivity vs. Specificity at Threshold = 0.50 Across Stage 5 Conditions", fontsize=14, fontweight="bold", pad=12)
    plt.legend(loc="lower right", framealpha=0.95)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "stage5_metrics_tradeoff.png")
    plt.close()

    logger.info("================================================================================")
    logger.info("                 STAGE 5 PIPELINE COMPLETED SUCCESSFULLY!                       ")
    logger.info("================================================================================")


if __name__ == "__main__":
    main()
