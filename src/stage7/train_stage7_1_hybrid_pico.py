"""
src/stage7/train_stage7_1_hybrid_pico.py - Stage 7.1: Hybrid PICO Cross-Encoder + 7 Clinical Rule Indicators.

Combines:
- Segment A: 73-token PICO Query
- Segment B: Title + Abstract of candidate article
- Interactive 12-layer Cross-Attention (max_length = 512)
- Late Fusion: Concatenates 768-d [CLS] embedding with 7-d clinical indicator vector -> 775-d
- 2-layer MLP Head: Linear(775, 128) -> LayerNorm(128) -> GELU -> Dropout(0.25) -> Linear(128, 1)
- Weighted BCEWithLogitsLoss (pos_weight = 1.7222)
- Evaluated STRICTLY on Validation set (val.csv, N=84).
- Held-out test set remains UNTOUCHED.
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
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup

# Safe stdout on Windows
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent.parent
SPLITS_DIR = BASE_DIR / "data" / "splits" / "stage5"
MODELS_DIR = BASE_DIR / "models" / "stage7_1"
RESULTS_DIR = BASE_DIR / "results" / "stage7_1"
LOGS_DIR = BASE_DIR / "logs"

for d in [MODELS_DIR, RESULTS_DIR, LOGS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOGS_DIR / "stage7_1_hybrid_pico.log", encoding="utf-8", mode="w"),
    ],
)
logger = logging.getLogger(__name__)

sys.path.insert(0, str(BASE_DIR / "src"))
from preprocessing import clean_clinical_text
from stage3.rule_engine import evaluate_pre_filter, get_rule_feature_vector

# Hyperparameters & Constants
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

# 73-token PICO Query
PICO_QUERY = (
    "Screening criteria for localized prostate cancer: Primary curative treatment "
    "(radical prostatectomy, EBRT >= 72 Gy, brachytherapy, cryo, HIFU). "
    "Must be a comparative trial (exclude single-arm audits, post-treatment salvage, metastatic, or palliative). "
    "Reports oncologic survival (BRFS, OS, CSS) with follow-up >= 5 years."
)


def set_all_seeds(seed: int = SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class HybridPICODataset(Dataset):
    """
    Dual-segment PICO Cross-Encoder + 7-dimensional clinical indicator features.
    Segment A: PICO Query
    Segment B: Title + Abstract
    """

    def __init__(
        self,
        titles: List[str],
        abstracts: List[str],
        labels: List[int],
        rule_vectors: List[List[float]],
        tokenizer: AutoTokenizer,
        pico_query: str = PICO_QUERY,
        max_length: int = MAX_LENGTH,
    ):
        self.labels = torch.tensor(labels, dtype=torch.float32)
        self.rule_vectors = torch.tensor(rule_vectors, dtype=torch.float32)
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
            "rule_vector": self.rule_vectors[idx],
            "label": self.labels[idx],
        }


class HybridPICOCrossEncoder(nn.Module):
    """
    Hybrid PICO Cross-Encoder Architecture:
    [CLS] from BERT (768) concatenated with 7-d clinical indicator features -> 775-d
    Head: Linear(775, 128) -> LayerNorm(128) -> GELU() -> Dropout(0.25) -> Linear(128, 1)
    """

    def __init__(
        self,
        pretrained_name: str = PRETRAINED_MODEL_NAME,
        rule_dim: int = 7,
        dropout_rate: float = DROPOUT_RATE,
    ):
        super().__init__()
        self.bert = AutoModel.from_pretrained(pretrained_name, local_files_only=True)
        self.dropout = nn.Dropout(dropout_rate)
        hidden_size = self.bert.config.hidden_size  # 768
        total_dim = hidden_size + rule_dim  # 775

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
        token_type_ids: torch.Tensor,
        rule_vector: torch.Tensor,
    ) -> torch.Tensor:
        bert_out = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        cls_rep = bert_out.last_hidden_state[:, 0, :]
        cls_rep = self.dropout(cls_rep)
        combined = torch.cat([cls_rep, rule_vector], dim=1)  # (batch_size, 775)
        logits = self.classifier(combined).squeeze(-1)
        return logits


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
            rule_vector = batch["rule_vector"].to(device)
            labels = batch["label"].to(device)

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


def compute_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
    operating_point: str = "Default (0.50)",
) -> Dict:
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
        "model_name": "Hybrid PICO Cross-Encoder (7 Rules, Weighted)",
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
) -> Tuple[Dict[float, float], pd.DataFrame]:
    threshold_grid = np.linspace(0.01, 0.99, 99)
    selected_thresholds = {}
    all_sweeps = []

    for t in threshold_grid:
        m = compute_metrics(y_val, val_probs, threshold=t, operating_point="Validation Sweep")
        all_sweeps.append(m)

    sweep_df = pd.DataFrame(all_sweeps)

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

    return selected_thresholds, sweep_df


def main():
    logger.info("================================================================================")
    logger.info("  STARTING STAGE 7.1: HYBRID PICO CROSS-ENCODER + 7 CLINICAL INDICATOR RULES   ")
    logger.info("================================================================================")

    set_all_seeds(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Execution Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    # 1. Load Train and Validation Splits ONLY (Held-out Test is NOT loaded or evaluated)
    train_df = pd.read_csv(SPLITS_DIR / "train.csv")
    val_df = pd.read_csv(SPLITS_DIR / "val.csv")
    logger.info(f"Loaded Splits: Train={len(train_df)}, Val={len(val_df)} (Held-out Test is completely excluded)")

    # 2. Enrich text & extract 7-d rule vectors
    def enrich_data(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["clean_title"] = [clean_clinical_text(str(t or "")) for t in df["title"]]
        df["clean_abstract"] = [clean_clinical_text(str(a or "")) for a in df["abstract"]]

        pub_types_list, is_ex_list, rule_vecs = [], [], []
        for _, row in df.iterrows():
            pts = [t.strip() for t in str(row.get("publication_types", "")).split(";") if t.strip()]
            pub_types_list.append(pts)
            is_ex, _ = evaluate_pre_filter(row["clean_title"], row["clean_abstract"], pts)
            is_ex_list.append(is_ex)
            vec = get_rule_feature_vector(row["clean_title"], row["clean_abstract"], pts)
            rule_vecs.append(vec)

        df["rule_vector"] = rule_vecs
        return df

    logger.info("Enriching clinical text and generating 7 Clinical Indicator vectors...")
    train_df = enrich_data(train_df)
    val_df = enrich_data(val_df)

    y_train = train_df["label"].to_numpy().astype(int)
    y_val = val_df["label"].to_numpy().astype(int)

    n_pos = int((y_train == 1).sum())
    n_neg = int((y_train == 0).sum())
    pos_weight = float(2.0 * (n_neg / n_pos))
    logger.info(f"Sub-train class balance: Pos={n_pos}, Neg={n_neg} -> Cost-Sensitive pos_weight = {pos_weight:.4f}")

    # 3. Tokenizer & DataLoaders
    tokenizer = AutoTokenizer.from_pretrained(PRETRAINED_MODEL_NAME, local_files_only=True)

    ds_train = HybridPICODataset(
        train_df["clean_title"].tolist(),
        train_df["clean_abstract"].tolist(),
        y_train.tolist(),
        train_df["rule_vector"].tolist(),
        tokenizer,
        pico_query=PICO_QUERY,
        max_length=MAX_LENGTH,
    )
    ds_val = HybridPICODataset(
        val_df["clean_title"].tolist(),
        val_df["clean_abstract"].tolist(),
        y_val.tolist(),
        val_df["rule_vector"].tolist(),
        tokenizer,
        pico_query=PICO_QUERY,
        max_length=MAX_LENGTH,
    )

    loader_train = DataLoader(ds_train, batch_size=BATCH_SIZE_TRAIN, shuffle=True)
    loader_val = DataLoader(ds_val, batch_size=BATCH_SIZE_EVAL, shuffle=False)

    # 4. Initialize Architecture
    model = HybridPICOCrossEncoder(
        pretrained_name=PRETRAINED_MODEL_NAME,
        rule_dim=7,
        dropout_rate=DROPOUT_RATE,
    ).to(device)

    weight_tensor = torch.tensor([pos_weight], device=device, dtype=torch.float32)
    criterion = nn.BCEWithLogitsLoss(pos_weight=weight_tensor)

    total_train_steps = (len(loader_train) // GRAD_ACCUM_STEPS) * EPOCHS
    warmup_steps = int(total_train_steps * 0.1)

    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_train_steps,
    )

    # 5. Training Loop with Early Stopping on Val Loss
    best_val_loss = float("inf")
    best_weights = None
    no_improve_epochs = 0
    training_history = []

    logger.info(f"Training for {EPOCHS} epochs (Patience = {PATIENCE})...")
    start_time = time.time()

    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss = 0.0
        train_samples = 0
        optimizer.zero_grad()

        for step, batch in enumerate(loader_train):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            token_type_ids = batch["token_type_ids"].to(device)
            rule_vector = batch["rule_vector"].to(device)
            labels = batch["label"].to(device)

            logits = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                rule_vector=rule_vector,
            )
            loss = criterion(logits, labels)
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

        val_rec = recall_score(val_y_true, (val_y_prob >= 0.5).astype(int), zero_division=0)
        cm_val = confusion_matrix(val_y_true, (val_y_prob >= 0.5).astype(int), labels=[0, 1])
        tn, fp, fn, tp = cm_val.ravel()
        val_spec = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0

        training_history.append({
            "epoch": epoch,
            "train_loss": train_mean_loss,
            "val_loss": val_mean_loss,
            "val_auroc": val_auroc,
            "val_recall": val_rec,
            "val_specificity": val_spec,
        })

        logger.info(
            f"Epoch {epoch:02d}/{EPOCHS:02d} | Train Loss: {train_mean_loss:.4f} | "
            f"Val Loss: {val_mean_loss:.4f} | Val Recall: {val_rec:.4f} | Val Spec: {val_spec:.4f} | Val AUROC: {val_auroc:.4f}"
        )

        if val_mean_loss < best_val_loss:
            best_val_loss = val_mean_loss
            best_weights = copy.deepcopy(model.state_dict())
            no_improve_epochs = 0
            torch.save(best_weights, MODELS_DIR / "pico_cross_encoder_hybrid_rules_weighted.pt")
            logger.info(f"  --> Saved new best checkpoint at epoch {epoch:02d} (Val Loss: {val_mean_loss:.4f})")
        else:
            no_improve_epochs += 1
            if no_improve_epochs >= PATIENCE:
                logger.info(f"Early stopping triggered at epoch {epoch:02d}.")
                break

    elapsed = time.time() - start_time
    logger.info(f"Training completed in {elapsed:.1f} seconds.")

    # Load best checkpoint
    model.load_state_dict(best_weights)

    # 6. Final Evaluation Strictly on Validation Set
    logger.info("Computing final validation probabilities from best checkpoint...")
    _, _, final_val_probs = evaluate_loader(model, loader_val, criterion, device)

    # Save validation probabilities
    val_preds_df = pd.DataFrame({
        "pmid": val_df["pmid"].tolist(),
        "title": val_df["clean_title"].tolist(),
        "true_label": y_val,
        "predicted_prob": final_val_probs,
    })
    val_preds_df.to_csv(RESULTS_DIR / "stage7_1_validation_predictions.csv", index=False)

    # Sweep thresholds strictly on validation set
    val_thresholds, sweep_df = sweep_validation_thresholds(y_val, final_val_probs, target_sensitivities=[0.90, 0.95, 1.00])
    sweep_df.to_csv(RESULTS_DIR / "stage7_1_validation_threshold_sweeps.csv", index=False)

    th_95 = val_thresholds[0.95]
    logger.info(f"Selected Validation Threshold for Recall >= 95%: {th_95:.2f}")

    # Metrics at Default (0.50) and at Recall >= 95%
    m_default = compute_metrics(y_val, final_val_probs, threshold=0.50, operating_point="Default (0.50)")
    m_95 = compute_metrics(y_val, final_val_probs, threshold=th_95, operating_point="Recall >= 95%")

    all_val_metrics = pd.DataFrame([m_default, m_95])
    all_val_metrics.to_csv(RESULTS_DIR / "stage7_1_validation_metrics.csv", index=False)

    # Save training history
    pd.DataFrame(training_history).to_csv(RESULTS_DIR / "stage7_1_training_history.csv", index=False)

    # 7. Print Output Table Matching User Specification
    print("\n" + "="*80)
    print("STAGE 7.1: HYBRID PICO CROSS-ENCODER + 7 RULES (VALIDATION SET N=84)")
    print("="*80)
    print(f"{'Operating Point':<20} | {'Threshold':<10} | {'Recall':<10} | {'Specificity':<12} | {'AUROC':<8}")
    print("-" * 80)
    for m in [m_default, m_95]:
        print(f"{m['operating_point']:<20} | {m['threshold']:<10.2f} | {m['recall_sensitivity']:<10.2%} | {m['specificity']:<12.2%} | {m['auroc']:<8.4f}")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
