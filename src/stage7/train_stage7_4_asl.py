"""
src/stage7/train_stage7_4_asl.py - Stage 7.4: Asymmetric Loss (Approach C) Screening Benchmark.

Evaluates a 2x2 factorial design across 4 distinct models:
1. Model 1: ASL Standalone, Unweighted (rule_dim = 0, pos_weight = 1.0)
2. Model 2: ASL Standalone, Weighted   (rule_dim = 0, pos_weight = 1.7222)
3. Model 3: ASL Hybrid, Unweighted     (rule_dim = 7, pos_weight = 1.0)
4. Model 4: ASL Hybrid, Weighted       (rule_dim = 7, pos_weight = 1.7222)

All models:
- Natural-language PICO Query (Stage 7.0/7.1 text)
- 512 tokens sequence budget
- Evaluated STRICTLY on Validation set (val.csv, N=84)
- Held-out test set remains UNTOUCHED
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
MODELS_DIR = BASE_DIR / "models" / "stage7_4"
RESULTS_DIR = BASE_DIR / "results" / "stage7_4"
LOGS_DIR = BASE_DIR / "logs"

for d in [MODELS_DIR, RESULTS_DIR, LOGS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOGS_DIR / "stage7_4_asl.log", encoding="utf-8", mode="w"),
    ],
)
logger = logging.getLogger(__name__)

sys.path.insert(0, str(BASE_DIR / "src"))
from preprocessing import clean_clinical_text
from stage3.rule_engine import evaluate_pre_filter, get_rule_feature_vector

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

# Natural-Language PICO Query (Stage 7.0 / 7.1 text)
PICO_QUERY_NATURAL = (
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
        if self.gamma_pos > 0:
            pos_modulating = (1.0 - probs) ** self.gamma_pos
        else:
            pos_modulating = 1.0
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
        rule_vectors: List[List[float]],
        tokenizer: AutoTokenizer,
        pico_query: str = PICO_QUERY_NATURAL,
        max_length: int = MAX_LENGTH,
    ):
        self.labels = torch.tensor(labels, dtype=torch.float32)
        self.rule_vectors = torch.tensor(np.array(rule_vectors), dtype=torch.float32)
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


class FlexibleCrossEncoder(nn.Module):
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
        token_type_ids: torch.Tensor,
        rule_vector: torch.Tensor = None,
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
    model_name: str = "",
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
    model_name: str,
    target_sensitivities: List[float] = [0.90, 0.95, 0.97, 1.00],
) -> Tuple[Dict[float, float], pd.DataFrame]:
    threshold_grid = np.linspace(0.01, 0.99, 99)
    selected_thresholds = {}
    all_sweeps = []

    for t in threshold_grid:
        m = compute_metrics(y_val, val_probs, threshold=t, model_name=model_name, operating_point="Validation Sweep")
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


def train_single_model(
    model_id_code: str,
    model_name: str,
    rule_dim: int,
    pos_weight: float,
    loader_train: DataLoader,
    loader_val: DataLoader,
    y_val: np.ndarray,
    device: torch.device,
) -> Tuple[Dict, Dict, np.ndarray, pd.DataFrame, pd.DataFrame]:
    logger.info("=" * 80)
    logger.info(f"TRAINING [{model_id_code}]: {model_name} (rule_dim={rule_dim}, pos_weight={pos_weight:.4f})")
    logger.info("=" * 80)

    set_all_seeds(SEED)
    model = FlexibleCrossEncoder(
        pretrained_name=PRETRAINED_MODEL_NAME,
        rule_dim=rule_dim,
        dropout_rate=DROPOUT_RATE,
    ).to(device)

    criterion = AsymmetricLossWithCostWeighting(
        gamma_pos=0.0,
        gamma_neg=2.0,
        margin=0.05,
        pos_weight=pos_weight,
    )

    total_train_steps = (len(loader_train) // GRAD_ACCUM_STEPS) * EPOCHS
    warmup_steps = int(total_train_steps * 0.1)

    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_train_steps,
    )

    best_val_loss = float("inf")
    best_weights = None
    no_improve_epochs = 0
    training_history = []
    checkpoint_file = MODELS_DIR / f"{model_id_code.lower()}.pt"

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
        except Exception:
            val_auroc = 0.5

        val_rec = recall_score(val_y_true, (val_y_prob >= 0.5).astype(int), zero_division=0)
        cm_val = confusion_matrix(val_y_true, (val_y_prob >= 0.5).astype(int), labels=[0, 1])
        tn, fp, fn, tp = cm_val.ravel()
        val_spec = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0

        training_history.append({
            "model_id": model_id_code,
            "epoch": epoch,
            "train_loss": train_mean_loss,
            "val_loss": val_mean_loss,
            "val_auroc": val_auroc,
            "val_recall": val_rec,
            "val_specificity": val_spec,
        })

        logger.info(
            f"[{model_id_code}] Epoch {epoch:02d}/{EPOCHS:02d} | Train Loss: {train_mean_loss:.4f} | "
            f"Val Loss: {val_mean_loss:.4f} | Val Recall: {val_rec:.4f} | Val Spec: {val_spec:.4f} | Val AUROC: {val_auroc:.4f}"
        )

        if val_mean_loss < best_val_loss:
            best_val_loss = val_mean_loss
            best_weights = copy.deepcopy(model.state_dict())
            no_improve_epochs = 0
            torch.save(best_weights, checkpoint_file)
            logger.info(f"  --> Saved new best checkpoint at epoch {epoch:02d} (Val Loss: {val_mean_loss:.4f})")
        else:
            no_improve_epochs += 1
            if no_improve_epochs >= PATIENCE:
                logger.info(f"[{model_id_code}] Early stopping triggered at epoch {epoch:02d}.")
                break

    elapsed = time.time() - start_time
    logger.info(f"[{model_id_code}] Training finished in {elapsed:.1f}s.")

    # Load best checkpoint
    model.load_state_dict(best_weights)

    # Validation predictions
    _, _, final_val_probs = evaluate_loader(model, loader_val, criterion, device)

    val_thresholds, sweep_df = sweep_validation_thresholds(y_val, final_val_probs, model_name=model_name)
    sweep_df["model_id"] = model_id_code

    th_95 = val_thresholds[0.95]
    logger.info(f"[{model_id_code}] Selected Validation Threshold for Recall >= 95%: {th_95:.2f}")

    m_default = compute_metrics(y_val, final_val_probs, threshold=0.50, model_name=model_name, operating_point="Default (0.50)")
    m_default["model_id"] = model_id_code

    m_95 = compute_metrics(y_val, final_val_probs, threshold=th_95, model_name=model_name, operating_point="Recall >= 95%")
    m_95["model_id"] = model_id_code

    # Cleanup GPU memory
    del model
    del optimizer
    del scheduler
    torch.cuda.empty_cache()
    gc.collect()

    return m_default, m_95, final_val_probs, sweep_df, pd.DataFrame(training_history)


def main():
    set_all_seeds(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Execution Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    # 1. Load Train and Validation Splits ONLY
    train_df = pd.read_csv(SPLITS_DIR / "train.csv")
    val_df = pd.read_csv(SPLITS_DIR / "val.csv")
    logger.info(f"Loaded Splits: Train={len(train_df)}, Val={len(val_df)} (Held-out Test is completely excluded)")

    # 2. Enrich text & generate 7-d rule vectors
    def enrich_data(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["clean_title"] = [clean_clinical_text(str(t or "")) for t in df["title"]]
        df["clean_abstract"] = [clean_clinical_text(str(a or "")) for a in df["abstract"]]

        pub_types_list, rule_vecs = [], []
        for _, row in df.iterrows():
            pts = [t.strip() for t in str(row.get("publication_types", "")).split(";") if t.strip()]
            pub_types_list.append(pts)
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
    calculated_pos_weight = float(2.0 * (n_neg / n_pos))  # 1.7222
    logger.info(f"Sub-train class balance: Pos={n_pos}, Neg={n_neg} -> Weighted pos_weight = {calculated_pos_weight:.4f}")

    tokenizer = AutoTokenizer.from_pretrained(PRETRAINED_MODEL_NAME, local_files_only=True)

    ds_train = PICODataset(
        train_df["clean_title"].tolist(),
        train_df["clean_abstract"].tolist(),
        y_train.tolist(),
        train_df["rule_vector"].tolist(),
        tokenizer,
        pico_query=PICO_QUERY_NATURAL,
        max_length=MAX_LENGTH,
    )
    ds_val = PICODataset(
        val_df["clean_title"].tolist(),
        val_df["clean_abstract"].tolist(),
        y_val.tolist(),
        val_df["rule_vector"].tolist(),
        tokenizer,
        pico_query=PICO_QUERY_NATURAL,
        max_length=MAX_LENGTH,
    )

    loader_train = DataLoader(ds_train, batch_size=BATCH_SIZE_TRAIN, shuffle=True)
    loader_val = DataLoader(ds_val, batch_size=BATCH_SIZE_EVAL, shuffle=False)

    # 4 Model configurations
    configs = [
        ("model_7_4_standalone_unweighted", "ASL Standalone (Unweighted)", 0, 1.0),
        ("model_7_4_standalone_weighted",   "ASL Standalone (Weighted)",   0, calculated_pos_weight),
        ("model_7_4_hybrid_unweighted",     "ASL Hybrid + 7 Rules (Unweighted)", 7, 1.0),
        ("model_7_4_hybrid_weighted",       "ASL Hybrid + 7 Rules (Weighted)",   7, calculated_pos_weight),
    ]

    all_metrics = []
    all_sweeps = []
    all_histories = []
    preds_dict = {
        "pmid": val_df["pmid"].tolist(),
        "title": val_df["clean_title"].tolist(),
        "true_label": y_val,
    }

    # Execute training for all 4 models sequentially
    for code, name, rule_dim, p_weight in configs:
        m_def, m_95, probs, sweep_df, hist_df = train_single_model(
            model_id_code=code,
            model_name=name,
            rule_dim=rule_dim,
            pos_weight=p_weight,
            loader_train=loader_train,
            loader_val=loader_val,
            y_val=y_val,
            device=device,
        )
        all_metrics.extend([m_def, m_95])
        all_sweeps.append(sweep_df)
        all_histories.append(hist_df)
        preds_dict[f"{code}_prob"] = probs

    # Save all results to disk
    df_metrics = pd.DataFrame(all_metrics)
    df_metrics.to_csv(RESULTS_DIR / "stage7_4_validation_metrics.csv", index=False)

    df_sweeps = pd.concat(all_sweeps, ignore_index=True)
    df_sweeps.to_csv(RESULTS_DIR / "stage7_4_validation_threshold_sweeps.csv", index=False)

    df_hist = pd.concat(all_histories, ignore_index=True)
    df_hist.to_csv(RESULTS_DIR / "stage7_4_training_histories.csv", index=False)

    df_preds = pd.DataFrame(preds_dict)
    df_preds.to_csv(RESULTS_DIR / "stage7_4_all_validation_predictions.csv", index=False)

    # Print summary table matching user specifications
    print("\n" + "="*95)
    print("STAGE 7.4 (APPROACH C: ASYMMETRIC LOSS) VALIDATION RESULTS (N = 84)")
    print("="*95)
    print(f"{'Model / Architecture':<42} | {'Operating Point':<16} | {'Threshold':<9} | {'Recall':<9} | {'Specificity':<11} | {'AUROC':<7}")
    print("-" * 95)
    for m in all_metrics:
        print(f"{m['model_name']:<42} | {m['operating_point']:<16} | {m['threshold']:<9.2f} | {m['recall_sensitivity']:<9.2%} | {m['specificity']:<11.2%} | {m['auroc']:<7.4f}")
    print("="*95 + "\n")


if __name__ == "__main__":
    main()
