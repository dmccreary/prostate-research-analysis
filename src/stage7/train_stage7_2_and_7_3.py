"""
src/stage7/train_stage7_2_and_7_3.py - Train and Evaluate Stage 7.2 & Stage 7.3 on Validation Set.

Stage 7.2:
- Approach B: Structured PICO Query (Population, Intervention, Comparator, Outcome)
- Standalone Cross-Encoder (no clinical rules, rule_dim = 0)
- Weighted Loss (pos_weight = 1.7222)
- Evaluated strictly on Validation set (val.csv, N=84)

Stage 7.3:
- Approach B: Structured PICO Query (Population, Intervention, Comparator, Outcome)
- Hybrid Cross-Encoder with 7-d Clinical Rule Indicators (Late Fusion, rule_dim = 7)
- Weighted Loss (pos_weight = 1.7222)
- Evaluated strictly on Validation set (val.csv, N=84)

Held-out test set remains UNTOUCHED.
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
MODELS_DIR_7_2 = BASE_DIR / "models" / "stage7_2"
RESULTS_DIR_7_2 = BASE_DIR / "results" / "stage7_2"
MODELS_DIR_7_3 = BASE_DIR / "models" / "stage7_3"
RESULTS_DIR_7_3 = BASE_DIR / "results" / "stage7_3"
LOGS_DIR = BASE_DIR / "logs"

for d in [MODELS_DIR_7_2, RESULTS_DIR_7_2, MODELS_DIR_7_3, RESULTS_DIR_7_3, LOGS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOGS_DIR / "stage7_2_and_7_3.log", encoding="utf-8", mode="w"),
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

# Approach B: Structured PICO Facets Query (71 tokens)
PICO_QUERY_STRUCTURED = (
    "Population: localized prostate cancer. "
    "Intervention: primary curative treatment (radical prostatectomy, EBRT >= 72 Gy, brachytherapy, cryo, HIFU). "
    "Comparator: comparative trial (exclude single-arm audits, salvage, metastatic, palliative). "
    "Outcome: oncologic survival (BRFS, OS, CSS), follow-up >= 5 years."
)


def set_all_seeds(seed: int = SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class StructuredPICODataset(Dataset):
    def __init__(
        self,
        titles: List[str],
        abstracts: List[str],
        labels: List[int],
        rule_vectors: List[List[float]],
        tokenizer: AutoTokenizer,
        pico_query: str = PICO_QUERY_STRUCTURED,
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
    """
    Cross-Encoder supporting:
    - rule_dim = 0: Standalone text (Stage 7.2)
    - rule_dim = 7: Hybrid text + 7 clinical indicators (Stage 7.3)
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


def train_and_eval(
    stage_name: str,
    model_name: str,
    rule_dim: int,
    loader_train: DataLoader,
    loader_val: DataLoader,
    val_df: pd.DataFrame,
    y_val: np.ndarray,
    pos_weight: float,
    models_dir: Path,
    results_dir: Path,
    device: torch.device,
) -> Tuple[Dict, Dict]:
    logger.info("=" * 80)
    logger.info(f"STARTING {stage_name}: {model_name} (rule_dim={rule_dim})")
    logger.info("=" * 80)

    set_all_seeds(SEED)
    model = FlexibleCrossEncoder(
        pretrained_name=PRETRAINED_MODEL_NAME,
        rule_dim=rule_dim,
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

    best_val_loss = float("inf")
    best_weights = None
    no_improve_epochs = 0
    training_history = []
    checkpoint_file = models_dir / f"{stage_name.lower().replace(' ', '_')}_checkpoint.pt"

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
            "epoch": epoch,
            "train_loss": train_mean_loss,
            "val_loss": val_mean_loss,
            "val_auroc": val_auroc,
            "val_recall": val_rec,
            "val_specificity": val_spec,
        })

        logger.info(
            f"[{stage_name}] Epoch {epoch:02d}/{EPOCHS:02d} | Train Loss: {train_mean_loss:.4f} | "
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
                logger.info(f"Early stopping triggered at epoch {epoch:02d}.")
                break

    elapsed = time.time() - start_time
    logger.info(f"[{stage_name}] Training finished in {elapsed:.1f}s.")

    # Load best checkpoint
    model.load_state_dict(best_weights)

    # Validation predictions
    _, _, final_val_probs = evaluate_loader(model, loader_val, criterion, device)

    # Save outputs
    val_preds_df = pd.DataFrame({
        "pmid": val_df["pmid"].tolist(),
        "title": val_df["clean_title"].tolist(),
        "true_label": y_val,
        "predicted_prob": final_val_probs,
    })
    val_preds_df.to_csv(results_dir / f"{stage_name.lower().replace(' ', '_')}_validation_predictions.csv", index=False)

    val_thresholds, sweep_df = sweep_validation_thresholds(y_val, final_val_probs, model_name=model_name)
    sweep_df.to_csv(results_dir / f"{stage_name.lower().replace(' ', '_')}_validation_threshold_sweeps.csv", index=False)
    pd.DataFrame(training_history).to_csv(results_dir / f"{stage_name.lower().replace(' ', '_')}_training_history.csv", index=False)

    th_95 = val_thresholds[0.95]
    logger.info(f"[{stage_name}] Selected Validation Threshold for Recall >= 95%: {th_95:.2f}")

    m_default = compute_metrics(y_val, final_val_probs, threshold=0.50, model_name=model_name, operating_point="Default (0.50)")
    m_95 = compute_metrics(y_val, final_val_probs, threshold=th_95, model_name=model_name, operating_point="Recall >= 95%")

    all_val_metrics = pd.DataFrame([m_default, m_95])
    all_val_metrics.to_csv(results_dir / f"{stage_name.lower().replace(' ', '_')}_validation_metrics.csv", index=False)

    # Free memory
    del model
    del optimizer
    del scheduler
    torch.cuda.empty_cache()
    gc.collect()

    return m_default, m_95


def main():
    set_all_seeds(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Execution Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    # 1. Load Train and Validation Splits ONLY
    train_df = pd.read_csv(SPLITS_DIR / "train.csv")
    val_df = pd.read_csv(SPLITS_DIR / "val.csv")
    logger.info(f"Loaded Splits: Train={len(train_df)}, Val={len(val_df)} (Held-out Test is completely excluded)")

    # 2. Enrich text & extract 7-d rule vectors
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
    pos_weight = float(2.0 * (n_neg / n_pos))
    logger.info(f"Sub-train class balance: Pos={n_pos}, Neg={n_neg} -> Cost-Sensitive pos_weight = {pos_weight:.4f}")

    tokenizer = AutoTokenizer.from_pretrained(PRETRAINED_MODEL_NAME, local_files_only=True)

    ds_train = StructuredPICODataset(
        train_df["clean_title"].tolist(),
        train_df["clean_abstract"].tolist(),
        y_train.tolist(),
        train_df["rule_vector"].tolist(),
        tokenizer,
        pico_query=PICO_QUERY_STRUCTURED,
        max_length=MAX_LENGTH,
    )
    ds_val = StructuredPICODataset(
        val_df["clean_title"].tolist(),
        val_df["clean_abstract"].tolist(),
        y_val.tolist(),
        val_df["rule_vector"].tolist(),
        tokenizer,
        pico_query=PICO_QUERY_STRUCTURED,
        max_length=MAX_LENGTH,
    )

    loader_train = DataLoader(ds_train, batch_size=BATCH_SIZE_TRAIN, shuffle=True)
    loader_val = DataLoader(ds_val, batch_size=BATCH_SIZE_EVAL, shuffle=False)

    # -------------------------------------------------------------------------
    # Train Model 7.2: Structured PICO Standalone (rule_dim = 0)
    # -------------------------------------------------------------------------
    m_def_7_2, m_95_7_2 = train_and_eval(
        stage_name="Stage 7.2",
        model_name="Structured PICO Cross-Encoder (Standalone, Weighted)",
        rule_dim=0,
        loader_train=loader_train,
        loader_val=loader_val,
        val_df=val_df,
        y_val=y_val,
        pos_weight=pos_weight,
        models_dir=MODELS_DIR_7_2,
        results_dir=RESULTS_DIR_7_2,
        device=device,
    )

    # -------------------------------------------------------------------------
    # Train Model 7.3: Structured PICO + 7 Rules (rule_dim = 7)
    # -------------------------------------------------------------------------
    m_def_7_3, m_95_7_3 = train_and_eval(
        stage_name="Stage 7.3",
        model_name="Structured PICO Cross-Encoder + 7 Rules (Hybrid, Weighted)",
        rule_dim=7,
        loader_train=loader_train,
        loader_val=loader_val,
        val_df=val_df,
        y_val=y_val,
        pos_weight=pos_weight,
        models_dir=MODELS_DIR_7_3,
        results_dir=RESULTS_DIR_7_3,
        device=device,
    )

    # -------------------------------------------------------------------------
    # Final Output Summary Table
    # -------------------------------------------------------------------------
    print("\n" + "="*95)
    print("STAGE 7.2 & 7.3 VALIDATION RESULTS SUMMARY (N = 84)")
    print("="*95)
    print(f"{'Model / Architecture':<45} | {'Operating Point':<16} | {'Threshold':<9} | {'Recall':<9} | {'Specificity':<11} | {'AUROC':<7}")
    print("-" * 95)
    all_summary = [m_def_7_2, m_95_7_2, m_def_7_3, m_95_7_3]
    for m in all_summary:
        print(f"{m['model_name']:<45} | {m['operating_point']:<16} | {m['threshold']:<9.2f} | {m['recall_sensitivity']:<9.2%} | {m['specificity']:<11.2%} | {m['auroc']:<7.4f}")
    print("="*95 + "\n")


if __name__ == "__main__":
    main()
