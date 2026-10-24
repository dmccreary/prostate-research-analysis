"""
src/eval_val_baseline.py - Evaluate All Models (Stage 5 to Stage 7) on Validation Set.
Strictly calculates Recall, Specificity, and AUROC at default threshold 0.50.
Zero retraining. Uses existing checkpoints on disk.
"""

import sys
from pathlib import Path
import logging
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics import recall_score, confusion_matrix, roc_auc_score

BASE_DIR = Path(__file__).resolve().parent.parent
SPLITS_DIR = BASE_DIR / "data" / "splits" / "stage5"
MODELS_DIR = BASE_DIR / "models"
EMB_DIR = BASE_DIR / "data" / "embeddings" / "specter2"
RESULTS_DIR = BASE_DIR / "results"

sys.path.insert(0, str(BASE_DIR / "src"))
from preprocessing import clean_clinical_text
from stage3.rule_engine import evaluate_pre_filter, get_rule_feature_vector

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

# 1. Load Validation Data
val_df = pd.read_csv(SPLITS_DIR / "val.csv")
logger.info(f"Validation set loaded: {len(val_df)} articles.")

# Clean texts
val_df["clean_title"] = [clean_clinical_text(str(t or "")) for t in val_df["title"]]
val_df["clean_abstract"] = [clean_clinical_text(str(a or "")) for a in val_df["abstract"]]

# Rule features & deterministic filtering
pub_types_list, is_ex_list, rule_vecs = [], [], []
for _, row in val_df.iterrows():
    pts = [t.strip() for t in str(row.get("publication_types", "")).split(";") if t.strip()]
    pub_types_list.append(pts)
    is_ex, _ = evaluate_pre_filter(row["clean_title"], row["clean_abstract"], pts)
    is_ex_list.append(is_ex)
    vec = get_rule_feature_vector(row["clean_title"], row["clean_abstract"], pts)
    rule_vecs.append(vec)

val_df["is_pre_filtered"] = is_ex_list
val_df["rule_vector"] = rule_vecs
val_is_filtered = np.array(is_ex_list, dtype=bool)

y_val = val_df["label"].to_numpy().astype(int)
logger.info(f"Val label distribution: Positives={y_val.sum()}, Negatives={(y_val == 0).sum()}")

# Metric helper
def calc_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5):
    y_pred = (y_prob >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    try:
        auroc = roc_auc_score(y_true, y_prob)
    except Exception:
        auroc = 0.5
    return rec, spec, auroc

results = []

# ==============================================================================
# STAGE 5 MODELS (384 tokens)
# ==============================================================================
class FlexiblePubMedBERT(nn.Module):
    def __init__(self, pretrained_name="microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract", rule_dim=0, dropout_rate=0.25):
        super().__init__()
        self.bert = AutoModel.from_pretrained(pretrained_name, local_files_only=True)
        self.dropout = nn.Dropout(dropout_rate)
        self.rule_dim = rule_dim
        hidden_size = self.bert.config.hidden_size
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size + rule_dim, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 1),
        )

    def forward(self, input_ids, attention_mask, token_type_ids=None, rule_vector=None):
        bert_out = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        cls_rep = self.dropout(bert_out.last_hidden_state[:, 0, :])
        if self.rule_dim > 0 and rule_vector is not None:
            combined = torch.cat([cls_rep, rule_vector], dim=1)
        else:
            combined = cls_rep
        return self.classifier(combined).squeeze(-1)

class Stage5Dataset(Dataset):
    def __init__(self, titles, abstracts, labels, rule_vectors, tokenizer, mode="title_abstract", max_length=384):
        self.labels = torch.tensor(labels, dtype=torch.float32)
        self.rule_vectors = rule_vectors
        if mode == "abstract_only":
            enc = tokenizer(abstracts, padding="max_length", truncation=True, max_length=max_length, return_tensors="pt")
        else:
            enc = tokenizer(titles, abstracts, padding="max_length", truncation=True, max_length=max_length, return_tensors="pt")
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

def predict_loader(model, loader, device):
    model.eval()
    all_prob = []
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            token_type_ids = batch["token_type_ids"].to(device)
            rule_vector = batch.get("rule_vector")
            if rule_vector is not None:
                rule_vector = rule_vector.to(device)
            logits = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, rule_vector=rule_vector)
            probs = torch.sigmoid(logits).cpu().numpy()
            all_prob.extend(probs.tolist())
    return np.array(all_prob, dtype=float)

logger.info("Evaluating Stage 5 models...")
tokenizer_pbert = AutoTokenizer.from_pretrained("microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract", local_files_only=True)

# Abstract-only
ds_val_abs = Stage5Dataset(val_df["clean_title"].tolist(), val_df["clean_abstract"].tolist(), y_val.tolist(), None, tokenizer_pbert, mode="abstract_only")
loader_val_abs = DataLoader(ds_val_abs, batch_size=8, shuffle=False)

model_abs = FlexiblePubMedBERT(rule_dim=0).to(device)
model_abs.load_state_dict(torch.load(MODELS_DIR / "stage5" / "pubmedbert_abstract_only_weighted.pt", map_location=device))
probs_abs = predict_loader(model_abs, loader_val_abs, device)
probs_abs_cascaded = np.where(val_is_filtered, 0.0, probs_abs)

del model_abs
torch.cuda.empty_cache()

# Title+Abstract
ds_val_ta = Stage5Dataset(val_df["clean_title"].tolist(), val_df["clean_abstract"].tolist(), y_val.tolist(), None, tokenizer_pbert, mode="title_abstract")
loader_val_ta = DataLoader(ds_val_ta, batch_size=8, shuffle=False)

model_ta = FlexiblePubMedBERT(rule_dim=0).to(device)
model_ta.load_state_dict(torch.load(MODELS_DIR / "stage5" / "pubmedbert_title_abstract_weighted.pt", map_location=device))
probs_ta = predict_loader(model_ta, loader_val_ta, device)
probs_ta_cascaded = np.where(val_is_filtered, 0.0, probs_ta)

del model_ta
torch.cuda.empty_cache()

# Hybrid
ds_val_hybrid = Stage5Dataset(val_df["clean_title"].tolist(), val_df["clean_abstract"].tolist(), y_val.tolist(), val_df["rule_vector"].tolist(), tokenizer_pbert, mode="title_abstract")
loader_val_hybrid = DataLoader(ds_val_hybrid, batch_size=8, shuffle=False)

model_hybrid = FlexiblePubMedBERT(rule_dim=7).to(device)
model_hybrid.load_state_dict(torch.load(MODELS_DIR / "stage5" / "pubmedbert_hybrid_rules_weighted.pt", map_location=device))
probs_hybrid = predict_loader(model_hybrid, loader_val_hybrid, device)

del model_hybrid
torch.cuda.empty_cache()

r, s, a = calc_metrics(y_val, probs_abs_cascaded)
results.append({"Model": "Condition 1: Abstract Only, Weighted, Cascaded (Stage 5)", "Recall": r, "Specificity": s, "AUROC": a})

r, s, a = calc_metrics(y_val, probs_abs)
results.append({"Model": "Condition 2: Abstract Only, Weighted, Standalone (Stage 5)", "Recall": r, "Specificity": s, "AUROC": a})

r, s, a = calc_metrics(y_val, probs_ta_cascaded)
results.append({"Model": "Condition 3: Title + Abstract, Weighted, Cascaded (Stage 5)", "Recall": r, "Specificity": s, "AUROC": a})

r, s, a = calc_metrics(y_val, probs_ta)
results.append({"Model": "Condition 4: Title + Abstract, Weighted, Standalone (Stage 5)", "Recall": r, "Specificity": s, "AUROC": a})

r, s, a = calc_metrics(y_val, probs_hybrid)
results.append({"Model": "Condition 5: Title + Abstract + 7 Rules Hybrid, Weighted (Stage 5)", "Recall": r, "Specificity": s, "AUROC": a})

# ==============================================================================
# STAGE 6 MODELS (SPECTER2)
# ==============================================================================
logger.info("Evaluating Stage 6 models...")

# Read frozen models validation sweep at threshold 0.50
sweep_frozen = pd.read_csv(RESULTS_DIR / "stage6" / "specter2_validation_threshold_sweeps.csv")
for m_name in ["SPECTER2 + Logistic Regression", "SPECTER2 + Linear SVM", "SPECTER2 + Small MLP"]:
    sub = sweep_frozen[(sweep_frozen["model_name"] == m_name) & (np.isclose(sweep_frozen["threshold"], 0.50))]
    if len(sub) > 0:
        row = sub.iloc[0]
        results.append({
            "Model": f"{m_name} (Stage 6 Frozen)",
            "Recall": float(row["recall_sensitivity"]),
            "Specificity": float(row["specificity"]),
            "AUROC": float(row["auroc"]),
        })

# Read fine-tuned SPECTER2 models validation sweep at threshold 0.50
sweep_ft = pd.read_csv(RESULTS_DIR / "stage6" / "specter2_finetuned_validation_thresholds.csv")
for m_name in ["SPECTER2 Fine-Tuned (Unweighted)", "SPECTER2 Fine-Tuned (Weighted)", "SPECTER2 Fine-Tuned (Gradual Unfreezing)"]:
    sub = sweep_ft[(sweep_ft["model_name"] == m_name) & (np.isclose(sweep_ft["threshold"], 0.50))]
    if len(sub) > 0:
        row = sub.iloc[0]
        results.append({
            "Model": f"{m_name} (Stage 6)",
            "Recall": float(row["recall_sensitivity"]),
            "Specificity": float(row["specificity"]),
            "AUROC": float(row["auroc"]),
        })

# ==============================================================================
# STAGE 5.5 MODELS (PubMedBERT 512 tokens)
# ==============================================================================
logger.info("Evaluating Stage 5.5 models...")

class PubMedBERT512Dataset(Dataset):
    def __init__(self, titles, abstracts, labels, tokenizer, max_length=512):
        self.labels = torch.tensor(labels, dtype=torch.float32)
        enc = tokenizer(titles, abstracts, padding="max_length", truncation=True, max_length=max_length, return_tensors="pt")
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

ds_val_512 = PubMedBERT512Dataset(val_df["clean_title"].tolist(), val_df["clean_abstract"].tolist(), y_val.tolist(), tokenizer_pbert, max_length=512)
loader_val_512 = DataLoader(ds_val_512, batch_size=8, shuffle=False)

# Model A: Unweighted
model_512_unweighted = FlexiblePubMedBERT(rule_dim=0).to(device)
model_512_unweighted.load_state_dict(torch.load(MODELS_DIR / "stage5_512" / "pubmedbert_512_unweighted.pt", map_location=device))
probs_512_unw = predict_loader(model_512_unweighted, loader_val_512, device)
del model_512_unweighted
torch.cuda.empty_cache()

r, s, a = calc_metrics(y_val, probs_512_unw)
results.append({"Model": "PubMedBERT 512 Unweighted (Stage 5.5)", "Recall": r, "Specificity": s, "AUROC": a})

# Model B: Weighted
model_512_weighted = FlexiblePubMedBERT(rule_dim=0).to(device)
model_512_weighted.load_state_dict(torch.load(MODELS_DIR / "stage5_512" / "pubmedbert_512_weighted.pt", map_location=device))
probs_512_w = predict_loader(model_512_weighted, loader_val_512, device)
del model_512_weighted
torch.cuda.empty_cache()

r, s, a = calc_metrics(y_val, probs_512_w)
results.append({"Model": "PubMedBERT 512 Weighted (Stage 5.5)", "Recall": r, "Specificity": s, "AUROC": a})

# ==============================================================================
# STAGE 7 MODEL (PICO Cross-Encoder 512 tokens)
# ==============================================================================
logger.info("Evaluating Stage 7 PICO Cross-Encoder...")

PICO_QUERY = (
    "Screening criteria for localized prostate cancer: Primary curative treatment "
    "(radical prostatectomy, EBRT >= 72 Gy, brachytherapy, cryo, HIFU). "
    "Must be a comparative trial (exclude single-arm audits, post-treatment salvage, metastatic, or palliative). "
    "Reports oncologic survival (BRFS, OS, CSS) with follow-up >= 5 years."
)

class PICOCrossEncoderDataset(Dataset):
    def __init__(self, titles, abstracts, labels, tokenizer, pico_query=PICO_QUERY, max_length=512):
        self.labels = torch.tensor(labels, dtype=torch.float32)
        pico_queries = [pico_query] * len(titles)
        paper_texts = [f"{t}. {a}" if t else a for t, a in zip(titles, abstracts)]
        enc = tokenizer(text=pico_queries, text_pair=paper_texts, padding="max_length", truncation=True, max_length=max_length, return_tensors="pt")
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

ds_val_pico = PICOCrossEncoderDataset(val_df["clean_title"].tolist(), val_df["clean_abstract"].tolist(), y_val.tolist(), tokenizer_pbert, max_length=512)
loader_val_pico = DataLoader(ds_val_pico, batch_size=8, shuffle=False)

model_pico = FlexiblePubMedBERT(rule_dim=0).to(device)
model_pico.load_state_dict(torch.load(MODELS_DIR / "stage7" / "pico_cross_encoder_weighted.pt", map_location=device))
probs_pico = predict_loader(model_pico, loader_val_pico, device)
del model_pico
torch.cuda.empty_cache()

r, s, a = calc_metrics(y_val, probs_pico)
results.append({"Model": "PICO Cross-Encoder 512 Weighted (Stage 7)", "Recall": r, "Specificity": s, "AUROC": a})

# ==============================================================================
# SUMMARY TABLE
# ==============================================================================
df_res = pd.DataFrame(results)
print("\n" + "="*80)
print("VALIDATION SET BASELINE PERFORMANCE (N = 84, Threshold = 0.50)")
print("="*80)
print(df_res.to_string(index=False))
print("="*80)

# Save to CSV
out_csv = RESULTS_DIR / "validation_baseline_comparison.csv"
df_res.to_csv(out_csv, index=False)
logger.info(f"Saved validation baseline table to {out_csv}")
