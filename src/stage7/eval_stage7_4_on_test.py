"""
src/stage7/eval_stage7_4_on_test.py - Evaluate ASL Hybrid + 7 Rules (Unweighted) on Held-Out Test Set.

Evaluates existing trained checkpoint:
models/stage7_4/model_7_4_hybrid_unweighted.pt

Evaluates on:
data/splits/stage5/test.csv (N = 105: 56 Positives, 49 Negatives)

Evaluates at:
- Default threshold: 0.50
- Calibrated 100% validation threshold: 0.46
"""

import sys
from pathlib import Path
import logging
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
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer

BASE_DIR = Path(__file__).resolve().parent.parent.parent
SPLITS_DIR = BASE_DIR / "data" / "splits" / "stage5"
MODELS_DIR = BASE_DIR / "models" / "stage7_4"
RESULTS_DIR = BASE_DIR / "results" / "stage7_4"

sys.path.insert(0, str(BASE_DIR / "src"))
from preprocessing import clean_clinical_text
from stage3.rule_engine import evaluate_pre_filter, get_rule_feature_vector

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

# 1. Load Test Data
test_df = pd.read_csv(SPLITS_DIR / "test.csv")
logger.info(f"Loaded Held-Out Test Set: {len(test_df)} articles.")

# Clean text & generate 7-d rule vector
test_df["clean_title"] = [clean_clinical_text(str(t or "")) for t in test_df["title"]]
test_df["clean_abstract"] = [clean_clinical_text(str(a or "")) for a in test_df["abstract"]]

rule_vecs = []
for _, row in test_df.iterrows():
    pts = [t.strip() for t in str(row.get("publication_types", "")).split(";") if t.strip()]
    vec = get_rule_feature_vector(row["clean_title"], row["clean_abstract"], pts)
    rule_vecs.append(vec)

test_df["rule_vector"] = rule_vecs
y_test = test_df["label"].to_numpy().astype(int)
logger.info(f"Test Class Balance: Positives={y_test.sum()} ({(y_test==1).mean():.1%}), Negatives={(y_test==0).sum()}")

# 2. Dataset and Architecture
PRETRAINED_MODEL_NAME = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract"
MAX_LENGTH = 512
PICO_QUERY_NATURAL = (
    "Screening criteria for localized prostate cancer: Primary curative treatment "
    "(radical prostatectomy, EBRT >= 72 Gy, brachytherapy, cryo, HIFU). "
    "Must be a comparative trial (exclude single-arm audits, post-treatment salvage, metastatic, or palliative). "
    "Reports oncologic survival (BRFS, OS, CSS) with follow-up >= 5 years."
)

class PICODataset(Dataset):
    def __init__(self, titles, abstracts, labels, rule_vectors, tokenizer, pico_query=PICO_QUERY_NATURAL, max_length=MAX_LENGTH):
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
    def __init__(self, pretrained_name=PRETRAINED_MODEL_NAME, rule_dim=7, dropout_rate=0.25):
        super().__init__()
        self.bert = AutoModel.from_pretrained(pretrained_name, local_files_only=True)
        self.dropout = nn.Dropout(dropout_rate)
        self.rule_dim = rule_dim
        hidden_size = self.bert.config.hidden_size  # 768
        total_dim = hidden_size + rule_dim          # 775

        self.classifier = nn.Sequential(
            nn.Linear(total_dim, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 1),
        )

    def forward(self, input_ids, attention_mask, token_type_ids, rule_vector=None):
        bert_out = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        cls_rep = self.dropout(bert_out.last_hidden_state[:, 0, :])
        if self.rule_dim > 0 and rule_vector is not None:
            combined = torch.cat([cls_rep, rule_vector], dim=1)
        else:
            combined = cls_rep
        return self.classifier(combined).squeeze(-1)

# 3. Load Model and Predict
tokenizer = AutoTokenizer.from_pretrained(PRETRAINED_MODEL_NAME, local_files_only=True)
ds_test = PICODataset(test_df["clean_title"].tolist(), test_df["clean_abstract"].tolist(), y_test.tolist(), test_df["rule_vector"].tolist(), tokenizer)
loader_test = DataLoader(ds_test, batch_size=8, shuffle=False)

model = FlexibleCrossEncoder(rule_dim=7).to(device)
checkpoint_path = MODELS_DIR / "model_7_4_hybrid_unweighted.pt"
logger.info(f"Loading checkpoint from: {checkpoint_path}")
model.load_state_dict(torch.load(checkpoint_path, map_location=device))
model.eval()

all_probs = []
with torch.no_grad():
    for batch in loader_test:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        token_type_ids = batch["token_type_ids"].to(device)
        rule_vector = batch["rule_vector"].to(device)
        logits = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, rule_vector=rule_vector)
        probs = torch.sigmoid(logits).cpu().numpy()
        all_probs.extend(probs.tolist())

test_probs = np.array(all_probs, dtype=float)

# Save test predictions
test_preds_df = pd.DataFrame({
    "pmid": test_df["pmid"].tolist(),
    "title": test_df["clean_title"].tolist(),
    "true_label": y_test,
    "predicted_prob": test_probs,
})
test_preds_df.to_csv(RESULTS_DIR / "stage7_4_test_predictions_asl_hybrid_unweighted.csv", index=False)

# Metric calculation function
def calc_all_metrics(y_true, y_prob, threshold, operating_point):
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
        "model_name": "ASL Hybrid + 7 Rules (Unweighted)",
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

m_def = calc_all_metrics(y_test, test_probs, threshold=0.50, operating_point="Default (0.50)")
m_46  = calc_all_metrics(y_test, test_probs, threshold=0.46, operating_point="Validation 100% Threshold (0.46)")

test_metrics_df = pd.DataFrame([m_def, m_46])
test_metrics_df.to_csv(RESULTS_DIR / "stage7_4_test_metrics_asl_hybrid_unweighted.csv", index=False)

print("\n" + "="*85)
print("ASL HYBRID + 7 RULES (UNWEIGHTED) - HELD-OUT TEST SET EVALUATION (N = 105)")
print("="*85)
print(f"{'Operating Point':<32} | {'Threshold':<10} | {'Recall':<10} | {'Specificity':<12} | {'AUROC':<8}")
print("-" * 85)
for m in [m_def, m_46]:
    print(f"{m['operating_point']:<32} | {m['threshold']:<10.2f} | {m['recall_sensitivity']:<10.2%} | {m['specificity']:<12.2%} | {m['auroc']:<8.4f}")
print("="*85 + "\n")
