"""
eval_all_false_positives.py - Comprehensive cross-model evaluation of all false positive papers.

Evaluates all candidate false positive papers identified across Stage 4 experiments:
1. Stage 4 Single-Seed First Model (Seed 42, Weighted)
2. Stage 4 Single-Seed First Model (Seed 42, Unweighted)
3. Stage 4 Multi-Seed Model 1 (Seed 101, Weighted)
4. Stage 4 Multi-Seed Model 2 (Seed 123, Weighted)
5. Stage 4 Multi-Seed Model 3 (Seed 456, Weighted)

Reports:
- Model confidence (predicted probability: sigmoid(logits))
- Predicted class at threshold = 0.50 (Positive / False Positive vs. Negative / True Negative)
- Partition status for each model (TRAIN, VAL, TEST)
- Abstract and title clinical analysis explaining why the paper was challenging.
"""

import json
import logging
import os
from pathlib import Path
import sys
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
import pandas as pd
import numpy as np

# Safe stdout on Windows
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"

BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(BASE_DIR / "src"))
from preprocessing import clean_clinical_text

PRETRAINED_MODEL_NAME = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract"
DATASET_PATH = BASE_DIR / "data" / "labeled-dataset-v2.csv"
OUTPUT_DIR = BASE_DIR / "results" / "stage4_multi_seed"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


class DualInputPubMedBERT(nn.Module):
    def __init__(self, pretrained_name=PRETRAINED_MODEL_NAME, dropout_rate=0.25):
        super().__init__()
        self.bert = AutoModel.from_pretrained(pretrained_name, local_files_only=True)
        self.dropout = nn.Dropout(dropout_rate)
        hidden_size = self.bert.config.hidden_size
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 1),
        )

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        cls_rep = self.dropout(out.last_hidden_state[:, 0, :])
        logits = self.classifier(cls_rep).squeeze(-1)
        return logits


def load_model(checkpoint_path: Path, device: torch.device) -> nn.Module:
    model = DualInputPubMedBERT()
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load dataset
    df = pd.read_csv(DATASET_PATH)

    # All candidate False Positives from Stage 4 splits (and historical recurring FP)
    target_pmids = [
        26581143,  # FP in Seed 101
        26399602,  # FP in Seed 101
        31964317,  # FP in Seed 101
        15887028,  # FP in Seed 101
        25819287,  # FP in Seed 101
        25600860,  # FP in Seed 123
        25729256,  # Recurring FP from Stage 3
    ]

    target_df = df[df["pmid"].isin(target_pmids)].copy()
    target_df = target_df.sort_values(by="pmid").reset_index(drop=True)
    print(f"Found {len(target_df)} target papers in dataset.")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(PRETRAINED_MODEL_NAME, local_files_only=True)

    # Pre-tokenize target papers
    titles = [clean_clinical_text(str(t or "")) for t in target_df["title"]]
    abstracts = [clean_clinical_text(str(a or "")) for a in target_df["abstract"]]

    enc = tokenizer(
        text=titles,
        text_pair=abstracts,
        max_length=384,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)
    token_type_ids = enc.get("token_type_ids", torch.zeros_like(input_ids)).to(device)

    # Models definition
    models_dict = {
        "Stage 4 First Model (Seed 42, Weighted)": BASE_DIR / "models" / "stage4" / "pubmedbert_stage4_weighted.pt",
        "Stage 4 First Model (Seed 42, Unweighted)": BASE_DIR / "models" / "stage4" / "pubmedbert_stage4_unweighted.pt",
        "Stage 4 Multi-Seed (Seed 101, Weighted)": BASE_DIR / "models" / "stage4_multi_seed" / "pubmedbert_weighted_seed_101.pt",
        "Stage 4 Multi-Seed (Seed 123, Weighted)": BASE_DIR / "models" / "stage4_multi_seed" / "pubmedbert_weighted_seed_123.pt",
        "Stage 4 Multi-Seed (Seed 456, Weighted)": BASE_DIR / "models" / "stage4_multi_seed" / "pubmedbert_weighted_seed_456.pt",
    }

    # Partition mappings
    splits_dir_s4 = BASE_DIR / "data" / "splits" / "stage4"
    train_42 = set(pd.read_csv(splits_dir_s4 / "train.csv").pmid)
    val_42 = set(pd.read_csv(splits_dir_s4 / "val.csv").pmid)
    test_42 = set(pd.read_csv(splits_dir_s4 / "test.csv").pmid)

    partitions = {
        "Stage 4 First Model (Seed 42, Weighted)": (train_42, val_42, test_42),
        "Stage 4 First Model (Seed 42, Unweighted)": (train_42, val_42, test_42),
    }

    for s in [101, 123, 456]:
        s_dir = BASE_DIR / "data" / "splits" / "stage4_multi_seed" / f"split_seed_{s}"
        tr = set(pd.read_csv(s_dir / "train.csv").pmid)
        va = set(pd.read_csv(s_dir / "val.csv").pmid)
        te = set(pd.read_csv(s_dir / "test.csv").pmid)
        partitions[f"Stage 4 Multi-Seed (Seed {s}, Weighted)"] = (tr, va, te)

    results = []

    # Run predictions
    for model_name, ckpt_path in models_dict.items():
        if not ckpt_path.exists():
            print(f"Warning: {ckpt_path} does not exist!")
            continue

        model = load_model(ckpt_path, device)
        with torch.no_grad():
            logits = model(input_ids, attention_mask, token_type_ids)
            probs = torch.sigmoid(logits).cpu().numpy()

        tr_set, va_set, te_set = partitions[model_name]

        for i, row in target_df.iterrows():
            pmid = row["pmid"]
            title = row["title"]
            prob = probs[i]
            pred_class = 1 if prob >= 0.5 else 0
            is_fp = (pred_class == 1)

            if pmid in te_set:
                part = "TEST"
            elif pmid in va_set:
                part = "VAL"
            elif pmid in tr_set:
                part = "TRAIN"
            else:
                part = "UNKNOWN"

            results.append({
                "pmid": pmid,
                "title": title,
                "true_label": int(row["label"]),
                "model_name": model_name,
                "partition": part,
                "predicted_probability": float(prob),
                "predicted_label": pred_class,
                "classification_result": "False Positive (FP)" if is_fp else "True Negative (TN)",
            })

        del model
        torch.cuda.empty_cache()

    res_df = pd.DataFrame(results)
    csv_save_path = OUTPUT_DIR / "false_positives_cross_model_evaluation.csv"
    res_df.to_csv(csv_save_path, index=False)
    print(f"Saved cross-model evaluation to {csv_save_path}")

    # Create pivot table for display
    pivot_prob = res_df.pivot(index=["pmid", "title"], columns="model_name", values="predicted_probability")
    pivot_part = res_df.pivot(index=["pmid", "title"], columns="model_name", values="partition")

    print("\n" + "=" * 100)
    print("ALL FALSE POSITIVE PAPERS - PREDICTED PROBABILITIES ACROSS ALL STAGE 4 MODELS")
    print("=" * 100)
    for pmid in target_pmids:
        paper_row = target_df[target_df.pmid == pmid].iloc[0]
        print(f"\nPMID: {pmid}")
        print(f"Title: {paper_row.title}")
        for m_name in models_dict.keys():
            m_res = res_df[(res_df.pmid == pmid) & (res_df.model_name == m_name)].iloc[0]
            status_tag = "[FP - Flagged]" if m_res.predicted_label == 1 else "[TN - Correctly Rejected]"
            print(f"  * {m_name:42s} | Prob: {m_res.predicted_probability:.4f} | Split: {m_res.partition:5s} | {status_tag}")


if __name__ == "__main__":
    main()
