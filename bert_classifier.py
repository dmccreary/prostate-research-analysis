"""
bert_classifier.py — Fine-tune PubMedBERT to classify prostate cancer abstracts.

Model:  microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract
        (pre-trained on PubMed abstracts — best fit for this task)
Input:  labeled_results.csv  (output from fetch_labeled.py)
Split:  70% train / 15% validation / 15% test  (stratified)

Outputs:
    best_model/               saved model + tokenizer (best val F1)
    training_log.csv          loss and metrics per epoch
    test_predictions.csv      true label, predicted label, confidence
    classification_report.txt precision / recall / F1 per class

Usage:
    pip install torch transformers scikit-learn pandas
    python bert_classifier.py [--data labeled_results.csv] [--epochs 10] [--batch-size 16]
"""

import argparse
import csv
import random
import sys
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

MODEL_NAME  = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract"
LABEL_MAP   = {"Negative": 0, "Positive": 1}
ID_TO_LABEL = {0: "Negative", 1: "Positive"}
SEED        = 42
MAX_LEN     = 512
DATA_DIR    = Path("data")


def clean_text(text: str) -> str:
    """Normalize Unicode and strip invisible/special characters."""
    text = unicodedata.normalize("NFKC", text)
    for char in (" ", "​", "‌", "‍",
                 "‎", "‏", "­", "﻿"):
        text = text.replace(char, " ")
    return " ".join(text.split())


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class AbstractDataset(Dataset):
    def __init__(self, texts: list[str], labels: list[int], tokenizer, max_len: int):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> dict:
        enc = self.tokenizer(
            self.texts[idx],
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids":      enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "label":          torch.tensor(self.labels[idx], dtype=torch.long),
        }


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_data(csv_path: Path) -> tuple[pd.DataFrame, list[str], list[int]]:
    df = pd.read_csv(csv_path, dtype=str, encoding="utf-8-sig")
    df.columns = df.columns.str.strip()          # remove BOM / stray whitespace
    # Validate required columns exist
    for col in ("label", "abstract"):
        if col not in df.columns:
            raise ValueError(
                f"Column '{col}' not found in {csv_path.name}.\n"
                f"  Available columns: {list(df.columns)}\n"
                "  Re-run fetch_labeled.py to regenerate the CSV."
            )
    before = len(df)
    df = df[df["label"].isin(["Positive", "Negative"])]
    df = df[df["abstract"].notna() & (df["abstract"].str.strip() != "")]
    dropped = before - len(df)
    if dropped:
        print(f"  Dropped {dropped} row(s) with missing abstract or unknown label.")
    # Clean special Unicode characters
    df["abstract"] = df["abstract"].str.strip().apply(clean_text)
    df = df.reset_index(drop=True)
    texts  = df["abstract"].tolist()
    labels = [LABEL_MAP[l] for l in df["label"]]
    return df, texts, labels


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(model, loader: DataLoader, device: torch.device) -> dict:
    model.eval()
    loss_fn = torch.nn.CrossEntropyLoss()
    total_loss = 0.0
    all_preds, all_labels, all_probs = [], [], []

    with torch.no_grad():
        for batch in loader:
            ids   = batch["input_ids"].to(device)
            mask  = batch["attention_mask"].to(device)
            lbls  = batch["label"].to(device)

            out   = model(input_ids=ids, attention_mask=mask)
            loss  = loss_fn(out.logits, lbls)
            total_loss += loss.item()

            probs = torch.softmax(out.logits, dim=1)[:, 1].cpu().numpy()
            preds = out.logits.argmax(dim=1).cpu().numpy()

            all_preds.extend(preds.tolist())
            all_labels.extend(lbls.cpu().numpy().tolist())
            all_probs.extend(probs.tolist())

    return {
        "loss":      total_loss / len(loader),
        "accuracy":  accuracy_score(all_labels, all_preds),
        "f1":        f1_score(all_labels, all_preds, average="binary"),
        "precision": precision_score(all_labels, all_preds, average="binary", zero_division=0),
        "recall":    recall_score(all_labels, all_preds, average="binary", zero_division=0),
        "auc_roc":   roc_auc_score(all_labels, all_probs),
        "preds":     all_preds,
        "labels":    all_labels,
        "probs":     all_probs,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fine-tune PubMedBERT for binary abstract classification."
    )
    parser.add_argument("--data",       default="data/labeled_results.csv",
                        help="Input CSV (default: data/labeled_results.csv)")
    parser.add_argument("--model-dir",  default="data/best_model",
                        help="Directory to save best model (default: data/best_model)")
    parser.add_argument("--epochs",     type=int,   default=10)
    parser.add_argument("--batch-size", type=int,   default=32)
    parser.add_argument("--lr",         type=float, default=2e-5)
    parser.add_argument("--patience",   type=int,   default=3,
                        help="Early-stopping patience in epochs (default: 3)")
    args = parser.parse_args()

    set_seed(SEED)
    DATA_DIR.mkdir(exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cpu":
        print("  Note: training on CPU will be slow. A GPU is recommended.")

    # ---- Load data --------------------------------------------------------
    data_path = Path(args.data)
    if not data_path.exists():
        sys.exit(f"[ERROR] Data file not found: {data_path}\n"
                 "Run fetch_labeled.py first to generate labeled_results.csv")

    print(f"\nLoading data from {data_path.name}...")
    df, texts, labels = load_data(data_path)
    n_pos = sum(labels)
    n_neg = len(labels) - n_pos
    print(f"  Total usable samples : {len(texts)}")
    print(f"  Positive             : {n_pos}")
    print(f"  Negative             : {n_neg}")

    # ---- 70 / 15 / 15 split on indices (stratified) -----------------------
    indices = list(range(len(texts)))
    train_idx, tmp_idx, t_y, tmp_y = train_test_split(
        indices, labels, test_size=0.30, random_state=SEED, stratify=labels
    )
    val_idx, test_idx, v_y, te_y = train_test_split(
        tmp_idx, tmp_y, test_size=0.50, random_state=SEED, stratify=tmp_y
    )
    t_x  = [texts[i] for i in train_idx]
    v_x  = [texts[i] for i in val_idx]
    te_x = [texts[i] for i in test_idx]
    print(f"\n  Train : {len(t_x)}  |  Val : {len(v_x)}  |  Test : {len(te_x)}")

    # ---- Save train / val sets to CSV for manual review -------------------
    df.iloc[train_idx].to_csv(DATA_DIR / "train_set.csv", index=False, encoding="utf-8")
    df.iloc[val_idx].to_csv(DATA_DIR / "val_set.csv",   index=False, encoding="utf-8")
    print("  Saved data/train_set.csv and data/val_set.csv")

    # ---- Tokenizer and model ----------------------------------------------
    print(f"\nDownloading / loading model: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=2
    )
    model.to(device)

    # ---- DataLoaders -------------------------------------------------------
    def make_loader(x, y, shuffle):
        ds = AbstractDataset(x, y, tokenizer, MAX_LEN)
        return DataLoader(ds, batch_size=args.batch_size, shuffle=shuffle,
                          num_workers=0, pin_memory=(device.type == "cuda"))

    train_loader = make_loader(t_x,  t_y,  shuffle=True)
    val_loader   = make_loader(v_x,  v_y,  shuffle=False)
    test_loader  = make_loader(te_x, te_y, shuffle=False)

    # ---- Optimizer + scheduler --------------------------------------------
    optimizer    = AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total_steps  = len(train_loader) * args.epochs
    warmup_steps = max(1, int(0.1 * total_steps))
    scheduler    = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )
    loss_fn = torch.nn.CrossEntropyLoss()

    # ---- Training loop ----------------------------------------------------
    model_dir = Path(args.model_dir)
    model_dir.mkdir(exist_ok=True)

    log_rows:       list[dict] = []
    best_val_f1:    float      = 0.0
    patience_count: int        = 0

    print(f"\nTraining  (epochs={args.epochs}, batch={args.batch_size}, "
          f"lr={args.lr}, patience={args.patience})\n")
    print(f"{'Epoch':>5}  {'Train Loss':>10}  {'Val Loss':>8}  "
          f"{'Val Acc':>7}  {'Val F1':>6}  {'Val AUC':>7}")
    print("-" * 55)

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0

        for batch in train_loader:
            ids   = batch["input_ids"].to(device)
            mask  = batch["attention_mask"].to(device)
            lbls  = batch["label"].to(device)

            optimizer.zero_grad()
            out  = model(input_ids=ids, attention_mask=mask)
            loss = loss_fn(out.logits, lbls)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            epoch_loss += loss.item()

        avg_train_loss = epoch_loss / len(train_loader)
        val = evaluate(model, val_loader, device)

        marker = ""
        if val["f1"] > best_val_f1:
            best_val_f1    = val["f1"]
            patience_count = 0
            model.save_pretrained(model_dir)
            tokenizer.save_pretrained(model_dir)
            marker = "  [saved]"
        else:
            patience_count += 1

        print(f"{epoch:>5}  {avg_train_loss:>10.4f}  {val['loss']:>8.4f}  "
              f"{val['accuracy']:>7.4f}  {val['f1']:>6.4f}  {val['auc_roc']:>7.4f}"
              + marker)

        log_rows.append({
            "epoch":        epoch,
            "train_loss":   round(avg_train_loss,   4),
            "val_loss":     round(val["loss"],       4),
            "val_accuracy": round(val["accuracy"],   4),
            "val_f1":       round(val["f1"],         4),
            "val_precision":round(val["precision"],  4),
            "val_recall":   round(val["recall"],     4),
            "val_auc_roc":  round(val["auc_roc"],    4),
        })

        if patience_count >= args.patience:
            print(f"\nEarly stopping: no val F1 improvement for {args.patience} epochs.")
            break

    # ---- Save training log ------------------------------------------------
    log_path = DATA_DIR / "training_log.csv"
    with log_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=log_rows[0].keys())
        writer.writeheader()
        writer.writerows(log_rows)

    # ---- Test evaluation (best checkpoint) --------------------------------
    print(f"\nLoading best model from '{model_dir}' for test evaluation...")
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.to(device)
    test = evaluate(model, test_loader, device)

    print("\n" + "=" * 45)
    print("TEST SET RESULTS")
    print("=" * 45)
    print(f"  Accuracy  : {test['accuracy']:.4f}")
    print(f"  F1        : {test['f1']:.4f}")
    print(f"  Precision : {test['precision']:.4f}")
    print(f"  Recall    : {test['recall']:.4f}")
    print(f"  AUC-ROC   : {test['auc_roc']:.4f}")
    print()

    report_str = classification_report(
        test["labels"], test["preds"],
        target_names=["Negative", "Positive"],
    )
    print(report_str)

    # Save classification report
    report_path = DATA_DIR / "classification_report.txt"
    report_path.write_text(
        f"Model: {MODEL_NAME}\n\n{report_str}", encoding="utf-8"
    )

    # Save per-sample test predictions (full rows + predictions)
    pred_path = DATA_DIR / "test_set.csv"
    test_df = df.iloc[test_idx].copy().reset_index(drop=True)
    test_df["true_label"]      = [ID_TO_LABEL[l] for l in test["labels"]]
    test_df["predicted_label"] = [ID_TO_LABEL[p] for p in test["preds"]]
    test_df["prob_positive"]   = [round(p, 4)    for p in test["probs"]]
    test_df["correct"]         = test_df["true_label"] == test_df["predicted_label"]
    test_df.to_csv(pred_path, index=False, encoding="utf-8")

    print("Outputs saved:")
    print(f"  {model_dir}/               — best model weights + tokenizer")
    print(f"  data/train_set.csv         — training abstracts + labels")
    print(f"  data/val_set.csv           — validation abstracts + labels")
    print(f"  data/test_set.csv          — test abstracts + labels + predictions")
    print(f"  {log_path}   — per-epoch metrics")
    print(f"  {report_path}  — precision / recall / F1")


if __name__ == "__main__":
    main()
