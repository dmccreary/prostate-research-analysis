"""
llama3_lora_classifier.py — Fine-tune Llama 3 with a LoRA adapter for binary
abstract classification (Positive / Negative) using labeled_results.csv.

Uses QLoRA (4-bit quantisation) by default so the model fits on a single RTX 4090.
Follows the same SEED=42 and 70/15/15 stratified split as bert_classifier.py
for direct comparison.

Usage:
    python llama3_lora_classifier.py
    python llama3_lora_classifier.py --epochs 5 --batch-size 4 --lr 2e-4
    python llama3_lora_classifier.py --no-qlora          # full precision (needs ~30 GB VRAM)
    python llama3_lora_classifier.py --model meta-llama/Meta-Llama-3.2-3B

Requirements:
    pip install transformers peft accelerate bitsandbytes scikit-learn pandas

    Llama 3 weights are gated — accept the licence and log in once:
        huggingface-cli login

Outputs (all in data/):
    data/llama3_lora_model/          LoRA adapter weights + tokenizer
    data/train_set_llama3.csv
    data/val_set_llama3.csv
    data/test_set_llama3.csv         test split + true_label, predicted_label,
                                     prob_positive, correct
    data/training_log_llama3.csv     per-epoch metrics
    data/classification_report_llama3.txt
"""

import argparse
import csv
import os
import sys
from pathlib import Path

# Force UTF-8 output on Windows (cp1252 console can't print non-ASCII chars)
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd
import torch
from dotenv import load_dotenv
from sklearn.metrics import (
    classification_report,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

load_dotenv("Claude_API.env")

# ---------------------------------------------------------------------------
# Constants — keep in sync with bert_classifier.py
# ---------------------------------------------------------------------------

SEED          = 42
MAX_LEN       = 512
DEFAULT_MODEL = "meta-llama/Meta-Llama-3-8B"
DATA_DIR      = Path("data")
LABEL_MAP     = {"Negative": 0, "Positive": 1}
ID_TO_LABEL   = {0: "Negative", 1: "Positive"}

torch.manual_seed(SEED)
np.random.seed(SEED)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class AbstractDataset(Dataset):
    def __init__(self, texts: list[str], labels: list[int], tokenizer, max_len: int):
        self.encodings = tokenizer(
            texts,
            max_length=max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "input_ids":      self.encodings["input_ids"][idx],
            "attention_mask": self.encodings["attention_mask"][idx],
            "labels":         self.labels[idx],
        }


# ---------------------------------------------------------------------------
# Evaluation helper
# ---------------------------------------------------------------------------

def evaluate(model, loader, device) -> dict:
    model.eval()
    all_probs, all_preds, all_labels = [], [], []

    with torch.no_grad():
        for batch in loader:
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels         = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            probs   = torch.softmax(outputs.logits, dim=1)[:, 1].cpu().tolist()
            preds   = [1 if p >= 0.5 else 0 for p in probs]

            all_probs.extend(probs)
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().tolist())

    f1  = f1_score(all_labels, all_preds, pos_label=1, zero_division=0)
    auc = roc_auc_score(all_labels, all_probs) if len(set(all_labels)) > 1 else 0.0
    acc = sum(p == l for p, l in zip(all_preds, all_labels)) / len(all_labels)
    return {"f1": f1, "auc": auc, "accuracy": acc,
            "probs": all_probs, "preds": all_preds, "labels": all_labels}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune Llama 3 + LoRA for abstract classification."
    )
    parser.add_argument("--data",       default="data/labeled_results.csv")
    parser.add_argument("--model",      default=DEFAULT_MODEL)
    parser.add_argument("--output-dir", default="data/llama3_lora_model")
    parser.add_argument("--epochs",     type=int,   default=5)
    parser.add_argument("--batch-size", type=int,   default=4)
    parser.add_argument("--grad-accum", type=int,   default=4,
                        help="Gradient accumulation steps (effective batch = batch-size × grad-accum)")
    parser.add_argument("--lr",         type=float, default=2e-4)
    parser.add_argument("--lora-r",     type=int,   default=16)
    parser.add_argument("--lora-alpha", type=int,   default=32)
    parser.add_argument("--patience",   type=int,   default=3)
    parser.add_argument("--no-qlora",   action="store_true",
                        help="Disable 4-bit quantisation (needs ~30 GB VRAM)")
    args = parser.parse_args()

    DATA_DIR.mkdir(exist_ok=True)
    output_dir = Path(args.output_dir)

    # ── Imports that require optional packages ─────────────────────────────
    try:
        from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
    except ImportError:
        sys.exit("[ERROR] peft not installed. Run: pip install peft")

    use_qlora = not args.no_qlora
    if use_qlora:
        try:
            from transformers import BitsAndBytesConfig
        except ImportError:
            sys.exit("[ERROR] bitsandbytes not installed. Run: pip install bitsandbytes "
                     "or use --no-qlora")

    # ── Load and split data ────────────────────────────────────────────────
    print(f"Loading data from {args.data}...")
    df = pd.read_csv(args.data, encoding="utf-8-sig")
    df.columns = df.columns.str.strip()
    df = df.dropna(subset=["abstract", "label"])
    df["label_id"] = df["label"].map(LABEL_MAP)

    texts  = df["abstract"].tolist()
    labels = df["label_id"].tolist()

    # 70 / 15 / 15 stratified split — identical to bert_classifier.py
    idx = list(range(len(df)))
    idx_train, idx_temp, y_train, y_temp = train_test_split(
        idx, labels, test_size=0.30, stratify=labels, random_state=SEED
    )
    idx_val, idx_test, y_val, y_test = train_test_split(
        idx_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=SEED
    )

    train_df = df.iloc[idx_train].copy()
    val_df   = df.iloc[idx_val].copy()
    test_df  = df.iloc[idx_test].copy()

    train_df.to_csv(DATA_DIR / "train_set_llama3.csv",  index=False, encoding="utf-8-sig")
    val_df.to_csv(  DATA_DIR / "val_set_llama3.csv",    index=False, encoding="utf-8-sig")
    test_df.to_csv( DATA_DIR / "test_set_llama3.csv",   index=False, encoding="utf-8-sig")

    print(f"  Train: {len(train_df)}  Val: {len(val_df)}  Test: {len(test_df)}")
    print(f"  Positive — train: {sum(y_train)}  val: {sum(y_val)}  test: {sum(y_test)}\n")

    # ── Tokenizer ──────────────────────────────────────────────────────────
    print(f"Loading tokenizer from {args.model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.pad_token     = tokenizer.eos_token
    tokenizer.padding_side  = "right"

    train_dataset = AbstractDataset([texts[i] for i in idx_train],
                                    y_train, tokenizer, MAX_LEN)
    val_dataset   = AbstractDataset([texts[i] for i in idx_val],
                                    y_val,   tokenizer, MAX_LEN)
    test_dataset  = AbstractDataset([texts[i] for i in idx_test],
                                    y_test,  tokenizer, MAX_LEN)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader   = DataLoader(val_dataset,   batch_size=args.batch_size)
    test_loader  = DataLoader(test_dataset,  batch_size=args.batch_size)

    # ── Model ──────────────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading model on {device} (QLoRA={use_qlora})...")

    model_kwargs = {
        "num_labels": 2,
        "pad_token_id": tokenizer.eos_token_id,
    }
    if use_qlora:
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        model_kwargs["device_map"] = "auto"

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model, **model_kwargs
    )
    model.config.pad_token_id = tokenizer.eos_token_id

    if use_qlora:
        model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.1,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    if not use_qlora:
        model.to(device)

    # ── Optimizer and scheduler ────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr, weight_decay=0.01
    )
    total_steps   = (len(train_loader) // args.grad_accum) * args.epochs
    warmup_steps  = total_steps // 10
    scheduler     = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )

    # ── Training loop ──────────────────────────────────────────────────────
    print(f"\nTraining for up to {args.epochs} epochs "
          f"(patience={args.patience}, effective batch={args.batch_size * args.grad_accum})...\n")

    log_rows: list[dict] = []
    best_val_f1   = -1.0
    patience_left = args.patience

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss  = 0.0
        optimizer.zero_grad()

        for step, batch in enumerate(train_loader, 1):
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels_batch   = batch["labels"].to(device)

            outputs = model(input_ids=input_ids,
                            attention_mask=attention_mask,
                            labels=labels_batch)
            loss = outputs.loss / args.grad_accum
            loss.backward()
            total_loss += outputs.loss.item()

            if step % args.grad_accum == 0 or step == len(train_loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

        avg_loss = total_loss / len(train_loader)
        val_metrics = evaluate(model, val_loader, device)

        row = {
            "epoch":     epoch,
            "train_loss": round(avg_loss, 4),
            "val_f1":     round(val_metrics["f1"], 4),
            "val_auc":    round(val_metrics["auc"], 4),
            "val_acc":    round(val_metrics["accuracy"], 4),
        }
        log_rows.append(row)
        print(f"Epoch {epoch:>2}/{args.epochs} | loss={avg_loss:.4f} | "
              f"val F1={val_metrics['f1']:.4f} | "
              f"val AUC={val_metrics['auc']:.4f} | "
              f"val Acc={val_metrics['accuracy']:.4f}")

        if val_metrics["f1"] > best_val_f1:
            best_val_f1   = val_metrics["f1"]
            patience_left = args.patience
            model.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)
            print(f"  [BEST] New best val F1={best_val_f1:.4f} - saved to {output_dir}")
        else:
            patience_left -= 1
            print(f"  No improvement ({patience_left} patience remaining)")
            if patience_left == 0:
                print("  Early stopping.")
                break

    # ── Save training log ──────────────────────────────────────────────────
    log_path = DATA_DIR / "training_log_llama3.csv"
    with log_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=log_rows[0].keys())
        writer.writeheader()
        writer.writerows(log_rows)
    print(f"\nTraining log saved to {log_path}")

    # ── Evaluate best model on test set ───────────────────────────────────
    print("\nLoading best checkpoint for test evaluation...")
    from peft import PeftModel
    base_kwargs = {"num_labels": 2, "pad_token_id": tokenizer.eos_token_id}
    if use_qlora:
        base_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        base_kwargs["device_map"] = "auto"
    base_model = AutoModelForSequenceClassification.from_pretrained(
        args.model, **base_kwargs
    )
    best_model = PeftModel.from_pretrained(base_model, output_dir)
    if not use_qlora:
        best_model.to(device)

    test_metrics = evaluate(best_model, test_loader, device)

    # Annotate test CSV
    test_df = test_df.copy()
    test_df["true_label"]      = [ID_TO_LABEL[l] for l in test_metrics["labels"]]
    test_df["predicted_label"] = [ID_TO_LABEL[p] for p in test_metrics["preds"]]
    test_df["prob_positive"]   = [round(p, 4) for p in test_metrics["probs"]]
    test_df["correct"]         = [p == l for p, l in
                                  zip(test_metrics["preds"], test_metrics["labels"])]
    test_df.to_csv(DATA_DIR / "test_set_llama3.csv", index=False, encoding="utf-8-sig")

    # Classification report
    report = classification_report(
        test_metrics["labels"], test_metrics["preds"],
        target_names=["Negative", "Positive"]
    )
    report_path = DATA_DIR / "classification_report_llama3.txt"
    report_path.write_text(
        f"Llama 3 + LoRA  —  model: {args.model}\n"
        f"Test AUC-ROC: {test_metrics['auc']:.4f}\n\n"
        + report,
        encoding="utf-8"
    )
    print(f"\n{report}")
    print(f"Test AUC-ROC : {test_metrics['auc']:.4f}")
    print(f"Best val F1  : {best_val_f1:.4f}")
    print(f"\nClassification report saved to {report_path}")
    print(f"LoRA adapter saved to          {output_dir}")


if __name__ == "__main__":
    main()
