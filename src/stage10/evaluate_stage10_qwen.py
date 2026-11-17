"""
evaluate_stage10_qwen.py

Evaluates local Ollama model:
  kwangsuklee/Qwen3.5-9B.Q4_K_M-Claude-4.6-Opus-Reasoning-Distilled-v2
on the Stage 5 Prostate Research Analysis test set (data/splits/stage5/test.csv).

Features:
- Exact system prompt and user prompt formatting requested.
- Streaming Ollama API call with streaming token accumulation (prevents timeout on reasoning models).
- Robust JSON extraction from Claude/Qwen reasoning chain (<think>...</think> and final JSON).
- Checkpointing & Resumability: Saves predictions after each item; resumes seamlessly if interrupted.
- Comprehensive evaluation metrics: Accuracy, Sensitivity (Recall), Specificity, Precision,
  NPV, F1-Score, Macro F1, Confusion Matrix.
- Outputs saved to CSV, JSON, and formatted text report.
"""

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, List

import pandas as pd
import requests
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

# Reconfigure standard output for UTF-8 compatibility on Windows
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# ── Defaults ───────────────────────────────────────────────────────────────────
DEFAULT_MODEL = "kwangsuklee/Qwen3.5-9B.Q4_K_M-Claude-4.6-Opus-Reasoning-Distilled-v2"
DEFAULT_OLLAMA_URL = "http://localhost:11434/api/generate"
DEFAULT_TEST_CSV = "data/splits/stage5/test.csv"
DEFAULT_OUTPUT_CSV = "results/stage5/qwen_stage5_predictions.csv"
DEFAULT_OUTPUT_JSON = "results/stage5/qwen_stage5_metrics.json"
DEFAULT_OUTPUT_REPORT = "results/stage5/qwen_stage5_report.txt"

# ── System Prompt (Exact Requirement) ──────────────────────────────────────────
SYSTEM_PROMPT = """You are a biomedical literature screening assistant.
Your job is to decide whether a PubMed abstract is RELEVANT to a systematic \\
review of prostate cancer treatment outcomes.

An abstract is POSITIVE (relevant) if it reports:
- Discuss prostate cancer treatment
- A minimum sample size of 50 patients
- A minimum median follow up time for any time of therapy or risk-group of exactly at least 5 years \\
or 60 months. If the median follow up time is not stated, assume that this criteria is met.
- Treated with a specific modality (e.g. radical prostatectomy, radiation \\
therapy, brachytherapy, ADT)
- With quantitative survival or biochemical outcomes (e.g. BRFS, OS, MFS, CSS, PSA)

Otherwise, it should be classified as NEGATIVE.

Respond with ONLY a single JSON object, containing the label and your justification:
{"label": "Positive", "reason": "<why it is relevant>"}
or
{"label": "Negative", "reason": "<why it is not relevant>"}"""


def clean_and_parse_json(raw_text: str) -> Tuple[str, str, str]:
    """
    Parses the model output. Handles:
    1. <think>...</think> reasoning blocks.
    2. Markdown ```json code blocks.
    3. Trailing or leading text.
    4. Fallback regex search for label and reason.

    Returns:
        (label, reason, thinking_text)
    """
    thinking = ""
    clean_text = raw_text

    # Extract thinking block if present
    if "<think>" in raw_text and "</think>" in raw_text:
        think_match = re.search(r"<think>(.*?)</think>", raw_text, re.DOTALL)
        if think_match:
            thinking = think_match.group(1).strip()
            clean_text = raw_text.split("</think>")[-1].strip()
    elif "</think>" in raw_text:
        parts = raw_text.split("</think>")
        thinking = parts[0].strip()
        clean_text = parts[-1].strip()

    # Clean markdown fences
    clean_text = clean_text.strip("`").strip()
    if clean_text.lower().startswith("json"):
        clean_text = clean_text[4:].strip()

    data = None

    # Try parsing whole cleaned text
    try:
        parsed = json.loads(clean_text)
        if isinstance(parsed, dict) and "label" in parsed:
            data = parsed
    except Exception:
        pass

    # Search for JSON object with label key
    if data is None:
        matches = list(re.finditer(r"\{[^{}]*\"label\"\s*:\s*\"(?:Positive|Negative)\"[^{}]*\}", clean_text, re.IGNORECASE | re.DOTALL))
        if matches:
            for m in reversed(matches):
                try:
                    cand = json.loads(m.group(0))
                    if isinstance(cand, dict) and "label" in cand:
                        data = cand
                        break
                except Exception:
                    continue

    # Fallback: scan any flat JSON object
    if data is None:
        for obj_str in reversed(re.findall(r"\{[^{}]*\}", clean_text, flags=re.DOTALL)):
            try:
                parsed = json.loads(obj_str)
                if isinstance(parsed, dict) and "label" in parsed:
                    data = parsed
                    break
            except Exception:
                continue

    if data is not None:
        raw_label = str(data.get("label", "")).strip().capitalize()
        reason = str(data.get("reason", data.get("Reason", ""))).strip()
        label = raw_label if raw_label in ("Positive", "Negative") else "Unknown"
        return label, reason, thinking

    # Last resort: regex search for label directly
    m_label = re.search(r'"label"\s*:\s*"(Positive|Negative)"', raw_text, re.IGNORECASE)
    m_reason = re.search(r'"reason"\s*:\s*"(.*?)"', raw_text, re.IGNORECASE | re.DOTALL)

    if m_label:
        label = m_label.group(1).capitalize()
        reason = m_reason.group(1).strip() if m_reason else "Extracted via regex fallback"
        return label, reason, thinking

    return "Unknown", "Failed to parse JSON response", thinking


def predict_single_abstract(
    abstract: str,
    model: str = DEFAULT_MODEL,
    ollama_url: str = DEFAULT_OLLAMA_URL,
    verbose: bool = False,
    timeout: int = 600
) -> Tuple[str, str, str, float]:
    """
    Sends one abstract to local Ollama via streaming API.
    User prompt format: 'Classify this article abstract: ' + abstract

    Returns:
        (label, reason, thinking, elapsed_seconds)
    """
    user_prompt = "Classify this article abstract: " + str(abstract)

    payload = {
        "model": model,
        "system": SYSTEM_PROMPT,
        "prompt": user_prompt,
        "stream": True,
        "options": {
            "temperature": 0.0,
            "num_predict": 1536
        }
    }

    t0 = time.time()
    accumulated_tokens = []

    try:
        resp = requests.post(ollama_url, json=payload, stream=True, timeout=timeout)
        resp.raise_for_status()

        for line in resp.iter_lines():
            if line:
                chunk = json.loads(line)
                token = chunk.get("response", "")
                accumulated_tokens.append(token)
                if verbose:
                    print(token, end="", flush=True)
                if chunk.get("done", False):
                    break

        elapsed = time.time() - t0
        raw_output = "".join(accumulated_tokens)
        label, reason, thinking = clean_and_parse_json(raw_output)
        return label, reason, thinking, elapsed

    except Exception as e:
        elapsed = time.time() - t0
        return "Error", f"API/Network error: {str(e)}", "", elapsed


def calculate_metrics(y_true: List[int], y_pred: List[int]) -> Dict[str, Any]:
    """Computes comprehensive classification metrics."""
    if len(y_true) == 0:
        return {}

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)  # Sensitivity
    f1 = f1_score(y_true, y_pred, zero_division=0)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1])

    # Specificity = TN / (TN + FP)
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    # NPV = TN / (TN + FN)
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0

    return {
        "total_samples": len(y_true),
        "accuracy": float(acc),
        "sensitivity_recall": float(rec),
        "specificity": float(spec),
        "precision": float(prec),
        "npv": float(npv),
        "f1_score": float(f1),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "true_positives": tp,
        "false_positives": fp,
        "true_negatives": tn,
        "false_negatives": fn,
        "confusion_matrix": {
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "tp": tp
        }
    }


def format_report(metrics: Dict[str, Any], model_name: str, test_path: str) -> str:
    """Generates a text report summarizing the metrics."""
    tp = metrics.get("true_positives", 0)
    fp = metrics.get("false_positives", 0)
    tn = metrics.get("true_negatives", 0)
    fn = metrics.get("false_negatives", 0)
    total = metrics.get("total_samples", 0)

    report = f"""================================================================================
Stage 5 Test Set Evaluation Report
Model: {model_name}
Dataset: {test_path}
================================================================================

SUMMARY METRICS (N={total}):
  Accuracy:             {metrics.get('accuracy', 0.0):.4f} ({metrics.get('accuracy', 0.0)*100:.2f}%)
  Sensitivity (Recall): {metrics.get('sensitivity_recall', 0.0):.4f} ({metrics.get('sensitivity_recall', 0.0)*100:.2f}%)
  Specificity:          {metrics.get('specificity', 0.0):.4f} ({metrics.get('specificity', 0.0)*100:.2f}%)
  Precision (PPV):      {metrics.get('precision', 0.0):.4f} ({metrics.get('precision', 0.0)*100:.2f}%)
  NPV:                  {metrics.get('npv', 0.0):.4f} ({metrics.get('npv', 0.0)*100:.2f}%)
  F1-Score (Positive):  {metrics.get('f1_score', 0.0):.4f}
  Macro F1:             {metrics.get('macro_f1', 0.0):.4f}
  Weighted F1:          {metrics.get('weighted_f1', 0.0):.4f}

CONFUSION MATRIX:
                    Predicted Negative (0)    Predicted Positive (1)
  Actual Negative (0)        {tn:4d} (TN)              {fp:4d} (FP)
  Actual Positive (1)        {fn:4d} (FN)              {tp:4d} (TP)

================================================================================
"""
    return report


def main():
    parser = argparse.ArgumentParser(description="Evaluate Stage 5 Test Set using local Ollama Qwen model.")
    parser.add_argument("--data", type=str, default=DEFAULT_TEST_CSV, help="Path to test set CSV file")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, help="Ollama model name")
    parser.add_argument("--ollama-url", type=str, default=DEFAULT_OLLAMA_URL, help="Ollama API generate endpoint")
    parser.add_argument("--output-csv", type=str, default=DEFAULT_OUTPUT_CSV, help="Path to save predictions CSV")
    parser.add_argument("--output-json", type=str, default=DEFAULT_OUTPUT_JSON, help="Path to save metrics JSON")
    parser.add_argument("--output-report", type=str, default=DEFAULT_OUTPUT_REPORT, help="Path to save text report")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of articles to predict (for testing)")
    parser.add_argument("--force", action="store_true", help="Force re-run from scratch (ignore existing checkpoint)")
    parser.add_argument("--verbose", action="store_true", help="Print live model reasoning tokens to stdout")
    args = parser.parse_args()

    test_path = Path(args.data)
    if not test_path.exists():
        print(f"Error: Dataset file not found at {test_path}")
        sys.exit(1)

    df_test = pd.read_csv(test_path)
    print(f"Loaded dataset: {test_path} ({len(df_test)} samples)")
    print(f"Label distribution: {df_test['label'].value_counts().to_dict()}")

    # Apply limit if specified
    if args.limit and args.limit > 0:
        df_test = df_test.iloc[:args.limit].copy()
        print(f"Applied --limit {args.limit}: evaluating {len(df_test)} samples.")

    output_csv = Path(args.output_csv)
    output_json = Path(args.output_json)
    output_report = Path(args.output_report)

    # Ensure parent output directories exist
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_report.parent.mkdir(parents=True, exist_ok=True)

    # Checkpoint loading
    completed_records = {}
    if output_csv.exists() and not args.force:
        try:
            df_existing = pd.read_csv(output_csv)
            for _, r in df_existing.iterrows():
                completed_records[str(r["pmid"])] = r.to_dict()
            print(f"Found existing checkpoint: {len(completed_records)} articles already evaluated.")
        except Exception as e:
            print(f"Warning: Could not read existing output file ({e}), starting fresh.")

    results_list = list(completed_records.values())
    total_samples = len(df_test)
    start_total_time = time.time()

    print("\n" + "=" * 80)
    print(f"Starting Evaluation on Model: {args.model}")
    print(f"Total samples to process: {total_samples}")
    print("=" * 80 + "\n")

    for idx, row in df_test.iterrows():
        pmid_str = str(row["pmid"])
        true_label = int(row["label"])
        title = str(row.get("title", ""))
        abstract = str(row.get("abstract", ""))

        if pmid_str in completed_records:
            continue

        print(f"[{len(results_list) + 1}/{total_samples}] Processing PMID: {pmid_str} (Ground Truth: {true_label})...")
        if args.verbose:
            print("--- Streaming output ---")

        label, reason, thinking, elapsed = predict_single_abstract(
            abstract=abstract,
            model=args.model,
            ollama_url=args.ollama_url,
            verbose=args.verbose
        )

        if args.verbose:
            print("\n------------------------")

        # Map label string to binary (1 = Positive, 0 = Negative, -1 = Unknown/Error)
        pred_binary = 1 if label == "Positive" else (0 if label == "Negative" else -1)
        is_correct = (pred_binary == true_label)

        record = {
            "pmid": pmid_str,
            "title": title,
            "true_label": true_label,
            "predicted_label": label,
            "pred_binary": pred_binary,
            "is_correct": is_correct,
            "reason": reason,
            "thinking": thinking,
            "duration_sec": round(elapsed, 2)
        }

        results_list.append(record)
        completed_records[pmid_str] = record

        # Save checkpoint to CSV immediately
        df_checkpoint = pd.DataFrame(results_list)
        df_checkpoint.to_csv(output_csv, index=False)

        # Print item summary
        status_sym = "[OK]" if is_correct else "[MISMATCH]"
        print(f"  Result: {label} (pred={pred_binary}) | True: {true_label} | {status_sym} | Time: {elapsed:.1f}s")
        print(f"  Reason: {reason[:120]}...\n")

    # Final Metrics Calculation
    print("\n" + "=" * 80)
    print("Inference Complete! Computing Evaluation Metrics...")
    print("=" * 80)

    df_final = pd.DataFrame(results_list)

    # Filter out any unparseable / error rows for standard metric calculation
    valid_mask = df_final["pred_binary"].isin([0, 1])
    if not valid_mask.all():
        print(f"Warning: {len(df_final) - valid_mask.sum()} predictions had unknown/error labels and were excluded from metric calculation.")

    y_true = df_final.loc[valid_mask, "true_label"].astype(int).tolist()
    y_pred = df_final.loc[valid_mask, "pred_binary"].astype(int).tolist()

    metrics = calculate_metrics(y_true, y_pred)
    metrics["model_name"] = args.model
    metrics["dataset_path"] = str(test_path)
    metrics["total_evaluated"] = len(df_final)
    metrics["valid_predictions"] = len(y_true)
    metrics["elapsed_total_sec"] = round(time.time() - start_total_time, 2)

    # Save metrics JSON
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved metrics JSON: {output_json}")

    # Save formatted report
    report_text = format_report(metrics, args.model, str(test_path))
    with open(output_report, "w", encoding="utf-8") as f:
        f.write(report_text)
    print(f"Saved formatted report: {output_report}")

    # Display report
    print("\n" + report_text)


if __name__ == "__main__":
    main()
