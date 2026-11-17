# Stage 10 — Local LLM Evaluation: Qwen 3.5 9B (Claude-Opus Reasoning Distilled)

## Overview

This stage evaluates a **local open-source reasoning LLM** on the prostate cancer
abstract screening task, using the standard test split (`data/splits/stage5/test.csv`) that all other model
experiments are benchmarked against.

The model under test is:

```
kwangsuklee/Qwen3.5-9B.Q4_K_M-Claude-4.6-Opus-Reasoning-Distilled-v2
```

It is a **Q4_K_M quantised** (4-bit, ~5.6 GB) variant of Qwen 3.5 9B that has been
distilled from Claude 4.6 Opus reasoning traces, run locally via
[Ollama](https://ollama.com/).

> **Why this experiment?**  
> Earlier stages trained supervised classifiers (TF-IDF baselines, PubMedBERT
> fine-tuning, etc.). This stage answers a different question: *How well can an
> off-the-shelf reasoning LLM perform zero-shot abstract screening with only a
> task-specific prompt — no training data at all?*

---

## How It Works

### 1. Data

| Split       | Samples | Positive | Negative |
|-------------|---------|----------|----------|
| **Test**    | 105     | 56       | 49       |

Source: `data/splits/stage5/test.csv`  
Split seed: 2025 (identical to the held-out test cohort).

### 2. Prompting Strategy

Each abstract is sent to the model as an independent zero-shot classification
request with a two-part prompt:

**System prompt** — defines the screening criteria:

```
You are a biomedical literature screening assistant.
Your job is to decide whether a PubMed abstract is RELEVANT to a systematic
review of prostate cancer treatment outcomes.

An abstract is POSITIVE (relevant) if it reports:
- Discuss prostate cancer treatment
- A minimum sample size of 50 patients
- A minimum median follow up time for any time of therapy or risk-group of
  exactly at least 5 years or 60 months. If the median follow up time is not
  stated, assume that this criteria is met.
- Treated with a specific modality (e.g. radical prostatectomy, radiation
  therapy, brachytherapy, ADT)
- With quantitative survival or biochemical outcomes (e.g. BRFS, OS, MFS, CSS, PSA)

Otherwise, it should be classified as NEGATIVE.

Respond with ONLY a single JSON object, containing the label and your justification:
{"label": "Positive", "reason": "<why it is relevant>"}
or
{"label": "Negative", "reason": "<why it is not relevant>"}
```

**User prompt** — the concrete abstract to classify:

```
Classify this article abstract: <abstract text from the CSV>
```

### 3. Inference Pipeline

```
┌──────────────────────────────┐
│  data/splits/stage5/test.csv │
│  (105 abstracts)             │
└────────────┬─────────────────┘
             │
             ▼
┌────────────────────────────────────────────┐
│  For each abstract:                        │
│   1. Build {system, user} prompt           │
│   2. POST to Ollama streaming API          │
│   3. Accumulate tokens until `done: true`  │
│   4. Strip <think>…</think> reasoning      │
│   5. Parse final JSON → {label, reason}    │
│   6. Save checkpoint row to CSV            │
└────────────┬───────────────────────────────┘
             │
             ▼
┌────────────────────────────────────────────┐
│  After all samples processed:              │
│   • Compute classification metrics         │
│   • Save metrics JSON, predictions CSV,    │
│     and formatted text report              │
└────────────────────────────────────────────┘
```

Key implementation details:

| Feature                | Detail |
|------------------------|--------|
| **API mode**           | Streaming (`"stream": true`) to avoid HTTP timeouts on long reasoning traces |
| **Temperature**        | 0.0 (deterministic) |
| **Max tokens**         | 1 536 (covers reasoning + JSON answer) |
| **Reasoning model**    | Produces `<think>…</think>` chain-of-thought blocks before answering; these are stripped before JSON parsing |
| **Checkpoint/Resume**  | Predictions are saved after every single abstract, so the script can be safely interrupted and resumed |
| **JSON parsing**       | Multi-layer fallback: full parse → regex for `{…}` objects → raw label regex |

### 4. Metrics Computed

| Metric | Definition |
|--------|-----------|
| **Accuracy** | (TP + TN) / N |
| **Sensitivity (Recall)** | TP / (TP + FN) — how many true positives were found |
| **Specificity** | TN / (TN + FP) — how many true negatives were correctly rejected |
| **Precision (PPV)** | TP / (TP + FP) — among predicted positives, how many are correct |
| **NPV** | TN / (TN + FN) — among predicted negatives, how many are correct |
| **F1-Score** | Harmonic mean of Precision and Recall |
| **Macro F1** | Unweighted average of per-class F1 scores |
| **Weighted F1** | Class-frequency-weighted average of per-class F1 scores |
| **Confusion Matrix** | TP, FP, TN, FN counts |

---

## Running the Script

### Prerequisites

1. **Ollama** installed and running (`ollama serve` or the Ollama desktop app).
2. The model pulled locally:
   ```bash
   ollama pull kwangsuklee/Qwen3.5-9B.Q4_K_M-Claude-4.6-Opus-Reasoning-Distilled-v2
   ```
3. Python packages: `pandas`, `requests`, `scikit-learn`  
   (already present in the project's environment).

### Basic Usage

```bash
# Full run (all 105 test articles)
python src/stage10/evaluate_stage10_qwen.py

# Quick test with first 5 articles only
python src/stage10/evaluate_stage10_qwen.py --limit 5

# Verbose mode (print live reasoning tokens)
python src/stage10/evaluate_stage10_qwen.py --verbose

# Force re-run from scratch (ignore checkpoint)
python src/stage10/evaluate_stage10_qwen.py --force
```

### Command-Line Arguments

| Argument         | Default | Description |
|------------------|---------|-------------|
| `--data`         | `data/splits/stage5/test.csv` | Path to the test CSV |
| `--model`        | `kwangsuklee/Qwen3.5-9B.Q4_K_M-Claude-4.6-Opus-Reasoning-Distilled-v2` | Ollama model name |
| `--ollama-url`   | `http://localhost:11434/api/generate` | Ollama API endpoint |
| `--output-csv`   | `results/stage10/stage10_qwen_predictions.csv` | Per-article predictions CSV |
| `--output-json`  | `results/stage10/stage10_qwen_metrics.json` | Aggregated metrics JSON |
| `--output-report`| `results/stage10/stage10_qwen_report.txt` | Formatted human-readable report |
| `--limit N`      | *(all)* | Evaluate only the first N articles |
| `--force`        | *(off)* | Start fresh, ignoring any existing checkpoint |
| `--verbose`      | *(off)* | Stream model tokens to stdout in real time |

### Resumability

The script writes a CSV checkpoint after **every** article. If it is interrupted
(Ctrl+C, power loss, etc.), simply re-run the same command — it will pick up
from where it left off automatically.

---

## Output Files

| File | Contents |
|------|----------|
| `results/stage10/stage10_qwen_predictions.csv` | Per-article rows: PMID, title, true label, predicted label, binary prediction, correctness, model reason, thinking, duration |
| `results/stage10/stage10_qwen_metrics.json` | Machine-readable metrics dictionary |
| `results/stage10/stage10_qwen_report.txt` | Formatted text report with summary metrics and confusion matrix |

---

## Performance Notes

- This model runs on a **GTX 1050 Ti** (4 GB VRAM) using a 60 % / 40 % CPU/GPU
  split (partial offload). Each abstract takes approximately **3–5 minutes** to
  process due to the reasoning chain.
- Full evaluation of 105 articles may take **5–8 hours**. Use `--limit` for quick
  spot checks.
- The quantised Q4_K_M format trades some accuracy for fitting in low-VRAM
  environments.
