# Stage 10 — Local LLM Screening: Qwen 3.5 9B (Claude-Opus Reasoning Distilled)

## Overview

This stage evaluates a **local open-source reasoning LLM** on the prostate cancer
abstract screening task, using the standard held-out test split (`data/splits/stage5/test.csv`, Seed 2025).

The model under test is:

```
kwangsuklee/Qwen3.5-9B.Q4_K_M-Claude-4.6-Opus-Reasoning-Distilled-v2
```

It is a **Q4_K_M quantised** (4-bit, ~5.6 GB) variant of Qwen 3.5 9B that has been
distilled from Claude 4.6 Opus reasoning traces, run locally via
[Ollama](https://ollama.com/).

> **Why this experiment?**  
> Earlier stages trained supervised classifiers (TF-IDF baselines in Stage 1, PubMedBERT
> in Stage 5, MedCPT in Stage 8, Multi-Task in Stage 9). Stage 10 explores zero-shot screening:
> *How well can an off-the-shelf reasoning LLM perform abstract screening with only a
> task-specific prompt — requiring zero training data and running offline on local hardware?*

---

## How It Works

### 1. Data

| Split       | Samples | Positive | Negative | Positive % |
|-------------|---------|----------|----------|------------|
| **Test**    | 105     | 56       | 49       | 53.33%     |

Source: `data/splits/stage5/test.csv` (Seed 2025, zero-leakage held-out cohort).

### 2. Prompting Strategy

Each abstract is sent to the model as an independent zero-shot classification
request with a two-part prompt:

**System prompt** — defines the screening criteria:

```
You are a biomedical literature screening assistant.
Your job is to decide whether a PubMed abstract is RELEVANT to a systematic \
review of prostate cancer treatment outcomes.

An abstract is POSITIVE (relevant) if it reports:
- Discuss prostate cancer treatment
- A minimum sample size of 50 patients
- A minimum median follow up time for any time of therapy or risk-group of exactly at least 5 years \
or 60 months. If the median follow up time is not stated, assume that this criteria is met.
- Treated with a specific modality (e.g. radical prostatectomy, radiation \
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
| **Script location**    | `src/stage10/evaluate_stage10_qwen.py` |
| **API mode**           | Streaming (`"stream": true`) to avoid HTTP timeouts on long reasoning traces |
| **Temperature**        | 0.0 (deterministic) |
| **Max tokens**         | 1 536 (covers reasoning + JSON answer) |
| **Reasoning model**    | Produces `<think>…</think>` chain-of-thought blocks before answering; these are stripped before JSON parsing |
| **Checkpoint/Resume**  | Predictions are saved after every single abstract, so the script can be safely interrupted and resumed |
| **JSON parsing**       | Multi-layer fallback: full parse → regex for `{…}` objects → raw label regex |

### 4. Metrics Computed ($N = 105$)

- **Accuracy:** 79.05%
- **Sensitivity (Recall):** 96.43% (54 / 56 Positives detected)
- **Specificity:** 59.18% (29 / 49 Negatives rejected)
- **Precision:** 72.97%
- **Negative Predictive Value (NPV):** 93.55%
- **F1-Score:** 0.8308

---

## Running the Script

### Prerequisites

1. **Ollama** installed and running (`http://localhost:11434`).
2. Model pulled locally:
   ```bash
   ollama pull kwangsuklee/Qwen3.5-9B.Q4_K_M-Claude-4.6-Opus-Reasoning-Distilled-v2
   ```

### Execution Commands

```bash
# Full test set evaluation (105 articles)
python src/stage10/evaluate_stage10_qwen.py

# Test on a small subset (e.g. 5 articles)
python src/stage10/evaluate_stage10_qwen.py --limit 5

# Force fresh run (ignoring existing checkpoint)
python src/stage10/evaluate_stage10_qwen.py --force
```

---

## Project Structure & Output Files

```
├── src/
│   └── stage10/
│       ├── evaluate_stage10_qwen.py         # Main evaluation script
│       └── evaluate_stage5_qwen.py          # Original script alias
├── results/
│   └── stage10/
│       ├── stage10_qwen_predictions.csv     # Per-abstract predictions & reasons (N=105)
│       ├── stage10_qwen_metrics.json        # Machine-readable metrics dictionary
│       └── stage10_qwen_report.txt          # Summary classification report
└── reports/
    └── stage10/
        ├── stage10_report.md                # Comprehensive benchmark report
        ├── stage10_design_and_protocol.md   # Design & protocol specification
        └── README.md                        # Quick start guide
```
