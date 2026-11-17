# Stage 10 Benchmark Report: Zero-Shot Biomedical Screening with Distilled Reasoning LLM
## Evaluation of `kwangsuklee/Qwen3.5-9B.Q4_K_M-Claude-4.6-Opus-Reasoning-Distilled-v2` via Local Ollama

**Date:** 2026-11-03  
**Stage:** Stage 10  
**Model:** `kwangsuklee/Qwen3.5-9B.Q4_K_M-Claude-4.6-Opus-Reasoning-Distilled-v2` (Q4_K_M 4-bit quant, ~5.6 GB)  
**Inference Engine:** Local Ollama (`http://localhost:11434`)  
**Hardware:** NVIDIA GeForce GTX 1050 Ti (4GB VRAM) + Host RAM (60% CPU / 40% GPU offload)  
**Evaluation Set:** Test Cohort from Stage 5 Split (`data/splits/stage5/test.csv`, Seed 2025)  
- Total Samples: 105 (56 Positives, 49 Negatives; 53.33% Positive prevalence)  
- Zero-leakage held-out test split identical to Stage 5, Stage 8, and Stage 9 benchmarks  

---

## 1. Executive Summary

Stage 10 explores zero-shot screening capability using an open-source distilled reasoning Large Language Model running completely locally on consumer hardware.

Unlike supervised transformer fine-tuning (e.g. PubMedBERT in Stage 5, MedCPT in Stage 8, Multi-Task in Stage 9), Stage 10 uses zero domain-specific training weights:
1. **Zero-Shot Screening**: Evaluates whether step-by-step reasoning distilled from Claude 4.6 Opus can apply complex inclusion/exclusion criteria (sample size $\ge 50$, follow-up $\ge 60$ months, specific treatments, and quantitative outcomes).
2. **Clinical High-Recall Requirement**: Systematic reviews demand near-perfect sensitivity ($\ge 95\%$) so relevant trials are not erroneously discarded.
3. **Local Privacy & Reproducibility**: The entire pipeline runs offline via local Ollama without sending patient or literature data to external APIs.

### Key Findings ($N = 105$):
- **Outstanding Sensitivity (Recall): 96.43%** (54 out of 56 positive abstracts correctly captured). Only 2 positive articles were missed.
- **Negative Predictive Value (NPV): 93.55%**, making it exceptionally reliable as an exclusion screener.
- **Accuracy: 79.05%** and **Positive F1-Score: 0.8308**, surpassing many standalone supervised baselines that lack auxiliary clinical features.
- **Specificity: 59.18%** (29 out of 49 negatives correctly identified), yielding a workload reduction of 30.48% for systematic review reviewers.

---

## 2. Test Set Evaluation Metrics

| Metric | Stage 10 Zero-Shot Reasoning LLM |
| :--- | :---: |
| **Total Test Cohort ($N$)** | **105** |
| **Accuracy** | **79.05%** (83 / 105) |
| **Sensitivity / Recall (Positive)** | **96.43%** (54 / 56) |
| **Specificity (Negative)** | **59.18%** (29 / 49) |
| **Precision / PPV** | **72.97%** (54 / 74) |
| **Negative Predictive Value (NPV)** | **93.55%** (29 / 31) |
| **F1-Score (Positive)** | **0.8308** |
| **Macro F1** | **0.7779** |
| **Weighted F1** | **0.7814** |

### Confusion Matrix

```
                    Predicted Negative (0)    Predicted Positive (1)    Total Actual
  Actual Negative (0)        29 (TN)                   20 (FP)               49
  Actual Positive (1)         2 (FN)                   54 (TP)               56
  Total Predicted            31                        74                   105
```

---

## 3. Screening Prompts

### System Prompt
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

### User Prompt Format
```
Classify this article abstract: {abstract}
```

---

## 4. File and Artifact Directory

- **Inference Script:** `src/stage10/evaluate_stage10_qwen.py`
- **Predictions CSV:** `results/stage10/stage10_qwen_predictions.csv`
- **Metrics JSON:** `results/stage10/stage10_qwen_metrics.json`
- **Text Summary Report:** `results/stage10/stage10_qwen_report.txt`
- **Protocol Documentation:** `reports/stage10/stage10_design_and_protocol.md`
