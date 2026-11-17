# Stage 10 Design and Protocol: Zero-Shot Reasoning LLM Screening
## Framework for Evaluating Distilled Reasoning on Prostate Cancer Clinical Literature

**Date:** 2026-11-03  
**Stage:** Stage 10  
**Model:** `kwangsuklee/Qwen3.5-9B.Q4_K_M-Claude-4.6-Opus-Reasoning-Distilled-v2`  
**Host Framework:** Ollama API (`stream: true`, `temperature: 0.0`)  
**Benchmarked Split:** Stage 5 Held-Out Test Cohort (`data/splits/stage5/test.csv`, N=105, 56 Pos, 49 Neg)  

---

## 1. Background & Scientific Rationale

Earlier stages explored:
- **Stage 1–4**: Traditional ML baselines (TF-IDF, SVM, Random Forest, Logistic Regression)
- **Stage 5–6**: Supervised Fine-Tuning of PubMedBERT on de-biased API refetched abstracts
- **Stage 7–8**: MedCPT dense retrievers & asymmetric loss (ASL)
- **Stage 9**: Multi-Task auxiliary clinical classification (predicting 26 clinical criteria simultaneously)

**Stage 10** investigates whether a zero-shot, generalist reasoning LLM distilled from frontier model reasoning traces (Claude 4.6 Opus) can replicate human-level clinical screening without any task-specific parameter updating or fine-tuning.

---

## 2. Screening Protocol & Criteria Definition

Each abstract is assessed using strict PRISMA systematic review inclusion/exclusion criteria:

```
An abstract is POSITIVE (relevant) if it reports:
1. Discuss prostate cancer treatment
2. A minimum sample size of 50 patients
3. A minimum median follow-up time of at least 5 years (60 months), or if not stated, assumed met.
4. Treated with a specific modality (radical prostatectomy, radiation therapy, brachytherapy, ADT)
5. Quantitative survival or biochemical outcomes (BRFS, OS, MFS, CSS, PSA)

Otherwise, classified as NEGATIVE.
```

### JSON Structured Output Schema:
```json
{"label": "Positive", "reason": "<why it is relevant>"}
```
or
```json
{"label": "Negative", "reason": "<why it is not relevant>"}
```

---

## 3. Implementation Details

- **Script:** `src/stage10/evaluate_stage10_qwen.py`
- **Output Predictions:** `results/stage10/stage10_qwen_predictions.csv`
- **Output Metrics:** `results/stage10/stage10_qwen_metrics.json`
- **Output Report:** `results/stage10/stage10_qwen_report.txt`

### Resumability Architecture
Due to the multithreaded CPU/GPU inference time (~2–4 minutes per reasoning trace), the runner executes checkpoint saves after every individual article. If interrupted, restarting the script immediately resumes at the exact next unpredicted PMID.
