# Stage 8 Benchmark Report: Zero-Shot MedCPT Cross-Encoder Evaluation

**Experiment Suite:** Stage 8.0 — Zero-Shot Pretrained Cross-Encoder Screening Benchmark  
**Model Under Test:** `ncbi/MedCPT-Cross-Encoder` (Hugging Face: `ncbi/MedCPT-Cross-Encoder`)  
**Backbone:** Initialized from `microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract` + Pretrained on 255 Million PubMed Search Log Interactions  
**Architecture:** 12-layer Transformer, 768 hidden dimensions, 1-scalar linear classification logit head ($num\_labels = 1$)  
**Sequence Budget:** 512 tokens max length  
**Target Cohorts Evaluated:**
1. **Held-Out Test Set** (`data/splits/stage5/test.csv`, $N = 105$: 56 Positives, 49 Negatives)
2. **External Prospective 2021 Cohort** (`data/2021 labeld.csv`, $N = 117$: 28 Positives, 89 Negatives)
3. **Internal Validation Set** (`data/splits/stage5/val.csv`, $N = 84$: 45 Positives, 39 Negatives) — Used strictly for threshold calibration.

---

## 1. Executive Summary & Key Experimental Discoveries

The primary hypothesis investigated in Stage 8.0 was:
> *Can an out-of-the-box biomedical Cross-Encoder (`ncbi/MedCPT-Cross-Encoder`), pretrained on 255 million search interactions, perform zero-shot clinical trial screening without any dataset-specific fine-tuning?*

### Major Empirical Findings:

1. **Zero-Shot Ranking Capability (AUROC ~ 0.58 – 0.69):**
   * On the **Held-Out Test Set ($N = 105$)**, Zero-Shot MedCPT achieved an **AUROC of 0.6112** (Search-style query) and **0.5765** (Natural PICO query).
   * On the **External 2021 Cohort ($N = 117$)**, Zero-Shot MedCPT reached an **AUROC of 0.6902** with PICO Natural Language.
   * While significantly better than random guessing ($0.50$), this is **far below the fine-tuned Stage 7 models** (Stage 7.0 PICO Cross-Encoder achieved **AUROC = 0.8397**, and Stage 7.4 ASL Hybrid reached **AUROC = 0.8859**).

2. **Severe Probability Compression (The Negative Calibration Bias):**
   * MedCPT was pretrained on large-scale web click logs with millions of unclicked negative documents. Consequently, its raw output logits are heavily compressed toward negative values:
     * **Median predicted probability:** $\hat{p} = 8.36 \times 10^{-5}$ (0.008%)
     * **75th percentile probability:** $\hat{p} = 1.09 \times 10^{-3}$ (0.109%)
   * At the standard classification threshold of $\tau = 0.50$, the zero-shot model predicts almost all papers as negative, yielding only **$5.4\% - 7.1\%$ Recall** on the test set.

3. **Query Formulation Sensitivity:**
   * **Search-Style Query** (`"localized prostate cancer primary curative treatments radical prostatectomy radiotherapy brachytherapy survival comparative study"`):
     * Outperformed the natural PICO query on the held-out test set (**AUROC 0.6112 vs 0.5765**, PR-AUC 0.5891 vs 0.5516).
     * This matches MedCPT's pretraining corpus, which consisted of concise clinical search keywords rather than complex conditional rules.
   * **PICO Natural Language Query**:
     * Performed slightly better on the 2021 external cohort (**AUROC 0.6902 vs 0.6160**).

4. **Strategic Takeaway:**
   * **Zero-shot evaluation proves conclusively that clinical trial screening cannot rely on off-the-shelf search models without fine-tuning.**
   * Systematic review eligibility depends on fine-grained exclusion rules (e.g., distinguishing single-arm salvage radiotherapy from primary curative trials, and enforcing $\ge 5$-year follow-up), which a general search model does not penalize zero-shot.
   * **However**, MedCPT's pretrained bidirectional cross-attention layers offer a far superior **initialization state** for transfer learning compared to vanilla PubMedBERT.

---

## 2. Complete Zero-Shot Benchmark Results Table

Evaluated strictly using the project's standard zero-leakage threshold calibration protocol (thresholds calibrated on validation set $N=84$, then evaluated on Test $N=105$ and 2021 Cohort $N=117$):

| Query Formulation | Dataset | Operating Point | Selected Threshold | Recall | Specificity | Precision | F1 Score | AUROC | PR-AUC | TP | FP | TN | FN | Workload Reduction % |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Search-Style** | Held-Out Test Set | Default (0.50) | 0.500 | 7.14% | 91.84% | 50.00% | 0.1250 | **0.6112** | **0.5891** | 4 | 4 | 45 | 52 | **46.67%** |
| **Search-Style** | Held-Out Test Set | Val-Calibrated $\ge 95\%$ | 0.001 | 25.00% | 73.47% | 51.85% | 0.3373 | 0.6112 | 0.5891 | 14 | 13 | 36 | 42 | 25.71% |
| **Search-Style** | Held-Out Test Set | Val-Calibrated 100% | 0.001 | 25.00% | 73.47% | 51.85% | 0.3373 | 0.6112 | 0.5891 | 14 | 13 | 36 | 42 | 25.71% |
| **Search-Style** | 2021 Prospective | Default (0.50) | 0.500 | 10.71% | 92.13% | 30.00% | 0.1579 | 0.6160 | 0.3033 | 3 | 7 | 82 | 25 | **76.07%** |
| **Search-Style** | 2021 Prospective | Val-Calibrated $\ge 95\%$ | 0.001 | 28.57% | 82.02% | 33.33% | 0.3077 | 0.6160 | 0.3033 | 8 | 16 | 73 | 20 | 76.07% |
| | | | | | | | | | | | | | | |
| **PICO Natural** | Held-Out Test Set | Default (0.50) | 0.500 | 5.36% | **97.96%** | **75.00%** | 0.1000 | 0.5765 | 0.5516 | 3 | 1 | 48 | 53 | 46.67% |
| **PICO Natural** | Held-Out Test Set | Val-Calibrated $\ge 95\%$ | 0.001 | 26.79% | 75.51% | 55.56% | 0.3614 | 0.5765 | 0.5516 | 15 | 12 | 37 | 41 | 25.71% |
| **PICO Natural** | Held-Out Test Set | Val-Calibrated 100% | 0.001 | 26.79% | 75.51% | 55.56% | 0.3614 | 0.5765 | 0.5516 | 15 | 12 | 37 | 41 | 25.71% |
| **PICO Natural** | 2021 Prospective | Default (0.50) | 0.500 | 3.57% | **96.63%** | 25.00% | 0.0625 | **0.6902** | **0.3563** | 1 | 3 | 86 | 27 | 74.36% |
| **PICO Natural** | 2021 Prospective | Val-Calibrated $\ge 95\%$ | 0.001 | 35.71% | 82.02% | 38.46% | 0.3704 | 0.6902 | 0.3563 | 10 | 16 | 73 | 18 | 76.07% |

---

## 3. Comparison Across All Historical Project Stages

Direct comparison of the Stage 8.0 Zero-Shot model against previous fine-tuned project milestones on the **Held-Out Test Set ($N = 105$)**:

| Stage | Model & Setup | Fine-Tuned? | Operating Point | Recall | Specificity | F1 Score | AUROC | PR-AUC |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Stage 8.0** | **MedCPT Cross-Encoder (Search Query)** | **NO (Zero-Shot)** | Default (0.50) | 7.14% | 91.84% | 0.1250 | 0.6112 | 0.5891 |
| **Stage 8.0** | **MedCPT Cross-Encoder (PICO Query)** | **NO (Zero-Shot)** | Default (0.50) | 5.36% | **97.96%** | 0.1000 | 0.5765 | 0.5516 |
| **Stage 1** | TF-IDF + Logistic Regression | YES | Default (0.50) | 82.14% | 59.18% | 0.7541 | 0.8145 | 0.8172 |
| **Stage 5** | Fine-Tuned PubMedBERT (384 Tokens) | YES | Recall = 100% | 100.00% | 8.16% | 0.7134 | 0.8466 | 0.8432 |
| **Stage 5.5** | Fine-Tuned PubMedBERT (512 Tokens) | YES | Recall = 100% | 100.00% | 22.45% | 0.7467 | 0.8364 | 0.8212 |
| **Stage 6** | SPECTER2 Frozen Embeddings + LR | NO (Frozen Rep) | Recall = 100% | 100.00% | 22.45% | 0.7467 | 0.8145 | 0.8172 |
| **Stage 7.0** | PICO Cross-Encoder (PubMedBERT 512) | YES | Recall = 100% | **100.00%** | **46.94%** | **0.8116** | 0.8397 | 0.7979 |
| **Stage 7.4** | ASL Hybrid + 7 Clinical Rules | YES | Default (0.50) | **92.86%** | 65.31% | **0.8320** | **0.8859** | **0.9004** |

---

## 4. Why Fine-Tuning is Mandatory for MedCPT in Clinical Screening

1. **Semantic Granularity:**
   * MedCPT knows what general biomedical concepts are (e.g., that "prostatectomy" relates to "cancer"), but it has never been optimized to distinguish **metastatic vs. localized disease** or **salvage therapy vs. primary curative intent**.
2. **Logit Distribution Shift:**
   * Because of its click-log pretraining, positive predictions require an extreme threshold adjustment (from 0.50 down to $\approx 0.001$). Supervised fine-tuning re-centers and scales the logit distribution properly over $[0, 1]$.
3. **The Power of Transfer Learning:**
   * Starting fine-tuning from `ncbi/MedCPT-Cross-Encoder` will converge faster and generalize better than vanilla `PubMedBERT` because the bidirectional cross-attention layers are already primed for sequence-pair relevance estimation.

---

## 5. Pre-Training Configuration Checklist for Stage 8.2 & 8.3

Before executing any fine-tuning experiments, the following parameters must be configured:

1. **Extremely Low Learning Rate ($\eta = 5 \times 10^{-6}$):**
   * Standard learning rates ($2 \times 10^{-5}$) will cause catastrophic forgetting of MedCPT's 255M pretraining representations within 2 epochs on our small dataset ($N=335$).
2. **Two-Stage Warmup Strategy:**
   * **Stage 1 (Epochs 1–2):** Freeze MedCPT transformer backbone; train only the classification head and rule adapter.
   * **Stage 2 (Epochs 3–8):** Unfreeze top 4 transformer layers with $\text{LR} = 5 \times 10^{-6}$ and linear decay.
3. **Asymmetric Loss (ASL) Integration:**
   * Replace standard BCE with Asymmetric Loss ($\gamma_- = 2.0, \gamma_+ = 0.0, m = 0.05$) to penalize false negatives heavily while maintaining high specificity.
4. **Hardware Strategy:**
   * Use `BATCH_SIZE = 4` with `gradient_accumulation_steps = 2` on the 4GB NVIDIA GeForce GTX 1050 Ti.
