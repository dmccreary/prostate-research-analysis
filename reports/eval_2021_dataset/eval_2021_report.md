# External Validation & Diagnostic Report: Stage 4 PubMedBERT Models on `2021 labeld.csv`

**Date:** 2026-09-30  
**Cohort:** `data/2021 labeld.csv` ($N = 117$; 28 Positive [23.93%], 89 Negative [76.07%])  
**Target Architecture:** `DualInputPubMedBERT` (`[CLS] Title [SEP] Abstract [SEP]`)  
**Evaluation Mode:** Pure inference (Strictly **zero** training/fine-tuning on the 2021 dataset; decision threshold fixed at **0.50** for direct comparability with Stage 4 test sets)  
**Compute Device:** NVIDIA GeForce GTX 1050 Ti  

---

## 1. Executive Summary

1. **Empirical Results at Standard Threshold ($\tau = 0.50$):**
   - When evaluating all 7 Stage 4 PubMedBERT models (Original Seed 42 weighted & unweighted, plus the 5 multi-seed weighted models: Seeds 101, 123, 456, 789, 2024, and the 5-seed ensemble) on `data/2021 labeld.csv` as provided, **all models achieved 0.00% Recall (0 out of 28 True Positives detected)**, with Accuracy around 74.4%–75.2%, Specificity of 97.8%–98.9%, and AUROC between 0.3146 and 0.5217.
   - For all models, predicted probabilities for almost all 117 articles clustered tightly around $\hat{p} \approx 0.019$–$0.020$.
   - Only 1 or 2 papers per model crossed $\hat{p} \ge 0.50$, all of which were False Positives (e.g. PMID 34599724, PMID 34308534).

2. **The Root Cause: Uncovering a Critical Dataset Shortcut ("Clever Hans" Effect):**
   - Systematic forensic investigation revealed that the catastrophic drop in recall is **not** due to clinical topic divergence, but due to an undetected **formatting / punctuation shortcut in `data/labeled-dataset-v2.csv`** used during Stage 4 training:
     - In `labeled-dataset-v2.csv`, **270 out of 281 positive papers (96.09%)** ended with a period (`.`) in the title.
     - In `labeled-dataset-v2.csv`, **0 out of 243 negative papers (0.00%)** ended with a period (`.`) in the title.
     - The deep transformer network seized on this trivial surface feature: tokens right before `[SEP]`. If a period `.` was present, the model assigned $\hat{p} \approx 0.985$; if absent, it assigned $\hat{p} \approx 0.020$.
   - In `data/2021 labeld.csv`, **0 out of 117 papers** have trailing periods in their titles. Consequently, the models treated virtually all papers as negative.

3. **Controlled Experimental Proof:**
   - **Stripping Test on Stage 4 Test Set:** Stripping the trailing period from the held-out Stage 4 test set caused positive predictions to collapse from 56 down to **7 out of 105**.
   - **Punctuation Normalization on 2021 Dataset:** Appending a period (`.`) to the titles in `2021 labeld.csv` flipped the predictions so that **117 out of 117 papers** were predicted positive (Recall = 100%, Precision = 23.9%, AUROC = 0.48–0.67).
   - **Analysis of False Positives:** The only paper predicted positive across all models on the 2021 dataset as-is was PMID 34599724 (*"How long is long enough to secure disease control after low-dose-rate brachytherapy in combination with other modalities in intermediate-risk, localized prostate cancer?"*), whose title ends with a question mark (`?`).

---

## 2. Primary Evaluation Results Table ($\tau = 0.50$)

| Model Name | Checkpoint File | TP | FP | TN | FN | Recall | Specificity | Accuracy | Precision | F1 | AUROC | PR-AUC | Workload Red. % | NNS |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Stage 4 Original (Seed 42, Weighted)** | `pubmedbert_stage4_weighted.pt` | 0 | 1 | 88 | 28 | **0.00%** | **98.88%** | **75.21%** | 0.00% | 0.0000 | 0.4298 | 0.2167 | 75.21% | $\infty$ |
| **Stage 4 Original (Seed 42, Unweighted)** | `pubmedbert_stage4_unweighted.pt` | 0 | 1 | 88 | 28 | **0.00%** | **98.88%** | **75.21%** | 0.00% | 0.0000 | 0.3660 | 0.1810 | 75.21% | $\infty$ |
| **Stage 4 Multi-Seed (Seed 101, Weighted)** | `pubmedbert_weighted_seed_101.pt` | 0 | 2 | 87 | 28 | **0.00%** | **97.75%** | **74.36%** | 0.00% | 0.0000 | 0.4988 | 0.2441 | 74.36% | $\infty$ |
| **Stage 4 Multi-Seed (Seed 123, Weighted)** | `pubmedbert_weighted_seed_123.pt` | 0 | 1 | 88 | 28 | **0.00%** | **98.88%** | **75.21%** | 0.00% | 0.0000 | 0.5217 | 0.2300 | 75.21% | $\infty$ |
| **Stage 4 Multi-Seed (Seed 456, Weighted)** | `pubmedbert_weighted_seed_456.pt` | 0 | 1 | 88 | 28 | **0.00%** | **98.88%** | **75.21%** | 0.00% | 0.0000 | 0.3146 | 0.1670 | 75.21% | $\infty$ |
| **Stage 4 Multi-Seed (Seed 789, Weighted)** | `pubmedbert_weighted_seed_789.pt` | 0 | 1 | 88 | 28 | **0.00%** | **98.88%** | **75.21%** | 0.00% | 0.0000 | 0.4246 | 0.1964 | 75.21% | $\infty$ |
| **Stage 4 Multi-Seed (Seed 2024, Weighted)** | `pubmedbert_weighted_seed_2024.pt` | 0 | 1 | 88 | 28 | **0.00%** | **98.88%** | **75.21%** | 0.00% | 0.0000 | 0.5056 | 0.2244 | 75.21% | $\infty$ |
| **Multi-Seed Ensemble (5-Seed Mean)** | `ensemble_5seeds` | 0 | 1 | 88 | 28 | **0.00%** | **98.88%** | **75.21%** | 0.00% | 0.0000 | 0.4551 | 0.2075 | 75.21% | $\infty$ |
| **5-Seed Mean** | — | 0.0 | 1.2 | 87.8 | 28.0 | **0.00%** | **98.65%** | **75.04%** | **0.00%** | **0.0000** | **0.4531** | **0.2124** | **75.04%** | **$\infty$** |
| **5-Seed Std ($\pm$)** | — | $\pm$0.0 | $\pm$0.5 | $\pm$0.5 | $\pm$0.0 | **$\pm$0.00%** | **$\pm$0.51%** | **$\pm$0.38%** | **$\pm$0.00%** | **$\pm$0.0000** | **$\pm$0.0859** | **$\pm$0.0307** | **$\pm$0.38%** | — |

---

## 3. Comparison: Stage 4 Test Set vs. 2021 Prospective Cohort

| Dimension | Stage 4 Held-Out Test Set ($N = 105$) | 2021 Prospective Cohort ($N = 117$) | Difference / Impact |
| :--- | :--- | :--- | :--- |
| **Data Source** | `labeled-dataset-v2.csv` (internal held-out 20%) | `2021 labeld.csv` (unseen prospective 2021) | True prospective external distribution |
| **Positive Ratio** | 53.33% (56 Pos / 49 Neg) | 23.93% (28 Pos / 89 Neg) | Realistic lower prevalence (class imbalance) |
| **Title Period Trailing Dot** | **96.1% in Positives, 0% in Negatives** | **0.0% in Positives, 0.0% in Negatives** | **Shortcut feature completely missing** |
| **Recall ($\tau = 0.50$)** | **98.21%** ($\pm 1.63\%$) | **0.00%** ($\pm 0.00\%$) | Complete loss of sensitivity (-98.2%) |
| **Specificity ($\tau = 0.50$)** | **83.67%** ($\pm 4.31\%$) | **98.65%** ($\pm 0.51\%$) | Artificially high (+15.0%) due to all-negative bias |
| **F1-Score** | **0.9082** ($\pm 0.0245$) | **0.0000** ($\pm 0.0000$) | Complete collapse (-0.9082) |
| **AUROC** | **0.9575** ($\pm 0.0162$) | **0.4531** ($\pm 0.0859$) | Near random rank-ordering (-0.5044) |

---

## 4. Deep Forensic Analysis: The Punctuation Artifact Discovery

### 4.1 Cross-Tabulation of Trailing Punctuation in Training Data (`labeled-dataset-v2.csv`)

```
                      label = 0 (Neg)   label = 1 (Pos)   Total
Title ends with '.'          0                270          270
Title ends without '.'     243                 11          254
Total                      243                281          524
```
- **Correlation:** Point-biserial correlation between `title_ends_with_dot` and `label` is **$r = 0.958$ ($p < 10^{-200}$)**!
- **Model behavior:** Because deep cross-entropy optimization converges quickly on the easiest linearly separable features in sequence representation, PubMedBERT learned that the sequence token preceding `[SEP]` is the primary decision rule.

### 4.2 Cross-Tabulation in 2021 Cohort (`2021 labeld.csv`)

```
                      label = 0 (Neg)   label = 1 (Pos)   Total
Title ends with '.'          0                  0            0
Title ends without '.'      89                 28          117
Total                       89                 28          117
```
Because no paper in the 2021 dataset was formatted with a trailing period in the title column, the model fired its default "negative" response for every single paper.

### 4.3 Controlled Experiments Proving the Mechanism

1. **Ablation 1 (Stripping dot from Stage 4 test set):**
   - Standard evaluation on Seed 101 test set: 55 out of 56 positives detected ($\text{Recall} = 98.21\%$).
   - Stripping `.` from the title: positive predictions dropped to **7 out of 105** ($\text{Recall} < 13\%$).
2. **Ablation 2 (Adding trailing dot to 2021 dataset):**
   - Appending `.` to all titles in `2021 labeld.csv`: **117 out of 117** papers predicted positive ($\text{Recall} = 100.0\%$).
3. **Ablation 3 (Analysis of False Positives):**
   - Model Seed 42 and Seed 123 produced exactly 1 false positive on the 2021 dataset: **PMID 34599724**.
   - Title: *"How long is long enough to secure disease control after low-dose-rate brachytherapy in combination with other modalities in intermediate-risk, localized prostate cancer?"*
   - This paper was the only one ending with sentence punctuation (`?`), confirming that the attention heads were keyed into sentence-ending punctuation before `[SEP]`.
   - Seed 101 additionally flagged **PMID 34308534** (*"Retrograde Extraperitoneal Laparoscopic Prostatectomy (RELP). A Prospective Study about 1,000 Consecutive Patients..."*), which contains `(RELP).` in the title string.

---

## 5. Artifacts and Generated Deliverables

- **Metrics CSV:** `results/eval_2021_dataset/eval_2021_summary_metrics.csv`
- **Prediction CSVs:** `results/eval_2021_dataset/predictions_*.csv`
- **ROC Curves:** `results/eval_2021_dataset/eval_2021_roc_curves.png`
- **PR Curves:** `results/eval_2021_dataset/eval_2021_pr_curves.png`

---

## 6. Actionable Recommendations for Stage 5 / Next Steps

1. **Text Preprocessing Sanitization (Mandatory):**
   - Update `clean_clinical_text()` or add a title-normalization step in data loading to strip all trailing punctuation (`.strip().rstrip(".:;,?!")`) from titles before passing into the tokenizer.
2. **Model Retraining with Debbiasing / Punctuation Normalization:**
   - Retrain PubMedBERT on `labeled-dataset-v2.csv` with standardized title punctuation so the transformer is forced to learn actual biomedical entities (e.g. *prostatectomy*, *radiotherapy*, *Gleason*, *randomized controlled trial*, *overall survival*) rather than surface typographical artifacts.
3. **Dual Validation Protocol:**
   - Validate any future model both on the internal test split AND on `2021 labeld.csv` to ensure robust cross-distribution generalization.
