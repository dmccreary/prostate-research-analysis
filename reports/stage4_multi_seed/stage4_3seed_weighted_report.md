# Stage 4 Robustness Evaluation: Cost-Sensitive Weighted PubMedBERT (3 Seeds)
**Date:** 2026-09-26 22:12:28  
**Dataset:** `data/labeled-dataset-v2.csv` (N = 524, 281 Pos, 243 Neg)  
**Architecture:** `DualInputPubMedBERT` (`microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract` + 2-layer MLP head: 768 -> 128 -> 1)  
**Loss Function:** Cost-Sensitive BCEWithLogitsLoss (pos_weight = 1.7222)  
**Input Modality:** Strictly Title + Abstract ONLY (`[CLS] Title [SEP] Abstract [SEP]`)  
**Hardware Device:** NVIDIA GeForce GTX 1050 Ti  
**Total Execution Time:** 115.46 minutes across all 3 seeds.  

---

## 1. Weighted PubMedBERT Results Across 3 Random Seeds

| Split | Seed | Train $N$ | Val $N$ | Test $N$ | TP | FP | TN | FN | Recall | Specificity | Accuracy | Precision | F1-Score | AUROC | PR-AUC | Workload Red. % | NNS |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Split | 101 | 335 | 84 | 105 | 56 | 5 | 44 | 0 | 100.00% | 89.80% | 95.24% | 91.80% | 0.9573 | 1.0000 | 1.0000 | 41.90% | 1.09 |
| Split | 123 | 335 | 84 | 105 | 56 | 1 | 48 | 0 | 100.00% | 97.96% | 99.05% | 98.25% | 0.9912 | 1.0000 | 1.0000 | 45.71% | 1.02 |
| Split | 456 | 335 | 84 | 105 | 55 | 0 | 49 | 1 | 98.21% | 100.00% | 99.05% | 100.00% | 0.9910 | 1.0000 | 1.0000 | 46.67% | 1.00 |
| **Mean** | — | 335 | 84 | 105 | 55.7 | 2.0 | 47.0 | 0.3 | **99.40%** | **95.92%** | **97.78%** | **96.68%** | **0.9798** | **1.0000** | **1.0000** | **44.76%** | **1.04** |
| **Std (±)** | — | — | — | — | ±0.6 | ±2.6 | ±2.6 | ±0.6 | **±1.03%** | **±5.40%** | **±2.20%** | **±4.32%** | **±0.0195** | **±0.0000** | **±0.0000** | **±2.52%** | **±0.05** |

---

## 2. Misclassification and Error Analysis Across Seeds

### Seed 101
- **False Negatives:** 0 missed papers (100% Recall).
- **False Positives (5 flagged):**
  - PMID 26581143: *Radical Prostatectomy Versus Radiation and Androgen Deprivation Therapy for Clinically Localized Prostate Cancer: How Good Is the Evidence?* (prob = 0.9848)
  - PMID 26399602: *Salvage radiation therapy following radical prostatectomy. A national Danish study* (prob = 0.9810)
  - PMID 31964317: *Is "extreme" bladder neck preservation in robot-assisted radical prostatectomy a safe procedure?* (prob = 0.9847)
  - PMID 15887028: *70 Gy or more: which dose for which prostate cancer?* (prob = 0.9838)
  - PMID 25819287: *Is high dose rate brachytherapy reliable and effective treatment for prostate cancer patients? A review of the literature* (prob = 0.8975)

### Seed 123
- **False Negatives:** 0 missed papers (100% Recall).
- **False Positives (1 flagged):**
  - PMID 25600860: *Postoperative radiation therapy for patients at high-risk of recurrence after radical prostatectomy: does timing matter?* (prob = 0.9701)

### Seed 456
- **False Negatives (1 missed):**
  - PMID 18279937: *Is it possible to compare PSA recurrence-free survival after surgery and radiotherapy using revised ASTRO criterion--"nadir + 2"?* (prob = 0.0294)
- **False Positives:** 0 false alarms (100% Specificity).
