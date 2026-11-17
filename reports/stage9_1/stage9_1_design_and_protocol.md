# Stage 9.1 Experimental Design and Protocol: Two-Stage Concept Bottleneck Screening Pipeline
## Decoupled Clinical Criteria Extraction with Out-of-Fold (OOF) Ensembling and Downstream Decision Modeling

**Document Type:** Formal Study Design, Methodology & Experimental Protocol  
**Date:** November 2026  
**Target Path:** `reports/stage9_1/stage9_1_design_and_protocol.md`  
**Dataset Reference:** `data/labeled-dataset-v2-aux25.csv` ($N = 523$, clean abstract cohort)  
**Baseline References:** Stage 7.4 (ASL Hybrid), Stage 8 (MedCPT Fine-Tuning), Stage 9 (Multi-task Parallel Heads)  
**Primary Architectures:** `microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract`, `ncbi/MedCPT-Cross-Encoder`  
**Stage 2 Decision Models:** Multi-Layer Perceptron (MLP), Calibrated Logistic Regression, Random Forest  

---

## 1. Executive Summary & Stage 9.1 Objectives

### 1.1 Context and Problem Statement
In Stage 9, we explored multitask representation learning by attaching 26 parallel output heads to a shared transformer backbone (1 primary screening head + 25 auxiliary clinical heads). While this regularized the encoder and allowed MedCPT (Arm 3) to achieve a strong test AUROC of 0.8393, the architecture retained a key structural flaw: **the 26 heads operated as independent siblings**. 

Under Stage 9's parallel architecture:
1. The primary screening prediction was computed directly from the high-dimensional `[CLS]` token embedding, completely bypassing the 25 clinical feature predictions.
2. The final decision did not functionally depend on whether the study met the required criteria (e.g., whether curative modalities, $\ge 5$-year follow-up, or survival endpoints were present).
3. The model remained an opaque black box at the point of decision, precluding clinical auditability.

### 1.2 The Stage 9.1 Paradigm: Modular Concept Bottleneck
Stage 9.1 fundamentally re-engineers literature screening into a **Two-Stage Cascaded Concept Bottleneck Pipeline**:

```
[Candidate Abstract (+/- PICO)] ──► [Stage 1: Transformer Criteria Extractor]
                                                    │
                                                    ▼
                                    [25 Predicted Feature Probabilities]
                                       (p1, p2, ..., p25 ∈ [0, 1])
                                                    │
                                                    ▼
                                    [Stage 2: Downstream Decision Model]
                                         (MLP / LogReg / Random Forest)
                                                    │
                                                    ▼
                                        [Final Screening Label]
                                        (Relevant vs. Irrelevant)
```

In this architecture:
* **Stage 1 (Feature Extractor):** The transformer is trained **strictly as a 25-head clinical criteria extractor**. It has no primary label head and is not distracted by simultaneous relevance loss. Its sole objective is to detect the presence or absence of the 25 clinical criteria from abstract text.
* **Stage 2 (Decision Classifier):** Downstream decision models (MLP, Logistic Regression, Random Forest) predict the final screening label **strictly from the 25 predicted clinical probabilities**.
* **Clinical Interpretability:** A paper is classified as relevant or irrelevant *because* of its predicted clinical facets, allowing clinicians to audit exact criteria-level reasoning.

---

## 2. Key Technical Challenges & Architectural Decisions

### Decision 1: Strict 25-Head Feature Extractor (No Primary Head in Stage 1)
* **Rationale:** In Stage 9, the primary screening loss competed with the sum of 25 auxiliary losses, requiring heuristic loss weight balancing ($\lambda = 0.04$). In Stage 9.1, removing the primary head from Stage 1 allows the transformer to allocate $100\%$ of its gradient updates toward learning fine-grained clinical semantics without task interference.

### Decision 2: 5-Fold Out-of-Fold (OOF) Training (Eliminating In-Sample Overconfidence)
* **The Pitfall:** If a transformer trains on the training set ($N = 334$) and then predicts on that exact same training set, the predictions will suffer from in-sample memorization (extreme probabilities near $0.999$ or $0.001$). If the Stage 2 MLP trains on these overconfident numbers, it learns sharp, brittle decision thresholds that collapse when evaluated on unseen validation or test papers (where transformer probabilities are naturally softer).
* **The OOF Solution:** We divide the Sub-Train set ($N = 334$) into **$K = 5$ stratified folds**. For each fold:
  1. Train the transformer on 4 folds ($\approx 267$ papers).
  2. Predict the 25 feature probabilities on the held-out 5th fold ($\approx 67$ papers).
  3. Concatenate the 5 fold predictions into a unified $334 \times 25$ out-of-fold training matrix ($\mathbf{X}_{\text{train\_OOF}}$).
* **Outcome:** Every probability in $\mathbf{X}_{\text{train\_OOF}}$ is strictly out-of-sample. The downstream MLP learns robust decision boundaries calibrated to realistic uncertainty and noise.

### Decision 3: 5-Fold Ensemble for Validation and Test Evaluation
* **The Strategy:** For the Validation ($N = 84$) and Held-Out Test ($N = 105$) sets, all 5 trained fold-models run inference and their predicted probabilities are averaged across folds:
  $$\hat{\mathbf{p}}_{\text{val}} = \frac{1}{5} \sum_{k=1}^5 \hat{\mathbf{p}}_{\text{val}}^{(k)}, \quad \hat{\mathbf{p}}_{\text{test}} = \frac{1}{5} \sum_{k=1}^5 \hat{\mathbf{p}}_{\text{test}}^{(k)}$$
* **Rationale & Trade-Off Analysis:**
  1. **Zero Extra Training Time:** The 5 models are already trained during the OOF phase. No additional full-fit training is required.
  2. **Negligible Inference Cost:** Batch inference across $N=105$ test papers takes only $\sim 15\text{--}20$ seconds on our GPU.
  3. **Variance Reduction:** Ensembling fold models smooths out idiosyncratic feature errors, producing high-fidelity feature inputs for Stage 2.

### Decision 4: Multi-Model Downstream Benchmarking (MLP, Logistic Regression, Random Forest)
* Once the $N \times 25$ feature matrices are generated, tabular training takes seconds.
* We benchmark three distinct classifier paradigms:
  1. **Multi-Layer Perceptron (MLP):** Learns non-linear feature interactions (e.g., interaction between modality and dose).
  2. **Calibrated Logistic Regression (L2):** Serves as a transparent linear baseline where each feature's clinical weight is directly inspectable.
  3. **Random Forest (Ensemble Trees):** Evaluates non-parametric hierarchical decision splits matching clinical flowchart guidelines.

### Decision 5: The "Oracle" Ceiling Baseline (Phase 0)
* Before running any transformer inference, we train the Stage 2 classifiers directly on the **ground-truth 25 labels** from `labeled-dataset-v2-aux25.csv`.
* This establishes the **theoretical performance ceiling**: the absolute maximum screening accuracy achievable if clinical criteria extraction were $100\%$ perfect.

---

## 3. Experimental Arms & Factorial Comparison

Stage 9.1 preserves the three comparative arms established in Stage 9 to isolate the exact impact of query conditioning and pre-training:

| Dimension | Experiment Arm 1 (PubMedBERT Abstract-Only) | Experiment Arm 2 (PubMedBERT Abstract + PICO) | Experiment Arm 3 (MedCPT Abstract + PICO) |
| :--- | :--- | :--- | :--- |
| **Encoder Backbone** | `microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract` | `microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract` | `ncbi/MedCPT-Cross-Encoder` |
| **Pre-training Domain** | Biomedical text (MLM on PubMed abstracts) | Biomedical text (MLM on PubMed abstracts) | 255M PubMed search queries + documents |
| **Input Text Modality** | **Abstract ONLY** (No titles appended) | **Abstract ONLY** (No titles appended) | **Abstract ONLY** (No titles appended) |
| **Query Conditioning** | **None** (Single sequence mode) | **Fixed Verbatim PICO Query** (73 words) | **Fixed Verbatim PICO Query** (73 words) |
| **Stage 1 Architecture** | Shared Backbone $\to 25 \times \text{Linear}(768, 1)$ | Shared Backbone $\to 25 \times \text{Linear}(768, 1)$ | Shared Backbone $\to 25 \times \text{Linear}(768, 1)$ |
| **Stage 1 Training Mode** | 5-Fold Stratified Cross-Validation ($N=334$) | 5-Fold Stratified Cross-Validation ($N=334$) | 5-Fold Stratified Cross-Validation ($N=334$) |
| **Stage 1 Models Trained** | 5 Fold Models | 5 Fold Models | 5 Fold Models |
| **Stage 2 Inputs** | $\mathbf{X}_{\text{train\_OOF}}, \mathbf{X}_{\text{val}}, \mathbf{X}_{\text{test}} \in \mathbb{R}^{N \times 25}$ | $\mathbf{X}_{\text{train\_OOF}}, \mathbf{X}_{\text{val}}, \mathbf{X}_{\text{test}} \in \mathbb{R}^{N \times 25}$ | $\mathbf{X}_{\text{train\_OOF}}, \mathbf{X}_{\text{val}}, \mathbf{X}_{\text{test}} \in \mathbb{R}^{N \times 25}$ |
| **Stage 2 Models** | MLP, Logistic Regression, Random Forest | MLP, Logistic Regression, Random Forest | MLP, Logistic Regression, Random Forest |
| **Primary Research Question** | *Can an abstract-only concept bottleneck match direct end-to-end screening?* | *Does explicit PICO cross-attention improve feature extraction accuracy and downstream decision performance?* | *Does MedCPT's search-log retrieval pre-training yield superior criteria extraction over PubMedBERT?* |

---

## 4. Input Text & Fixed PICO Query Specification

### 4.1 Fixed Verbatim PICO Query (Arms 2 & 3)
In Arms 2 and 3, Segment A consists of the exact 73-word clinical eligibility criteria query established in previous stages, used without alteration:

```text
Find prostate cancer studies stratified by pretreatment low/intermediate/high risk (D’Amico, Zelefsky, or NCCN criteria)
evaluating open/robotic radical prostatectomy; conventional external-beam radiotherapy ≥72 Gy; hypofractionated
radiotherapy/SBRT; LDR/HDR brachytherapy; proton therapy; androgen-deprivation therapy combinations; cryotherapy;
or HIFU. Retain all patients regardless of post-surgical pathological findings. Outcomes include long-term biochemical
recurrence-free survival, overall survival, metastasis-free survival, or cancer-specific survival. When reported,
consider risk-group-specific patient counts and median follow-up ≥5 years.
```

### 4.2 Sequence Formatting & Truncation Controls
* **Arm 1 (Single-Sequence):**
  $$\text{Input} = \mathbf{[CLS]} \quad \text{Abstract Tokens} \quad \mathbf{[SEP]}$$
  Total sequence budget = 512 tokens. Truncation rate on raw abstracts $< 2\%$.
* **Arms 2 & 3 (Dual-Sequence Cross-Encoder):**
  $$\text{Input} = \mathbf{[CLS]} \quad \text{Fixed PICO Query (86 tokens)} \quad \mathbf{[SEP]} \quad \text{Abstract Tokens (up to 426 tokens)} \quad \mathbf{[SEP]}$$
  Token type IDs: $0$ for Segment A, $1$ for Segment B.

---

## 5. Mathematical Formulations & Model Architectures

### 5.1 Stage 1: 25-Head Feature Extractor
For each candidate paper $i$, the transformer outputs a contextual representation. The pooled `[CLS]` embedding is extracted with dropout:
$$\mathbf{h}_{\text{CLS}, i} = \text{Dropout}(\mathbf{H}_{\text{last\_layer}, i}[:, 0, :], \; p = 0.25) \in \mathbb{R}^{768}$$

The 25 auxiliary clinical feature logits are computed via 25 parallel linear projections:
$$\hat{z}_{i, k} = \mathbf{w}_k^T \mathbf{h}_{\text{CLS}, i} + b_k, \quad \hat{p}_{i, k} = \sigma(\hat{z}_{i, k}), \quad k \in \{1, \dots, 25\}$$

#### Loss Function: Multi-Label Asymmetric Loss (ASL)
To handle severe class imbalances across features (e.g., `proton_therapy` with $1.5\%$ positive prevalence), each head is supervised via Asymmetric Loss with margin truncation:
$$\mathcal{L}_{\text{ASL}}(\hat{z}_{i, k}, y_{i, k}) = - y_{i, k} \log(\sigma(\hat{z}_{i, k})) - (1 - y_{i, k}) (p_{m, i, k})^{\gamma_-} \log(1 - p_{m, i, k} + \epsilon)$$
$$p_{m, i, k} = \max(\sigma(\hat{z}_{i, k}) - m, 0)$$
* Hyperparameters: $\gamma_+ = 0.0$, $\gamma_- = 2.0$, $m = 0.05$, $\epsilon = 1 \times 10^{-7}$.
* Total Stage 1 Loss:
  $$\mathcal{L}_{\text{Stage1}} = \frac{1}{25} \sum_{k=1}^{25} \mathcal{L}_{\text{ASL}}(\hat{z}_{i, k}, y_{i, k})$$

### 5.2 Stage 2: Downstream Decision Models

#### Model A: Multi-Layer Perceptron (MLP)
$$\mathbf{x}_i = [\hat{p}_{i, 1}, \hat{p}_{i, 2}, \dots, \hat{p}_{i, 25}]^T \in [0, 1]^{25}$$
$$\mathbf{h}_1 = \text{Dropout}(\text{GELU}(\text{LayerNorm}(\mathbf{W}_1 \mathbf{x}_i + \mathbf{b}_1)), \; p = 0.20), \quad \mathbf{W}_1 \in \mathbb{R}^{32 \times 25}$$
$$\hat{z}_{\text{final}, i} = \mathbf{W}_2 \mathbf{h}_1 + b_2, \quad \mathbf{W}_2 \in \mathbb{R}^{1 \times 32}$$
$$\hat{p}_{\text{final}, i} = \sigma(\hat{z}_{\text{final}, i})$$
* Supervised with Binary Cross-Entropy / ASL against ground-truth primary label $y_i \in \{0, 1\}$.

#### Model B: Calibrated Logistic Regression (L2)
$$\hat{z}_{\text{final}, i} = \mathbf{w}_{\text{LR}}^T \mathbf{x}_i + b_{\text{LR}}, \quad \min_{\mathbf{w}} \sum_i \mathcal{L}_{\text{BCE}}(\hat{z}_i, y_i) + \frac{1}{2C} \|\mathbf{w}\|_2^2$$
* Provides exact linear feature importance weights for clinical interpretability.

#### Model C: Random Forest Classifier
* Ensemble of 100 decision trees trained with Gini impurity, max depth $= 5$, minimum leaf samples $= 4$.

---

## 6. Dataset, Splits, and Strict Leakage Prevention

### 6.1 Dataset Census (`labeled-dataset-v2-aux25.csv`)
* **Total Cohort Size:** $N = 523$ articles (PMID 21056265 excluded due to blank abstract).
* **Missing Values:** Zero nulls across all 25 auxiliary columns and primary label.
* **Stratified Partitions (Matching Stage 5 Splits via PMID Inner Join):**
  * **Sub-Train:** $N = 334$ (180 Positives [53.89%], 154 Negatives [46.11%])
  * **Validation:** $N = 84$ (45 Positives [53.57%], 39 Negatives [46.43%])
  * **Held-Out Test:** $N = 105$ (56 Positives [53.33%], 49 Negatives [46.67%])

### 6.2 5-Fold Stratification Inside Sub-Train ($N = 334$)
Using `StratifiedKFold(n_splits=5, shuffle=True, random_state=2025)`:
* Folds 1–4: $N = 67$ samples (36 Pos, 31 Neg)
* Fold 5: $N = 66$ samples (36 Pos, 30 Neg)
* Zero sample overlap between training folds and out-of-fold evaluation slices.

### 6.3 Leakage Guardrails
1. **Validation Locking:** The validation set ($N = 84$) is strictly used for model checkpoint selection and threshold sweeps ($\tau \in [0.01, 0.99]$).
2. **Held-Out Test Locking:** The test set ($N = 105$) is evaluated **strictly once** per arm using frozen validation-calibrated thresholds.

---

## 7. Execution Architecture & Step-by-Step Training Workflow

The complete execution workflow for Stage 9.1 is structured as follows:

```
========================================================================================
PHASE 0: THE ORACLE CEILING TEST (Zero GPU Compute)
----------------------------------------------------------------------------------------
- Load ground-truth 25 features for Train (N=334), Val (N=84), Test (N=105).
- Fit MLP, Logistic Regression, Random Forest on ground-truth training features.
- Evaluate on ground-truth Test features.
- Output: Theoretical upper bound on screening accuracy from these 25 features.
========================================================================================
PHASE 1: 5-FOLD STAGE 1 FEATURE EXTRACTION (Arms 1, 2, 3)
----------------------------------------------------------------------------------------
For each Arm in [Arm 1, Arm 2, Arm 3]:
    For fold k in [1, 2, 3, 4, 5]:
        - Train 25-head Transformer on 4 folds (N ≈ 267).
        - Track validation loss on fold k (N ≈ 67); early stopping patience = 4.
        - Predict out-of-sample feature probabilities on fold k.
        - Save fold model checkpoint.
    - Assemble X_train_OOF (334 × 25).
    - Run 5-model ensemble on Val (N=84) -> X_val (84 × 25).
    - Run 5-model ensemble on Test (N=105) -> X_test (105 × 25).
    - Save extracted feature matrices to results/stage9_1/.
========================================================================================
PHASE 2: STAGE 2 DECISION CLASSIFIER TRAINING & EVALUATION
----------------------------------------------------------------------------------------
For each Arm:
    - Train MLP, Logistic Regression, Random Forest on X_train_OOF.
    - Sweep decision thresholds (0.01 to 0.99) on X_val to lock operating points:
        * Default: tau = 0.50
        * High Recall Target: tau @ Recall >= 90%
        * Clinical Target: tau @ Recall >= 95%
        * Zero-Miss Target: tau @ Recall = 100%
    - Evaluate on X_test with frozen thresholds.
    - Compute AUROC, PR-AUC, Sensitivity, Specificity, NNS, and Workload Reduction.
========================================================================================
PHASE 3: COMPREHENSIVE BENCHMARK REPORTING
----------------------------------------------------------------------------------------
- Compare Arm 1 vs. Arm 2 vs. Arm 3 vs. Oracle vs. Stage 9 Multitask.
- Extract feature importance coefficients (interpreting criteria weights).
- Generate reports/stage9_1/stage9_1_benchmark_report.md.
========================================================================================
```

---

## 8. Comprehensive Evaluation Metrics

### 8.1 Primary Task Metrics (Literature Screening)
* **AUROC & PR-AUC:** Area under ROC and Precision-Recall curves.
* **Clinical Safety Points:** Specificity and False Negatives at $\text{Recall} = 100\%$ and $\text{Recall} \ge 95\%$.
* **Workload Reduction %:** $\frac{N - (\text{TP} + \text{FP})}{N} \times 100\%$
* **Number Needed to Screen (NNS):** $\frac{\text{TP} + \text{FP}}{\text{TP}}$

### 8.2 Stage 1 Feature Extraction Quality
* Per-feature AUROC and PR-AUC across all 25 clinical targets.
* Macro-Averaged F1 and Micro-Averaged F1 across the 25 features.

---

## 9. Project Directory & Artifact Coherence

To maintain complete architectural coherence with preceding stages:
* **Protocol:** `reports/stage9_1/stage9_1_design_and_protocol.md`
* **Benchmark Report:** `reports/stage9_1/stage9_1_benchmark_report.md`
* **Training Script:** `src/stage9_1/train_stage9_1_cascade.py`
* **Model Checkpoints:** `models/stage9_1/arm_{arm_id}_fold_{fold_k}.pt`, `models/stage9_1/arm_{arm_id}_mlp.pt`
* **Feature Matrices:** `results/stage9_1/arm_{arm_id}_oof_features.csv`, `val_features.csv`, `test_features.csv`
* **Metric Tables:** `results/stage9_1/stage9_1_test_metrics.csv`, `stage9_1_stage1_feature_metrics.csv`
* **Execution Logs:** `logs/stage9_1/stage9_1_cascade_training.log`
