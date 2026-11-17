# Stage 9 Experimental Design and Protocol: Multitask Parallel-Head Screening
## Benchmarking Shared Encoders with 25 Auxiliary Clinical Objectives (PubMedBERT vs. MedCPT)

**Document Type:** Formal Study Design & Protocol  
**Date:** October 2026  
**Target Path:** `reports/stage9/stage9_design_and_protocol.md`  
**Dataset Reference:** `data/labeled-dataset-v2-aux25.csv` ($N = 523$)  
**Baseline Stages:** Stage 7 (Cross-Encoder & ASL), Stage 8 (MedCPT Zero-Shot & Fine-Tuning)  
**Primary Architectures:** `microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract`, `ncbi/MedCPT-Cross-Encoder`

---

## 1. Executive Summary & Stage 9 Objectives

### 1.1 Context and Problem Statement
In systematic review literature screening for localized prostate cancer clinical trials (under PRISMA and Prostate Cancer Results Study Group [PCRSG] criteria), automated models must achieve extreme recall ($\ge 95\%$ to $100\%$) while eliminating false alarms to reduce manual human review workload. In Stages 7 and 8, fine-tuned cross-encoders demonstrated that explicitly conditioning the encoder on clinical criteria text substantially improves high-recall specificity. 

However, prior stages treated literature screening primarily as a **monolithic binary classification task** (relevant vs. irrelevant). While a single binary label indicates whether a study satisfies all clinical criteria, it forces the neural encoder to implicitly learn diverse clinical facets—risk stratification methods, 17 distinct curative treatment modalities, numeric radiation dose thresholds, 5-year follow-up adequacy, and hard oncologic survival endpoints—without direct supervision on those individual concepts.

### 1.2 Stage 9 Objectives
Stage 9 introduces a **multitask representation learning paradigm**. Each model consists of a **shared transformer encoder** coupled with **26 parallel output heads**:
1. **One Primary Head:** Predicts the original binary screening decision (Relevant = 1 vs. Irrelevant = 0).
2. **25 Auxiliary Heads:** Simultaneously predict 25 fine-grained clinical criteria features extracted from the abstract (e.g., specific treatment modalities, risk stratification systems, survival endpoints, follow-up status).

Stage 9 is designed to execute a rigorous three-arm empirical comparison:
* **Arm 1 (PubMedBERT, Abstract-Only):** Shared PubMedBERT encoder trained on the raw abstract text without a PICO query, simultaneously driving the primary screening head and 25 auxiliary clinical heads.
* **Arm 2 (PubMedBERT, Abstract + PICO Query):** Shared PubMedBERT encoder conditioned on a fixed natural-language PICO query paired with the abstract via bidirectional cross-attention, driving all 26 heads.
* **Arm 3 (MedCPT Cross-Encoder, Abstract + PICO Query):** Shared MedCPT Cross-Encoder (pre-trained on 255M PubMed search logs) conditioned on the exact same PICO query and abstract text, driving all 26 heads.

### 1.3 Core Hypotheses
1. **Multitask Regularization Hypothesis:** Jointly training the shared encoder to predict 25 explicit clinical attributes will prevent the encoder from latching onto spurious superficial patterns, regularizing the latent space and improving primary screening AUROC and high-recall specificity.
2. **Query Conditioning in Multitask Hypothesis (Arm 1 vs. Arm 2):** Providing an explicit PICO criteria query in Segment A will allow the cross-attention layers to contextualize the abstract against clinical guidelines, enhancing both auxiliary feature detection and primary screening performance compared to abstract-only input.
3. **Pre-training Domain Alignment Hypothesis (Arm 2 vs. Arm 3):** MedCPT's query-document interaction pre-training from search logs will yield superior query-to-abstract cross-attention alignment for multi-attribute matching compared to PubMedBERT's masked-language-modeling pre-training.

> [!IMPORTANT]
> **Architectural Constraint:** Multitask in Stage 9 strictly denotes **parallel classification heads branching from a single shared encoder**. This design is distinct from a *cascaded architecture* (where auxiliary feature predictions are fed into a separate downstream classifier). The primary screening head and all 25 auxiliary heads are optimized jointly via backpropagation into the shared transformer backbone.

> [!IMPORTANT]
> **Input Constraint:** Across all three arms, the article input text consists of the **Abstract ONLY**. Article titles are strictly excluded from the input sequence across all Stage 9 experiments.

---

## 2. Review and Synthesis of Stage 8 Evidence

To establish empirical baselines and justify Stage 9 design parameters, this section synthesizes the quantitative results from Stage 8. 

Stage 8 investigated the `ncbi/MedCPT-Cross-Encoder` across zero-shot retrieval and multiple fine-tuning configurations. **A critical methodological distinction must be maintained regarding datasets**:
* **Stage 8.0 (Zero-Shot):** Evaluated across both the held-out test cohort ($N = 105$) and the 2021 prospective validation cohort ($N = 117$).
* **Stage 8.1 through Stage 8.2.2.1 (Fine-Tuning):** Evaluated **STRICTLY on the internal validation split ($N = 84$)**. As recorded in the training logs, the held-out test set ($N = 105$) and the 2021 cohort ($N = 117$) were completely untouched and excluded from Stage 8.1 onward to prevent data leakage. Validation metrics must not be conflated with held-out test results.

### 2.1 Stage 8.0: Zero-Shot Baseline Performance
As documented in [`results/stage8/stage8_zero_shot_summary.csv`](file:///d:/prostate-research-analysis/prostate-research-analysis/results/stage8/stage8_zero_shot_summary.csv) and [`reports/stage8/stage8_zero_shot_report.md`](file:///d:/prostate-research-analysis/prostate-research-analysis/reports/stage8/stage8_zero_shot_report.md), off-the-shelf MedCPT without task-specific fine-tuning showed poor calibration for systematic literature screening:

| Cohort | Query Formulation | Operating Point | Recall | Specificity | AUROC | PR-AUC | Source |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Held-Out Test ($N=105$)** | Search-Style Query | Default (0.50) | 7.14% | 91.84% | **0.6112** | 0.5891 | `stage8_zero_shot_summary.csv` |
| **Held-Out Test ($N=105$)** | PICO Natural Language | Default (0.50) | 5.36% | 97.96% | **0.5765** | 0.5516 | `stage8_zero_shot_summary.csv` |
| **2021 Cohort ($N=117$)** | PICO Natural Language | Default (0.50) | 3.57% | 96.63% | **0.6902** | 0.3563 | `stage8_zero_shot_summary.csv` |
| **2021 Cohort ($N=117$)** | Search-Style Query | Default (0.50) | 10.71% | 92.13% | **0.6160** | 0.3033 | `stage8_zero_shot_summary.csv` |

*Takeaway:* Zero-shot MedCPT collapses sensitivity under default decision boundaries because general search-log relevance differs fundamentally from clinical trial inclusion screening. Task-specific fine-tuning is mandatory.

### 2.2 Stage 8.1 to 8.2.2.1: Supervised Fine-Tuning Progression (Validation Cohort, $N = 84$)
Supervised fine-tuning of MedCPT was executed across six iterative experiments using Asymmetric Loss (ASL) and conservative two-stage unfreezing (Epoch 1 head-only warmup, Epochs 2–8 full backbone fine-tuning with backbone LR $= 5 \times 10^{-6}$, head LR $= 2 \times 10^{-5}$).

The table below summarizes the recorded validation performance across these runs:

| Stage ID | Model & Objective | Query Type | Loss Weighting | Best Epoch (Val Loss) | Val Loss | Val AUROC | Val PR-AUC | Val Specificity @ 100% Recall | Verified Sources |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **8.1A** | MedCPT Standalone ASL | Natural PICO (73 tok) | Unweighted ($1.0$) | Ep 6 | 0.3424 | **0.8063** | 0.7636 | 33.33% (26 FP) | [`stage8_1_validation_metrics.csv`](file:///d:/prostate-research-analysis/prostate-research-analysis/results/stage8_1/stage8_1_validation_metrics.csv), [`stage8_1_medcpt_finetuning.log`](file:///d:/prostate-research-analysis/prostate-research-analysis/logs/stage8_1_medcpt_finetuning.log) |
| **8.1B** | MedCPT Standalone ASL | Natural PICO (73 tok) | Cost-Weighted ($1.722$) | Ep 6 | 0.4820 | **0.7960** | 0.7455 | 28.21% (28 FP) | [`stage8_1_validation_metrics.csv`](file:///d:/prostate-research-analysis/prostate-research-analysis/results/stage8_1/stage8_1_validation_metrics.csv), [`stage8_1_medcpt_finetuning.log`](file:///d:/prostate-research-analysis/prostate-research-analysis/logs/stage8_1_medcpt_finetuning.log) |
| **8.1.1** | MedCPT Standalone ASL | Natural PICO (73 tok) | Unweighted ($1.0$) | Ep 6 | 0.3129 | **0.8615** | 0.8002 | 38.46% (24 FP) | [`stage8_1_1_validation_metrics.csv`](file:///d:/prostate-research-analysis/prostate-research-analysis/results/stage8_1_1/stage8_1_1_validation_metrics.csv), [`stage8_1_1_medcpt_finetuning.log`](file:///d:/prostate-research-analysis/prostate-research-analysis/logs/stage8_1_1_medcpt_finetuning.log) |
| **8.2** | MedCPT Standalone ASL | Detailed Criteria | Unweighted ($1.0$) | Ep 6 | 0.3439 | **0.8405** | 0.8129 | 35.90% (25 FP) | [`stage8_2_validation_metrics.csv`](file:///d:/prostate-research-analysis/prostate-research-analysis/results/stage8_2/stage8_2_validation_metrics.csv), [`stage8_2_medcpt_finetuning.log`](file:///d:/prostate-research-analysis/prostate-research-analysis/logs/stage8_2_medcpt_finetuning.log) |
| **8.2.1** | MedCPT Standalone ASL | Search Query Style | Unweighted ($1.0$) | **Ep 6** | **0.2998** | **0.8769** | **0.8656** | 28.21% (28 FP) | [`stage8_2_1_validation_metrics.csv`](file:///d:/prostate-research-analysis/prostate-research-analysis/results/stage8_2_1/stage8_2_1_validation_metrics.csv), [`stage8_2_1_medcpt_finetuning.log`](file:///d:/prostate-research-analysis/prostate-research-analysis/logs/stage8_2_1_medcpt_finetuning.log) |
| **8.2.2** | MedCPT Standalone ASL | Restored Clinical Details | Unweighted ($1.0$) | Ep 6 | 0.3204 | **0.8490** | 0.7950 | **43.59% (22 FP)** | [`stage8_2_2_validation_metrics.csv`](file:///d:/prostate-research-analysis/prostate-research-analysis/results/stage8_2_2/stage8_2_2_validation_metrics.csv), [`stage8_2_2_medcpt_finetuning.log`](file:///d:/prostate-research-analysis/prostate-research-analysis/logs/stage8_2_2_medcpt_finetuning.log) |
| **8.2.2.1**| MedCPT Standalone ASL | Compact Verbatim Criteria | Unweighted ($1.0$) | **Ep 6** | **0.3252** | **0.8484** | **0.7985** | 35.90% (25 FP) | [`stage8_2_2_1_validation_metrics.csv`](file:///d:/prostate-research-analysis/prostate-research-analysis/results/stage8_2_2_1/stage8_2_2_1_validation_metrics.csv), [`stage8_2_2_1_medcpt_finetuning.log`](file:///d:/prostate-research-analysis/prostate-research-analysis/logs/stage8_2_2_1_medcpt_finetuning.log) |

### 2.3 Critical Verification and Discrepancy Analysis in Stage 8 Runs
A rigorous inspection of the Stage 8 CSV tables versus raw execution logs reveals critical subtleties:

1. **Checkpoint Selection vs. Final Epoch Metrics in Stage 8.2.1:**
   - In [`stage8_2_1_validation_metrics.csv`](file:///d:/prostate-research-analysis/prostate-research-analysis/results/stage8_2_1/stage8_2_1_validation_metrics.csv), the selected checkpoint exhibits **AUROC = 0.8769** and **PR-AUC = 0.8656**. This checkpoint was chosen at **Epoch 6** because it attained the global minimum validation loss ($\mathcal{L}_{\text{val}} = 0.2998$).
   - In [`logs/stage8_2_1_medcpt_finetuning.log`](file:///d:/prostate-research-analysis/prostate-research-analysis/logs/stage8_2_1_medcpt_finetuning.log#L17-L21), the model continued training through Epoch 8, achieving a higher raw AUROC of **0.8798** at Epoch 8. However, validation loss at Epoch 8 rose slightly to $0.3022$. Under the strict early-stopping protocol (tracking validation loss), Epoch 8 was not saved.
   - *Protocol Standard:* Model checkpoints must always be identified by their saved weights (Epoch 6, AUROC 0.8769), distinguishing them from later epoch artifacts.
2. **Checkpoint Selection vs. Final Epoch Metrics in Stage 8.2.2.1:**
   - Similarly, in [`stage8_2_2_1_validation_metrics.csv`](file:///d:/prostate-research-analysis/prostate-research-analysis/results/stage8_2_2_1/stage8_2_2_1_validation_metrics.csv), the saved checkpoint (Epoch 6, $\mathcal{L}_{\text{val}} = 0.3252$) achieved **AUROC = 0.8484** and **PR-AUC = 0.7985**.
   - In [`logs/stage8_2_2_1_medcpt_finetuning.log`](file:///d:/prostate-research-analysis/prostate-research-analysis/logs/stage8_2_2_1_medcpt_finetuning.log#L17-L21), Epoch 8 reached a raw AUROC of **0.8507**, but with a higher validation loss ($0.3325$).
3. **High-Recall Operating Point Shift (Stage 8.2.1 vs. Stage 8.2.2):**
   - While Stage 8.2.1 achieved the highest global ranking metrics (AUROC 0.8769), Stage 8.2.2 performed substantially better at the extreme clinical operating point ($\text{Recall} = 100\%$), producing only **22 false positives** compared to **28 false positives** in Stage 8.2.1.
4. **Attribution Methodological Warning:**
   - The query transitions across Stages 8.1, 8.2, 8.2.1, 8.2.2, and 8.2.2.1 altered multiple syntactic, structural, and lexical elements simultaneously (e.g., query length, inclusion of explicit numeric thresholds, formatting of risk stratification names). Performance variations across these sub-stages **cannot be attributed solely to isolated wording choices**, but reflect compound changes in sequence truncation, prompt density, and token distribution.

---

## 3. The 25 Auxiliary Features: Definitions, Distribution, and Labeling Audit

### 3.1 Source Dataset & Dimensionality
The dataset designated for Stage 9 multitask modeling is [`data/labeled-dataset-v2-aux25.csv`](file:///d:/prostate-research-analysis/prostate-research-analysis/data/labeled-dataset-v2-aux25.csv).
- **Total Articles:** $N = 523$
- **Total Columns:** 40 columns (15 original metadata and text columns + 1 original binary screening `label` + 24 auxiliary feature columns).
- **Auxiliary Feature Set:** 25 binary indicators covering PCRSG screening criteria.

### 3.2 Methodological Audit: Rule-Based vs. AI-Assisted Labeling
> [!WARNING]
> **Discrepancy Audit Regarding Labeling Methodology:**
> While external descriptions or user prompts may refer to "AI-assisted labeling," the official project documentation in [`data/auxiliary-labeling-26-method.md`](file:///d:/prostate-research-analysis/prostate-research-analysis/data/auxiliary-labeling-26-method.md#L15-L21) explicitly records that:
> *"Binary values were generated through rule-based extraction from abstract text. This is a reproducible first-pass dataset for research review, not a substitute for manual expert adjudication of ambiguous cases."*
> 
> Stage 9 protocol formally treats these 25 features as **rule-based algorithmic annotations** derived from regular expressions and term matching over PubMed abstracts.

### 3.3 Semantic Meaning of Label Values ($0$ vs. $1$)
The auxiliary labels follow an asymmetric semantic definition that is critical for multitask loss formulation:
* **Value = `1` (Explicit Positive Evidence):** The abstract explicitly mentions or satisfies the defined clinical criterion.
* **Value = `0` (Lack of Explicit Evidence / Unreported):** The abstract does **not** contain sufficient explicit textual evidence to confirm the feature. 
* **Critical Machine Learning Implication:** A value of `0` **does not establish that the clinical feature is truly absent in the underlying clinical trial**. Scientific abstracts frequently omit specific risk criteria, median follow-up months, or secondary treatment arms due to strict journal word limits. Treating every `0` as a hard ground-truth negative risks penalizing models that correctly infer clinical context.

### 3.4 Feature Definitions and Empirical Distribution
The 25 retained auxiliary features, categorized into five clinical domains, with exact positive prevalence out of $N = 523$ records (verified directly against `labeled-dataset-v2-aux25.csv` and [`auxiliary-labeling-26-method.md`](file:///d:/prostate-research-analysis/prostate-research-analysis/data/auxiliary-labeling-26-method.md#L68-L98)):

| Category | Feature Name | Positive Count ($N=523$) | Prevalence (%) | Defining Evidence Required for Value = 1 |
| :--- | :--- | :---: | :---: | :--- |
| **Risk Stratification** | `risk_method_damico` | 35 | 6.69% | The explicit name "D'Amico" appears in the abstract. |
| | `risk_method_nccn` | 48 | 9.18% | The explicit name or abbreviation "NCCN" appears in the abstract. |
| | `pretreatment_risk_stratification` | 58 | 11.09% | Pretreatment risk classification is explicit, or baseline risk groups are tracked in results. |
| | `low_risk_group_status` | 31 | 5.93% | Explicitly reported low-risk group contains $\ge 100$ patients (0 if count unclear). |
| | `intermediate_risk_group_status`| 36 | 6.88% | Explicitly reported intermediate-risk group contains $\ge 100$ patients (0 if count unclear). |
| | `high_risk_group_status` | 47 | 8.99% | Explicitly reported high-risk group contains $\ge 50$ patients (0 if count unclear). |
| | `retains_baseline_risk_groups` | 78 | 14.91% | Clinical outcomes are reported separately by at least two baseline risk groups. |
| **Treatment Modality** | `radical_prostatectomy` | 126 | 24.09% | Radical prostatectomy (open/robotic) is an active study treatment/intervention. |
| | `standard_ebrt` | 189 | 36.14% | EBRT is used and abstract does not specify a hypofractionated or SBRT-only regimen. |
| | `hypofractionated_ebrt` | 30 | 5.74% | Hypofractionated EBRT is explicitly reported in methods or intervention. |
| | `ultrahypofractionated_ebrt` | 21 | 4.02% | Ultrahypofractionation, SBRT, or stereotactic body radiation is explicitly stated. |
| | `proton_therapy` | 8 | 1.53% | Proton beam therapy is evaluated as a primary study intervention. |
| | `ldr_brachytherapy` | 60 | 11.47% | Low-dose-rate (LDR) seed brachytherapy (e.g., I-125, Pd-103) is an active intervention. |
| | `hdr_brachytherapy` | 60 | 11.47% | High-dose-rate (HDR) brachytherapy is explicitly reported. |
| | `adt_combination` | 107 | 20.46% | Androgen deprivation therapy (ADT) is combined with local therapy. |
| | `cryotherapy` | 16 | 3.06% | Cryotherapy / cryoablation is evaluated as a study intervention. |
| | `hifu` | 24 | 4.59% | High-Intensity Focused Ultrasound (HIFU) is evaluated as a study intervention. |
| **Clinical Endpoints** | `has_brfs` | 221 | 42.26% | Biochemical recurrence-free survival, biochemical control, or bPFS is assessed. |
| | `has_os` | 142 | 27.15% | Overall survival (OS) is evaluated or reported. |
| | `has_mfs` | 39 | 7.46% | Metastasis-free survival (MFS) or distant metastasis control is evaluated. |
| | `has_css` | 121 | 23.14% | Cancer-specific survival (CSS), cause-specific, or disease-specific survival is reported. |
| **Numerical Compliance**| `standard_ebrt_dose_status` | 21 | 4.02% | Explicit radiation dose for standard non-hypofractionated EBRT is $\ge 72\text{ Gy}$. |
| | `median_followup_status` | 145 | 27.72% | Explicit median follow-up duration is $\ge 60\text{ months}$ (5.0 years). |
| **Exclusion / Scope** | `advanced_metastatic_cohort` | 9 | 1.72% | Study population is primarily advanced/metastatic or castration-resistant (out-of-scope). |
| | `salvage_postfailure_cohort` | 42 | 8.03% | Evaluates salvage therapy following primary treatment failure (out-of-scope). |

### 3.5 Omitted Features Audit
Per [`data/auxiliary-labeling-26-method.md`](file:///d:/prostate-research-analysis/prostate-research-analysis/data/auxiliary-labeling-26-method.md#L7-L9):
1. `postop_pathology_exclusion`: Omitted because positive prevalence was $0/523$ (zero variance).
2. `risk_method_zelefsky`: Omitted because no abstract explicitly cited the Zelefsky criteria.

---

## 4. Three-Arm Multitask Experimental Design

Stage 9 executes a controlled factorial comparison across three model arms. All three arms share identical training splits, seeds, optimizer policies, and evaluation protocols.

### 4.1 Master Experiment Comparison Table

| Dimension | Experiment Arm 1 (PubMedBERT Abstract-Only) | Experiment Arm 2 (PubMedBERT Abstract + PICO) | Experiment Arm 3 (MedCPT Abstract + PICO) |
| :--- | :--- | :--- | :--- |
| **Encoder Backbone** | `microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract` | `microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract` | `ncbi/MedCPT-Cross-Encoder` |
| **Pre-training Domain** | Biomedical literature (PubMed abstracts) via MLM | Biomedical literature (PubMed abstracts) via MLM | PubMed search queries + articles (255M search log pairs) |
| **Query Input (Segment A)** | **None** (Single-sequence mode) | **Fixed Verbatim PICO Query** (73 words) | **Fixed Verbatim PICO Query** (73 words) |
| **Document Input (Segment B)**| **Abstract ONLY** (Segment A in single-seq) | **Abstract ONLY** (Segment B in cross-encoder) | **Abstract ONLY** (Segment B in cross-encoder) |
| **Article Titles Used?** | **No** (Strictly excluded) | **No** (Strictly excluded) | **No** (Strictly excluded) |
| **Sequence Budget** | 512 tokens max | 512 tokens max (PICO Query + Abstract) | 512 tokens max (PICO Query + Abstract) |
| **Cross-Attention Mode** | Self-attention within abstract only | Full cross-attention between PICO and abstract | Full cross-attention between PICO and abstract |
| **Output Heads** | **26 parallel heads** (1 primary + 25 auxiliary) | **26 parallel heads** (1 primary + 25 auxiliary) | **26 parallel heads** (1 primary + 25 auxiliary) |
| **Head Architecture** | Shared `[CLS]` $\to 26 \times \text{Linear}(768, 1)$ | Shared `[CLS]` $\to 26 \times \text{Linear}(768, 1)$ | Shared `[CLS]` $\to 26 \times \text{Linear}(768, 1)$ |
| **Primary Research Question**| *Does multitask auxiliary supervision enable pure abstract-level screening without prompt conditioning?* | *Does bidirectional cross-attention against explicit PICO criteria boost multi-facet clinical extraction in PubMedBERT?* | *Does search-log pre-training (MedCPT) provide better criteria-to-abstract alignment than PubMedBERT under identical multitask objectives?* |

---

## 5. Input Formatting & Fixed PICO Query Specification

### 5.1 Fixed PICO Query Specification (Arms 2 & 3)
For Experiment Arm 2 and Experiment Arm 3, the criteria text in Segment A must match the following text **verbatim, without quotation marks, paraphrasing, or structural alteration**:

```text
Find prostate cancer studies stratified by pretreatment low/intermediate/high risk (D’Amico, Zelefsky, or NCCN criteria)
evaluating open/robotic radical prostatectomy; conventional external-beam radiotherapy ≥72 Gy; hypofractionated
radiotherapy/SBRT; LDR/HDR brachytherapy; proton therapy; androgen-deprivation therapy combinations; cryotherapy;
or HIFU. Retain all patients regardless of post-surgical pathological findings. Outcomes include long-term biochemical
recurrence-free survival, overall survival, metastasis-free survival, or cancer-specific survival. When reported,
consider risk-group-specific patient counts and median follow-up ≥5 years.
```

### 5.2 Tokenization and Input Sequence Assembly

```
Arm 1 (Single-Sequence Abstract):
[CLS] Abstract Tokens [SEP] [PAD] ... [PAD]
(token_type_ids = 0 for all active tokens)

Arms 2 & 3 (Dual-Sequence Cross-Encoder):
[CLS] Fixed PICO Query Tokens [SEP] Abstract Tokens [SEP] [PAD] ... [PAD]
(token_type_ids = 0 for Query, token_type_ids = 1 for Abstract)
```

1. **Arm 1 Tokenization:**
   - Input format: `tokenizer(abstract, max_length=512, padding="max_length", truncation=True)`
   - Token type IDs: All $0$.
   - Truncation profile: Dedicated entirely to abstract text. Since the 99th percentile PubMed abstract length is $\sim 380$ words ($\sim 450$ tokens), abstract truncation in Arm 1 is $< 2\%$.
2. **Arms 2 & 3 Tokenization:**
   - Input format: `tokenizer(text=PICO_QUERY, text_pair=abstract, max_length=512, padding="max_length", truncation=True)`
   - Token type IDs: $0$ for PICO Query, $1$ for Abstract text.
   - Truncation profile: The fixed PICO query consumes exactly **86 WordPiece tokens** (including `[CLS]` and `[SEP]`). This leaves **426 tokens** dedicated strictly to the abstract. Based on the Stage 5.5 token audit, 426 tokens is sufficient to cover $> 85\%$ of all abstracts in full without losing terminal conclusion sentences.

---

## 6. Shared Multi-Head Architecture & Training Methodology

### 6.1 Multi-Head Neural Architecture
All three arms utilize a unified shared-backbone architecture. The transformer backbone generates contextualized representations, and the pooled `[CLS]` embedding is projected into 26 parallel scalar logits:

```
                          ┌───────────────────────────┐
                          │   Tokenized Input Text    │
                          └─────────────┬─────────────┘
                                        │
                                        ▼
                          ┌───────────────────────────┐
                          │  Shared Transformer Body  │
                          │   (PubMedBERT / MedCPT)   │
                          │      12 Layers, d=768     │
                          └─────────────┬─────────────┘
                                        │
                                        ▼
                          ┌───────────────────────────┐
                          │   Pooled [CLS] Dropout    │
                          │       (p_drop = 0.25)     │
                          └─────────────┬─────────────┘
                                        │
             ┌──────────────────────────┼──────────────────────────┐
             │                          │                          │
             ▼                          ▼                          ▼
   ┌───────────────────┐      ┌───────────────────┐      ┌───────────────────┐
   │   Primary Head    │      │  Aux Head 1 (b1)  │ ...  │  Aux Head 25 (b25)│
   │  Linear(768 -> 1) │      │  Linear(768 -> 1) │      │  Linear(768 -> 1) │
   └─────────┬─────────┘      └─────────┬─────────┘      └─────────┬─────────┘
             │                          │                          │
             ▼                          ▼                          ▼
       Primary Logit               Aux Logit 1                Aux Logit 25
      (Screening Label)       (risk_method_damico)      (salvage_postfailure)
```

$$\mathbf{h}_{\text{CLS}} = \text{Dropout}(\mathbf{H}_{\text{last\_layer}}[:, 0, :], \; p = 0.25)$$
$$\hat{z}_{\text{main}} = \mathbf{W}_{\text{main}} \mathbf{h}_{\text{CLS}} + b_{\text{main}}, \quad \mathbf{W}_{\text{main}} \in \mathbb{R}^{1 \times 768}$$
$$\hat{z}_{\text{aux}, k} = \mathbf{W}_{\text{aux}, k} \mathbf{h}_{\text{CLS}} + b_{\text{aux}, k}, \quad \mathbf{W}_{\text{aux}, k} \in \mathbb{R}^{1 \times 768}, \quad k \in \{1, \dots, 25\}$$

### 6.2 Loss Formulation: Asymmetric Loss (ASL)
To handle severe class imbalances across both the primary task and the 25 auxiliary tasks (where positive prevalence drops as low as $1.53\%$ for `proton_therapy`), all heads are supervised via **Asymmetric Loss (ASL)**:

$$\mathcal{L}_{\text{ASL}}(p, y) = - y (1 - p)^{\gamma_+} \log(p) - (1 - y) (p_m)^{\gamma_-} \log(1 - p_m + \epsilon)$$
$$p = \sigma(\hat{z}), \quad p_m = \max(p - m, 0)$$

* Starting Hyperparameters (carried over from Stage 8):
  * Positive focusing parameter: $\gamma_+ = 0.0$ (preserves full gradient on rare positive cases).
  * Negative focusing parameter: $\gamma_- = 2.0$ (down-weights easy negatives).
  * Probability margin shift: $m = 0.05$ (completely zeros out gradients from very easy negative samples where $p < 0.05$).

### 6.3 Multitask Loss Balancing and Aggregation
The joint training objective combines the primary screening loss with the sum of the auxiliary task losses:

$$\mathcal{L}_{\text{total}} = \lambda_{\text{main}} \mathcal{L}_{\text{ASL}}(\hat{z}_{\text{main}}, y_{\text{main}}) + \sum_{k=1}^{25} \lambda_{k} \mathcal{L}_{\text{ASL}}(\hat{z}_{\text{aux}, k}, y_{\text{aux}, k})$$

> [!WARNING]
> **Multitask Loss Weighting Caveat:**
> In Stage 8, models were trained exclusively on a single binary objective ($\lambda_{\text{main}} = 1.0$). In Stage 9, summing 25 auxiliary losses with unscaled weights ($\lambda_k = 1.0$) would cause the auxiliary tasks to contribute $96.1\%$ of the total gradient norm, swamping the primary screening objective.
> 
> **Proposed Starting Configurations:**
> 1. **Baseline Equalized Weighting:** $\lambda_{\text{main}} = 1.0$, $\lambda_k = \frac{1}{25} = 0.04$ (ensures the combined auxiliary tasks have equal gradient influence to the primary task).
> 2. **Screening-Prioritized Weighting:** $\lambda_{\text{main}} = 1.0$, $\lambda_k = 0.01$ (auxiliary tasks act strictly as a mild regularizer).

### 6.4 Proposed Training Hyperparameters (Starting Points)
Carried over from Stage 8 protocols as disciplined starting points (to be empirically validated):
* **Batch Size:** Physical batch size = $4$, Gradient accumulation steps = $2$ (Effective Batch Size = $8$).
* **Optimizer:** AdamW with decoupled weight decay ($\text{weight\_decay} = 1 \times 10^{-4}$).
* **Learning Rates:**
  * Transformer Backbone: $\text{LR}_{\text{backbone}} = 5 \times 10^{-6}$
  * Parallel Classification Heads: $\text{LR}_{\text{heads}} = 2 \times 10^{-5}$
* **Two-Stage Unfreezing Schedule:**
  * *Epoch 1 (Warmup):* Backbone frozen (`requires_grad = False`); only the 26 classification heads are trained to prevent random initial head gradients from distorting pre-trained weights.
  * *Epochs 2–8:* Full backbone unfreezed with linear learning rate warmup ($10\%$ of steps) followed by cosine/linear decay.
* **Epoch Budget & Early Stopping:** Maximum 8 epochs; early stopping patience = 4 epochs tracking validation loss.
* **Checkpoint Selection Policy:** The saved checkpoint is chosen **strictly by the minimum total validation loss** across epochs.

---

## 7. Dataset, Zero-Leakage Splits & Missing Record Audit

### 7.1 Dataset Verification
An inspection of `data/labeled-dataset-v2-aux25.csv` confirms:
* **Total Rows:** Exactly 523 records.
* **Label Integrity:** Complete labels (zero nulls / missing values) across all 25 auxiliary columns and the original `label` column.
* **Excluded Record Audit:** 
  The original cohort (`labeled-dataset-v2.csv`) contained 524 records. Exactly one record—**PMID 21056265** (an *Editorial Comment* with a completely blank abstract, ground-truth label = 0)—was excluded during auxiliary generation.

### 7.2 Matching Records to Existing Splits
Matching the 523 records of `labeled-dataset-v2-aux25.csv` against the established Stage 5 zero-leakage partitions (`data/splits/stage5/`):
* **Stage 5 Train Split (`train.csv`, $N = 335$):**
  * Exactly 334 records match.
  * The single non-matching record is the excluded blank-abstract PMID 21056265.
* **Stage 5 Validation Split (`val.csv`, $N = 84$):**
  * Exactly 84 records match ($100.0\%$ match rate).
* **Stage 5 Held-Out Test Split (`test.csv`, $N = 105$):**
  * Exactly 105 records match ($100.0\%$ match rate).

### 7.3 Limitation of Existing Split Files & Proposed Leakage-Safe Split
> [!IMPORTANT]
> **Data Limitation:** The existing split CSV files in `data/splits/stage5/` (`train.csv`, `val.csv`, `test.csv`) contain only the original 15 metadata columns and primary `label`. **They do not include the 25 auxiliary features.**

**Proposed Leakage-Safe Integration Protocol:**
1. Do not create new arbitrary splits.
2. In the Stage 9 data loader, load `data/labeled-dataset-v2-aux25.csv` and perform an inner join on `pmid` against the Stage 5 split manifests:
   * **Stage 9 Training Set:** $N = 334$ (180 Positives, 154 Negatives)
   * **Stage 9 Validation Set:** $N = 84$ (45 Positives, 39 Negatives)
   * **Stage 9 Held-Out Test Set:** $N = 105$ (56 Positives, 49 Negatives)
3. Total cohort size across all three partitions = $334 + 84 + 105 = 523$ articles.
4. This preserves complete historical compatibility, ensures zero test leakage, and guarantees that Stage 9 test evaluations are directly comparable to Stages 5, 5.5, 6, 7, and 8.

---

## 8. Leakage Prevention & Two-Stage Evaluation Protocol

To maintain clinical scientific integrity and prevent optimistic bias:

```
┌────────────────────────────────────────────────────────────────────────┐
│                        STAGE 1: VALIDATION PHASE                       │
│  - Train on Train Set (N = 334)                                        │
│  - Evaluate each epoch on Validation Set (N = 84)                      │
│  - Select best checkpoint by minimum Validation Loss                   │
│  - Sweep thresholds (0.01 to 0.99) on Validation Set                   │
│  - Lock operating thresholds:                                          │
│      * Default: tau = 0.50                                             │
│      * High-Recall Target: tau @ Recall >= 90%                         │
│      * Clinical Target: tau @ Recall >= 95%                            │
│      * Zero-Miss Target: tau @ Recall = 100%                           │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    │ (Freeze model weights & thresholds)
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│                        STAGE 2: HELD-OUT TEST PHASE                    │
│  - Unlock Held-Out Test Set (N = 105) STRICTLY ONCE                    │
│  - Apply frozen validation thresholds                                  │
│  - Compute un-calibrated test metrics                                  │
│  - Generate definitive clinical screening performance                  │
└────────────────────────────────────────────────────────────────────────┘
```

1. **Parameter & Threshold Freezing:** All hyperparameter tuning, checkpoint selection, and decision threshold calibration ($\tau$) must occur strictly on the validation set ($N = 84$).
2. **Single-Pass Test Evaluation:** The held-out test cohort ($N = 105$) must remain completely untouched until model architectures and validation thresholds are finalized.

---

## 9. Comprehensive Evaluation Metrics

Evaluation is decoupled into primary screening performance and auxiliary feature extraction quality.

### 9.1 Primary Task Metrics (Literature Screening)
Evaluated across four standardized operating thresholds:
* $\tau = 0.50$ (Default decision boundary)
* $\tau_{90}$ (Validation-calibrated threshold for $\text{Recall} \ge 90\%$)
* $\tau_{95}$ (Validation-calibrated threshold for $\text{Recall} \ge 95\%$)
* $\tau_{100}$ (Validation-calibrated threshold for $\text{Recall} = 100\%$)

**Reported Primary Metrics:**
1. **AUROC (Area Under ROC Curve):** Primary metric for overall ranking discrimination.
2. **PR-AUC (Average Precision):** Definitive metric under class imbalance.
3. **Sensitivity / Recall:** Proportion of true clinical trials captured ($\frac{\text{TP}}{\text{TP} + \text{FN}}$).
4. **Specificity:** Proportion of irrelevant studies correctly rejected ($\frac{\text{TN}}{\text{TN} + \text{FP}}$).
5. **Precision:** Positive predictive value ($\frac{\text{TP}}{\text{TP} + \text{FP}}$).
6. **F1-Score:** Harmonic mean of precision and recall.
7. **Workload Reduction %:** $\frac{N - (\text{TP} + \text{FP})}{N} \times 100\%$ (Percentage of papers human reviewers can safely skip).
8. **Number Needed to Screen (NNS):** $\frac{\text{TP} + \text{FP}}{\text{TP}}$ (Total papers screened per relevant trial identified; ideal = 1.0).

### 9.2 Auxiliary Task Metrics (25 Clinical Features)
1. **Per-Feature Performance:**
   - For each feature $k \in \{1, \dots, 25\}$: Binary AUROC, PR-AUC, Sensitivity, Specificity, Precision, and F1-score evaluated at threshold $0.50$.
2. **Global Aggregate Summaries:**
   - **Macro-Averaged F1 ($\text{Macro-F1}$):** Unweighted mean of F1 scores across all 25 features:
     $$\text{Macro-F1} = \frac{1}{25} \sum_{k=1}^{25} \text{F1}_k$$
     (Heavily penalizes poor performance on rare features like `proton_therapy`).
   - **Micro-Averaged F1 ($\text{Micro-F1}$):** Pooled instance-level F1 across all $25 \times N$ prediction instances:
     $$\text{Micro-F1} = \frac{2 \sum \text{TP}_k}{2 \sum \text{TP}_k + \sum \text{FP}_k + \sum \text{FN}_k}$$
   - **Macro-Averaged AUROC:** Mean AUROC across features with valid positive and negative instances.

---

## 10. Technical Limitations, Risks & Open Decisions

### 10.1 Negative Transfer & Task Conflict Risk
* **Risk:** The 25 auxiliary features may compete for representation capacity in the 768-dimensional `[CLS]` embedding. Gradients from auxiliary tasks (e.g., detecting whether an EBRT dose is $\ge 72\text{ Gy}$) could conflict with gradients for general study relevance, leading to *negative transfer* and degraded primary screening recall.
* **Mitigation Protocol:** If validation tracking shows primary AUROC degrading below Stage 8 single-task baselines (AUROC $< 0.84$), implement auxiliary loss decay ($\lambda_k \to 0.005$) or gradient surgery (e.g., PCGrad).

### 10.2 Handling the Asymmetric "0" Label in Training
* **The Problem:** As established in Section 3.3, a label of `0` in `labeled-dataset-v2-aux25.csv` denotes *absence of explicit reporting in the abstract*, not verified clinical absence. Standard cross-entropy penalizes the model if it infers a feature that is clinically present in the trial but omitted from the abstract text.
* **Open Decision for Stage 9:**
  * *Option A (Standard Asymmetric Loss):* Train on all binary values as-is using ASL margin shifting ($m = 0.05$) to suppress easy negative gradients.
  * *Option B (Positive-Unlabeled / Label Smoothing):* Apply asymmetric label smoothing to negative instances ($y = 0 \to y = 0.05$) to reflect reporting uncertainty.
  * *Decision:* Option A is selected as the primary baseline for comparability, with Option B reserved for ablation if validation loss fails to converge.

### 10.3 Absence of Titles
* **The Trade-Off:** In Stage 5, removing article titles decreased AUROC by $\sim 0.02$ because titles often state primary trial intent (e.g., "A Randomized Phase III Trial..."). However, enforcing **Abstract-ONLY** input across all three arms guarantees an unconfounded assessment of multi-task information extraction from dense scientific text.

---

## 11. Summary Protocol Checklist for Stage 9 Execution

- [x] **File Creation:** Markdown protocol established at `reports/stage9/stage9_design_and_protocol.md`.
- [ ] **Data Pipeline Verification:** Ensure data loader matches `data/labeled-dataset-v2-aux25.csv` to `data/splits/stage5/` by PMID, confirming $N=334$ (Train), $N=84$ (Val), $N=105$ (Test).
- [ ] **Wording Preservation:** Enforce the exact verbatim 73-word PICO query for Arm 2 and Arm 3.
- [ ] **Modality Isolation:** Confirm abstract-only text input (no titles appended) across Arms 1, 2, and 3.
- [ ] **Head Parallelism:** Implement 26 parallel `nn.Linear(768, 1)` heads branching from a single shared encoder.
- [ ] **Leakage Isolation:** Maintain strict separation between validation threshold sweeping and final held-out test evaluation.
