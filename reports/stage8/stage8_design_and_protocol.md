# Stage 8 Design & Protocol: MedCPT Cross-Encoder Literature Screening

**Experiment Suite:** Stage 8 — Domain-Specialized Cross-Encoder Literature Screening for Localized Prostate Cancer  
**Model Family:** `ncbi/MedCPT-Cross-Encoder` (NCBI / NLM, Jin et al., *Bioinformatics* 2023)  
**Underlying Architecture:** 12-layer Transformer, 768 hidden dimensions, 12 attention heads, 512 max tokens  
**Pre-Training Heritage:** Initialized from `microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract` + Pre-trained on **255M PubMed search-log relevance interactions**  
**Evaluation Target Cohorts:**
1. **Held-Out Test Set**: `data/splits/stage5/test.csv` ($N = 105$, 56 Positives, 49 Negatives, Seed 2025)
2. **External Prospective 2021 Cohort**: `data/2021 labeld.csv` ($N = 117$, 28 Positives, 89 Negatives)
3. **Internal Validation Cohort (Threshold Calibration Only)**: `data/splits/stage5/val.csv` ($N = 84$, 45 Positives, 39 Negatives)

---

## 1. Rationale & Research Objectives

In systematic reviews (PRISMA framework), literature screening requires identifying clinical trials that meet multi-factorial criteria:
- **P (Population):** Localized prostate cancer (low, intermediate, or high risk).
- **I (Interventions):** Primary curative treatment modalities (Radical Prostatectomy, EBRT $\ge 72$ Gy, Brachytherapy, Cryotherapy, HIFU).
- **C (Comparisons):** Comparative cohort or randomized study designs (excluding single-arm observational audits).
- **O (Outcomes):** Long-term oncologic survival outcomes (BRFS, OS, CSS, MFS) with minimum follow-up $\ge 5.0$ years.

### Why MedCPT Cross-Encoder?
1. **Pretrained Query-Article Interaction:** While vanilla PubMedBERT was pretrained solely on masked language modeling (predicting masked words in isolated abstracts), `ncbi/MedCPT-Cross-Encoder` was trained explicitly to evaluate whether a PubMed article is relevant to a user query.
2. **Solving the Small-Data ($N=335$) Bottleneck:** Training a 110M parameter Cross-Encoder from scratch on 335 examples forced vanilla PubMedBERT to learn bidirectional cross-attention patterns de novo. MedCPT provides pre-optimized cross-attention weights.
3. **Native 512-Token Sequence Budget:** As proven in Stage 5.5, clinical survival endpoints cluster at the end of abstracts. MedCPT supports 512 tokens natively, avoiding the 63.9% truncation penalty of 384-token baselines.

---

## 2. Multi-Phase Roadmap for Stage 8

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ Stage 8.0: Zero-Shot Evaluation (No Training)                               │
│ - Evaluate out-of-the-box ncbi/MedCPT-Cross-Encoder                         │
│ - Contrast PICO Natural Language Query vs. PubMed Search-Style Query        │
│ - Zero-leakage threshold calibration on Val (N=84) -> Test (N=105) & 2021   │
└─────────────────────────────────────┬───────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ Stage 8.1: Query Formulation & Prompt Adaptation                            │
│ - Test dense semantic representations vs. rule-augmented text queries       │
│ - Determine optimal token allocation between Query (Seg A) and Paper (Seg B)│
└─────────────────────────────────────┬───────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ Stage 8.2: Conservative Fine-Tuning (Preventing Catastrophic Forgetting)    │
│ - Two-stage warmup: Head-only warmup followed by end-to-end Low-LR (5e-6)   │
│ - Cost-sensitive weighted BCE vs. Asymmetric Loss (ASL)                     │
└─────────────────────────────────────┬───────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ Stage 8.3: Full Hybrid Architecture (MedCPT + 7 Clinical Rules + ASL)       │
│ - Concatenate 768-d [CLS] from MedCPT with 7-d structured rule vector       │
│ - Joint classification via MLP Head + Asymmetric Loss                       │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Historical Benchmark Reference Matrix

To contextualize Stage 8 performance, the table below provides the authoritative test-set metrics across all previous project milestones:

| Stage | Model Description | Operating Point | Recall | Specificity | Precision | F1 Score | AUROC | PR-AUC | Workload Reduction % |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Stage 1** | TF-IDF + Logistic Regression | Default (0.50) | 82.14% | 59.18% | 69.70% | 0.7541 | 0.8145 | 0.8172 | 34.29% |
| **Stage 5** | PubMedBERT 384 tokens | Recall = 100% | 100.00% | 8.16% | 55.45% | 0.7134 | 0.8466 | 0.8432 | 3.81% |
| **Stage 5.5** | PubMedBERT 512 tokens | Recall = 100% | 100.00% | 22.45% | 59.57% | 0.7467 | 0.8364 | 0.8212 | 10.48% |
| **Stage 6** | SPECTER2 + Logistic Regression | Recall = 100% | 100.00% | 22.45% | 59.57% | 0.7467 | 0.8145 | 0.8172 | 10.48% |
| **Stage 7.0** | PICO Cross-Encoder (512, Weighted) | Default (0.50) | 94.64% | 61.22% | 73.61% | 0.8281 | 0.8397 | 0.7979 | 31.43% |
| **Stage 7.0** | PICO Cross-Encoder (512, Weighted) | Recall = 100% | **100.00%** | **46.94%** | **68.29%** | **0.8116** | 0.8397 | 0.7979 | **21.90%** |
| **Stage 7.4** | ASL Hybrid + 7 Clinical Rules | Default (0.50) | 92.86% | 65.31% | 75.36% | **0.8320** | **0.8859** | **0.9004** | **34.29%** |
| **Stage 7.4** | ASL Hybrid + 7 Clinical Rules | Recall = 100% | 100.00% | 28.57% | 61.54% | 0.7619 | **0.8859** | **0.9004** | 13.33% |

---

## 4. Stage 8.0: Zero-Shot Protocol Specifications

### Input Sequence Formulation
For every candidate study, the input sequence is packed as:
$$\mathbf{x} = \text{[CLS]} \circ \mathbf{q} \circ \text{[SEP]} \circ \mathbf{d} \circ \text{[SEP]}$$

Where:
* Segment A (`token_type_ids = 0`): Query $\mathbf{q}$
  * **Option A (Natural Language PICO, 73 tokens):**  
    `"Screening criteria for localized prostate cancer: Primary curative treatment (radical prostatectomy, EBRT >= 72 Gy, brachytherapy, cryo, HIFU). Must be a comparative trial (exclude single-arm audits, post-treatment salvage, metastatic, or palliative). Reports oncologic survival (BRFS, OS, CSS) with follow-up >= 5 years."`
  * **Option B (PubMed Search Query, 18 tokens):**  
    `"localized prostate cancer primary curative treatments radical prostatectomy radiotherapy brachytherapy survival comparative study"`
* Segment B (`token_type_ids = 1`): Document $\mathbf{d} = \text{Cleaned Title} + \text{". "} + \text{Cleaned Abstract}$ (up to 430 tokens).

### Inference & Scoring
$$\hat{z} = \text{ClassifierHead}(\mathbf{h}_{\text{CLS}}) \in \mathbb{R}$$
$$\hat{p} = \sigma(\hat{z}) = \frac{1}{1 + e^{-\hat{z}}} \in [0, 1]$$

---

## 5. Critical Pre-Training Checklist (Key Points Before Fine-Tuning)

Before running any fine-tuning in Stage 8.2 or 8.3, the following configurations must be set:

1. **Learning Rate & Optimizer Strategy (Catastrophic Forgetting Risk):**
   * *Problem:* MedCPT has spent 255M steps learning ranking representations. Training with standard BERT LR ($2 \times 10^{-5}$ or $3 \times 10^{-5}$) on only 335 examples will rapidly destroy these pre-trained attention features.
   * *Action:* Use a strictly conservative LR: $\eta \in [3 \times 10^{-6}, 8 \times 10^{-6}]$ with Linear Warmup ($10\%$ of steps) and AdamW (`weight_decay = 0.01`).
2. **Layer-Freezing / Staged Training:**
   * *Phase 1 (Epochs 1–2):* Freeze all 12 BERT transformer layers (`requires_grad = False`). Update only the classifier head / projection layers.
   * *Phase 2 (Epochs 3–6):* Unfreeze top 4 transformer layers (Layers 9–12) with a tiny LR ($5 \times 10^{-6}$).
3. **Loss Function Selection:**
   * Standard BCE suffers from extreme False Positive drift when pursuing 100% recall.
   * Use **Asymmetric Loss (ASL)** with $\gamma_- = 2.0$, $\gamma_+ = 0.0$, and probability margin clip $m = 0.05$ (established in Stage 7.4).
4. **VRAM Optimization on GTX 1050 Ti (4GB):**
   * Maintain physical `BATCH_SIZE = 4` with `gradient_accumulation_steps = 2` (effective batch size = 8).
   * Sequence length locked at `MAX_LENGTH = 512`.
   * Clear PyTorch CUDA cache between validation sweeps (`torch.cuda.empty_cache()`, `gc.collect()`).
