# ML Workflow for Paper Classification

This diagram shows the high-level steps to build a machine learning model that predicts whether a prostate cancer research paper should be recommended to patients based on the [inclusion criteria](../../inclusion-criteria.md) and [exclusion criteria](../../exclusion-criteria.md).

## Workflow Overview

```mermaid
flowchart TB
    subgraph DataCollection["1. Data Collection"]
        A1[("Accepted Papers\n(Positive)\n191 papers")]
        A2[("Rejected Papers\n(Negative)\n244 papers")]
        A3[Fetch Abstracts\nfrom PubMed]
        A1 --> A3
        A2 --> A3
    end

    subgraph FeatureExtraction["2. Feature Extraction"]
        B1[Text Features\n- Title\n- Abstract\n- Journal]
        B2[Metadata Features\n- Publication Year\n- Sample Size\n- Follow-up Duration]
        B3[Clinical Features\n- Treatment Modality\n- Risk Stratification\n- Endpoints Reported]
        A3 --> B1
        A3 --> B2
        A3 --> B3
    end

    subgraph Preprocessing["3. Preprocessing"]
        C1[Text Vectorization\nTF-IDF / Embeddings]
        C2[Handle Missing Data]
        C3[Balance Classes\nSMOTE / Undersampling]
        C4[Train/Test Split\n80/20]
        B1 --> C1
        B2 --> C2
        B3 --> C2
        C1 --> C3
        C2 --> C3
        C3 --> C4
    end

    subgraph ModelTraining["4. Model Training"]
        D1[Baseline Models\n- Logistic Regression\n- Random Forest]
        D2[Advanced Models\n- XGBoost\n- BERT Fine-tuning]
        D3[Hyperparameter\nTuning]
        C4 --> D1
        C4 --> D2
        D1 --> D3
        D2 --> D3
    end

    subgraph Evaluation["5. Evaluation"]
        E1[Metrics\n- Precision\n- Recall\n- F1 Score]
        E2[Cross-Validation\n5-Fold]
        E3[Error Analysis\n- False Positives\n- False Negatives]
        D3 --> E1
        D3 --> E2
        E1 --> E3
        E2 --> E3
    end

    subgraph Deployment["6. Prediction Pipeline"]
        F1[New Paper\nPMID Input]
        F2[Extract Features]
        F3[Apply Model]
        F4{Accept or\nReject?}
        F5[Recommend\nto Patients]
        F6[Flag for\nManual Review]
        F1 --> F2
        F2 --> F3
        F3 --> F4
        F4 -->|Accept| F5
        F4 -->|Reject| F6
    end

    E3 --> F1

    style DataCollection fill:#e8f5e9
    style FeatureExtraction fill:#e3f2fd
    style Preprocessing fill:#fff3e0
    style ModelTraining fill:#fce4ec
    style Evaluation fill:#f3e5f5
    style Deployment fill:#e0f7fa
```

## Inclusion Criteria Features

The model learns to identify papers that meet these criteria:

| Criterion | Feature Extraction Method |
|-----------|---------------------------|
| **Treatment Modality** | Keyword matching for 19 treatment types (prostatectomy, EBRT, brachytherapy, etc.) |
| **Risk Stratification** | Detection of D'Amico, NCCN, or Zelefsky classification mentions |
| **Proper Endpoints** | Keywords: BRFS, OS, MFS, CSS, biochemical recurrence |
| **EBRT Dose ≥72Gy** | Regex extraction of radiation dose values |
| **Sample Size** | Regex extraction: n=X, X patients, cohort of X |
| **Follow-up ≥5 years** | Regex extraction of median follow-up duration |
| **Peer Reviewed** | Journal name matching against known peer-reviewed journals |

## Exclusion Criteria Detection

Papers are flagged for rejection if they contain:

| Exclusion Reason | Detection Method |
|------------------|------------------|
| No risk stratification | Absence of D'Amico/NCCN/Zelefsky terms |
| Missing endpoints | No BRFS/OS/MFS/CSS mentions |
| Pathologic staging only | Keywords: "pathologic staging", "surgical staging" |
| Low EBRT dose | Extracted dose < 72Gy |
| Insufficient patients | Extracted n < 100 (low/int) or n < 50 (high risk) |
| Short follow-up | Extracted follow-up < 5 years |
| Not peer reviewed | Conference abstract, poster, presentation keywords |

## Model Performance Targets

| Metric | Target | Rationale |
|--------|--------|-----------|
| **Recall** | ≥95% | Don't miss good papers for patients |
| **Precision** | ≥80% | Minimize manual review burden |
| **F1 Score** | ≥85% | Balanced performance |

## Known Challenges

1. **Temporal Bias**: Positive papers (1999-2013) vs negative (2005-2022) - see [Year Distribution Report](../../reports/data-profiles/year-distribution.md)
2. **Class Imbalance**: 191 positive vs 244 negative papers
3. **Missing Abstracts**: Some papers lack abstracts in PubMed
4. **Implicit Criteria**: Some acceptance decisions may involve domain expertise not captured in text
