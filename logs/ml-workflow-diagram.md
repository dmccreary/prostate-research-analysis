# ML Workflow Diagram Session Log

**Date:** 2025-12-12
**Task:** Create Mermaid diagram showing ML workflow for paper classification

## Summary

Created a MicroSim with a Mermaid flowchart diagram illustrating the machine learning pipeline for predicting whether prostate cancer research papers should be recommended to patients based on the inclusion/exclusion criteria.

## Files Created

### MicroSim
- `docs/sims/ml-workflow/index.md` - Main diagram and documentation

### Updated Files
- `docs/sims/index.md` - Added link to ML Workflow
- `mkdocs.yml` - Added MicroSims navigation section and Mermaid support

## Diagram Structure

The flowchart contains 6 main stages:

```
1. Data Collection
   └── Positive papers (191) + Negative papers (244)
   └── Fetch abstracts from PubMed

2. Feature Extraction
   └── Text features (title, abstract, journal)
   └── Metadata features (year, sample size, follow-up)
   └── Clinical features (treatment, risk stratification, endpoints)

3. Preprocessing
   └── TF-IDF / Embeddings vectorization
   └── Handle missing data
   └── Class balancing (SMOTE/undersampling)
   └── Train/test split (80/20)

4. Model Training
   └── Baseline: Logistic Regression, Random Forest
   └── Advanced: XGBoost, BERT fine-tuning
   └── Hyperparameter tuning

5. Evaluation
   └── Metrics: Precision, Recall, F1
   └── 5-fold cross-validation
   └── Error analysis (FP/FN)

6. Prediction Pipeline
   └── New paper PMID input
   └── Feature extraction → Model → Accept/Reject decision
```

## Inclusion Criteria Mapped to Features

| Criterion | Feature Extraction |
|-----------|-------------------|
| Treatment Modality | Keyword matching (19 types) |
| Risk Stratification | D'Amico/NCCN/Zelefsky detection |
| Proper Endpoints | BRFS, OS, MFS, CSS keywords |
| EBRT Dose ≥72Gy | Regex extraction |
| Sample Size | Regex: n=X, X patients |
| Follow-up ≥5 years | Regex extraction |
| Peer Reviewed | Journal name matching |

## MkDocs Configuration Updates

### Navigation Added
```yaml
- MicroSims:
  - Intro: sims/index.md
  - ML Workflow: sims/ml-workflow/index.md
```

### Mermaid Support Added
```yaml
markdown_extensions:
  - pymdownx.superfences:
      custom_fences:
        - name: mermaid
          class: mermaid
          format: !!python/name:pymdownx.superfences.fence_code_format

extra_javascript:
  - https://unpkg.com/mermaid@10/dist/mermaid.min.js
```

## Model Performance Targets

| Metric | Target | Rationale |
|--------|--------|-----------|
| Recall | ≥95% | Don't miss good papers |
| Precision | ≥80% | Minimize manual review |
| F1 Score | ≥85% | Balanced performance |

## Known Challenges Documented

1. **Temporal Bias** - Positive papers end at 2013, negative continue to 2022
2. **Class Imbalance** - 191 positive vs 244 negative
3. **Missing Abstracts** - Some papers lack PubMed abstracts
4. **Implicit Criteria** - Domain expertise not captured in text

## Preview

```bash
mkdocs serve
```

Navigate to **MicroSims > ML Workflow** to view the interactive diagram.
