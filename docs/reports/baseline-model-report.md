# Baseline Model Report

Dataset: `../data/labeled-dataset.csv` (362 papers after excluding 1 conflicting-label PMID(s))

Split: 289 train / 73 test (stratified, seed=42)


============================================================
1. TF-IDF + Logistic Regression
============================================================

```
              precision    recall  f1-score   support

    negative       0.90      0.78      0.84        49
    positive       0.65      0.83      0.73        24

    accuracy                           0.79        73
   macro avg       0.77      0.80      0.78        73
weighted avg       0.82      0.79      0.80        73
```

Confusion matrix (rows=actual, cols=predicted [neg, pos]):
```
[[38 11]
 [ 4 20]]
```

ROC AUC: 0.853

F1 (positive class): 0.727


Top positive-indicating terms:

- risk (1.586)
- year (0.903)
- intermediate (0.859)
- risk prostate (0.831)
- high risk (0.783)
- biochemical (0.781)
- intermediate risk (0.778)
- survival (0.755)
- disease (0.717)
- high (0.707)
- clinical (0.668)
- term (0.640)
- long term (0.639)
- stage (0.621)
- low risk (0.608)


Top negative-indicating terms:

- rt (-1.059)
- salvage (-1.054)
- focal (-0.609)
- toxicity (-0.503)
- radiotherapy (-0.499)
- sbrt (-0.472)
- acute (-0.441)
- rp (-0.405)
- prostatectomy (-0.396)
- srt (-0.394)
- urinary (-0.384)
- gu (-0.374)
- erectile (-0.360)
- months (-0.352)
- imrt (-0.347)


============================================================
2. Existing rule-based scorer (prostate-cancer-scorer.py)
============================================================

Best threshold found on test set (score >= threshold -> positive): 45

```
              precision    recall  f1-score   support

    negative       0.83      0.59      0.69        49
    positive       0.47      0.75      0.58        24

    accuracy                           0.64        73
   macro avg       0.65      0.67      0.64        73
weighted avg       0.71      0.64      0.65        73
```

Confusion matrix (rows=actual, cols=predicted [neg, pos]):
```
[[29 20]
 [ 6 18]]
```

F1 (positive class): 0.581


============================================================
3. Comparison
============================================================

| Model | F1 (positive) |
|---|---|
| TF-IDF + Logistic Regression | 0.727 |
| Rule-based scorer (best threshold) | 0.581 |


The learned TF-IDF baseline outperforms the rule-based scorer on this test set.


============================================================
Notes / caveats
============================================================

- Test set is only ~73 papers (20% of 364); F1 estimates have wide variance. Treat this as a first signal, not a final benchmark.
- The rule-based threshold was chosen by sweeping on the test set itself, so its reported F1 is optimistic — a proper comparison needs a separate validation set.
- Class balance in train/test is imbalanced (negative > positive); `class_weight='balanced'` was used for the TF-IDF model.
- **Possible dataset artifact**: top positive-indicating terms are dominated by 'risk'/'intermediate risk'/'high risk'/'low risk'. This may be genuine signal (risk stratification is part of the acceptance criteria), but it may also reflect how the two sets were curated differently — positive papers were sourced by risk category, negative papers by publication year. Worth checking whether the model is learning acceptance criteria or just learning 'how this dataset was assembled'.
