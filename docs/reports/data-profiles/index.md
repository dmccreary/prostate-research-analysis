# Data Profiles

Comparative analysis of positive (accepted) and negative (rejected) paper datasets to ensure balanced training data for classifiers.

## Reports

| Report | Description |
|--------|-------------|
| [Year Distribution](year-distribution.md) | Publication year comparison between datasets |

## Data Sources

| Dataset | File | Papers |
|---------|------|--------|
| Positive (Accepted) | `data/accepted-articles.xlsx` | 191 papers |
| Negative (Rejected) | `data/negative-data-set.json` | 244 papers |

## Key Findings

### Temporal Mismatch Warning

The positive and negative datasets have significantly different year distributions:

- **Positive papers**: 1999-2013 (mean: 2009)
- **Negative papers**: 2005-2022 (mean: 2013)
- **No positive papers from 2014-2024**

This temporal bias could cause a classifier to incorrectly reject newer papers simply because they are more recent. Consider:

1. Adding more recent positive papers (2014-2024)
2. Removing older positive papers (pre-2005) for balanced comparison
3. Using year as a feature to control for temporal effects
