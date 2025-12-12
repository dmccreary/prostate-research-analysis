# Data Profile Reports Session Log

**Date:** 2025-12-12
**Task:** Create data profile reports comparing positive and negative datasets

## Summary

Created a new reports section to analyze and compare the positive (accepted) and negative (rejected) paper datasets, focusing on year distribution to identify potential biases for classifier training.

## Files Created

### Reports Directory
```
docs/reports/data-profiles/
├── index.md                    # Overview and key findings
├── year-distribution.md        # Detailed year comparison tables
└── year-distribution-chart.md  # Interactive Chart.js visualization
```

### Script
- `src/generate-data-profiles.py` - Generates year distribution reports by:
  - Loading positive papers from `data/accepted-articles.xlsx`
  - Loading negative papers from `data/negative-data-set.json`
  - Fetching publication years from PubMed for positive papers
  - Generating markdown reports with tables and ASCII charts

## Data Sources Analyzed

| Dataset | File | Papers | Year Range |
|---------|------|--------|------------|
| Positive | `data/accepted-articles.xlsx` | 191 (120 with years) | 1999-2013 |
| Negative | `data/negative-data-set.json` | 244 | 2005-2022 |

## Key Finding: Temporal Mismatch

**Critical issue identified:** The datasets have significantly different year distributions.

| Metric | Positive | Negative |
|--------|----------|----------|
| Year Range | 1999-2013 | 2005-2022 |
| Mean Year | 2009.1 | 2013.1 |
| Median Year | 2010 | 2015 |
| Papers 2015+ | **0 (0%)** | **128 (52%)** |

This temporal bias could cause a classifier to:
- Reject newer papers regardless of content quality
- Learn "year" as a feature instead of actual acceptance criteria

### Recommendations in Report
1. Add more recent positive papers (2014-2024)
2. Filter to overlapping years (2005-2013) for initial training
3. Use publication year as a control variable

## Chart.js Visualization

Created interactive scatter plot (`year-distribution-chart.md`):
- X-axis: Publication year (1998-2024)
- Y-axis: Number of papers (0-70)
- Green dots: Positive papers with connecting line
- Red dots: Negative papers with connecting line
- Hover tooltips showing paper counts

## MkDocs Navigation Updated

Added to `mkdocs.yml`:
```yaml
- Reports:
  - Data Profiles: reports/data-profiles/index.md
  - Year Distribution: reports/data-profiles/year-distribution.md
  - Year Distribution Chart: reports/data-profiles/year-distribution-chart.md
```

## Script Usage

```bash
cd src
python generate-data-profiles.py --email dan.mccreary@gmail.com
```

### Options
- `--positive` - Path to positive dataset (default: `../data/accepted-articles.xlsx`)
- `--negative` - Path to negative dataset JSON (default: `../data/negative-data-set.json`)
- `--output-dir` - Output directory for reports (default: `../docs/reports/data-profiles`)

## Positive Dataset Structure

The `accepted-articles.xlsx` file contains:
- Single column of PubMed URLs
- Category headers embedded in data: "LOW RISK", "INTERMEDIATE RISK", "HIGH RISK"
- 191 total papers across risk categories

## Notes

- 71 positive papers (37%) missing year data - PMIDs may be invalid or have incomplete PubMed metadata
- Negative dataset years come from citation parsing (already in JSON)
- Positive dataset years fetched live from PubMed API
