# Year Distribution: Positive vs Negative Datasets

**Generated:** 2025-12-12 15:52

## Summary

| Metric | Positive (Accepted) | Negative (Rejected) |
|--------|---------------------|---------------------|
| Total Papers | 120 | 244 |
| Year Range | 1999-2013 | 2005-2022 |
| Mean Year | 2009.1 | 2013.1 |
| Median Year | 2010 | 2015 |

## Year Distribution Table

| Year | Positive | Negative | Pos % | Neg % |
|------|----------|----------|-------|-------|
| 1999 | 1 | 0 | 0.8% | 0.0% |
| 2000 | 1 | 0 | 0.8% | 0.0% |
| 2001 | 1 | 0 | 0.8% | 0.0% |
| 2002 | 3 | 0 | 2.5% | 0.0% |
| 2003 | 3 | 0 | 2.5% | 0.0% |
| 2004 | 5 | 0 | 4.2% | 0.0% |
| 2005 | 8 | 52 | 6.7% | 21.3% |
| 2006 | 8 | 4 | 6.7% | 1.6% |
| 2007 | 9 | 0 | 7.5% | 0.0% |
| 2008 | 7 | 0 | 5.8% | 0.0% |
| 2009 | 9 | 0 | 7.5% | 0.0% |
| 2010 | 14 | 41 | 11.7% | 16.8% |
| 2011 | 7 | 13 | 5.8% | 5.3% |
| 2012 | 21 | 5 | 17.5% | 2.0% |
| 2013 | 23 | 1 | 19.2% | 0.4% |
| 2015 | 0 | 43 | 0.0% | 17.6% |
| 2016 | 0 | 17 | 0.0% | 7.0% |
| 2020 | 0 | 62 | 0.0% | 25.4% |
| 2021 | 0 | 5 | 0.0% | 2.0% |
| 2022 | 0 | 1 | 0.0% | 0.4% |
| **Total** | **120** | **244** | **100%** | **100%** |

## 5-Year Period Distribution

| Period | Positive | Negative | Pos % | Neg % |
|--------|----------|----------|-------|-------|
| 2000-2004 | 13 | 0 | 10.8% | 0.0% |
| 2005-2009 | 41 | 56 | 34.2% | 23.0% |
| 2010-2014 | 65 | 60 | 54.2% | 24.6% |
| 2015-2019 | 0 | 60 | 0.0% | 24.6% |
| 2020-2024 | 0 | 68 | 0.0% | 27.9% |

## Visual Distribution (by 5-year periods)

```
Period      Positive                          Negative
2000-2004  █████                                                        
2005-2009  ██████████████████             ████████████████████████      
2010-2014  ████████████████████████████   ██████████████████████████    
2015-2019                                 ██████████████████████████    
2020-2024                                 ██████████████████████████████
```

## Temporal Mismatch Warning

**The datasets have significantly different year distributions.** This could bias a classifier to reject newer papers simply because they are more recent:

- Positive papers end in 2013; negative papers continue through 2022
- 52% of negative papers (128) are from 2015-2022, with **zero** positive papers in this range
- A classifier might learn "newer = rejected" rather than actual quality criteria

### Recommendations

1. Add more recent positive papers (2014-2024) to balance the datasets
2. Consider filtering to overlapping years (2005-2013) for initial model training
3. Use publication year as a control variable to prevent temporal confounding

## Notes

- **Positive dataset**: 191 papers from `data/accepted-articles.xlsx` (120 with retrievable years)
- **Negative dataset**: 244 papers from `data/negative-data-set.json`
- Publication years fetched from PubMed metadata
- 71 positive papers missing year data (PMIDs may be invalid or have incomplete metadata)
