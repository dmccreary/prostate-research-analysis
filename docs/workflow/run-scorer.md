# Run Scorer

```sh
python prostate-cancer-scorer.py
```

## Test Run on first 99 Papers

```
Loading data from output100.csv...
Successfully loaded 99 papers

Processing papers...
================================================================================
Paper 1: Score 52 - Outcomes of Observation vs Stereotactic Ablative Radiation f...
Paper 2: Score 28 - Considering the role of radical prostatectomy in 21st centur...
Paper 3: Score 43 - MRI-Targeted, Systematic, and Combined Biopsy for Prostate C...
Paper 4: Score 24 - HDR Prostate Brachytherapy...
Paper 5: Score 54 - Testosterone replacement therapy (TRT) and prostate cancer: ...

================================================================================
✓ Scoring completed successfully!
✓ Results saved to: output100_scored.csv

Score Statistics:
  Mean Score: 35.1
  Median Score: 35.0
  Standard Deviation: 16.4
  Range: 0 - 68

Score Distribution:
  High Quality (≥70):      0 papers (0.0%)
  Good Quality (50-69):   22 papers (22.2%)
  Moderate Quality (30-49): 39 papers (39.4%)
  Low Quality (<30):      38 papers (38.4%)

Top 10 Highest Scoring Papers:
Rank Score  Title
---- ------ ------------------------------------------------------------
1    68     Both comorbidity and worse performance status are assoc...
2    68     Evolution of definitive external beam radiation therapy...
3    66     Focal therapy for localized prostate cancer in the era ...
4    63     Techniques and Outcomes of Salvage Robot-Assisted Radic...
5    62     Patient-Reported Outcomes Through 5 Years for Active Su...
6    62     Cesium-131 prostate brachytherapy: A single institution...
7    58     Radical cytoreductive prostatectomy in men with prostat...
8    58     Pathologically Node-Positive Prostate Cancer: Casting f...
9    58     Definition of high-risk prostate cancer impacts oncolog...
10   58     Recent advances in de-intensification of radiotherapy i...

================================================================================
Analysis complete! Check output100_scored.csv for full results.
```

## Full Run on the Entire FIle

```sh
python prostate-cancer-scorer.py
```

```
Loading data from ../data/output.csv...
Successfully loaded 7338 papers

Processing papers...
================================================================================
Paper 1: Score 52 - Outcomes of Observation vs Stereotactic Ablative Radiation f...
Paper 2: Score 28 - Considering the role of radical prostatectomy in 21st centur...
Paper 3: Score 43 - MRI-Targeted, Systematic, and Combined Biopsy for Prostate C...
Paper 4: Score 24 - HDR Prostate Brachytherapy...
Paper 5: Score 54 - Testosterone replacement therapy (TRT) and prostate cancer: ...

================================================================================
✓ Scoring completed successfully!
✓ Results saved to: output-full-scored.csv

Score Statistics:
  Mean Score: 16.4
  Median Score: 10.0
  Standard Deviation: 14.7
  Range: 0 - 88

Score Distribution:
  High Quality (≥70):     26 papers (0.4%)
  Good Quality (50-69):   282 papers (3.8%)
  Moderate Quality (30-49): 1026 papers (14.0%)
  Low Quality (<30):      6004 papers (81.8%)

Top 10 Highest Scoring Papers:
Rank Score  Title
---- ------ ------------------------------------------------------------
1    88     Stereotactic body radiotherapy versus conventional/mode...
2    86     Metastasis, Mortality, and Quality of Life for Men With...
3    86     The role of radical prostatectomy and definitive extern...
4    86     Metastasis, Mortality, and Quality of Life for Men With...
5    85     Quantifying treatment selection bias effect on survival...
6    84     Comparative effectiveness of surgery versus external be...
7    80     External Beam Radiation Therapy (EBRT) and High-Dose-Ra...
8    79     Benefits and Risks of Primary Treatments for High-risk ...
9    79     Cohort study of high-intensity focused ultrasound in th...
10   79     Cohort study of high-intensity focused ultrasound in th...

================================================================================
Analysis complete! Check output-full-scored.csv for full results.
```