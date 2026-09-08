# Data Folder

Sample data for the prostate cancer treatment research paper classifier.

## Files

- `Pubmed-Exports_2021_Final.xlsx` — raw PubMed export (source data)
- `positive-data-set.xlsx` — **positive examples**: PMIDs of papers that met the acceptance criteria, grouped by risk category
- `positive-data-set.json` — positive examples with full metadata (title, abstract, journal, year, risk_category), fetched via `src/build-positive-dataset.py`
- `negative-data-set.xlsx` / `negative-data-set.json` — **negative examples**: papers that did not meet criteria, with abstracts and metadata
- `labeled-dataset.csv` — positive + negative merged into one table (`label` column: 1=positive, 0=negative), built via `src/build-labeled-dataset.py`. This is the file to use for training/evaluating a classifier.
- `data.csv`, `output.csv`, `output100.csv`, `output-full-scored.csv`, `scored-sample-10.csv` — intermediate/scored outputs from the extraction and scoring pipeline

## Positive vs. negative example counts

| Set | Count |
|---|---|
| Positive (`positive-data-set.xlsx`) | 140 rows (deduplicated), **120 unique PMIDs** |
| Negative (`negative-data-set.json`) | 244 papers (243 with abstracts), split across 4 sheets: 2005 (56), 2010 (60), 2015 (60), 2020 (68) |

## Data quality findings (as of 2026-09-08)

### `positive-data-set.xlsx` structure
Single unnamed column containing a mix of:
- PubMed URLs (`pubmed.ncbi.nlm.nih.gov/<pmid>`)
- A few non-PubMed links (PMC article pages, one `redjournal.org` fulltext link)
- Section header rows dividing the list into risk groups: `LOW RISK`, `INTERMEDIATE RISK`, `HIGH RISK`

### Duplicates within the positive set — resolved
Originally 191 PubMed links resolved to only 120 unique PMIDs (46 PMIDs appeared 2–3 times each, likely from being cross-listed under multiple risk-group sections). **Deduplicated on 2026-09-08**: 71 duplicate rows removed (211 → 140 rows), keeping the first occurrence of each PMID and preserving the risk-group header rows.

### Overlap between positive and negative sets
Only **1 PMID overlaps** between the positive and negative sets: [`15774239`](https://pubmed.ncbi.nlm.nih.gov/15774239/). This paper is currently labeled as both accepted and negative — worth manually reviewing to resolve the conflicting label before using both sets for training/evaluation.

## Recommendations
- Resolve the label conflict for PMID `15774239`.
- Consider converting `positive-data-set.xlsx` into a structured format (PMID + risk group) similar to `negative-data-set.json`, rather than a flat list of URLs with inline section headers.

## Building the labeled training set

```bash
cd src
python build-positive-dataset.py --email dan.mccreary@gmail.com   # fetches abstracts for positive-data-set.xlsx -> positive-data-set.json
python build-labeled-dataset.py                                    # merges positive + negative -> ../data/labeled-dataset.csv
```

`labeled-dataset.csv` (364 rows: 120 positive, 244 negative) is the input for training a classifier.

A baseline TF-IDF + Logistic Regression model has been trained and compared against the existing rule-based scorer — see [docs/reports/baseline-model-report.md](../docs/reports/baseline-model-report.md). Re-run it with:

```bash
cd src
python train-baseline-model.py
```

## Untapped raw data (found 2026-09-08)

`Pubmed-Exports_2021_Final.xlsx` contains **10,834 unique PMIDs**, but only 6,622 of them have ever been through abstract extraction/scoring (`output-full-scored.csv`). **9,665 PMIDs (89%) were never extracted.**

`src/filter-unextracted-papers.py` fetches abstracts for those unextracted PMIDs and applies the pre-screening filters agreed by Alex (Richard Hsi) and Mark Nguyen in `TODO.md` (June 2026):
- Exclude if the abstract contains "palliative", "metastatic", or "hormone resistant"
- Require the abstract to contain "prostate cancer", "prostate neoplasia", or "prostate carcinoma"

Run results (2026-09-08, all 9,665 unextracted PMIDs):

| Outcome | Count | % |
|---|---|---|
| Passed filters | 4,211 | 43.6% |
| Missing required phrase | 3,083 | 31.9% |
| Excluded — "metastatic" | 1,291 | 13.4% |
| No abstract available | 1,019 | 10.5% |
| Excluded — "palliative" | 60 | 0.6% |
| Excluded — "hormone resistant" | 1 | <0.1% |

Outputs:
- `unextracted-with-abstracts.csv` — all 9,665 fetched PMIDs with abstracts and `filter_pass`/`filter_reason` columns
- `filtered-candidates.csv` — the 4,211 papers that passed the filters, ready to run through `prostate-cancer-scorer.py` and/or be selected for manual labeling to grow the training set beyond the current 364 papers

Re-run with:
```bash
cd src
python filter-unextracted-papers.py --email dan.mccreary@gmail.com
```

### Rule-based scoring of the filtered candidates

`filtered-candidates.csv` (4,211 papers) was scored with the existing rule-based scorer (`prostate-cancer-scorer.py`) — output saved to `filtered-candidates-scored.csv`, sorted by score descending.

| Score range | Count |
|---|---|
| 0–10 | 2,073 |
| 11–20 | 854 |
| 21–30 | 449 |
| 31–40 | 268 |
| 41–50 | 152 |
| 51–60 | 91 |
| 61–70 | 44 |
| 71–80 | 20 |
| 81–90 | 4 |

Mean score 15.2, median 10. Top-scoring papers (score ≥ 80) are randomized/phase-II trials with clear treatment modality + endpoint language, e.g. PMID `34740768` (score 90, salvage brachytherapy phase 2 trial) and `33909100` (score 86, trimodal therapy trial). Bottom-scoring papers are mostly off-topic reviews that slipped past the phrase filter (breast cancer, lncRNA reviews, etc.) or narrow technique write-ups without quality/endpoint language.

This ranked list is a good source for manual labeling: the top ~100 are strong positive-candidates, the bottom few hundred are strong negative-candidates, and the middle band (score 30–60) is where manual review adds the most value.
