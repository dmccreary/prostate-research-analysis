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

`labeled-dataset.csv` (364 rows: 120 positive, 244 negative) is the input for training a classifier — see project TODO for next steps (train/test split, baseline TF-IDF model, evaluation against the existing rule-based scorer).
