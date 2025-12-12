# Negative Papers Excel to JSON Conversion

**Date:** 2025-12-12
**Task:** Convert `data/negative-data-set.xlsx` to JSON with PubMed abstracts

## Summary

Created a Python script to convert the negative dataset Excel file (with multiple sheets) into a structured JSON file, fetching abstracts from PubMed via the NCBI E-utilities API.

## Input File

- **File:** `data/negative-data-set.xlsx`
- **Sheets:** 4 tabs (2005, 2010, 2015, 2020)
- **Columns:** row_id, pmid, title, citation, author, url (no header row)

| Sheet | Papers |
|-------|--------|
| 2005  | 56     |
| 2010  | 60     |
| 2015  | 60     |
| 2020  | 68     |
| **Total** | **244** |

## Output File

- **File:** `data/negative-data-set.json`
- **Size:** ~500 KB
- **Papers with abstracts:** 243/244 (one editorial comment has no abstract in PubMed)

### JSON Structure

```json
{
  "metadata": {
    "source": "../data/negative-data-set.xlsx",
    "total_papers": 244,
    "papers_with_abstracts": 243,
    "dataset_type": "negative",
    "sheets": ["2005", "2010", "2015", "2020"],
    "papers_per_sheet": {"2005": 56, "2010": 60, "2015": 60, "2020": 68}
  },
  "papers": [
    {
      "pmid": 15758704,
      "title": "Laparoscopic radical prostatectomy",
      "author": "Trabulsi EJ",
      "abstract": "PURPOSE: After the pioneering period...",
      "journal": "J Urol",
      "year": "2005",
      "volume": "173",
      "pages": "1072-9",
      "doi": "doi: 10.1097/01.ju.0000154970.63147.90",
      "url": "https://www.ncbi.nlm.nih.gov/pubmed/15758704",
      "year_group": "2005",
      "dataset": "negative"
    }
  ]
}
```

## Script Created

- **Location:** `src/xlsx-to-json.py`
- **Dependencies:** pandas, openpyxl, biopython, tqdm

### Usage

```bash
cd src
python xlsx-to-json.py --email dan.mccreary@gmail.com

# With custom input/output:
python xlsx-to-json.py --email dan.mccreary@gmail.com \
  --input ../data/your-file.xlsx \
  --output ../data/your-output.json
```

### Features

- Reads all sheets/tabs from Excel file
- Parses citation strings into structured fields (journal, year, volume, pages, DOI)
- Fetches abstracts from PubMed in batches (50 PMIDs per request)
- Respects NCBI rate limits (0.34s delay between requests)
- Tracks source sheet as `year_group` field
- Outputs UTF-8 encoded JSON

## Missing Abstract

One paper (PMID 21056265) has no abstract - it's an editorial comment:
> "Editorial comment. Evaluation of combined oncological and functional outcomes after radical prostatectomy: trifecta rate of achieving continence, potency and cancer control--a literature review"

## Requirements Added

```bash
pip install openpyxl  # Required for reading .xlsx files
```
