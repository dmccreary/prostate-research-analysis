# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Python-based classifier for prostate cancer treatment research papers. Analyzes PubMed articles and scores them based on treatment modality identification and study quality criteria. Includes MkDocs documentation site.

## Development Environment

```bash
conda create -n "doc-classifier" python=3
conda activate doc-classifier
pip install pandas tqdm Bio chardet scikit-learn umap-learn numpy mkdocs mkdocs-material
```

## Commands

### Data Pipeline (run from `src/`)

```bash
# Extract abstracts from PubMed (requires email for NCBI API)
python extractor3.py --input_file ../data/input.csv --output_file output.csv --email your@email.com

# Optional: specify encoding for non-UTF8 input files (default: cp1252)
python extractor3.py --input_file ../data/input.csv --output_file output.csv --email your@email.com --encoding utf-8

# Score papers (reads ../data/output.csv, outputs output-full-scored.csv)
python prostate-cancer-scorer.py

# Generate embeddings for visualization
cd create-embedding && python create-embeddings.py
```

### Documentation Site

```bash
mkdocs serve        # Local preview at http://127.0.0.1:8000
mkdocs build        # Build static site
mkdocs gh-deploy    # Deploy to GitHub Pages
```

## Architecture

### Scoring System (0-100 scale)

The scorer (`prostate-cancer-scorer.py`) evaluates papers on three criteria:
- **Treatment modality** (40 pts): Identifies 17 treatments from radical prostatectomy to HIFU
- **Study quality** (40 pts): Peer review, proper endpoints (BRFS, OS, MFS, CSS), stratification (D'Amico, NCCN)
- **Numerical criteria** (20 pts): Sample size (≥100 for low/int risk, ≥50 for high), follow-up (≥5 years), EBRT dose (≥72Gy)

Treatment modalities and acceptance criteria are defined in `docs/summary-criteria.md`.

### Data Flow

1. **Input**: CSV with columns `pmid`, `title`, `author`, `details`
2. **Extract**: `extractor3.py` fetches abstracts via NCBI E-utilities API (0.34s rate limit)
3. **Score**: `prostate-cancer-scorer.py` adds `score` column based on treatment/quality criteria
4. **Visualize**: `create-embeddings.py` generates TF-IDF + UMAP embeddings for `papers-embeddings.json`

### Key Files

- `src/extractor3.py` - PubMed abstract extraction with batch processing and logging
- `src/prostate-cancer-scorer.py` - Main scoring algorithm with regex-based criteria matching
- `src/create-embedding/create-embeddings.py` - TF-IDF vectorization + dimensionality reduction
- `docs/data-viewers/` - Interactive HTML/JS visualizations (similarity plots, Sankey diagrams)