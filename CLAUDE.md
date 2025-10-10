# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This repository implements a Python-based classifier for prostate cancer treatment research papers. The system analyzes research articles from PubMed and scores them based on treatment modality identification and study quality criteria. The project includes data extraction, scoring algorithms, visualization tools, and a documentation website built with MkDocs.

## Development Environment Setup

Create and activate a Conda environment:
```bash
conda deactivate
conda create -n "doc-classifier" python=3
conda activate doc-classifier
pip install pandas tqdm Bio chardet scikit-learn umap-learn numpy
```

Additional packages for web functionality:
```bash
pip install mkdocs mkdocs-material
```

## Core Development Commands

### Python Scripts (run from `src/` directory)

**Extract abstracts from PubMed data:**
```bash
cd src
python extractor3.py --input_file ../data/input.csv --output_file output.csv --email your@email.com
```

**Score research papers:**
```bash
cd src
python prostate-cancer-scorer.py
```

**Generate embeddings for visualization:**
```bash
cd src/create-embedding
python create-embeddings.py
```

### Documentation Website

**Serve documentation locally:**
```bash
mkdocs serve
```

**Build documentation site:**
```bash
mkdocs build
```

**Deploy to GitHub Pages:**
```bash
mkdocs gh-deploy
```

## Architecture Overview

### Core Components

**Data Pipeline (src/):**
- `extractor3.py` - PubMed abstract extraction using NCBI E-utilities API
- `extractor4.py` - Enhanced version with additional features  
- `prostate-cancer-scorer.py` - Main scoring algorithm implementation
- `check-special-chars.py` - Data validation utility

**Scoring System:**
The scoring algorithm evaluates papers on three criteria (0-100 total):
- Treatment modality identification (40 points max) - Based on 17 treatment modalities from radical prostatectomy to HIFU
- Study quality indicators (40 points max) - Peer review status, endpoints, stratification
- Numerical criteria compliance (20 points max) - Sample size, follow-up duration, radiation dose

**Visualization & Analysis (src/create-embedding/):**
- `create-embeddings.py` - TF-IDF + UMAP/t-SNE dimensionality reduction for 2D visualization
- Generates `papers-embeddings.json` for interactive web visualizations

**Data Viewers (docs/data-viewers/):**
- Interactive HTML/JavaScript visualizations for similarity analysis
- Sankey diagrams for treatment flow analysis
- 2D scatter plots of paper embeddings

### Data Flow

1. Input: CSV files with PubMed IDs (PMIDs)
2. Extract abstracts via `extractor3.py` using NCBI API
3. Score papers using `prostate-cancer-scorer.py` based on treatment and quality criteria
4. Generate embeddings for similarity analysis via `create-embeddings.py`
5. Visualize results through interactive web tools

### File Organization

```
├── src/                     # Core Python scripts
│   ├── create-embedding/    # Visualization and similarity analysis
│   └── score-papers/        # Alternative scoring implementations
├── data/                    # CSV data files and outputs
├── docs/                    # MkDocs documentation source
│   ├── workflow/           # Step-by-step process documentation
│   └── data-viewers/       # Interactive visualization tools
└── site/                   # Generated MkDocs site
```

### Key Configuration Files

- `mkdocs.yml` - Documentation site configuration with Material theme
- `docs/summary-criteria.md` - Defines the 17 treatment modalities and 8 acceptance criteria used by scoring algorithms

## Input/Output Specifications

**Expected Input Format:**
CSV files with columns: `pmid`, `title`, `author`, `details`

**Output Format:**
Enhanced CSV with additional columns: `abstract`, `score`, `cluster` (if embeddings generated)

**Scoring Criteria:**
Based on treatment modalities (1-17) and acceptance/rejection criteria (1-8) defined in summary-criteria.md

## Data Processing Notes

- Uses UTF-8 encoding for output to handle international characters
- Implements rate limiting for NCBI API compliance (0.34s delay between requests)
- Supports batch processing for large datasets
- Includes error handling and logging for robust data extraction