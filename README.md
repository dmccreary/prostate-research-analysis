# Prostate Research Analysis

[![License: CC BY-NC-SA 4.0](https://img.shields.io/badge/License-CC_BY--NC--SA_4.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc-sa/4.0/)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Documentation](https://img.shields.io/badge/docs-mkdocs-blue.svg)](https://dmccreary.github.io/prostate-research-analysis/)
[![GitHub stars](https://img.shields.io/github/stars/dmccreary/prostate-research-analysis.svg)](https://github.com/dmccreary/prostate-research-analysis/stargazers)

A comprehensive Python-based classifier for prostate cancer treatment research papers that automatically extracts, analyzes, and scores scientific articles based on treatment modalities and study quality criteria.

## 🎯 Overview

This project provides tools to systematically evaluate prostate cancer research literature by:
- **Extracting abstracts** from PubMed using NCBI E-utilities API
- **Scoring papers** based on 17 treatment modalities and 8 quality criteria (0-100 scale)
- **Visualizing similarities** using machine learning embeddings and clustering
- **Generating insights** through interactive data visualizations

**📖 Full Documentation:** [https://dmccreary.github.io/prostate-research-analysis/](https://dmccreary.github.io/prostate-research-analysis/)

## ✨ Features

- **Automated Data Extraction**: Fetch abstracts from PubMed using PMIDs
- **Intelligent Scoring**: Multi-criteria evaluation of research quality and relevance
- **Machine Learning Analysis**: TF-IDF + UMAP/t-SNE for similarity visualization
- **Interactive Visualizations**: Web-based tools for exploring paper relationships
- **Comprehensive Documentation**: Step-by-step workflow guides

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- Conda (recommended for environment management)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/dmccreary/prostate-research-analysis.git
   cd prostate-research-analysis
   ```

2. **Set up Python environment**
   ```bash
   conda create -n doc-classifier python=3.8
   conda activate doc-classifier
   pip install pandas tqdm Bio chardet scikit-learn umap-learn numpy mkdocs mkdocs-material
   ```

3. **Extract abstracts from PubMed**
   ```bash
   cd src
   python extractor3.py --input_file ../data/input.csv --output_file output.csv --email your@email.com
   ```

4. **Score the papers**
   ```bash
   python prostate-cancer-scorer.py
   ```

5. **Generate visualizations**
   ```bash
   cd create-embedding
   python create-embeddings.py
   ```

## 📊 Scoring System

Papers are evaluated on three components (100 points total):

| Component | Points | Description |
|-----------|--------|-------------|
| **Treatment Modality** | 40 | Identification of 17 specific treatments (prostatectomy, radiation, etc.) |
| **Study Quality** | 40 | Peer review status, endpoints, proper stratification |
| **Numerical Criteria** | 20 | Sample size, follow-up duration, radiation dose compliance |

### Treatment Modalities Evaluated

1. Radical Prostatectomy (Open/Robotic)
2. External Beam Radiation Therapy variants
3. Brachytherapy (LDR/HDR)
4. Hormone therapy combinations
5. Cryotherapy and HIFU

## 📁 Project Structure

```
├── src/                     # Core Python scripts
│   ├── extractor3.py       # PubMed abstract extraction
│   ├── prostate-cancer-scorer.py  # Main scoring algorithm
│   ├── create-embedding/   # Visualization tools
│   └── score-papers/       # Alternative implementations
├── data/                   # Input/output CSV files
├── docs/                   # Documentation source
│   ├── workflow/          # Step-by-step guides
│   └── data-viewers/      # Interactive visualizations
└── site/                  # Generated documentation
```

## 🔬 Research Applications

This tool has been used for:
- Systematic literature reviews in prostate cancer treatment
- Comparative effectiveness research
- Treatment modality trend analysis
- Quality assessment of clinical studies

## 📈 Example Results

From analysis of 7,338 papers:
- **Mean Score**: 16.4/100
- **High Quality (≥70)**: 26 papers (0.4%)
- **Good Quality (50-69)**: 282 papers (3.8%)
- **Top Score**: 88/100 (Stereotactic body radiotherapy study)

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

## 📄 License

This project is licensed under the Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License - see the [LICENSE](license.md) file for details.

## 🙏 Acknowledgments

This project builds upon several excellent open-source libraries:

- **[Biopython](https://biopython.org/)** - For NCBI E-utilities API access
- **[pandas](https://pandas.pydata.org/)** - Data manipulation and analysis
- **[scikit-learn](https://scikit-learn.org/)** - Machine learning algorithms
- **[UMAP](https://umap-learn.readthedocs.io/)** - Dimensionality reduction
- **[MkDocs](https://www.mkdocs.org/)** - Documentation generation
- **[Material for MkDocs](https://squidfunk.github.io/mkdocs-material/)** - Documentation theme

Special thanks to the [NCBI](https://www.ncbi.nlm.nih.gov/) for providing free access to PubMed data through their E-utilities API.

## 📞 Contact

**Author**: Dan McCreary  
**Website**: [https://dmccreary.github.io/prostate-research-analysis/](https://dmccreary.github.io/prostate-research-analysis/)  
**Issues**: [GitHub Issues](https://github.com/dmccreary/prostate-research-analysis/issues)