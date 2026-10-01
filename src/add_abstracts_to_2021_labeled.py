#!/usr/bin/env python3
"""
add_abstracts_to_2021_labeled.py

Adds an 'abstract' column to data/2021 labeld.csv by fetching PubMed abstracts
via NCBI E-Utilities (Bio.Entrez.efetch) using the project's standard extraction pipeline:
- Batch fetching (batch size = 100)
- Preserving structured section labels (e.g., BACKGROUND: ..., METHODS: ...)
- Removing HTML/XML formatting tags while preserving text and inequalities
- Replacing thin spaces (\u2009) with standard spaces
- Preserving all existing columns, row order, labels, and duplicate PMIDs
- Saving to data/2021 labeld.csv with UTF-8 encoding and index=False
"""

import logging
import re
import sys
import time
from pathlib import Path
from typing import Dict, List

import pandas as pd
from Bio import Entrez

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

# Regex for HTML formatting tags (same as src/preprocessing.py)
HTML_TAG_REGEX = re.compile(r"</?(?:[a-zA-Z]+[0-9]*)\b[^>]*>", re.IGNORECASE)

CSV_PATH = Path("data/2021 labeld.csv")


def clean_abstract_part(part_str: str) -> str:
    """Strip XML/HTML tags and replace thin spaces."""
    cleaned = HTML_TAG_REGEX.sub("", part_str)
    return cleaned


def fetch_pubmed_abstracts(
    pmids: List[int],
    email: str = "researcher@example.com",
    batch_size: int = 100,
    delay: float = 0.34
) -> Dict[str, str]:
    """
    Fetch abstracts for a list of PMIDs using Bio.Entrez in batches of 100.
    """
    Entrez.email = email
    
    # Get ordered unique list of PMIDs
    unique_pmids = list(dict.fromkeys([str(p).strip() for p in pmids if pd.notna(p) and str(p).strip()]))
    logger.info(f"Total unique PMIDs to fetch: {len(unique_pmids)}")
    
    abstracts_by_pmid: Dict[str, str] = {}
    
    for i in range(0, len(unique_pmids), batch_size):
        batch = unique_pmids[i : i + batch_size]
        logger.info(f"Fetching PubMed batch {i // batch_size + 1}/{(len(unique_pmids) + batch_size - 1) // batch_size} ({len(batch)} PMIDs)...")
        
        try:
            handle = Entrez.efetch(db="pubmed", id=",".join(batch), rettype="xml", retmode="text")
            records = Entrez.read(handle)
            handle.close()
        except Exception as e:
            logger.error(f"Error fetching batch starting at index {i}: {e}")
            raise
        
        for article in records.get("PubmedArticle", []):
            try:
                pmid = str(article["MedlineCitation"]["PMID"])
                article_data = article["MedlineCitation"]["Article"]
                
                if "Abstract" in article_data and "AbstractText" in article_data["Abstract"]:
                    abstract_field = article_data["Abstract"]["AbstractText"]
                    parts = []
                    
                    if isinstance(abstract_field, (str, bytes)):
                        part_clean = clean_abstract_part(str(abstract_field))
                        if part_clean.strip():
                            parts.append(part_clean)
                    else:
                        for part in abstract_field:
                            part_clean = clean_abstract_part(str(part))
                            if hasattr(part, "attributes") and part.attributes and "Label" in part.attributes:
                                label = part.attributes["Label"]
                                parts.append(f"{label}: {part_clean}")
                            else:
                                parts.append(part_clean)
                    
                    full_abstract = " ".join(parts).replace("\u2009", " ").strip()
                    abstracts_by_pmid[pmid] = full_abstract
                else:
                    abstracts_by_pmid[pmid] = ""
            except Exception as e:
                logger.warning(f"Error extracting abstract for article: {e}")
                
        time.sleep(delay)
        
    return abstracts_by_pmid


def process_and_update_dataset(csv_path: Path = CSV_PATH) -> pd.DataFrame:
    """
    Loads 2021 labeld.csv, fetches abstracts, performs quality audits, and saves back.
    """
    if not csv_path.exists():
        raise FileNotFoundError(f"Input file not found at {csv_path}")
        
    df_original = pd.read_csv(csv_path)
    initial_row_count = len(df_original)
    original_columns = df_original.columns.tolist()
    original_pmids = df_original["pmid"].tolist()
    original_labels = df_original["label"].tolist()
    
    logger.info(f"Loaded {initial_row_count} rows from {csv_path}")
    logger.info(f"Existing columns: {original_columns}")
    
    # Fetch abstracts
    abstracts_map = fetch_pubmed_abstracts(original_pmids)
    
    # Map abstracts back to each row preserving exact order and duplicate PMIDs
    abstract_list = [abstracts_map.get(str(p), "") for p in original_pmids]
    
    # Assign abstract column
    df_updated = df_original.copy()
    df_updated["abstract"] = abstract_list
    
    # ---------------------------------------------------------------------------
    # Quality Verification Checks
    # ---------------------------------------------------------------------------
    # 1. Row count verification
    assert len(df_updated) == initial_row_count, f"Row count changed: {len(df_updated)} vs {initial_row_count}"
    
    # 2. PMID verification
    assert df_updated["pmid"].tolist() == original_pmids, "PMID order or values changed"
    
    # 3. Label verification
    assert df_updated["label"].tolist() == original_labels, "Label values changed"
    
    # 4. Column verification
    assert "abstract" in df_updated.columns, "Abstract column not found"
    for col in original_columns:
        assert col in df_updated.columns, f"Original column '{col}' missing"
        
    # 5. Abstract availability stats
    with_abstract = (df_updated["abstract"].str.strip() != "").sum()
    without_abstract = (df_updated["abstract"].str.strip() == "").sum()
    duplicate_pmid_count = df_updated["pmid"].duplicated().sum()
    
    logger.info(f"Quality Check Results:")
    logger.info(f"  - Total rows: {len(df_updated)} (unchanged)")
    logger.info(f"  - Unique PMIDs: {df_updated['pmid'].nunique()}")
    logger.info(f"  - Duplicate PMIDs: {duplicate_pmid_count}")
    logger.info(f"  - Articles with abstract: {with_abstract}/{len(df_updated)}")
    logger.info(f"  - Articles without abstract: {without_abstract}/{len(df_updated)}")
    
    # Save to CSV
    df_updated.to_csv(csv_path, index=False, encoding="utf-8")
    logger.info(f"Successfully saved updated dataset with abstracts to {csv_path}")
    
    return df_updated


if __name__ == "__main__":
    process_and_update_dataset()
