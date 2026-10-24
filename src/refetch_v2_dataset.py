#!/usr/bin/env python3
"""
refetch_v2_dataset.py

Refetches titles and abstracts for all 524 papers (both positive and negative)
in data/labeled-dataset-v2.csv directly from the NCBI PubMed Entrez API.

- Safely backs up the existing CSV before writing.
- Uses NCBI Entrez efetch in batches of 100.
- Extracts ArticleTitle and Abstract with structured section label preservation.
- Updates the same CSV file: data/labeled-dataset-v2.csv while preserving all metadata columns,
  row order, and labels.
- Validates data integrity and logs before/after punctuation statistics.
"""

import copy
import logging
from pathlib import Path
import re
import shutil
import sys
import time
from typing import Dict, List

import pandas as pd
from Bio import Entrez

# Safe stdout on Windows
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent
CSV_PATH = BASE_DIR / "data" / "labeled-dataset-v2.csv"
BACKUP_PATH = BASE_DIR / "data" / "labeled-dataset-v2_before_api_refetch.csv"

HTML_TAG_REGEX = re.compile(r"</?(?:[a-zA-Z]+[0-9]*)\b[^>]*>", re.IGNORECASE)


def parse_abstract(article_data: dict) -> str:
    """Parses PubMed XML abstract preserving section labels and cleaning HTML tags."""
    if "Abstract" not in article_data:
        return ""
    parts = []
    abstract_text = article_data["Abstract"].get("AbstractText", [])
    if isinstance(abstract_text, (str, bytes)):
        cleaned = HTML_TAG_REGEX.sub("", str(abstract_text)).strip()
        if cleaned:
            parts.append(cleaned)
    else:
        for part in abstract_text:
            cleaned = HTML_TAG_REGEX.sub("", str(part)).strip()
            if hasattr(part, "attributes") and part.attributes and "Label" in part.attributes:
                label = part.attributes["Label"]
                parts.append(f"{label}: {cleaned}")
            else:
                parts.append(cleaned)
    text = " ".join(parts)
    return text.replace("\u2009", " ").strip()


def parse_title(article_data: dict) -> str:
    """Parses PubMed XML ArticleTitle and strips HTML tags."""
    raw_title = str(article_data.get("ArticleTitle", ""))
    cleaned = HTML_TAG_REGEX.sub("", raw_title)
    return cleaned.replace("\u2009", " ").strip()


def fetch_all_pubmed_records(pmids: List[int], email: str = "researcher@example.com", batch_size: int = 100) -> Dict[int, dict]:
    """Fetches PubMed XML records for all PMIDs in batches."""
    Entrez.email = email
    records_by_pmid = {}
    
    unique_pmids = list(dict.fromkeys(pmids))
    total_batches = (len(unique_pmids) + batch_size - 1) // batch_size
    
    for i in range(0, len(unique_pmids), batch_size):
        batch = [str(p) for p in unique_pmids[i : i + batch_size]]
        batch_num = i // batch_size + 1
        logger.info(f"Fetching PubMed batch {batch_num}/{total_batches} ({len(batch)} PMIDs)...")
        
        max_retries = 3
        for attempt in range(1, max_retries + 1):
            try:
                handle = Entrez.efetch(db="pubmed", id=",".join(batch), rettype="xml", retmode="text")
                records = Entrez.read(handle)
                handle.close()
                break
            except Exception as e:
                logger.warning(f"Batch {batch_num} attempt {attempt} failed: {e}")
                if attempt == max_retries:
                    raise
                time.sleep(2 * attempt)
        
        for pubmed_article in records.get("PubmedArticle", []):
            try:
                mc = pubmed_article["MedlineCitation"]
                pmid = int(str(mc["PMID"]))
                article = mc["Article"]
                
                records_by_pmid[pmid] = {
                    "title": parse_title(article),
                    "abstract": parse_abstract(article),
                }
            except Exception as e:
                logger.warning(f"Error parsing article in batch {batch_num}: {e}")
        
        time.sleep(0.35)
        
    return records_by_pmid


def main():
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"Dataset not found at {CSV_PATH}")
    
    logger.info(f"Loading existing dataset: {CSV_PATH}")
    df = pd.read_csv(CSV_PATH)
    logger.info(f"Loaded {len(df)} rows across {len(df.columns)} columns.")
    
    # Audit before refetch
    pos_dot_before = df[df["label"] == 1]["title"].astype(str).str.strip().str.endswith(".").sum()
    neg_dot_before = df[df["label"] == 0]["title"].astype(str).str.strip().str.endswith(".").sum()
    logger.info(f"BEFORE refetch - Titles ending with '.' -> Pos: {pos_dot_before}/{(df['label']==1).sum()}, Neg: {neg_dot_before}/{(df['label']==0).sum()}")
    
    # 1. Create safety backup
    logger.info(f"Creating safety backup at: {BACKUP_PATH}")
    shutil.copy2(CSV_PATH, BACKUP_PATH)
    
    # 2. Fetch all PubMed records
    pmids = df["pmid"].tolist()
    records = fetch_all_pubmed_records(pmids)
    logger.info(f"Successfully retrieved API records for {len(records)}/{len(pmids)} PMIDs.")
    
    # 3. Update titles and abstracts
    missing_count = 0
    updated_titles = []
    updated_abstracts = []
    
    for idx, row in df.iterrows():
        p = int(row["pmid"])
        if p in records and records[p]["title"]:
            new_title = records[p]["title"]
            new_abstract = records[p]["abstract"]
            # If API abstract is empty but existing was not, keep existing
            if not new_abstract and pd.notna(row["abstract"]) and str(row["abstract"]).strip():
                new_abstract = str(row["abstract"]).strip()
        else:
            missing_count += 1
            logger.warning(f"PMID {p} was not in API records or had empty title; keeping original.")
            new_title = row["title"]
            new_abstract = row["abstract"]
            
        updated_titles.append(new_title)
        updated_abstracts.append(new_abstract)
        
    df["title"] = updated_titles
    df["abstract"] = updated_abstracts
    
    # 4. Save updated dataset to the same CSV file
    df.to_csv(CSV_PATH, index=False)
    logger.info(f"Saved updated dataset to {CSV_PATH}")
    
    # 5. Validation and Audit After Refetch
    df_check = pd.read_csv(CSV_PATH)
    assert len(df_check) == len(df) == 524, f"Row count mismatch: {len(df_check)} != 524"
    assert df_check["pmid"].nunique() == 524, "Duplicate PMIDs detected!"
    assert (df_check["label"] == 1).sum() == 281, "Positive count changed!"
    assert (df_check["label"] == 0).sum() == 243, "Negative count changed!"
    
    pos_dot_after = df_check[df_check["label"] == 1]["title"].astype(str).str.strip().str.endswith(".").sum()
    neg_dot_after = df_check[df_check["label"] == 0]["title"].astype(str).str.strip().str.endswith(".").sum()
    logger.info(f"AFTER refetch - Titles ending with '.' -> Pos: {pos_dot_after}/281 ({pos_dot_after/281*100:.2f}%), Neg: {neg_dot_after}/243 ({neg_dot_after/243*100:.2f}%)")
    logger.info("Dataset successfully updated and verified!")


if __name__ == "__main__":
    main()
