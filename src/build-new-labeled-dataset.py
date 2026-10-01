#!/usr/bin/env python3
"""
build-new-labeled-dataset.py

Constructs a new labeled dataset (data/labeled-dataset-v2.csv) ready for model training:
- Positives: Sourced from data/accepted_articles_unique.csv with complete PubMed metadata
  (PMID, title, author, abstract, journal, year, volume, pages, doi, url).
- Negatives: Sourced from previous negative papers in data/labeled-dataset.csv, with conflicting
  papers (those now in the positive set) properly resolved.

Usage:
    python src/build-new-labeled-dataset.py
"""

import logging
import re
import sys
import time
from pathlib import Path
from typing import Dict, List

import pandas as pd
from Bio import Entrez

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def parse_abstract(article: dict) -> str:
    if "Abstract" not in article:
        return ""
    parts = []
    abstract_text = article["Abstract"].get("AbstractText", [])
    if isinstance(abstract_text, str):
        parts.append(abstract_text)
    else:
        for part in abstract_text:
            if hasattr(part, "attributes") and "Label" in part.attributes:
                parts.append(f"{part.attributes['Label']}: {part}")
            else:
                parts.append(str(part))
    text = " ".join(parts)
    return text.replace("\u2009", " ")


def parse_author(article: dict) -> str:
    authors = article.get("AuthorList")
    if not authors:
        return ""
    first = authors[0]
    last = first.get("LastName", "")
    initials = first.get("Initials", "")
    return f"{last} {initials}".strip()


def parse_doi(pubmed_article: dict) -> str:
    for aid in pubmed_article.get("PubmedData", {}).get("ArticleIdList", []):
        if aid.attributes.get("IdType") == "doi":
            return str(aid)
    return ""


def fetch_pubmed_records(pmids: List[int], email: str = "researcher@example.com", batch_size: int = 100) -> Dict[int, dict]:
    Entrez.email = email
    records_by_pmid = {}

    for i in range(0, len(pmids), batch_size):
        batch = [str(p) for p in pmids[i : i + batch_size]]
        logger.info(f"Fetching PubMed batch {i // batch_size + 1} ({len(batch)} PMIDs)...")
        handle = Entrez.efetch(db="pubmed", id=",".join(batch), rettype="xml", retmode="text")
        records = Entrez.read(handle)
        handle.close()

        for pubmed_article in records["PubmedArticle"]:
            mc = pubmed_article["MedlineCitation"]
            pmid = int(str(mc["PMID"]))
            article = mc["Article"]
            journal = article.get("Journal", {})
            journal_issue = journal.get("JournalIssue", {})
            pub_date = journal_issue.get("PubDate", {})
            pagination = article.get("Pagination", {})

            year = pub_date.get("Year")
            if not year and "MedlineDate" in pub_date:
                m = re.search(r"(\d{4})", pub_date["MedlineDate"])
                year = m.group(1) if m else ""

            records_by_pmid[pmid] = {
                "pmid": pmid,
                "title": str(article.get("ArticleTitle", "")),
                "author": parse_author(article),
                "abstract": parse_abstract(article),
                "journal": str(journal.get("ISOAbbreviation") or journal.get("Title", "")),
                "year": int(year) if str(year).isdigit() else year,
                "volume": str(journal_issue.get("Volume", "")),
                "pages": str(pagination.get("MedlinePgn", "")),
                "doi": parse_doi(pubmed_article),
                "url": f"https://www.ncbi.nlm.nih.gov/pubmed/{pmid}",
            }
        time.sleep(0.34)

    return records_by_pmid


# Physician-verified label resolutions
PHYSICIAN_OVERRIDE_POSITIVE = {20215039}
PHYSICIAN_OVERRIDE_NEGATIVE = {15774239}


def build_new_labeled_dataset(
    old_labeled_path: str = "data/labeled-dataset.csv",
    new_accepted_path: str = "data/accepted_articles_unique.csv",
    output_path: str = "data/labeled-dataset-v2.csv",
):
    logger.info("Loading existing dataset from %s", old_labeled_path)
    df_old = pd.read_csv(old_labeled_path)
    old_pos = df_old[df_old["label"] == 1].copy()
    old_neg = df_old[df_old["label"] == 0].copy()

    # Map previous positive risk categories if known
    old_risk_map = dict(zip(old_pos["pmid"], old_pos["risk_category"]))

    logger.info("Loading accepted articles from %s", new_accepted_path)
    df_new = pd.read_csv(new_accepted_path)

    # Clean and extract unique valid PMIDs from accepted_articles_unique
    clean_pmids = []
    for p in df_new["PMID"]:
        if pd.notna(p):
            s = str(p).strip().replace(".0", "")
            if s.isdigit() and int(s) not in clean_pmids:
                clean_pmids.append(int(s))

    # Apply physician determinations:
    # 1. PMID 15774239 confirmed negative -> exclude from positives
    clean_pmids = [p for p in clean_pmids if p not in PHYSICIAN_OVERRIDE_NEGATIVE]

    # 2. PMID 20215039 confirmed positive -> ensure in positive set
    for p in PHYSICIAN_OVERRIDE_POSITIVE:
        if p not in clean_pmids:
            clean_pmids.append(p)

    logger.info(f"Found {len(clean_pmids)} unique valid positive PMIDs from accepted articles (after physician overrides).")

    # Fetch PubMed metadata for all unique positive PMIDs
    pubmed_records = fetch_pubmed_records(clean_pmids)

    pos_rows = []
    for pmid in clean_pmids:
        if pmid in pubmed_records:
            rec = pubmed_records[pmid].copy()
            rec["risk_category"] = old_risk_map.get(pmid, float("nan"))
            rec["dataset"] = "positive"
            rec["label"] = 1
            rec["year_group"] = float("nan")
            pos_rows.append(rec)
        else:
            logger.warning(f"PMID {pmid} was not returned by PubMed.")

    df_pos_new = pd.DataFrame(pos_rows)
    logger.info(f"Constructed {len(df_pos_new)} positive rows from accepted articles.")

    # 3. Add the 42 positive papers from the previous dataset that were not in accepted_articles_unique
    missing_pos_df = old_pos[
        ~old_pos["pmid"].isin(clean_pmids) & ~old_pos["pmid"].isin(PHYSICIAN_OVERRIDE_NEGATIVE)
    ].copy()
    logger.info(f"Adding {len(missing_pos_df)} positive papers from previous dataset not in new positive set.")

    df_pos = pd.concat([df_pos_new, missing_pos_df], ignore_index=True)
    logger.info(f"Total positive papers: {len(df_pos)}")

    # 4. Prepare negative papers from previous dataset
    all_positive_pmids = set(df_pos["pmid"])
    df_neg_filtered = old_neg[~old_neg["pmid"].isin(all_positive_pmids)].copy()
    df_neg_filtered = df_neg_filtered.drop_duplicates(subset=["pmid"]).copy()

    # Ensure PMID 15774239 is included in negatives (per physician confirmation)
    if 15774239 not in df_neg_filtered["pmid"].values:
        row_15774239 = df_old[(df_old["pmid"] == 15774239) & (df_old["label"] == 0)].copy()
        if len(row_15774239) == 0:
            row_15774239 = df_old[df_old["pmid"] == 15774239].iloc[0:1].copy()
            row_15774239["label"] = 0
            row_15774239["dataset"] = "negative"
            row_15774239["risk_category"] = float("nan")
        df_neg_filtered = pd.concat([df_neg_filtered, row_15774239], ignore_index=True)

    logger.info(f"Total negative papers: {len(df_neg_filtered)}")

    # Combine positives and negatives
    combined = pd.concat([df_pos, df_neg_filtered], ignore_index=True, sort=False)

    # Match exact columns of labeled-dataset.csv
    expected_cols = df_old.columns.tolist()
    combined = combined[expected_cols]

    # Sort deterministically: positives first (label descending), then by pmid ascending
    combined = combined.sort_values(["label", "pmid"], ascending=[False, True]).reset_index(drop=True)

    # Ensure output directory exists
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(output_path, index=False)
    logger.info(f"Saved updated labeled dataset ({len(combined)} rows) to {output_path}")

    # Validation checks
    assert len(combined) == len(df_pos) + len(df_neg_filtered), "Row count mismatch!"
    assert combined["pmid"].duplicated().sum() == 0, "Duplicate PMIDs detected in combined dataset!"
    pos_count = (combined["label"] == 1).sum()
    neg_count = (combined["label"] == 0).sum()
    logger.info(f"Validation successful: {pos_count} Positives, {neg_count} Negatives, Total {len(combined)}.")


if __name__ == "__main__":
    build_new_labeled_dataset()

