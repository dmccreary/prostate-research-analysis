"""
fetch_metadata.py - Batch fetcher and cacher for PubMed official publication types.

Uses NCBI Entrez E-Utilities (efetch.fcgi) to query official MeSH PublicationType tags.
Caches results locally to data/metadata/pubmed_publication_types.json for offline reproducibility.
"""

import json
import logging
import time
import urllib.request
import urllib.error
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

EUTILS_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"


def fetch_publication_types_batch(pmids: List[str], batch_size: int = 100, delay_sec: float = 0.35) -> Dict[str, List[str]]:
    """
    Fetches official publication types from PubMed in batches.
    Returns mapping: {pmid_str: [list of publication type strings]}
    """
    clean_pmids = [str(p).strip().replace(".0", "") for p in pmids if str(p).strip()]
    unique_pmids = sorted(list(set(clean_pmids)))
    logger.info("Fetching publication types for %d unique PMIDs from NCBI...", len(unique_pmids))

    results = {}
    for i in range(0, len(unique_pmids), batch_size):
        batch = unique_pmids[i : i + batch_size]
        id_str = ",".join(batch)
        url = f"{EUTILS_BASE}?db=pubmed&id={id_str}&rettype=xml&retmode=xml"

        success = False
        for attempt in range(3):
            try:
                req = urllib.request.Request(url, headers={"User-Agent": "ProstateResearchScreening/1.0"})
                with urllib.request.urlopen(req, timeout=30) as resp:
                    xml_data = resp.read()
                root = ET.fromstring(xml_data)

                for article in root.findall(".//PubmedArticle") + root.findall(".//PubmedBookArticle"):
                    pmid_elem = article.findtext(".//PMID")
                    if pmid_elem:
                        types = [pt.text.strip() for pt in article.findall(".//PublicationType") if pt.text and pt.text.strip()]
                        results[str(pmid_elem).strip()] = types

                success = True
                break
            except Exception as e:
                logger.warning("Batch %d-%d attempt %d failed: %s. Retrying...", i, i + len(batch), attempt + 1, e)
                time.sleep(1.0 + attempt)

        if not success:
            logger.error("Failed to fetch batch %d-%d after 3 attempts.", i, i + len(batch))

        time.sleep(delay_sec)

    # Ensure all requested PMIDs have at least an empty list
    for p in unique_pmids:
        if p not in results:
            results[p] = []

    logger.info("Successfully retrieved publication types for %d / %d PMIDs.", len([p for p in results if results[p]]), len(unique_pmids))
    return results


def get_publication_types(data_dir: Optional[Path] = None, force_refresh: bool = False) -> Dict[str, List[str]]:
    """
    Retrieves publication types from local cache if available; otherwise fetches and caches.
    """
    if data_dir is None:
        base_dir = Path(__file__).resolve().parent.parent.parent
        data_dir = base_dir / "data"
    else:
        data_dir = Path(data_dir)

    cache_file = data_dir / "metadata" / "pubmed_publication_types.json"
    cache_file.parent.mkdir(parents=True, exist_ok=True)

    # Check cache
    if cache_file.exists() and not force_refresh:
        try:
            with open(cache_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            if len(data) >= 300:
                logger.info("Loaded %d publication types from local cache: %s", len(data), cache_file)
                return data
        except Exception as e:
            logger.warning("Failed to read cache file (%s), refetching...", e)

    # Load PMIDs from splits
    train_file = data_dir / "splits" / "train.csv"
    test_file = data_dir / "splits" / "test.csv"
    pmids = []
    if train_file.exists():
        pmids.extend(pd.read_csv(train_file)["pmid"].tolist())
    if test_file.exists():
        pmids.extend(pd.read_csv(test_file)["pmid"].tolist())

    if not pmids:
        labeled_file = data_dir / "labeled-dataset.csv"
        if labeled_file.exists():
            pmids.extend(pd.read_csv(labeled_file)["pmid"].tolist())

    data = fetch_publication_types_batch(pmids)

    # Save cache
    with open(cache_file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    logger.info("Saved publication types cache to %s", cache_file)
    return data


if __name__ == "__main__":
    get_publication_types(force_refresh=True)
