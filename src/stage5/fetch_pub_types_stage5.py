"""
fetch_pub_types_stage5.py - Fetch publication types for all PMIDs in labeled-dataset-v2.csv.
Updates data/metadata/pubmed_publication_types.json and adds 'publication_types' column to labeled-dataset-v2.csv.
"""

import json
import logging
import time
import urllib.request
import urllib.error
import xml.etree.ElementTree as ET
from pathlib import Path
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_PATH = BASE_DIR / "data" / "labeled-dataset-v2.csv"
JSON_CACHE_PATH = BASE_DIR / "data" / "metadata" / "pubmed_publication_types.json"

EUTILS_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"


def fetch_batch(pmids, batch_size=100, delay=0.35):
    results = {}
    for i in range(0, len(pmids), batch_size):
        batch = pmids[i : i + batch_size]
        id_str = ",".join(batch)
        url = f"{EUTILS_BASE}?db=pubmed&id={id_str}&rettype=xml&retmode=xml"

        success = False
        for attempt in range(3):
            try:
                req = urllib.request.Request(url, headers={"User-Agent": "ProstateResearchStage5/1.0"})
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
                logger.warning(f"Batch {i}-{i+len(batch)} attempt {attempt+1} failed: {e}. Retrying...")
                time.sleep(1.0 + attempt)

        if not success:
            logger.error(f"Failed to fetch batch {i}-{i+len(batch)} after 3 attempts.")

        time.sleep(delay)

    for p in pmids:
        if p not in results:
            results[p] = []

    return results


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"{DATA_PATH} not found.")

    df = pd.read_csv(DATA_PATH)
    clean_pmids = [str(p).strip().replace(".0", "") for p in df["pmid"] if str(p).strip()]
    unique_pmids = sorted(list(set(clean_pmids)))
    logger.info(f"Loaded {len(df)} rows from {DATA_PATH}, {len(unique_pmids)} unique PMIDs.")

    pt_map = {}
    if JSON_CACHE_PATH.exists():
        with open(JSON_CACHE_PATH, "r", encoding="utf-8") as f:
            pt_map = json.load(f)
        logger.info(f"Loaded {len(pt_map)} publication types from {JSON_CACHE_PATH}.")

    missing = [p for p in unique_pmids if p not in pt_map or not pt_map[p]]
    logger.info(f"Missing / unpopulated publication types for {len(missing)} PMIDs.")

    if missing:
        logger.info(f"Fetching {len(missing)} PMIDs from NCBI...")
        fetched = fetch_batch(missing)
        pt_map.update(fetched)

        # Save back to json cache
        JSON_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(JSON_CACHE_PATH, "w", encoding="utf-8") as f:
            json.dump(pt_map, f, indent=2)
        logger.info(f"Updated cache with {len(pt_map)} total entries.")

    # Add publication_types column to dataframe
    pub_types_col = []
    for p in df["pmid"]:
        p_str = str(p).strip().replace(".0", "")
        types_list = pt_map.get(p_str, [])
        pub_types_col.append("; ".join(types_list))

    df["publication_types"] = pub_types_col
    df.to_csv(DATA_PATH, index=False)
    logger.info(f"Saved updated dataset with 'publication_types' column to {DATA_PATH}.")
    logger.info(f"Non-empty publication types count: {(df['publication_types'] != '').sum()} / {len(df)}")


if __name__ == "__main__":
    main()
