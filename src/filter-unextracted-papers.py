#!/usr/bin/env python3
"""
Apply the June 2026 TODO.md pre-screening filters to the raw PubMed export,
for the PMIDs that have not yet been through abstract extraction.

TODO.md filters (agreed by Alex/Richard Hsi and Mark Nguyen):
  - Exclude if abstract contains "palliative", "metastatic", or "hormone resistant"
  - Require abstract to contain "prostate cancer", "prostate neoplasia", or "prostate carcinoma"

Since these filters operate on abstract text, this script first fetches abstracts
for the not-yet-extracted PMIDs, then applies the filters, and reports how much
the candidate pool shrinks.

Usage:
    python filter-unextracted-papers.py --email dan.mccreary@gmail.com
"""

import argparse
import time
import pandas as pd
from Bio import Entrez
from tqdm import tqdm
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()

EXCLUDE_TERMS = ['palliative', 'metastatic', 'hormone resistant']
REQUIRE_TERMS = ['prostate cancer', 'prostate neoplasia', 'prostate carcinoma']


def get_unextracted_pmids(raw_path, extracted_path):
    raw = pd.read_excel(raw_path, header=0)
    extracted = pd.read_csv(extracted_path)

    raw = raw.dropna(subset=['pmid']).copy()
    raw['pmid'] = raw['pmid'].astype(int)
    extracted_pmids = set(extracted['pmid'].dropna().astype(int))

    raw_dedup = raw.drop_duplicates(subset='pmid')
    unextracted = raw_dedup[~raw_dedup['pmid'].isin(extracted_pmids)].reset_index(drop=True)
    return unextracted


def fetch_abstract_batch(pmid_list, email):
    Entrez.email = email
    pmid_list = [str(p) for p in pmid_list]
    try:
        handle = Entrez.efetch(db="pubmed", id=",".join(pmid_list), rettype="xml", retmode="text")
        records = Entrez.read(handle)
        handle.close()
    except Exception as e:
        logger.error(f"Batch fetch failed: {e}")
        return {}

    abstracts = {}
    for article in records.get('PubmedArticle', []):
        pmid = str(article['MedlineCitation']['PMID'])
        art = article['MedlineCitation']['Article']
        if 'Abstract' in art:
            parts = []
            for part in art['Abstract']['AbstractText']:
                if hasattr(part, 'attributes') and 'Label' in part.attributes:
                    parts.append(f"{part.attributes['Label']}: {part}")
                else:
                    parts.append(str(part))
            text = " ".join(parts).replace(' ', ' ')
            abstracts[pmid] = text
        else:
            abstracts[pmid] = ""
    return abstracts


def passes_filters(abstract):
    text = str(abstract).lower()
    if not text:
        return False, "no_abstract"
    for term in EXCLUDE_TERMS:
        if term in text:
            return False, f"excluded:{term}"
    if not any(term in text for term in REQUIRE_TERMS):
        return False, "missing_required_phrase"
    return True, "pass"


def main():
    parser = argparse.ArgumentParser(description='Apply TODO.md filters to unextracted PMIDs')
    parser.add_argument('--raw', default='../data/Pubmed-Exports_2021_Final.xlsx')
    parser.add_argument('--extracted', default='../data/output-full-scored.csv')
    parser.add_argument('--email', required=True)
    parser.add_argument('--batch_size', type=int, default=200)
    parser.add_argument('--delay', type=float, default=0.34)
    parser.add_argument('--limit', type=int, default=None, help='Optional cap on PMIDs to fetch, for a quick test run')
    parser.add_argument('--out_all', default='../data/unextracted-with-abstracts.csv')
    parser.add_argument('--out_filtered', default='../data/filtered-candidates.csv')
    args = parser.parse_args()

    unextracted = get_unextracted_pmids(args.raw, args.extracted)
    logger.info(f"Unextracted PMIDs: {len(unextracted)}")

    if args.limit:
        unextracted = unextracted.iloc[:args.limit].reset_index(drop=True)
        logger.info(f"Limiting to {len(unextracted)} for this run")

    pmids = unextracted['pmid'].tolist()
    all_abstracts = {}

    for i in tqdm(range(0, len(pmids), args.batch_size), desc="Fetching abstracts"):
        batch = pmids[i:i + args.batch_size]
        fetched = fetch_abstract_batch(batch, args.email)
        all_abstracts.update(fetched)
        time.sleep(args.delay)

    unextracted['abstract'] = unextracted['pmid'].astype(str).map(all_abstracts).fillna("")
    unextracted.to_csv(args.out_all, index=False)

    results = unextracted['abstract'].apply(passes_filters)
    unextracted['filter_pass'] = results.apply(lambda r: r[0])
    unextracted['filter_reason'] = results.apply(lambda r: r[1])

    reason_counts = unextracted['filter_reason'].value_counts()

    filtered = unextracted[unextracted['filter_pass']].drop(columns=['filter_pass', 'filter_reason'])
    filtered.to_csv(args.out_filtered, index=False)

    logger.info(f"Fetched abstracts for {len(unextracted)} PMIDs (saved to {args.out_all})")
    logger.info(f"Filter breakdown:\n{reason_counts}")
    logger.info(f"Passed filters: {len(filtered)} / {len(unextracted)} ({len(filtered)/len(unextracted)*100:.1f}%)")
    logger.info(f"Saved filtered candidates to {args.out_filtered}")


if __name__ == '__main__':
    main()
