#!/usr/bin/env python3
"""
Build a fully-populated positive dataset (title, abstract, journal, year, etc.)
from the PMIDs listed in data/positive-data-set.xlsx, matching the schema of
data/negative-data-set.json so the two sets can be merged for model training.

Usage:
    python build-positive-dataset.py --email dan.mccreary@gmail.com
"""

import argparse
import json
import re
import time
import pandas as pd
from Bio import Entrez
from tqdm import tqdm
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()


def load_pmids_with_risk_category(filepath):
    """Read positive-data-set.xlsx and return [(pmid, risk_category, url), ...]."""
    df = pd.read_excel(filepath, sheet_name='Sheet1', header=None)

    entries = []
    current_category = "UNKNOWN"

    for val in df[0]:
        val_str = str(val).strip()
        if 'pubmed' not in val_str.lower():
            if 'LOW' in val_str.upper():
                current_category = 'LOW RISK'
            elif 'INTERMEDIATE' in val_str.upper():
                current_category = 'INTERMEDIATE RISK'
            elif 'HIGH' in val_str.upper():
                current_category = 'HIGH RISK'
            continue

        match = re.search(r'(\d{6,9})', val_str)
        if match:
            entries.append((match.group(1), current_category, val_str))

    return entries


def parse_abstract(article):
    if 'Abstract' not in article:
        return ""
    parts = []
    for part in article['Abstract']['AbstractText']:
        if hasattr(part, 'attributes') and 'Label' in part.attributes:
            parts.append(f"{part.attributes['Label']}: {part}")
        else:
            parts.append(str(part))
    text = " ".join(parts)
    return text.replace(' ', ' ')


def parse_author(article):
    authors = article.get('AuthorList')
    if not authors:
        return ""
    first = authors[0]
    last = first.get('LastName', '')
    initials = first.get('Initials', '')
    return f"{last} {initials}".strip()


def parse_doi(pubmed_article):
    for aid in pubmed_article.get('PubmedData', {}).get('ArticleIdList', []):
        if aid.attributes.get('IdType') == 'doi':
            return str(aid)
    return ""


def fetch_batch(pmids, email):
    Entrez.email = email
    handle = Entrez.efetch(db="pubmed", id=",".join(pmids), rettype="xml", retmode="text")
    records = Entrez.read(handle)
    handle.close()

    results = {}
    for pubmed_article in records['PubmedArticle']:
        mc = pubmed_article['MedlineCitation']
        pmid = str(mc['PMID'])
        article = mc['Article']
        journal = article.get('Journal', {})
        journal_issue = journal.get('JournalIssue', {})
        pub_date = journal_issue.get('PubDate', {})
        pagination = article.get('Pagination', {})

        year = pub_date.get('Year')
        if not year and 'MedlineDate' in pub_date:
            m = re.search(r'(\d{4})', pub_date['MedlineDate'])
            year = m.group(1) if m else ""

        results[pmid] = {
            'pmid': int(pmid),
            'title': str(article.get('ArticleTitle', '')),
            'author': parse_author(article),
            'abstract': parse_abstract(article),
            'journal': str(journal.get('ISOAbbreviation') or journal.get('Title', '')),
            'year': year or "",
            'volume': str(journal_issue.get('Volume', '')),
            'pages': str(pagination.get('MedlinePgn', '')),
            'doi': parse_doi(pubmed_article),
            'url': f"https://www.ncbi.nlm.nih.gov/pubmed/{pmid}",
        }
    return results


def main():
    parser = argparse.ArgumentParser(description='Build positive dataset JSON with abstracts')
    parser.add_argument('--input', default='../data/positive-data-set.xlsx')
    parser.add_argument('--output', default='../data/positive-data-set.json')
    parser.add_argument('--email', required=True, help='Email for NCBI API')
    parser.add_argument('--batch_size', type=int, default=50)
    parser.add_argument('--delay', type=float, default=0.34)
    args = parser.parse_args()

    entries = load_pmids_with_risk_category(args.input)
    logger.info(f"Loaded {len(entries)} PMID entries from {args.input}")

    risk_by_pmid = {pmid: risk for pmid, risk, _ in entries}
    unique_pmids = list(risk_by_pmid.keys())
    logger.info(f"Fetching {len(unique_pmids)} unique PMIDs from PubMed")

    papers = []
    failed = []
    for i in tqdm(range(0, len(unique_pmids), args.batch_size), desc="Fetching abstracts"):
        batch = unique_pmids[i:i + args.batch_size]
        try:
            fetched = fetch_batch(batch, args.email)
        except Exception as e:
            logger.error(f"Batch starting at {i} failed: {e}")
            fetched = {}

        for pmid in batch:
            if pmid in fetched:
                record = fetched[pmid]
                record['risk_category'] = risk_by_pmid[pmid]
                record['dataset'] = 'positive'
                papers.append(record)
            else:
                failed.append(pmid)

        time.sleep(args.delay)

    if failed:
        logger.warning(f"Failed to fetch {len(failed)} PMIDs: {failed}")

    with_abstracts = sum(1 for p in papers if p['abstract'])

    output = {
        'metadata': {
            'source': args.input,
            'total_papers': len(papers),
            'papers_with_abstracts': with_abstracts,
            'papers_failed': len(failed),
            'failed_pmids': failed,
            'dataset_type': 'positive',
            'risk_category_counts': pd.Series([p['risk_category'] for p in papers]).value_counts().to_dict(),
        },
        'papers': papers,
    }

    with open(args.output, 'w') as f:
        json.dump(output, f, indent=2)

    logger.info(f"Saved {len(papers)} papers ({with_abstracts} with abstracts) to {args.output}")


if __name__ == '__main__':
    main()
