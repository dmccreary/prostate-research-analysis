#!/usr/bin/env python3
"""
Convert negative-data-set.xlsx to JSON with abstracts fetched from PubMed.

Usage:
    python xlsx-to-json.py --email your@email.com
    python xlsx-to-json.py --email your@email.com --output ../data/negative-data-set.json
"""

import argparse
import json
import time
import pandas as pd
from Bio import Entrez
from tqdm import tqdm
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()

def fetch_abstracts_batch(pmids, email, delay=0.34):
    """Fetch abstracts for a list of PMIDs from PubMed."""
    Entrez.email = email
    pmids = [str(p) for p in pmids if p]

    if not pmids:
        return {}

    try:
        handle = Entrez.efetch(db="pubmed", id=",".join(pmids), rettype="xml", retmode="text")
        records = Entrez.read(handle)
        handle.close()
        time.sleep(delay)

        abstracts = {}
        for article in records['PubmedArticle']:
            pmid = str(article['MedlineCitation']['PMID'])

            if 'Abstract' in article['MedlineCitation']['Article']:
                abstract_parts = []
                for part in article['MedlineCitation']['Article']['Abstract']['AbstractText']:
                    if hasattr(part, 'attributes') and 'Label' in part.attributes:
                        abstract_parts.append(f"{part.attributes['Label']}: {part}")
                    else:
                        abstract_parts.append(str(part))
                abstracts[pmid] = " ".join(abstract_parts).replace('\u2009', ' ')
            else:
                abstracts[pmid] = ""

        return abstracts
    except Exception as e:
        logger.error(f"Error fetching abstracts: {e}")
        return {}

def parse_citation(citation):
    """Extract journal, year, volume, pages, and DOI from citation string."""
    parts = {'journal': '', 'year': '', 'volume': '', 'pages': '', 'doi': ''}

    if not citation:
        return parts

    # Extract DOI
    if 'doi:' in citation.lower():
        doi_idx = citation.lower().find('doi:')
        parts['doi'] = citation[doi_idx:].strip().rstrip('.')
        citation = citation[:doi_idx].strip()

    # Try to parse "Journal. Year Mon;Vol(Issue):Pages"
    import re
    match = re.match(r'^(.+?)\.\s*(\d{4})\s*\w*;?\s*(\d+)?(?:\([^)]+\))?:?([\d\-]+)?', citation)
    if match:
        parts['journal'] = match.group(1).strip()
        parts['year'] = match.group(2) if match.group(2) else ''
        parts['volume'] = match.group(3) if match.group(3) else ''
        parts['pages'] = match.group(4) if match.group(4) else ''
    else:
        parts['journal'] = citation.split('.')[0] if '.' in citation else citation

    return parts

def main():
    parser = argparse.ArgumentParser(description='Convert Excel to JSON with PubMed abstracts')
    parser.add_argument('--input', default='../data/negative-data-set.xlsx', help='Input Excel file')
    parser.add_argument('--output', default='../data/negative-data-set.json', help='Output JSON file')
    parser.add_argument('--email', required=True, help='Email for NCBI API')
    parser.add_argument('--batch-size', type=int, default=50, help='PMIDs per API request')
    args = parser.parse_args()

    # Read Excel without headers
    logger.info(f"Reading {args.input}")
    df = pd.read_excel(args.input, header=None)
    df.columns = ['row_id', 'pmid', 'title', 'citation', 'author', 'url']

    logger.info(f"Found {len(df)} papers")

    # Fetch abstracts in batches
    all_pmids = df['pmid'].tolist()
    abstracts = {}

    for i in tqdm(range(0, len(all_pmids), args.batch_size), desc="Fetching abstracts"):
        batch = all_pmids[i:i + args.batch_size]
        batch_abstracts = fetch_abstracts_batch(batch, args.email)
        abstracts.update(batch_abstracts)

    # Build JSON structure
    papers = []
    for _, row in df.iterrows():
        pmid = str(row['pmid'])
        citation_parts = parse_citation(row['citation'])

        paper = {
            'pmid': int(row['pmid']),
            'title': row['title'],
            'author': row['author'],
            'abstract': abstracts.get(pmid, ''),
            'journal': citation_parts['journal'],
            'year': citation_parts['year'],
            'volume': citation_parts['volume'],
            'pages': citation_parts['pages'],
            'doi': citation_parts['doi'],
            'url': row['url'],
            'dataset': 'negative'
        }
        papers.append(paper)

    # Write JSON
    output_data = {
        'metadata': {
            'source': args.input,
            'total_papers': len(papers),
            'papers_with_abstracts': sum(1 for p in papers if p['abstract']),
            'dataset_type': 'negative'
        },
        'papers': papers
    }

    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    logger.info(f"Saved {len(papers)} papers to {args.output}")
    logger.info(f"Papers with abstracts: {output_data['metadata']['papers_with_abstracts']}/{len(papers)}")

if __name__ == "__main__":
    main()
