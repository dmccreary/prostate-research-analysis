#!/usr/bin/env python3
"""
Generate data profile reports comparing positive and negative datasets.

Usage:
    python generate-data-profiles.py --email dan.mccreary@gmail.com
"""

import argparse
import json
import re
import time
from collections import Counter
from datetime import datetime
import pandas as pd
from Bio import Entrez
from tqdm import tqdm
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()

def fetch_pub_years(pmids, email, batch_size=50, delay=0.34):
    """Fetch publication years for a list of PMIDs from PubMed."""
    Entrez.email = email
    pmids = [str(p) for p in pmids if p]
    years = {}

    for i in tqdm(range(0, len(pmids), batch_size), desc="Fetching publication years"):
        batch = pmids[i:i + batch_size]
        try:
            handle = Entrez.efetch(db="pubmed", id=",".join(batch), rettype="xml", retmode="text")
            records = Entrez.read(handle)
            handle.close()
            time.sleep(delay)

            for article in records['PubmedArticle']:
                pmid = str(article['MedlineCitation']['PMID'])
                # Try to get publication year
                pub_date = article['MedlineCitation']['Article'].get('ArticleDate', [])
                if pub_date:
                    years[pmid] = int(pub_date[0]['Year'])
                else:
                    # Fall back to journal pub date
                    journal_date = article['MedlineCitation']['Article']['Journal']['JournalIssue'].get('PubDate', {})
                    if 'Year' in journal_date:
                        years[pmid] = int(journal_date['Year'])
                    elif 'MedlineDate' in journal_date:
                        # Parse "2005 Jan-Feb" style dates
                        match = re.search(r'(\d{4})', journal_date['MedlineDate'])
                        if match:
                            years[pmid] = int(match.group(1))
        except Exception as e:
            logger.error(f"Error fetching batch: {e}")

    return years

def load_positive_dataset(filepath):
    """Load positive dataset and extract PMIDs and risk categories."""
    xlsx = pd.ExcelFile(filepath)
    df = pd.read_excel(xlsx, sheet_name='Sheet1', header=None)

    papers = []
    current_category = "UNKNOWN"

    for val in df[0]:
        val_str = str(val).strip()
        if not val_str.startswith('http'):
            # Category header
            if 'LOW' in val_str.upper():
                current_category = 'LOW RISK'
            elif 'INTERMEDIATE' in val_str.upper():
                current_category = 'INTERMEDIATE RISK'
            elif 'HIGH' in val_str.upper():
                current_category = 'HIGH RISK'
        else:
            # Extract PMID from URL
            match = re.search(r'/(\d+)/?$', val_str)
            if match:
                papers.append({
                    'pmid': int(match.group(1)),
                    'risk_category': current_category,
                    'url': val_str
                })

    return papers

def load_negative_dataset(filepath):
    """Load negative dataset from JSON."""
    with open(filepath) as f:
        data = json.load(f)
    return data['papers']

def generate_year_distribution_report(positive_years, negative_years, output_path):
    """Generate markdown report comparing year distributions."""

    pos_counter = Counter(positive_years.values())
    neg_counter = Counter([int(p['year']) for p in negative_years if p.get('year')])

    all_years = sorted(set(pos_counter.keys()) | set(neg_counter.keys()))

    # Calculate statistics
    pos_year_list = list(positive_years.values())
    neg_year_list = [int(p['year']) for p in negative_years if p.get('year')]

    report = f"""# Year Distribution: Positive vs Negative Datasets

**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M')}

## Summary

| Metric | Positive (Accepted) | Negative (Rejected) |
|--------|---------------------|---------------------|
| Total Papers | {len(pos_year_list)} | {len(neg_year_list)} |
| Year Range | {min(pos_year_list)}-{max(pos_year_list)} | {min(neg_year_list)}-{max(neg_year_list)} |
| Mean Year | {sum(pos_year_list)/len(pos_year_list):.1f} | {sum(neg_year_list)/len(neg_year_list):.1f} |
| Median Year | {sorted(pos_year_list)[len(pos_year_list)//2]} | {sorted(neg_year_list)[len(neg_year_list)//2]} |

## Year Distribution Table

| Year | Positive | Negative | Pos % | Neg % |
|------|----------|----------|-------|-------|
"""

    for year in all_years:
        pos_count = pos_counter.get(year, 0)
        neg_count = neg_counter.get(year, 0)
        pos_pct = (pos_count / len(pos_year_list) * 100) if pos_year_list else 0
        neg_pct = (neg_count / len(neg_year_list) * 100) if neg_year_list else 0
        report += f"| {year} | {pos_count} | {neg_count} | {pos_pct:.1f}% | {neg_pct:.1f}% |\n"

    report += f"| **Total** | **{len(pos_year_list)}** | **{len(neg_year_list)}** | **100%** | **100%** |\n"

    # Add 5-year bins
    report += """
## 5-Year Period Distribution

| Period | Positive | Negative | Pos % | Neg % |
|--------|----------|----------|-------|-------|
"""

    bins = [(2000, 2004), (2005, 2009), (2010, 2014), (2015, 2019), (2020, 2024)]
    for start, end in bins:
        pos_count = sum(1 for y in pos_year_list if start <= y <= end)
        neg_count = sum(1 for y in neg_year_list if start <= y <= end)
        pos_pct = (pos_count / len(pos_year_list) * 100) if pos_year_list else 0
        neg_pct = (neg_count / len(neg_year_list) * 100) if neg_year_list else 0
        report += f"| {start}-{end} | {pos_count} | {neg_count} | {pos_pct:.1f}% | {neg_pct:.1f}% |\n"

    # Add ASCII bar chart
    report += """
## Visual Distribution (by 5-year periods)

```
Period      Positive                          Negative
"""

    max_count = max(
        max(sum(1 for y in pos_year_list if start <= y <= end) for start, end in bins),
        max(sum(1 for y in neg_year_list if start <= y <= end) for start, end in bins)
    )
    scale = 30 / max_count if max_count > 0 else 1

    for start, end in bins:
        pos_count = sum(1 for y in pos_year_list if start <= y <= end)
        neg_count = sum(1 for y in neg_year_list if start <= y <= end)
        pos_bar = '█' * int(pos_count * scale)
        neg_bar = '█' * int(neg_count * scale)
        report += f"{start}-{end}  {pos_bar:<30} {neg_bar:<30}\n"

    report += "```\n"

    # Notes
    report += """
## Notes

- **Positive dataset**: Papers accepted for the systematic review (from `data/accepted-articles.xlsx`)
- **Negative dataset**: Papers rejected/excluded (from `data/negative-data-set.xlsx`)
- Publication years fetched from PubMed metadata
- Year distribution similarity is important for training unbiased classifiers
"""

    with open(output_path, 'w') as f:
        f.write(report)

    return report

def main():
    parser = argparse.ArgumentParser(description='Generate data profile reports')
    parser.add_argument('--email', required=True, help='Email for NCBI API')
    parser.add_argument('--positive', default='../data/accepted-articles.xlsx', help='Positive dataset')
    parser.add_argument('--negative', default='../data/negative-data-set.json', help='Negative dataset JSON')
    parser.add_argument('--output-dir', default='../docs/reports/data-profiles', help='Output directory')
    args = parser.parse_args()

    # Load datasets
    logger.info("Loading positive dataset...")
    positive_papers = load_positive_dataset(args.positive)
    logger.info(f"Found {len(positive_papers)} positive papers")

    logger.info("Loading negative dataset...")
    negative_papers = load_negative_dataset(args.negative)
    logger.info(f"Found {len(negative_papers)} negative papers")

    # Fetch publication years for positive papers
    logger.info("Fetching publication years for positive papers...")
    pos_pmids = [p['pmid'] for p in positive_papers]
    positive_years = fetch_pub_years(pos_pmids, args.email)
    logger.info(f"Got years for {len(positive_years)}/{len(pos_pmids)} positive papers")

    # Generate year distribution report
    output_path = f"{args.output_dir}/year-distribution.md"
    logger.info(f"Generating year distribution report: {output_path}")
    report = generate_year_distribution_report(positive_years, negative_papers, output_path)

    print("\n" + "="*60)
    print(report)
    print("="*60)
    logger.info(f"Report saved to {output_path}")

if __name__ == "__main__":
    main()
