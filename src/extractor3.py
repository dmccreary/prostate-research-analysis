#!/usr/bin/env python3
"""
PubMed Abstract Extractor

This script extracts abstracts for articles listed in a CSV file containing PubMed IDs (PMIDs).
It uses the Entrez Programming Utilities (E-utilities) API from NCBI to fetch the abstracts.

Requirements:
- pandas
- biopython (Bio package)
- tqdm (for progress bar)

Usage:
python pubmed_abstract_extractor.py --input_file "Filtered 2020 Articles.csv" --output_file "articles_with_abstracts.csv" --email "your_email@example.com"

Note on encoding: This script handles encoding issues by reading with the file's original encoding
but writing the output in UTF-8 format which can represent all Unicode characters.
"""

import argparse
import csv
import time
import pandas as pd
from tqdm import tqdm
from Bio import Entrez
from concurrent.futures import ThreadPoolExecutor
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("pubmed_extractor.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger()

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Extract abstracts from PubMed articles.')
    parser.add_argument('--input_file', type=str, required=True, help='Path to input CSV file')
    parser.add_argument('--output_file', type=str, required=True, help='Path to output CSV file')
    parser.add_argument('--email', type=str, required=True, help='Email address for NCBI API')
    parser.add_argument('--batch_size', type=int, default=100, help='Number of PMIDs to process in each batch')
    parser.add_argument('--max_workers', type=int, default=1, help='Number of parallel workers')
    parser.add_argument('--delay', type=float, default=0.34, help='Delay between API calls in seconds')
    parser.add_argument('--encoding', type=str, default='cp1252', help='File encoding (default: cp1252)')
    return parser.parse_args()

def fetch_abstract_batch(pmid_list, email):
    """
    Fetch abstracts for a batch of PMIDs.
    
    Args:
        pmid_list: List of PMIDs to fetch
        email: Email address for NCBI API
        
    Returns:
        Dictionary mapping PMIDs to abstracts
    """
    Entrez.email = email
    pmid_list = [str(pmid) for pmid in pmid_list if pmid]  # Ensure all PMIDs are strings and not None
    
    # Filter out any empty strings
    pmid_list = [pmid for pmid in pmid_list if pmid.strip()]
    
    if not pmid_list:
        return {}
    
    try:
        # Fetch the articles
        handle = Entrez.efetch(db="pubmed", id=",".join(pmid_list), rettype="xml", retmode="text")
        records = Entrez.read(handle)
        handle.close()
        
        # Extract abstracts
        abstracts = {}
        for article in records['PubmedArticle']:
            pmid = str(article['MedlineCitation']['PMID'])
            
            # Check if abstract exists
            if 'Abstract' in article['MedlineCitation']['Article']:
                abstract_text_list = article['MedlineCitation']['Article']['Abstract']['AbstractText']
                
                # The abstract text can be a list of labeled sections or just a list of text
                abstract_parts = []
                for abstract_part in abstract_text_list:
                    if hasattr(abstract_part, 'attributes') and 'Label' in abstract_part.attributes:
                        label = abstract_part.attributes['Label']
                        abstract_parts.append(f"{label}: {abstract_part}")
                    else:
                        abstract_parts.append(str(abstract_part))
                
                # Clean the abstract text to remove problematic characters
                abstract_text = " ".join(abstract_parts)
                # Replace thin spaces and other potentially problematic characters
                abstract_text = abstract_text.replace('\u2009', ' ')  # Replace thin space with regular space
                abstracts[pmid] = abstract_text
            else:
                abstracts[pmid] = ""
                
        return abstracts
    
    except Exception as e:
        logger.error(f"Error fetching abstracts for batch: {e}")
        return {}

def process_csv(input_file, output_file, email, batch_size=100, max_workers=1, delay=0.34, encoding='cp1252'):
    """
    Process the CSV file, fetch abstracts, and save to a new CSV file.
    
    Args:
        input_file: Path to input CSV file
        output_file: Path to output CSV file
        email: Email address for NCBI API
        batch_size: Number of PMIDs to process in each batch
        max_workers: Number of parallel workers
        delay: Delay between API calls in seconds
        encoding: File encoding (default: cp1252)

        $ python extractor3 -input_file ../data/data.csv -output_file output.csv
    """
    try:
        # Read the CSV file with the specified encoding
        df = pd.read_csv(input_file, encoding=encoding)
        total_articles = len(df)
        logger.info(f"Processing {total_articles} articles")
        
        # Check if 'pmid' column exists
        if 'pmid' not in df.columns:
            raise ValueError("Input CSV must contain a 'pmid' column")
        
        # Add abstract column
        df['abstract'] = ""
        
        # Process in batches
        pmid_batches = [df['pmid'][i:i+batch_size].tolist() for i in range(0, len(df), batch_size)]
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            results = list(tqdm(
                executor.map(
                    lambda batch: (batch, fetch_abstract_batch(batch, email)),
                    pmid_batches
                ),
                total=len(pmid_batches),
                desc="Fetching abstracts"
            ))
            
            # Update the dataframe with abstracts
            for pmid_batch, abstracts in results:
                for pmid in pmid_batch:
                    if str(pmid) in abstracts:
                        df.loc[df['pmid'] == pmid, 'abstract'] = abstracts[str(pmid)]
                time.sleep(delay)  # Be respectful to the NCBI API
        
        # Save to new CSV with UTF-8 encoding to handle all Unicode characters
        # Use 'replace' as errors strategy to substitute characters that can't be encoded
        try:
            df.to_csv(output_file, index=False, quoting=csv.QUOTE_ALL, encoding='utf-8')
            logger.info(f"File saved with UTF-8 encoding")
        except UnicodeEncodeError:
            # If UTF-8 somehow fails, try with the original encoding but replace problematic characters
            df.to_csv(output_file, index=False, quoting=csv.QUOTE_ALL, encoding=encoding, errors='replace')
            logger.info(f"File saved with {encoding} encoding and character replacement")
        logger.info(f"Successfully saved {total_articles} articles with abstracts to {output_file}")
        
        # Log statistics
        articles_with_abstracts = len(df[df['abstract'] != ""])
        logger.info(f"Articles with abstracts: {articles_with_abstracts}/{total_articles} ({articles_with_abstracts/total_articles*100:.2f}%)")
        
    except Exception as e:
        logger.error(f"Error processing CSV: {e}")
        raise

def main():
    """Main function."""
    args = parse_arguments()
    
    logger.info("Starting PubMed abstract extraction")
    logger.info(f"Input file: {args.input_file}")
    logger.info(f"Output file: {args.output_file}")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info(f"Max workers: {args.max_workers}")
    logger.info(f"File encoding: {args.encoding}")
    
    process_csv(
        args.input_file,
        args.output_file,
        args.email,
        args.batch_size,
        args.max_workers,
        args.delay,
        args.encoding
    )
    
    logger.info("Extraction completed")

if __name__ == "__main__":
    main()
