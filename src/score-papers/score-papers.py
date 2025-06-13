#!/usr/bin/env python3
"""
Prostate Cancer Treatment Study Paper Analysis Program

This program analyzes papers from a CSV file containing PubMed abstracts
and applies multiple filters and quality scoring for prostate cancer research.
"""

import pandas as pd
import numpy as np
import re
import argparse
import logging
from pathlib import Path

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ProstateStudyAnalyzer:
    def __init__(self):
        """Initialize the analyzer with scoring criteria."""
        # Positive quality indicators (based on summary criteria)
        self.positive_terms = {
            'randomized': 15,
            'prospective': 10,
            'multicenter': 8,
            'long-term': 5,
            'outcomes': 5,
            'survival': 8,
            'biochemical recurrence': 8,
            'disease-free survival': 10,
            'overall survival': 10,
            'cancer-specific survival': 8,
            'metastasis-free survival': 8,
            'radical prostatectomy': 5,
            'radiation therapy': 5,
            'brachytherapy': 5,
            'clinical staging': 8,
            'd\'amico': 10,
            'nccn': 10,
            'risk stratification': 8,
            'intermediate risk': 5,
            'high risk': 5,
            'low risk': 5
        }

        # Negative quality indicators (rejection criteria)
        self.negative_terms = {
            'abstract only': -50,
            'conference abstract': -50,
            'meeting abstract': -50,
            'case report': -10,
            'case series': -8,
            'review': 0,
            'editorial': -50,
            'commentary': -50,
            'letter': -50,
            'pathologic staging only': -30,
            'no stratification': -6
        }
        
        # Prostate keywords for filter one
        self.prostate_keywords = [
            'prostate cancer',
            'prostate neoplasm',
            'prostate neoplasia'
        ]

    def filter_one_prostate_keywords(self, abstract):
        """
        Filter One: Check for prostate-related keywords in abstract.
        
        Args:
            abstract (str): The abstract text
            
        Returns:
            str: 'TRUE' if keywords found, 'FALSE' otherwise
        """
        if pd.isna(abstract) or not isinstance(abstract, str):
            return 'FALSE'
        
        abstract_lower = abstract.lower()
        for keyword in self.prostate_keywords:
            if keyword.lower() in abstract_lower:
                return 'TRUE'
        return 'FALSE'

    def filter_two_sample_size(self, abstract, title):
        """
        Filter Two: Check sample size indicators.
        
        Args:
            abstract (str): The abstract text
            title (str): The paper title
            
        Returns:
            str: 'TRUE', 'FALSE', or 'UNKNOWN'
        """
        if pd.isna(abstract) and pd.isna(title):
            return 'UNKNOWN'
        
        # Combine abstract and title for analysis
        text = str(abstract or '') + ' ' + str(title or '')
        text_lower = text.lower()
        
        # Look for explicit mentions of high-risk and intermediate-risk patient numbers
        # Pattern to find numbers followed by relevant terms
        patterns = [
            r'(\d+)\s*(?:patients?|men|participants?|cases?)\s*(?:with\s*)?(?:high[- ]?risk)',
            r'(\d+)\s*high[- ]?risk\s*(?:patients?|men|participants?|cases?)',
            r'(\d+)\s*(?:patients?|men|participants?|cases?)\s*(?:with\s*)?(?:intermediate[- ]?risk)',
            r'(\d+)\s*intermediate[- ]?risk\s*(?:patients?|men|participants?|cases?)',
            r'(?:high[- ]?risk)(?:\s*[\w\s]*?)(\d+)\s*(?:patients?|men|participants?|cases?)',
            r'(?:intermediate[- ]?risk)(?:\s*[\w\s]*?)(\d+)\s*(?:patients?|men|participants?|cases?)',
            # General patient numbers
            r'(\d+)\s*(?:patients?|men|participants?|subjects?|cases?)',
            r'n\s*=\s*(\d+)',
            r'total\s*(?:of\s*)?(\d+)\s*(?:patients?|men|participants?|cases?)'
        ]
        
        numbers = []
        for pattern in patterns:
            matches = re.findall(pattern, text_lower)
            for match in matches:
                try:
                    numbers.append(int(match))
                except ValueError:
                    continue
        
        if not numbers:
            return 'UNKNOWN'
        
        # Check if any number suggests adequate sample size
        # For high-risk: need ≥50, for intermediate: need ≥100
        # If we can't distinguish risk levels, use conservative approach
        max_number = max(numbers)
        
        # Look for specific risk stratification
        has_high_risk = any(term in text_lower for term in ['high risk', 'high-risk'])
        has_intermediate_risk = any(term in text_lower for term in ['intermediate risk', 'intermediate-risk'])
        
        if has_high_risk and any(n >= 50 for n in numbers):
            return 'TRUE'
        if has_intermediate_risk and any(n >= 100 for n in numbers):
            return 'TRUE'
        if not has_high_risk and not has_intermediate_risk and max_number >= 100:
            return 'TRUE'
        if max_number < 50:
            return 'FALSE'
        
        return 'UNKNOWN'

    def filter_three_followup_time(self, abstract, title):
        """
        Filter Three: Check median followup time.
        
        Args:
            abstract (str): The abstract text
            title (str): The paper title
            
        Returns:
            str: 'TRUE', 'FALSE', or 'UNKNOWN'
        """
        if pd.isna(abstract) and pd.isna(title):
            return 'UNKNOWN'
        
        text = str(abstract or '') + ' ' + str(title or '')
        text_lower = text.lower()
        
        # Patterns to find followup time
        followup_patterns = [
            r'median\s+(?:follow[- ]?up|followup)\s*(?:time|period|duration)?\s*(?:was|of|is)?\s*(\d+(?:\.\d+)?)\s*(years?|months?|yrs?|mo)',
            r'median\s+(?:follow[- ]?up|followup)\s*[:\-]?\s*(\d+(?:\.\d+)?)\s*(years?|months?|yrs?|mo)',
            r'(?:follow[- ]?up|followup)\s*(?:time|period|duration)?\s*(?:was|of|is)?\s*(\d+(?:\.\d+)?)\s*(years?|months?|yrs?|mo)',
            r'(?:after|with)\s+(?:a\s+)?(?:median\s+)?(?:follow[- ]?up|followup)\s+(?:of\s+)?(\d+(?:\.\d+)?)\s*(years?|months?|yrs?|mo)',
            r'(\d+(?:\.\d+)?)\s*(years?|months?|yrs?|mo)\s+(?:of\s+)?(?:median\s+)?(?:follow[- ]?up|followup)'
        ]
        
        followup_years = []
        for pattern in followup_patterns:
            matches = re.findall(pattern, text_lower)
            for match in matches:
                try:
                    value = float(match[0])
                    unit = match[1].lower()
                    
                    # Convert to years
                    if unit.startswith('month') or unit == 'mo':
                        value = value / 12
                    
                    followup_years.append(value)
                except (ValueError, IndexError):
                    continue
        
        if not followup_years:
            return 'UNKNOWN'
        
        # Use the maximum followup time found
        max_followup = max(followup_years)
        
        if max_followup >= 5.0:
            return 'TRUE'
        else:
            return 'FALSE'

    def filter_four_pathologic_staging(self, abstract, title):
        """
        Filter Four: Check for pathologic staging without clinical staging.
        
        Args:
            abstract (str): The abstract text
            title (str): The paper title
            
        Returns:
            str: 'TRUE', 'FALSE', or 'UNKNOWN'
        """
        if pd.isna(abstract) and pd.isna(title):
            return 'UNKNOWN'
        
        text = str(abstract or '') + ' ' + str(title or '')
        text_lower = text.lower()
        
        # Check for clinical staging indicators
        clinical_staging_terms = [
            'clinical staging',
            'clinical stage',
            'clinical t',
            'ct stage',
            'ct1', 'ct2', 'ct3', 'ct4',
            'clinical tumor stage',
            'preoperative staging'
        ]
        
        # Check for pathologic staging indicators
        pathologic_staging_terms = [
            'pathologic staging',
            'pathologic stage',
            'pathological staging',
            'pathological stage',
            'pathologic t',
            'pt stage',
            'pt1', 'pt2', 'pt3', 'pt4',
            'surgical staging',
            'postoperative staging'
        ]
        
        has_clinical_staging = any(term in text_lower for term in clinical_staging_terms)
        has_pathologic_staging = any(term in text_lower for term in pathologic_staging_terms)
        
        if has_clinical_staging:
            return 'TRUE'
        elif has_pathologic_staging and not has_clinical_staging:
            return 'FALSE'
        else:
            return 'UNKNOWN'

    def calculate_keyword_score(self, abstract, title, details):
        """
        Calculate keyword quality score based on presence of positive and negative terms.
        
        Args:
            abstract (str): The abstract text
            title (str): The paper title
            details (str): The paper details
            
        Returns:
            int: Score from 0 to 100
        """
        if pd.isna(abstract) and pd.isna(title) and pd.isna(details):
            return 0
        
        # Combine all text for analysis
        text = str(abstract or '') + ' ' + str(title or '') + ' ' + str(details or '')
        text_lower = text.lower()
        
        score = 0
        matched_terms = []
        
        # Check positive terms
        for term, weight in self.positive_terms.items():
            if term.lower() in text_lower:
                score += weight
                matched_terms.append(f"+{term}({weight})")
        
        # Check negative terms
        for term, weight in self.negative_terms.items():
            if term.lower() in text_lower:
                score += weight  # weight is already negative
                matched_terms.append(f"{term}({weight})")
        
        # Ensure score is between 0 and 100
        score = max(0, min(100, score))
        
        # Log detailed scoring for debugging (optional)
        if matched_terms:
            logger.debug(f"Score: {score}, Terms: {', '.join(matched_terms)}")
        
        return score

    def analyze_paper(self, row):
        """
        Analyze a single paper (row) and return updated row with new columns.
        
        Args:
            row (pd.Series): A row from the dataframe
            
        Returns:
            pd.Series: Updated row with new analysis columns
        """
        # Extract fields
        abstract = row.get('abstract', '')
        title = row.get('title', '')
        details = row.get('details', '')
        
        # Apply filters
        row['prostate_keyword'] = self.filter_one_prostate_keywords(abstract)
        row['sample_size_indicator'] = self.filter_two_sample_size(abstract, title)
        row['followup_indicator'] = self.filter_three_followup_time(abstract, title)
        row['pathologic_staging_filter'] = self.filter_four_pathologic_staging(abstract, title)
        row['keyword_score'] = self.calculate_keyword_score(abstract, title, details)
        
        return row

    def process_csv(self, input_file, output_file=None):
        """
        Process the entire CSV file.
        
        Args:
            input_file (str): Path to input CSV file
            output_file (str): Path to output CSV file (optional)
            
        Returns:
            pd.DataFrame: Processed dataframe
        """
        logger.info(f"Reading CSV file: {input_file}")
        
        try:
            df = pd.read_csv(input_file)
            logger.info(f"Loaded {len(df)} papers for analysis")
        except Exception as e:
            logger.error(f"Error reading CSV file: {e}")
            raise
        
        # Analyze each paper
        logger.info("Analyzing papers...")
        df = df.apply(self.analyze_paper, axis=1)
        
        # Log statistics
        self.log_analysis_statistics(df)
        
        # Save results if output file specified
        if output_file:
            logger.info(f"Saving results to: {output_file}")
            df.to_csv(output_file, index=False)
            logger.info("Analysis complete!")
        
        return df

    def log_analysis_statistics(self, df):
        """Log analysis statistics."""
        total_papers = len(df)
        
        # Filter statistics
        prostate_true = len(df[df['prostate_keyword'] == 'TRUE'])
        sample_size_true = len(df[df['sample_size_indicator'] == 'TRUE'])
        followup_true = len(df[df['followup_indicator'] == 'TRUE'])
        staging_true = len(df[df['pathologic_staging_filter'] == 'TRUE'])
        
        logger.info("\n=== ANALYSIS STATISTICS ===")
        logger.info(f"Total papers analyzed: {total_papers}")
        logger.info(f"Prostate keywords found: {prostate_true} ({prostate_true/total_papers*100:.1f}%)")
        logger.info(f"Adequate sample size: {sample_size_true} ({sample_size_true/total_papers*100:.1f}%)")
        logger.info(f"Adequate followup: {followup_true} ({followup_true/total_papers*100:.1f}%)")
        logger.info(f"Appropriate staging: {staging_true} ({staging_true/total_papers*100:.1f}%)")
        
        # Score statistics
        scores = df['keyword_score']
        logger.info(f"\nKeyword Score Statistics:")
        logger.info(f"Mean: {scores.mean():.1f}")
        logger.info(f"Median: {scores.median():.1f}")
        logger.info(f"Range: {scores.min()}-{scores.max()}")
        logger.info(f"Standard deviation: {scores.std():.1f}")
        
        # Score distribution
        score_ranges = [
            (0, 20, "Very Low"),
            (21, 40, "Low"), 
            (41, 60, "Medium"),
            (61, 80, "High"),
            (81, 100, "Very High")
        ]
        
        logger.info("\nScore Distribution:")
        for min_score, max_score, label in score_ranges:
            count = len(df[(df['keyword_score'] >= min_score) & (df['keyword_score'] <= max_score)])
            logger.info(f"{label} ({min_score}-{max_score}): {count} ({count/total_papers*100:.1f}%)")

def main():
    """Main function with command line interface."""
    parser = argparse.ArgumentParser(description='Analyze prostate cancer treatment study papers')
    parser.add_argument('--input', '-i', required=True, help='Input CSV file path')
    parser.add_argument('--output', '-o', help='Output CSV file path (optional)')
    parser.add_argument('--verbose', '-v', action='store_true', help='Enable verbose logging')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Initialize analyzer
    analyzer = ProstateStudyAnalyzer()
    
    # Process the CSV file
    try:
        result_df = analyzer.process_csv(args.input, args.output)
        
        if not args.output:
            print("\nFirst 5 rows of results:")
            print(result_df[['title', 'prostate_keyword', 'sample_size_indicator', 
                           'followup_indicator', 'pathologic_staging_filter', 'keyword_score']].head())
            
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())