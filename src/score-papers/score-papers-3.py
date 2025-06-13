#!/usr/bin/env python3
"""
Prostate Cancer Treatment Study Paper Analysis Program

This program analyzes papers from a CSV file containing PubMed articles about prostate cancer treatment.
It applies multiple filters and calculates quality scores based on specified criteria.

Usage:
python prostate_paper_analyzer.py --input_file "output100.csv" --output_file "analyzed_papers.csv"
"""

import argparse
import pandas as pd
import re
import logging
from typing import Dict, List, Tuple, Union

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("prostate_analysis.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger()

class ProstateStudyAnalyzer:
    def __init__(self):
        """Initialize the analyzer with predefined criteria and scoring weights."""
        
        # Filter One: Prostate cancer keywords
        self.prostate_keywords = [
            "prostate cancer",
            "prostate neoplasm", 
            "prostate neoplasia"
        ]
        
        # Negative quality indicators with weights for scoring
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
        
        # Sample size patterns for extraction
        self.sample_size_patterns = [
            r'(?:n\s*=\s*|n\s+|patients?\s*[:\s]\s*)(\d+)',
            r'(\d+)\s+(?:patients?|men|subjects?|participants?)',
            r'total\s+of\s+(\d+)',
            r'cohort\s+of\s+(\d+)',
            r'sample\s+size\s+(?:of\s+)?(\d+)',
            r'study\s+(?:included|enrolled)\s+(\d+)',
            r'(\d+)\s+(?:underwent|received|treated)'
        ]
        
        # Follow-up time patterns
        self.followup_patterns = [
            r'median\s+follow[-\s]?up\s+(?:time\s+)?(?:of\s+)?(\d+(?:\.\d+)?)\s*years?',
            r'mean\s+follow[-\s]?up\s+(?:time\s+)?(?:of\s+)?(\d+(?:\.\d+)?)\s*years?',
            r'follow[-\s]?up\s+(?:time\s+)?(?:of\s+)?(\d+(?:\.\d+)?)\s*years?',
            r'(\d+(?:\.\d+)?)\s*years?\s+(?:of\s+)?follow[-\s]?up',
            r'(\d+(?:\.\d+)?)\s*(?:years?\s+)?median\s+follow[-\s]?up'
        ]
        
        # Risk stratification patterns
        self.risk_patterns = [
            r'(?:high[-\s]?risk|high\s+risk)\s+(?:group\s+)?(?:n\s*=\s*|patients?\s*[:\s]\s*)?(\d+)',
            r'(?:intermediate[-\s]?risk|intermediate\s+risk)\s+(?:group\s+)?(?:n\s*=\s*|patients?\s*[:\s]\s*)?(\d+)',
            r'(\d+)\s+(?:high[-\s]?risk|high\s+risk)',
            r'(\d+)\s+(?:intermediate[-\s]?risk|intermediate\s+risk)'
        ]
        
    def filter_one_prostate_keywords(self, text: str) -> bool:
        """
        Filter One: Check if prostate cancer keywords are present in the abstract.
        
        Args:
            text: Text to search (typically the abstract)
            
        Returns:
            bool: True if prostate keywords found, False otherwise
        """
        if pd.isna(text) or not isinstance(text, str):
            return False
            
        text_lower = text.lower()
        return any(keyword.lower() in text_lower for keyword in self.prostate_keywords)
    
    def filter_two_sample_size(self, text: str) -> str:
        """
        Filter Two: Check sample size requirements.
        High-risk patients: >= 50, Intermediate-risk patients: >= 100
        
        Args:
            text: Combined text from title, details, and abstract
            
        Returns:
            str: "TRUE", "FALSE", or "UNKNOWN"
        """
        if pd.isna(text) or not isinstance(text, str):
            return "UNKNOWN"
            
        text_lower = text.lower()
        
        # Extract all potential sample sizes
        sample_sizes = []
        for pattern in self.sample_size_patterns:
            matches = re.findall(pattern, text_lower, re.IGNORECASE)
            sample_sizes.extend([int(match) for match in matches if match.isdigit()])
        
        # Look for risk-specific sample sizes
        high_risk_sizes = []
        intermediate_risk_sizes = []
        
        for pattern in self.risk_patterns:
            matches = re.findall(pattern, text_lower, re.IGNORECASE)
            for match in matches:
                if isinstance(match, tuple):
                    # Handle tuple results from regex groups
                    for m in match:
                        if m.isdigit():
                            size = int(m)
                            if 'high' in pattern:
                                high_risk_sizes.append(size)
                            elif 'intermediate' in pattern:
                                intermediate_risk_sizes.append(size)
                elif match.isdigit():
                    size = int(match)
                    if 'high' in pattern:
                        high_risk_sizes.append(size)
                    elif 'intermediate' in pattern:
                        intermediate_risk_sizes.append(size)
        
        # Check criteria
        # If we have specific risk group sizes, use those
        if high_risk_sizes or intermediate_risk_sizes:
            high_risk_ok = not high_risk_sizes or max(high_risk_sizes) >= 50
            intermediate_risk_ok = not intermediate_risk_sizes or max(intermediate_risk_sizes) >= 100
            return "TRUE" if high_risk_ok and intermediate_risk_ok else "FALSE"
        
        # If no specific risk stratification, check general sample size
        if sample_sizes:
            max_size = max(sample_sizes)
            # Use conservative approach: require at least 100 for general studies
            return "TRUE" if max_size >= 100 else "FALSE"
        
        return "UNKNOWN"
    
    def filter_three_followup(self, text: str) -> str:
        """
        Filter Three: Check if median follow-up time is >= 5 years.
        
        Args:
            text: Combined text from title, details, and abstract
            
        Returns:
            str: "TRUE", "FALSE", or "UNKNOWN"
        """
        if pd.isna(text) or not isinstance(text, str):
            return "UNKNOWN"
            
        text_lower = text.lower()
        
        # Extract follow-up times
        followup_times = []
        for pattern in self.followup_patterns:
            matches = re.findall(pattern, text_lower, re.IGNORECASE)
            for match in matches:
                try:
                    followup_times.append(float(match))
                except (ValueError, TypeError):
                    continue
        
        # Check for months and convert to years
        month_patterns = [
            r'median\s+follow[-\s]?up\s+(?:time\s+)?(?:of\s+)?(\d+(?:\.\d+)?)\s*months?',
            r'(\d+(?:\.\d+)?)\s*months?\s+(?:of\s+)?follow[-\s]?up'
        ]
        
        for pattern in month_patterns:
            matches = re.findall(pattern, text_lower, re.IGNORECASE)
            for match in matches:
                try:
                    months = float(match)
                    years = months / 12.0
                    followup_times.append(years)
                except (ValueError, TypeError):
                    continue
        
        if followup_times:
            max_followup = max(followup_times)
            return "TRUE" if max_followup >= 5.0 else "FALSE"
        
        return "UNKNOWN"
    
    def filter_four_staging(self, text: str) -> str:
        """
        Filter Four: Check for pathologic staging without clinical staging.
        
        Args:
            text: Combined text from title, details, and abstract
            
        Returns:
            str: "TRUE", "FALSE", or "UNKNOWN"
        """
        if pd.isna(text) or not isinstance(text, str):
            return "UNKNOWN"
            
        text_lower = text.lower()
        
        # Check for staging terms
        has_pathologic_staging = bool(re.search(r'pathologic(?:al)?\s+stag', text_lower))
        has_clinical_staging = bool(re.search(r'clinical\s+stag', text_lower))
        
        # If pathologic staging without clinical staging, mark as FALSE
        if has_pathologic_staging and not has_clinical_staging:
            return "FALSE"
        
        # If clinical staging is mentioned, mark as TRUE
        if has_clinical_staging:
            return "TRUE"
        
        # If no clear indication of staging
        return "UNKNOWN"
    
    def calculate_keyword_score(self, text: str) -> int:
        """
        Calculate keyword quality score based on negative indicators.
        
        Args:
            text: Combined text from title, details, and abstract
            
        Returns:
            int: Score from 0 to 100
        """
        if pd.isna(text) or not isinstance(text, str):
            return 0
            
        text_lower = text.lower()
        
        # Start with base score
        score = 100
        
        # Apply negative scoring
        for term, weight in self.negative_terms.items():
            if term.lower() in text_lower:
                score += weight  # weight is negative
        
        # Ensure score stays within bounds
        score = max(0, min(100, score))
        
        return score
    
    def get_combined_text(self, row: pd.Series) -> str:
        """
        Combine title, details, and abstract into a single text for analysis.
        
        Args:
            row: Pandas Series representing a row from the CSV
            
        Returns:
            str: Combined text
        """
        texts = []
        for col in ['title', 'details', 'abstract']:
            if col in row and pd.notna(row[col]) and isinstance(row[col], str):
                texts.append(row[col])
        
        return ' '.join(texts)
    
    def analyze_paper(self, row: pd.Series) -> Dict[str, Union[str, bool, int]]:
        """
        Analyze a single paper and return all filter results.
        
        Args:
            row: Pandas Series representing a row from the CSV
            
        Returns:
            dict: Dictionary with all analysis results
        """
        # Get text for analysis
        abstract_text = row.get('abstract', '') if pd.notna(row.get('abstract', '')) else ''
        combined_text = self.get_combined_text(row)
        
        # Apply all filters
        results = {
            'prostate_keyword': self.filter_one_prostate_keywords(abstract_text),
            'sample_size_indicator': self.filter_two_sample_size(combined_text),
            'followup_indicator': self.filter_three_followup(combined_text),
            'pathologic_staging_filter': self.filter_four_staging(combined_text),
            'keyword_score': self.calculate_keyword_score(combined_text)
        }
        
        return results
    
    def process_csv(self, input_file: str, output_file: str) -> None:
        """
        Process the entire CSV file and add analysis columns.
        
        Args:
            input_file: Path to input CSV file
            output_file: Path to output CSV file
        """
        try:
            # Read the CSV file
            logger.info(f"Reading input file: {input_file}")
            df = pd.read_csv(input_file, encoding='utf-8')
            
            initial_count = len(df)
            logger.info(f"Processing {initial_count} papers")
            
            # Initialize new columns
            df['prostate_keyword'] = False
            df['sample_size_indicator'] = "UNKNOWN"
            df['followup_indicator'] = "UNKNOWN"
            df['pathologic_staging_filter'] = "UNKNOWN"
            df['keyword_score'] = 0
            
            # Process each row
            for idx, row in df.iterrows():
                try:
                    results = self.analyze_paper(row)
                    
                    # Update the dataframe
                    for column, value in results.items():
                        df.loc[idx, column] = value
                        
                except Exception as e:
                    logger.warning(f"Error processing row {idx}: {e}")
                    continue
            
            # Log statistics
            prostate_match_count = len(df[df['prostate_keyword'] == True])
            adequate_size_count = len(df[df['sample_size_indicator'] == "TRUE"])
            adequate_followup_count = len(df[df['followup_indicator'] == "TRUE"])
            clinical_staging_count = len(df[df['pathologic_staging_filter'] == "TRUE"])
            
            logger.info(f"Analysis Results:")
            logger.info(f"  Papers with prostate keywords: {prostate_match_count}/{initial_count} ({prostate_match_count/initial_count*100:.2f}%)")
            logger.info(f"  Papers with adequate sample size: {adequate_size_count}/{initial_count} ({adequate_size_count/initial_count*100:.2f}%)")
            logger.info(f"  Papers with adequate follow-up: {adequate_followup_count}/{initial_count} ({adequate_followup_count/initial_count*100:.2f}%)")
            logger.info(f"  Papers with clinical staging: {clinical_staging_count}/{initial_count} ({clinical_staging_count/initial_count*100:.2f}%)")
            logger.info(f"  Average keyword score: {df['keyword_score'].mean():.2f}")
            
            # Save results
            logger.info(f"Saving results to: {output_file}")
            df.to_csv(output_file, index=False, encoding='utf-8')
            
            logger.info("Analysis completed successfully")
            
        except Exception as e:
            logger.error(f"Error processing CSV file: {e}")
            raise

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Analyze prostate cancer treatment study papers.')
    parser.add_argument('--input_file', type=str, required=True, help='Path to input CSV file')
    parser.add_argument('--output_file', type=str, required=True, help='Path to output CSV file')
    return parser.parse_args()

def main():
    """Main function."""
    args = parse_arguments()
    
    logger.info("Starting Prostate Cancer Treatment Study Paper Analysis")
    logger.info(f"Input file: {args.input_file}")
    logger.info(f"Output file: {args.output_file}")
    
    # Initialize analyzer and process the file
    analyzer = ProstateStudyAnalyzer()
    analyzer.process_csv(args.input_file, args.output_file)
    
    logger.info("Analysis completed")

if __name__ == "__main__":
    main()