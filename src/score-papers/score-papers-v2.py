#!/usr/bin/env python3
"""
Prostate Cancer Treatment Study Paper Analysis Program

This program analyzes papers from a CSV file containing prostate cancer treatment studies
and applies multiple filters and scoring criteria to evaluate study quality and relevance.

Requirements:
- pandas
- re (for regex operations)
- numpy (for numerical operations)

Usage:
python prostate_analysis.py

Input: output100.csv
Output: analyzed_output.csv
"""

import pandas as pd
import re
import numpy as np
import logging
from typing import Tuple, Dict, Any

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
    """Main class for analyzing prostate cancer treatment studies."""
    
    def __init__(self):
        """Initialize the analyzer with negative quality indicators."""
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

    def extract_sample_sizes(self, text: str) -> Tuple[int, int, int]:
        """
        Extract sample sizes for low, intermediate, and high-risk patients.
        
        Args:
            text: Combined text from title, details, and abstract
            
        Returns:
            Tuple of (low_risk_n, intermediate_risk_n, high_risk_n)
        """
        if not isinstance(text, str):
            return 0, 0, 0
            
        text_lower = text.lower()
        
        # Initialize counts
        low_risk_n = 0
        intermediate_risk_n = 0
        high_risk_n = 0
        
        # Patterns for different risk groups with sample sizes
        patterns = {
            'low': [
                r'low[\-\s]*risk[^\d]*(\d+)',
                r'(\d+)[^\d]*low[\-\s]*risk',
                r'low[\-\s]*grade[^\d]*(\d+)',
                r'(\d+)[^\d]*low[\-\s]*grade'
            ],
            'intermediate': [
                r'intermediate[\-\s]*risk[^\d]*(\d+)',
                r'(\d+)[^\d]*intermediate[\-\s]*risk',
                r'medium[\-\s]*risk[^\d]*(\d+)',
                r'(\d+)[^\d]*medium[\-\s]*risk'
            ],
            'high': [
                r'high[\-\s]*risk[^\d]*(\d+)',
                r'(\d+)[^\d]*high[\-\s]*risk',
                r'high[\-\s]*grade[^\d]*(\d+)',
                r'(\d+)[^\d]*high[\-\s]*grade'
            ]
        }
        
        # Extract numbers for each risk category
        for risk_type, pattern_list in patterns.items():
            for pattern in pattern_list:
                matches = re.findall(pattern, text_lower)
                if matches:
                    try:
                        # Take the largest number found (most likely total patients)
                        numbers = [int(match) for match in matches if match.isdigit()]
                        if numbers:
                            if risk_type == 'low':
                                low_risk_n = max(low_risk_n, max(numbers))
                            elif risk_type == 'intermediate':
                                intermediate_risk_n = max(intermediate_risk_n, max(numbers))
                            elif risk_type == 'high':
                                high_risk_n = max(high_risk_n, max(numbers))
                    except ValueError:
                        continue
        
        # Also look for general patient numbers and D'Amico stratification
        general_patterns = [
            r'(\d+)\s*patients?',
            r'n\s*=\s*(\d+)',
            r'total[^\d]*(\d+)',
            r'cohort[^\d]*(\d+)'
        ]
        
        general_numbers = []
        for pattern in general_patterns:
            matches = re.findall(pattern, text_lower)
            general_numbers.extend([int(m) for m in matches if m.isdigit() and int(m) > 10])
        
        # If we found D'Amico classification mentioned, try to estimate distribution
        if 'd\'amico' in text_lower or 'damico' in text_lower:
            if general_numbers and not any([low_risk_n, intermediate_risk_n, high_risk_n]):
                total = max(general_numbers)
                # Rough estimation based on typical D'Amico distributions
                low_risk_n = int(total * 0.3)
                intermediate_risk_n = int(total * 0.4)
                high_risk_n = int(total * 0.3)
        
        return low_risk_n, intermediate_risk_n, high_risk_n

    def extract_followup_time(self, text: str) -> float:
        """
        Extract median followup time in years.
        
        Args:
            text: Combined text from title, details, and abstract
            
        Returns:
            Followup time in years (0 if not found)
        """
        if not isinstance(text, str):
            return 0.0
            
        text_lower = text.lower()
        
        # Patterns for followup time
        followup_patterns = [
            r'median\s+follow[\-\s]*up[^\d]*(\d+(?:\.\d+)?)\s*years?',
            r'follow[\-\s]*up[^\d]*(\d+(?:\.\d+)?)\s*years?',
            r'median[^\d]*(\d+(?:\.\d+)?)\s*years?\s*follow[\-\s]*up',
            r'(\d+(?:\.\d+)?)\s*years?\s*median\s*follow[\-\s]*up',
            r'median\s+fu[^\d]*(\d+(?:\.\d+)?)\s*y',
            r'fu[^\d]*(\d+(?:\.\d+)?)\s*years?',
            r'followup[^\d]*(\d+(?:\.\d+)?)\s*months?',
        ]
        
        for pattern in followup_patterns:
            matches = re.findall(pattern, text_lower)
            if matches:
                try:
                    time_value = float(matches[0])
                    # Convert months to years if the pattern included months
                    if 'month' in pattern:
                        time_value = time_value / 12.0
                    return time_value
                except ValueError:
                    continue
        
        return 0.0

    def check_staging_criteria(self, text: str) -> str:
        """
        Check for pathologic and clinical staging criteria.
        
        Args:
            text: Combined text from title, details, and abstract
            
        Returns:
            'TRUE' if clinical staging present, 'FALSE' if pathologic only, 'UNKNOWN' otherwise
        """
        if not isinstance(text, str):
            return 'UNKNOWN'
            
        text_lower = text.lower()
        
        # Check for pathologic staging indicators
        pathologic_indicators = [
            'pathologic staging',
            'pathological staging', 
            'pstage',
            'pt stage',
            'pathologic t',
            'pathological t',
            'surgical specimen',
            'prostatectomy specimen'
        ]
        
        # Check for clinical staging indicators
        clinical_indicators = [
            'clinical staging',
            'cstage',
            'ct stage',
            'clinical t',
            'pre-operative staging',
            'preoperative staging',
            'imaging staging',
            'mri staging',
            'ct staging',
            'pet staging'
        ]
        
        has_pathologic = any(indicator in text_lower for indicator in pathologic_indicators)
        has_clinical = any(indicator in text_lower for indicator in clinical_indicators)
        
        if has_pathologic and not has_clinical:
            return 'FALSE'  # Pathologic staging without clinical staging
        elif has_clinical:
            return 'TRUE'   # Has clinical staging
        else:
            return 'UNKNOWN'  # No clear indication

    def calculate_keyword_score(self, text: str) -> int:
        """
        Calculate keyword quality score based on positive and negative indicators.
        
        Args:
            text: Combined text from title, details, and abstract
            
        Returns:
            Score from 0 to 100
        """
        if not isinstance(text, str):
            return 0
            
        text_lower = text.lower()
        score = 50  # Start with baseline score
        
        # Apply negative terms
        for term, penalty in self.negative_terms.items():
            if term in text_lower:
                score += penalty  # penalty is negative
                logger.debug(f"Applied penalty for '{term}': {penalty}")
        
        # Apply positive terms
        for term, bonus in self.positive_terms.items():
            if term in text_lower:
                score += bonus
                logger.debug(f"Applied bonus for '{term}': {bonus}")
        
        # Ensure score is within 0-100 range
        score = max(0, min(100, score))
        
        return score

    def apply_sample_size_filter(self, row: pd.Series) -> str:
        """
        Apply Filter 1: Sample size filter.
        
        Args:
            row: Pandas series representing a row of data
            
        Returns:
            'TRUE', 'FALSE', or 'UNKNOWN'
        """
        # Combine all text fields for analysis
        text = f"{row.get('title', '')} {row.get('details', '')} {row.get('abstract', '')}"
        
        low_risk_n, intermediate_risk_n, high_risk_n = self.extract_sample_sizes(text)
        
        # Check if we have any sample size information
        if low_risk_n == 0 and intermediate_risk_n == 0 and high_risk_n == 0:
            return 'UNKNOWN'
        
        # Apply the criteria: high-risk <50 OR intermediate-risk <100
        if (high_risk_n > 0 and high_risk_n < 50) or (intermediate_risk_n > 0 and intermediate_risk_n < 100):
            return 'FALSE'
        
        # If we have numbers and they meet the criteria
        if (high_risk_n >= 50) or (intermediate_risk_n >= 100):
            return 'TRUE'
        
        return 'UNKNOWN'

    def apply_followup_filter(self, row: pd.Series) -> str:
        """
        Apply Filter 2: Followup time filter.
        
        Args:
            row: Pandas series representing a row of data
            
        Returns:
            'TRUE', 'FALSE', or 'UNKNOWN'
        """
        # Combine all text fields for analysis
        text = f"{row.get('title', '')} {row.get('details', '')} {row.get('abstract', '')}"
        
        followup_time = self.extract_followup_time(text)
        
        if followup_time == 0.0:
            return 'UNKNOWN'
        elif followup_time >= 5.0:
            return 'TRUE'
        else:
            return 'FALSE'

    def apply_pathologic_staging_filter(self, row: pd.Series) -> str:
        """
        Apply Filter 3: Pathologic staging filter.
        
        Args:
            row: Pandas series representing a row of data
            
        Returns:
            'TRUE', 'FALSE', or 'UNKNOWN'
        """
        # Combine all text fields for analysis
        text = f"{row.get('title', '')} {row.get('details', '')} {row.get('abstract', '')}"
        
        return self.check_staging_criteria(text)

    def calculate_row_keyword_score(self, row: pd.Series) -> int:
        """
        Calculate keyword score for a single row.
        
        Args:
            row: Pandas series representing a row of data
            
        Returns:
            Keyword score (0-100)
        """
        # Combine all text fields for analysis
        text = f"{row.get('title', '')} {row.get('details', '')} {row.get('abstract', '')}"
        
        return self.calculate_keyword_score(text)

    def analyze_csv(self, input_file: str = 'output100.csv', output_file: str = 'analyzed_output.csv'):
        """
        Main function to analyze the CSV file and add new columns.
        
        Args:
            input_file: Path to input CSV file
            output_file: Path to output CSV file
        """
        try:
            # Read the CSV file
            logger.info(f"Reading {input_file}")
            df = pd.read_csv(input_file)
            logger.info(f"Loaded {len(df)} rows")
            
            # Initialize new columns
            df['sample_size_indicator'] = ''
            df['followup_indicator'] = ''
            df['pathologic_staging_filter'] = ''
            df['keyword_score'] = 0
            
            # Process each row
            logger.info("Applying filters and calculating scores...")
            
            for idx, row in df.iterrows():
                # Apply Filter 1: Sample size
                df.at[idx, 'sample_size_indicator'] = self.apply_sample_size_filter(row)
                
                # Apply Filter 2: Followup time (only for rows that passed Filter 1)
                if df.at[idx, 'sample_size_indicator'] in ['TRUE', 'UNKNOWN']:
                    df.at[idx, 'followup_indicator'] = self.apply_followup_filter(row)
                else:
                    df.at[idx, 'followup_indicator'] = 'N/A'  # Skip if failed Filter 1
                
                # Apply Filter 3: Pathologic staging
                df.at[idx, 'pathologic_staging_filter'] = self.apply_pathologic_staging_filter(row)
                
                # Calculate keyword score
                df.at[idx, 'keyword_score'] = self.calculate_row_keyword_score(row)
                
                if (idx + 1) % 10 == 0:
                    logger.info(f"Processed {idx + 1}/{len(df)} rows")
            
            # Save the results
            logger.info(f"Saving results to {output_file}")
            df.to_csv(output_file, index=False)
            
            # Print summary statistics
            self.print_summary_statistics(df)
            
            logger.info("Analysis completed successfully")
            
        except Exception as e:
            logger.error(f"Error during analysis: {e}")
            raise

    def print_summary_statistics(self, df: pd.DataFrame):
        """
        Print summary statistics of the analysis results.
        
        Args:
            df: Analyzed DataFrame
        """
        logger.info("\n=== SUMMARY STATISTICS ===")
        
        # Sample size filter results
        sample_size_counts = df['sample_size_indicator'].value_counts()
        logger.info(f"Sample Size Filter Results:")
        for value, count in sample_size_counts.items():
            logger.info(f"  {value}: {count} ({count/len(df)*100:.1f}%)")
        
        # Followup filter results (for eligible papers)
        eligible_papers = df[df['sample_size_indicator'].isin(['TRUE', 'UNKNOWN'])]
        if len(eligible_papers) > 0:
            followup_counts = eligible_papers['followup_indicator'].value_counts()
            logger.info(f"\nFollowup Filter Results (for eligible papers):")
            for value, count in followup_counts.items():
                logger.info(f"  {value}: {count} ({count/len(eligible_papers)*100:.1f}%)")
        
        # Pathologic staging filter results
        staging_counts = df['pathologic_staging_filter'].value_counts()
        logger.info(f"\nPathologic Staging Filter Results:")
        for value, count in staging_counts.items():
            logger.info(f"  {value}: {count} ({count/len(df)*100:.1f}%)")
        
        # Keyword score statistics
        logger.info(f"\nKeyword Score Statistics:")
        logger.info(f"  Mean: {df['keyword_score'].mean():.1f}")
        logger.info(f"  Median: {df['keyword_score'].median():.1f}")
        logger.info(f"  Min: {df['keyword_score'].min()}")
        logger.info(f"  Max: {df['keyword_score'].max()}")
        
        # High-quality papers (passed all filters and high score)
        high_quality = df[
            (df['sample_size_indicator'] == 'TRUE') & 
            (df['followup_indicator'] == 'TRUE') & 
            (df['pathologic_staging_filter'] == 'TRUE') & 
            (df['keyword_score'] >= 70)
        ]
        logger.info(f"\nHigh-quality papers (passed all filters + score ≥70): {len(high_quality)}")


def main():
    """Main function to run the analysis."""
    analyzer = ProstateStudyAnalyzer()
    analyzer.analyze_csv()


if __name__ == "__main__":
    main()