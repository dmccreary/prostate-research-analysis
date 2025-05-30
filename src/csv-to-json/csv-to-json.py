#!/usr/bin/env python3
"""
Convert Scored Prostate Cancer Papers to JSON for Interactive Visualization

This script converts the output100_scored.csv file to a JSON format suitable
for interactive web visualization with detailed paper information.
"""

import pandas as pd
import json
import re
import numpy as np
from datetime import datetime

def convert_numpy_types(obj):
    """Convert numpy types to native Python types for JSON serialization."""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif pd.isna(obj):
        return None
    return obj
def extract_year_from_details(details):
    """Extract publication year from details field."""
    if pd.isna(details) or not details:
        return None
    
    # Look for year patterns like "2020", "2019", etc.
    year_match = re.search(r'\b(20\d{2})\b', str(details))
    if year_match:
        return int(year_match.group(1))
    return None

def extract_journal_from_details(details):
    """Extract journal name from details field."""
    if pd.isna(details) or not details:
        return "Unknown"
    
    # Journal name is typically before the first period or before the year
    details_str = str(details)
    
    # Split by period and take first part, then clean up
    journal_part = details_str.split('.')[0].strip()
    
    # Remove common prefixes and clean
    journal_part = re.sub(r'^\d+\s*', '', journal_part)  # Remove leading numbers
    
    return journal_part[:50] if journal_part else "Unknown"

def categorize_treatment_type(title, abstract, details):
    """Categorize the primary treatment type mentioned in the paper."""
    combined_text = f"{title} {abstract} {details}".lower()
    
    # Treatment categories in order of specificity
    if any(term in combined_text for term in ['robotic prostatectomy', 'robotic', 'robot-assisted']):
        return 'Robotic Prostatectomy'
    elif 'radical prostatectomy' in combined_text or 'prostatectomy' in combined_text:
        return 'Radical Prostatectomy'
    elif any(term in combined_text for term in ['stereotactic', 'sbrt', 'sabr']):
        return 'Stereotactic Radiation'
    elif any(term in combined_text for term in ['brachytherapy', 'seed implant', 'hdr', 'ldr']):
        return 'Brachytherapy'
    elif any(term in combined_text for term in ['proton', 'proton beam']):
        return 'Proton Therapy'
    elif any(term in combined_text for term in ['ebrt', 'external beam', 'radiation therapy']):
        return 'External Beam Radiation'
    elif any(term in combined_text for term in ['adt', 'androgen deprivation', 'hormone']):
        return 'Hormone Therapy'
    elif any(term in combined_text for term in ['cryotherapy', 'cryoablation']):
        return 'Cryotherapy'
    elif 'hifu' in combined_text or 'focused ultrasound' in combined_text:
        return 'HIFU'
    elif any(term in combined_text for term in ['surveillance', 'observation', 'watchful waiting']):
        return 'Active Surveillance'
    else:
        return 'Other/Multiple'

def categorize_study_type(title, abstract, details):
    """Categorize the study type based on content."""
    combined_text = f"{title} {abstract} {details}".lower()
    
    if 'randomized' in combined_text and ('controlled' in combined_text or 'trial' in combined_text):
        return 'Randomized Controlled Trial'
    elif 'prospective' in combined_text:
        return 'Prospective Study'
    elif 'retrospective' in combined_text:
        return 'Retrospective Study'
    elif any(term in combined_text for term in ['meta-analysis', 'systematic review']):
        return 'Meta-analysis/Review'
    elif 'cohort' in combined_text:
        return 'Cohort Study'
    elif any(term in combined_text for term in ['case series', 'case report']):
        return 'Case Series/Report'
    else:
        return 'Other'

def convert_csv_to_json(csv_file, json_file):
    """Convert CSV to JSON with enhanced metadata for visualization."""
    try:
        # Load the scored CSV
        df = pd.read_csv(csv_file)
        print(f"Loaded {len(df)} papers from {csv_file}")
        
        # Create enhanced dataset
        papers = []
        
        for idx, row in df.iterrows():
            # Extract basic information with type conversion
            paper = {
                'id': int(idx),
                'entry_number': convert_numpy_types(row.get('entry_number', 0)),
                'pmid': str(row.get('pmid', '')),
                'title': str(row.get('title', '')),
                'abstract': str(row.get('abstract', ''))[:500] + "..." if len(str(row.get('abstract', ''))) > 500 else str(row.get('abstract', '')),
                'details': str(row.get('details', '')),
                'author': str(row.get('author', '')),
                'url': str(row.get('url', '')),
                'score': convert_numpy_types(row.get('score', 0))
            }
            
            # Add derived metadata
            paper['year'] = extract_year_from_details(paper['details'])
            paper['journal'] = extract_journal_from_details(paper['details'])
            paper['treatment_type'] = categorize_treatment_type(
                paper['title'], paper['abstract'], paper['details']
            )
            paper['study_type'] = categorize_study_type(
                paper['title'], paper['abstract'], paper['details']
            )
            
            # Add quality categories
            if paper['score'] >= 70:
                paper['quality_category'] = 'High Quality'
            elif paper['score'] >= 50:
                paper['quality_category'] = 'Good Quality'
            elif paper['score'] >= 30:
                paper['quality_category'] = 'Moderate Quality'
            else:
                paper['quality_category'] = 'Low Quality'
            
            papers.append(paper)
        
        # Create summary statistics with type conversion
        summary = {
            'total_papers': int(len(papers)),
            'score_stats': {
                'mean': float(df['score'].mean()),
                'median': float(df['score'].median()),
                'std': float(df['score'].std()),
                'min': float(df['score'].min()),
                'max': float(df['score'].max())
            },
            'quality_distribution': {
                'High Quality (≥70)': int(len([p for p in papers if p['score'] >= 70])),
                'Good Quality (50-69)': int(len([p for p in papers if 50 <= p['score'] < 70])),
                'Moderate Quality (30-49)': int(len([p for p in papers if 30 <= p['score'] < 50])),
                'Low Quality (<30)': int(len([p for p in papers if p['score'] < 30]))
            },
            'treatment_types': {},
            'study_types': {},
            'year_range': {
                'earliest': min([p['year'] for p in papers if p['year']], default=None),
                'latest': max([p['year'] for p in papers if p['year']], default=None)
            }
        }
        
        # Count treatment and study types
        for paper in papers:
            treatment = paper['treatment_type']
            study = paper['study_type']
            
            summary['treatment_types'][treatment] = summary['treatment_types'].get(treatment, 0) + 1
            summary['study_types'][study] = summary['study_types'].get(study, 0) + 1
        
        # Create final JSON structure
        output_data = {
            'metadata': {
                'generated_at': datetime.now().isoformat(),
                'source_file': csv_file,
                'description': 'Prostate Cancer Treatment Comparative Effectiveness Study Scores'
            },
            'summary': summary,
            'papers': papers
        }
        
        # Save to JSON
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        print(f"✓ Successfully converted to {json_file}")
        print(f"✓ {len(papers)} papers with enhanced metadata")
        
        # Display summary
        print(f"\nDataset Summary:")
        print(f"  Score range: {summary['score_stats']['min']:.0f} - {summary['score_stats']['max']:.0f}")
        print(f"  Mean score: {summary['score_stats']['mean']:.1f}")
        print(f"  Year range: {summary['year_range']['earliest']} - {summary['year_range']['latest']}")
        
        print(f"\nTop Treatment Types:")
        sorted_treatments = sorted(summary['treatment_types'].items(), key=lambda x: x[1], reverse=True)
        for treatment, count in sorted_treatments[:5]:
            print(f"  {treatment}: {count} papers")
        
        return output_data
        
    except Exception as e:
        print(f"Error converting CSV to JSON: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    """Main function to convert CSV to JSON."""
    input_file = '../../data/output-full-scored.csv'  # Updated to match your file name
    output_file = 'prostate-papers-data.json'  # Updated to match your naming convention
    
    print("Converting prostate cancer papers to JSON for visualization...")
    convert_csv_to_json(input_file, output_file)
    print(f"\nReady for visualization! Use {output_file} with the interactive chart.")

if __name__ == "__main__":
    main()