#!/usr/bin/env python3
"""
Prostate Cancer Treatment Comparative Effectiveness Study Scoring Program

This program analyzes prostate cancer research papers from output100.csv and scores them 
based on treatment modality identification and study quality criteria.

Scoring Components:
1. Treatment modality presence (40 points max)
2. Study quality indicators (40 points max) 
3. Numerical criteria compliance (20 points max)

Score Range: 0-100 where:
- 0 = None of the criteria match
- 100 = All criteria match perfectly

Usage: python prostate_cancer_scorer.py

Output: Creates output100_scored.csv with added 'score' column
"""

import pandas as pd
import re

def calculate_treatment_score(text):
    """
    Score based on treatment modality mentions (0-40 points).
    
    Based on the 17 treatment modalities from summary-criteria.md:
    - Prostatectomy variants: 8 points each
    - Radiation therapy variants: 6 points each  
    - Specialized treatments: 6-8 points each
    - Combination therapies: Additive scoring
    """
    text = text.lower()
    
    # Treatment modality scoring based on summary-criteria.md
    treatments = {
        # Prostatectomy (modalities 1-2)
        'radical prostatectomy': 8,
        'robotic prostatectomy': 8,
        'robotic': 6,  # Catch robotic mentions
        'prostatectomy': 6,  # General prostatectomy
        
        # Radiation therapy (modalities 3-7)
        'ebrt': 6,
        'external beam': 6,
        'radiation therapy': 6,
        'hypofractionated': 6,
        'stereotactic': 8,  # SBRT/ultrahypofx
        'proton': 6,
        
        # Brachytherapy (modalities 8-15)
        'brachytherapy': 8,
        'seed implant': 6,
        'ldr': 6,
        'hdr': 6,
        
        # Hormonal therapy (present in many combinations)
        'adt': 4,
        'androgen deprivation': 4,
        'hormone therapy': 4,
        'hormonal': 4,
        
        # Other modalities (16-17)
        'cryotherapy': 6,
        'cryoablation': 6,
        'hifu': 6,
        'focused ultrasound': 6
    }
    
    score = 0
    found_treatments = []
    
    for treatment, points in treatments.items():
        if treatment in text:
            score += points
            found_treatments.append(treatment)
    
    # Bonus for combination therapies (common in prostate cancer)
    if len(found_treatments) >= 2:
        score += 4  # Bonus for comparative or combination studies
    
    return min(score, 40)  # Cap at 40 points

def calculate_quality_score(text):
    """
    Score based on study quality indicators (0-40 points).
    
    Based on acceptance criteria from summary-criteria.md:
    1. Accepted (peer-reviewed with proper endpoints)
    2. Proper stratification (D'Amico, NCCN)
    3. Adequate sample size and follow-up
    4. Clinical staging included
    """
    text = text.lower()
    
    # Positive quality indicators
    positive_terms = {
        # Study design quality
        'randomized': 10,
        'controlled trial': 8,
        'rct': 8,
        'prospective': 6,
        'multicenter': 4,
        'phase ii': 6,
        'phase iii': 8,
        
        # Proper endpoints (from criteria #2)
        'biochemical recurrence': 8,
        'brfs': 8,
        'overall survival': 8,
        'os': 6,
        'metastasis-free survival': 8,
        'mfs': 6,
        'cancer-specific survival': 6,
        'css': 6,
        'disease-free survival': 6,
        'progression-free': 6,
        
        # Stratification indicators (criteria #3)
        "d'amico": 6,
        'nccn': 6,
        'risk group': 4,
        'low risk': 3,
        'intermediate risk': 3,
        'high risk': 3,
        'stratified': 4,
        
        # General quality indicators
        'outcomes': 4,
        'efficacy': 4,
        'comparative': 6,
        'long-term': 4,
        'follow-up': 3
    }
    
    # Negative quality indicators (rejection criteria)
    negative_terms = {
        'abstract only': -15,
        'conference abstract': -15,
        'meeting abstract': -12,
        'case report': -10,
        'case series': -8,
        'review': -6,
        'editorial': -8,
        'commentary': -6,
        'letter': -8,
        'pathologic staging only': -8,
        'no stratification': -6,
        'retrospective': -4
    }
    
    score = 0
    
    # Add positive points
    for term, points in positive_terms.items():
        if term in text:
            score += points
    
    # Subtract negative points
    for term, penalty in negative_terms.items():
        if term in text:
            score += penalty  # penalty is already negative
    
    return max(0, min(score, 40))  # Keep between 0-40

def calculate_numerical_score(text):
    """
    Score based on numerical criteria (0-20 points).
    
    Based on rejection criteria from summary-criteria.md:
    4. EBRT dose <72Gy (penalty)
    5. Patient number <100 (low, int); <50 (high) (penalty) 
    6. Median follow-up < 5 years (penalty)
    """
    score = 0
    
    # Sample size scoring (criteria #5)
    n_patterns = [r'n\s*=\s*(\d+)', r'(\d+)\s*patients?', r'cohort.*?(\d+)', r'(\d+)\s*men']
    sample_size = None
    
    for pattern in n_patterns:
        n_match = re.search(pattern, text, re.IGNORECASE)
        if n_match:
            sample_size = int(n_match.group(1))
            break
    
    if sample_size:
        if sample_size >= 500:
            score += 8  # Large study bonus
        elif sample_size >= 200:
            score += 6  # Medium-large study
        elif sample_size >= 100:
            score += 4  # Meets minimum for low/intermediate risk
        elif sample_size >= 50:
            score += 2  # Meets minimum for high risk only
        else:
            score -= 4  # Below minimum thresholds
    
    # Follow-up duration scoring (criteria #6)
    followup_patterns = [
        r'median.*?follow-?up.*?(\d+\.?\d*)\s*years?',
        r'follow-?up.*?(\d+\.?\d*)\s*years?',
        r'(\d+\.?\d*)\s*years?.*?follow',
        r'(\d+)\s*months.*?follow'
    ]
    
    followup_years = None
    for pattern in followup_patterns:
        followup_match = re.search(pattern, text, re.IGNORECASE)
        if followup_match:
            years = float(followup_match.group(1))
            # Convert months to years if pattern indicates months
            if 'month' in followup_match.group(0).lower():
                years = years / 12
            followup_years = years
            break
    
    if followup_years:
        if followup_years >= 10:
            score += 6  # Excellent long-term follow-up
        elif followup_years >= 5:
            score += 4  # Meets minimum requirement
        elif followup_years >= 2:
            score += 2  # Reasonable follow-up
        else:
            score -= 4  # Below minimum requirement
    
    # Radiation dose scoring (criteria #4)
    dose_patterns = [r'(\d+\.?\d*)\s*Gy', r'dose.*?(\d+\.?\d*)\s*Gy']
    
    for pattern in dose_patterns:
        dose_match = re.search(pattern, text, re.IGNORECASE)
        if dose_match:
            dose = float(dose_match.group(1))
            if dose >= 78:
                score += 4  # High dose, good outcomes expected
            elif dose >= 72:
                score += 2  # Meets minimum requirement
            elif dose >= 60:
                score += 0  # Reasonable but suboptimal
            else:
                score -= 4  # Below recommended dose
            break
    
    return max(0, min(score, 20))  # Keep between 0-20

def score_paper(row):
    """
    Calculate total score for a paper (0-100 scale).
    
    Scoring breakdown:
    - Treatment modality identification: 40 points
    - Study quality indicators: 40 points  
    - Numerical criteria compliance: 20 points
    """
    # Combine all text fields for comprehensive analysis
    text_fields = [
        row.get('title', ''),
        row.get('abstract', ''),
        row.get('details', ''),
        row.get('author', '')  # Sometimes contains relevant info
    ]
    
    combined_text = ' '.join(str(field) for field in text_fields if field)
    
    # Calculate component scores
    treatment_score = calculate_treatment_score(combined_text)
    quality_score = calculate_quality_score(combined_text)
    numerical_score = calculate_numerical_score(combined_text)
    
    # Total score with ceiling at 100
    total_score = treatment_score + quality_score + numerical_score
    final_score = min(100, max(0, total_score))
    
    return final_score

def main():
    """Main scoring function that processes output100.csv and adds score column."""
    try:
        # Load the data
        input_file = '../data/output.csv'
        output_file = 'output-full-scored.csv'
        
        print(f"Loading data from {input_file}...")
        df = pd.read_csv(input_file)
        print(f"Successfully loaded {len(df)} papers")
        
        # Display first few papers for verification
        print(f"\nProcessing papers...")
        print("=" * 80)
        
        # Calculate scores for all papers
        scores = []
        high_score_papers = []
        
        for idx, row in df.iterrows():
            score = score_paper(row)
            scores.append(score)
            
            # Track high-scoring papers
            if score >= 60:
                high_score_papers.append((idx, score, row.get('title', 'N/A')))
            
            # Show progress for first few papers
            if idx < 5:
                print(f"Paper {idx+1}: Score {score:2.0f} - {row.get('title', 'N/A')[:60]}...")
        
        # Add score column to dataframe
        df['score'] = scores
        
        # Save results
        df.to_csv(output_file, index=False)
        print(f"\n{'='*80}")
        print(f"✓ Scoring completed successfully!")
        print(f"✓ Results saved to: {output_file}")
        
        # Display comprehensive statistics
        print(f"\nScore Statistics:")
        print(f"  Mean Score: {df['score'].mean():.1f}")
        print(f"  Median Score: {df['score'].median():.1f}")
        print(f"  Standard Deviation: {df['score'].std():.1f}")
        print(f"  Range: {df['score'].min():.0f} - {df['score'].max():.0f}")
        
        # Score distribution
        high_quality = len(df[df['score'] >= 70])
        good_quality = len(df[(df['score'] >= 50) & (df['score'] < 70)])
        moderate_quality = len(df[(df['score'] >= 30) & (df['score'] < 50)])
        low_quality = len(df[df['score'] < 30])
        
        print(f"\nScore Distribution:")
        print(f"  High Quality (≥70):     {high_quality:2d} papers ({high_quality/len(df)*100:.1f}%)")
        print(f"  Good Quality (50-69):   {good_quality:2d} papers ({good_quality/len(df)*100:.1f}%)")
        print(f"  Moderate Quality (30-49): {moderate_quality:2d} papers ({moderate_quality/len(df)*100:.1f}%)")
        print(f"  Low Quality (<30):      {low_quality:2d} papers ({low_quality/len(df)*100:.1f}%)")
        
        # Display top-scoring papers
        print(f"\nTop 10 Highest Scoring Papers:")
        print(f"{'Rank':<4} {'Score':<6} {'Title'}")
        print(f"{'-'*4} {'-'*6} {'-'*60}")
        
        top_papers = df.nlargest(10, 'score')
        for i, (_, row) in enumerate(top_papers.iterrows(), 1):
            title = row['title'][:55] + "..." if len(row['title']) > 55 else row['title']
            print(f"{i:<4} {row['score']:<6.0f} {title}")
        
        print(f"\n{'='*80}")
        print(f"Analysis complete! Check {output_file} for full results.")
        
        return df
        
    except FileNotFoundError:
        print(f"Error: Could not find {input_file}")
        print("Please ensure the file exists in the current directory.")
        return None
    except Exception as e:
        print(f"Error during processing: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    main()