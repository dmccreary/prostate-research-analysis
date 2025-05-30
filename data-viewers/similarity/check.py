#!/usr/bin/env python3
"""
Debug Helper for Embedding Visualization

This script helps diagnose issues with the visualization by:
1. Checking if required files exist
2. Validating JSON structure
3. Creating a minimal test dataset if needed
"""

import os
import json
import pandas as pd
import numpy as np

def check_files():
    """Check if required files exist."""
    print("File Check:")
    print("=" * 30)
    
    files_to_check = [
        '../../output-full-scored.csv',
        'papers-embeddings.json',
        'main.html'
    ]
    
    for filename in files_to_check:
        exists = os.path.exists(filename)
        print(f"  {filename}: {'✓ EXISTS' if exists else '✗ MISSING'}")
        
        if exists and filename.endswith('.json'):
            try:
                with open(filename, 'r') as f:
                    data = json.load(f)
                print(f"    - Valid JSON with {len(data.get('papers', []))} papers")
            except Exception as e:
                print(f"    - JSON ERROR: {e}")
    
    return all(os.path.exists(f) for f in files_to_check[:2])

def validate_json_structure(filename='papers-embeddings.json'):
    """Validate the JSON structure."""
    print(f"\nValidating {filename}:")
    print("=" * 30)
    
    try:
        with open(filename, 'r') as f:
            data = json.load(f)
        
        required_keys = ['papers', 'bounds', 'clusters']
        for key in required_keys:
            if key in data:
                print(f"  ✓ {key} exists")
            else:
                print(f"  ✗ {key} MISSING")
                return False
        
        papers = data['papers']
        if papers:
            sample_paper = papers[0]
            required_paper_keys = ['id', 'x', 'y', 'title', 'score']
            print(f"  Sample paper keys: {list(sample_paper.keys())}")
            
            for key in required_paper_keys:
                if key in sample_paper:
                    print(f"    ✓ {key}: {type(sample_paper[key])}")
                else:
                    print(f"    ✗ {key} MISSING")
        
        bounds = data['bounds']
        print(f"  Bounds: {bounds}")
        
        return True
        
    except Exception as e:
        print(f"  Error: {e}")
        return False

def create_test_dataset():
    """Create a minimal test dataset for debugging."""
    print("\nCreating test dataset...")
    print("=" * 30)
    
    # Create test papers with random embeddings
    np.random.seed(42)
    n_papers = 50
    
    papers = []
    for i in range(n_papers):
        papers.append({
            'id': i,
            'title': f'Test Paper {i+1}: Prostate Cancer Research',
            'abstract': f'This is a test abstract for paper {i+1} about prostate cancer treatment.',
            'score': np.random.randint(20, 90),
            'x': np.random.normal(0, 2),
            'y': np.random.normal(0, 2),
            'cluster': np.random.randint(0, 4),
            'treatment_type': np.random.choice(['Prostatectomy', 'Radiation', 'Brachytherapy', 'Hormone Therapy']),
            'year': np.random.randint(2015, 2024),
            'pmid': f'1234567{i}',
            'url': f'https://pubmed.ncbi.nlm.nih.gov/1234567{i}/'
        })
    
    # Calculate bounds
    x_coords = [p['x'] for p in papers]
    y_coords = [p['y'] for p in papers]
    
    bounds = {
        'x_min': float(np.min(x_coords)),
        'x_max': float(np.max(x_coords)),
        'y_min': float(np.min(y_coords)),
        'y_max': float(np.max(y_coords))
    }
    
    # Create clusters info
    clusters = {}
    for cluster_id in range(4):
        cluster_papers = [p for p in papers if p['cluster'] == cluster_id]
        clusters[str(cluster_id)] = {
            'size': len(cluster_papers),
            'avg_score': np.mean([p['score'] for p in cluster_papers]),
            'top_terms': [f'term{cluster_id}_1', f'term{cluster_id}_2', f'term{cluster_id}_3'],
            'top_treatment': f'Treatment{cluster_id}'
        }
    
    # Create test JSON
    test_data = {
        'metadata': {
            'total_papers': len(papers),
            'embedding_method': 'TEST_DATA'
        },
        'papers': papers,
        'bounds': bounds,
        'clusters': clusters
    }
    
    # Save test data
    with open('test-papers-embeddings.json', 'w') as f:
        json.dump(test_data, f, indent=2)
    
    print(f"  ✓ Created test-papers-embeddings.json with {len(papers)} papers")
    print(f"  ✓ Bounds: {bounds}")
    
    return test_data

def main():
    """Main debug function."""
    print("Embedding Visualization Debug Helper")
    print("=" * 50)
    
    # Check files
    files_ok = check_files()
    
    # Validate JSON if it exists
    if os.path.exists('papers-embeddings.json'):
        json_ok = validate_json_structure()
    else:
        json_ok = False
        print("\npapers-embeddings.json not found!")
    
    # Create test dataset if needed
    if not files_ok or not json_ok:
        print("\nCreating test dataset for debugging...")
        create_test_dataset()
        
        print("\n" + "=" * 50)
        print("NEXT STEPS:")
        print("1. Rename test-papers-embeddings.json to papers-embeddings.json")
        print("2. Open papers_similarity_map.html in your browser")
        print("3. If test data works, re-run the embedding analysis")
    else:
        print("\n✓ All files look good!")
        print("The visualization should work. Check browser console for errors.")

if __name__ == "__main__":
    main()