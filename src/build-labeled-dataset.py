#!/usr/bin/env python3
"""
Merge positive-data-set.json and negative-data-set.json into a single
labeled dataset (data/labeled-dataset.csv) ready for training a classifier.

Usage:
    python build-labeled-dataset.py
"""

import argparse
import json
import pandas as pd


def load_papers(filepath, label):
    with open(filepath) as f:
        data = json.load(f)
    df = pd.DataFrame(data['papers'])
    df['label'] = label
    return df


def main():
    parser = argparse.ArgumentParser(description='Merge positive/negative datasets into one labeled table')
    parser.add_argument('--positive', default='../data/positive-data-set.json')
    parser.add_argument('--negative', default='../data/negative-data-set.json')
    parser.add_argument('--output', default='../data/labeled-dataset.csv')
    args = parser.parse_args()

    pos_df = load_papers(args.positive, 1)
    neg_df = load_papers(args.negative, 0)

    combined = pd.concat([pos_df, neg_df], ignore_index=True, sort=False)

    dup_pmids = combined[combined.duplicated('pmid', keep=False)]['pmid'].unique().tolist()
    if dup_pmids:
        print(f"WARNING: {len(dup_pmids)} PMIDs appear in both sets (conflicting labels): {dup_pmids}")

    combined = combined.sort_values(['label', 'pmid'], ascending=[False, True]).reset_index(drop=True)
    combined.to_csv(args.output, index=False)

    print(f"Positive: {len(pos_df)}  Negative: {len(neg_df)}  Total: {len(combined)}")
    print(f"Saved merged labeled dataset to {args.output}")


if __name__ == '__main__':
    main()
