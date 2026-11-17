import os
import sys

if sys.platform.startswith('win'):
    sys.stdout.reconfigure(encoding='utf-8')

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression

def main():
    # 1. Load data
    raw_df = pd.read_csv('data/labeled-dataset-v2-aux25.csv')
    oof_df = pd.read_csv('results/stage9_1/arm_3_oof_features.csv')
    test_df = pd.read_csv('results/stage9_1/arm_3_test_features.csv')

    features = [c for c in oof_df.columns if c not in ['pmid', 'label', 'study_id', 'fold']]

    # 2. Fit Stage 2 Logistic Regression on OOF features
    lr = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
    lr.fit(oof_df[features].values, oof_df['label'].values)

    intercept = lr.intercept_[0]
    coefs = dict(zip(features, lr.coef_[0]))

    # Predict on test
    probs = lr.predict_proba(test_df[features].values)[:, 1]
    preds = (probs >= 0.50).astype(int)

    test_df['predicted_prob'] = probs
    test_df['predicted_label'] = preds

    # Merge with raw text & ground truth auxiliary labels
    gt_cols = ['pmid', 'title', 'abstract', 'label'] + features
    raw_sub = raw_df[gt_cols].copy()
    raw_sub = raw_sub.rename(columns={f: f'{f}_gt' for f in features})
    raw_sub = raw_sub.rename(columns={'label': 'ground_truth_label'})

    merged = test_df.merge(raw_sub, on='pmid', how='left')

    # Identify FP and FN
    fp_df = merged[(merged['ground_truth_label'] == 0) & (merged['predicted_label'] == 1)].copy()
    fn_df = merged[(merged['ground_truth_label'] == 1) & (merged['predicted_label'] == 0)].copy()

    print(f'Total Test Samples: {len(merged)}')
    print(f'False Positives (FP): {len(fp_df)}')
    print(f'False Negatives (FN): {len(fn_df)}')

    def analyze_contributions(row):
        contribs = []
        for f in features:
            pred_p = row[f]
            w = coefs[f]
            contrib = w * pred_p
            contribs.append((f, contrib, pred_p, row[f'{f}_gt'], w))
        # Sort by contribution
        contribs_sorted = sorted(contribs, key=lambda x: x[1], reverse=True)
        return contribs_sorted

    # Save detailed error reports
    rows_out = []
    
    print('\n' + '='*80)
    print('FALSE POSITIVES ANALYSIS (Ground Truth = 0, Predicted = 1)')
    print('='*80)
    for idx, row in fp_df.iterrows():
        contribs = analyze_contributions(row)
        top_pos = [f'{c[0]} (contrib: +{c[1]:.2f}, prob: {c[2]:.2f}, gt: {c[3]})' for c in contribs if c[1] > 0.2][:3]
        top_neg = [f'{c[0]} (contrib: {c[1]:.2f}, prob: {c[2]:.2f}, gt: {c[3]})' for c in contribs if c[1] < -0.1][:2]
        
        print(f"\nPMID: {row['pmid']} | Prob: {row['predicted_prob']:.4f}")
        print(f"Title: {row['title']}")
        print(f"Top Positive Drivers: {'; '.join(top_pos)}")
        print(f"Top Negative Mitigators: {'; '.join(top_neg) if top_neg else 'None'}")
        
        rows_out.append({
            'error_type': 'False Positive',
            'pmid': row['pmid'],
            'title': row['title'],
            'predicted_prob': round(row['predicted_prob'], 4),
            'ground_truth_label': 0,
            'top_drivers': '; '.join(top_pos),
            'abstract_snippet': str(row['abstract'])[:200] + '...'
        })

    print('\n' + '='*80)
    print('FALSE NEGATIVES ANALYSIS (Ground Truth = 1, Predicted = 0)')
    print('='*80)
    for idx, row in fn_df.iterrows():
        contribs = analyze_contributions(row)
        # For FN, why was it low? Either negative contributions or lack of positive contributions
        top_pos = [f'{c[0]} (contrib: +{c[1]:.2f}, prob: {c[2]:.2f}, gt: {c[3]})' for c in contribs if c[1] > 0.2][:3]
        top_neg = [f'{c[0]} (contrib: {c[1]:.2f}, prob: {c[2]:.2f}, gt: {c[3]})' for c in contribs if c[1] < -0.1][:2]
        missed_gt = [f'{c[0]} (gt=1 but pred_p={c[2]:.2f}, lost contrib: +{c[4]:.2f})' for c in contribs if c[3] == 1 and c[2] < 0.35][:3]
        
        print(f"\nPMID: {row['pmid']} | Prob: {row['predicted_prob']:.4f}")
        print(f"Title: {row['title']}")
        print(f"Present Positive Drivers: {'; '.join(top_pos) if top_pos else 'None'}")
        print(f"Missed Features (gt=1, pred low): {'; '.join(missed_gt) if missed_gt else 'None'}")
        print(f"Negative Penalty Drivers: {'; '.join(top_neg) if top_neg else 'None'}")

        rows_out.append({
            'error_type': 'False Negative',
            'pmid': row['pmid'],
            'title': row['title'],
            'predicted_prob': round(row['predicted_prob'], 4),
            'ground_truth_label': 1,
            'top_drivers': '; '.join(top_pos),
            'missed_features': '; '.join(missed_gt),
            'abstract_snippet': str(row['abstract'])[:200] + '...'
        })

    out_df = pd.DataFrame(rows_out)
    out_df.to_csv('results/stage9_1/arm3_error_analysis.csv', index=False)
    print(f"\nSaved full error breakdown to results/stage9_1/arm3_error_analysis.csv")

if __name__ == '__main__':
    main()
