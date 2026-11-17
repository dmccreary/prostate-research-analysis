import os
import sys

# Ensure UTF-8 output on Windows
if sys.platform.startswith('win'):
    sys.stdout.reconfigure(encoding='utf-8')

import pandas as pd
import numpy as np
from scipy import stats
from sklearn.linear_model import LogisticRegression

def main():
    # 1. Load data
    aux_df = pd.read_csv('data/labeled-dataset-v2-aux25.csv')
    train_df = pd.read_csv('data/splits/stage5/train.csv')
    test_df = pd.read_csv('data/splits/stage5/test.csv')

    id_col = 'pmid' if 'pmid' in train_df.columns else 'study_id'
    train_aux = aux_df[aux_df[id_col].isin(train_df[id_col])].copy()
    test_aux = aux_df[aux_df[id_col].isin(test_df[id_col])].copy()

    # Load Stage 1 test feature metrics for each arm
    arm1_metrics = pd.read_csv('results/stage9_1/arm_1_stage1_test_feature_metrics.csv')
    arm2_metrics = pd.read_csv('results/stage9_1/arm_2_stage1_test_feature_metrics.csv')
    arm3_metrics = pd.read_csv('results/stage9_1/arm_3_stage1_test_feature_metrics.csv')

    features = arm3_metrics['feature'].tolist()

    # Load OOF features to get Stage 2 Logistic Regression coefficients
    oof_arm3 = pd.read_csv('results/stage9_1/arm_3_oof_features.csv')
    oof_arm2 = pd.read_csv('results/stage9_1/arm_2_oof_features.csv')

    feats_oof = [c for c in oof_arm3.columns if c not in ['pmid', 'label', 'study_id', 'fold']]

    lr3 = LogisticRegression(max_iter=1000, random_state=42)
    lr3.fit(oof_arm3[feats_oof], oof_arm3['label'])
    coef3_map = dict(zip(feats_oof, lr3.coef_[0]))

    lr2 = LogisticRegression(max_iter=1000, random_state=42)
    lr2.fit(oof_arm2[feats_oof], oof_arm2['label'])
    coef2_map = dict(zip(feats_oof, lr2.coef_[0]))

    # Compile records
    records = []
    for feat in features:
        tr_cnt = int(train_aux[feat].sum())
        tr_prev = tr_cnt / len(train_aux)
        te_cnt = int(test_aux[feat].sum())
        te_prev = te_cnt / len(test_aux)

        m1 = arm1_metrics[arm1_metrics['feature'] == feat].iloc[0]
        m2 = arm2_metrics[arm2_metrics['feature'] == feat].iloc[0]
        m3 = arm3_metrics[arm3_metrics['feature'] == feat].iloc[0]

        c3 = coef3_map.get(feat, 0.0)
        c2 = coef2_map.get(feat, 0.0)

        records.append({
            'feature': feat,
            'train_positive_count': tr_cnt,
            'train_prevalence_pct': round(tr_prev * 100, 2),
            'test_positive_count': te_cnt,
            'test_prevalence_pct': round(te_prev * 100, 2),
            'arm3_test_auroc': round(m3['auroc'], 4),
            'arm3_test_prauc': round(m3['pr_auc'], 4),
            'arm3_test_f1': round(m3['f1'], 4),
            'arm2_test_auroc': round(m2['auroc'], 4),
            'arm2_test_prauc': round(m2['pr_auc'], 4),
            'arm1_test_auroc': round(m1['auroc'], 4),
            'arm1_test_prauc': round(m1['pr_auc'], 4),
            'arm3_stage2_coef': round(c3, 4),
            'arm3_abs_importance': round(abs(c3), 4),
            'arm2_stage2_coef': round(c2, 4),
            'arm2_abs_importance': round(abs(c2), 4),
        })

    df = pd.DataFrame(records)
    out_csv = 'results/stage9_1/feature_fidelity_correlation_analysis.csv'
    df.to_csv(out_csv, index=False)
    print(f'Saved analysis table to {out_csv}')

    # Compute Correlations
    print('\n======================================================')
    print('CORRELATION ANALYSIS: PREVALENCE vs FIDELITY vs IMPORTANCE')
    print('======================================================')

    # 1. Train prevalence vs Arm 3 AUROC
    r_p_auroc, p_val_p_auroc = stats.pearsonr(df['train_prevalence_pct'], df['arm3_test_auroc'])
    rho_s_auroc, p_val_s_auroc = stats.spearmanr(df['train_prevalence_pct'], df['arm3_test_auroc'])
    print(f'1. Train Prevalence vs Arm 3 Test AUROC:')
    print(f'   Pearson r  = {r_p_auroc:.4f} (p-value = {p_val_p_auroc:.4f})')
    print(f'   Spearman ρ = {rho_s_auroc:.4f} (p-value = {p_val_s_auroc:.4f})')

    # 2. Train prevalence vs Arm 3 PR-AUC
    r_p_prauc, p_val_p_prauc = stats.pearsonr(df['train_prevalence_pct'], df['arm3_test_prauc'])
    rho_s_prauc, p_val_s_prauc = stats.spearmanr(df['train_prevalence_pct'], df['arm3_test_prauc'])
    print(f'\n2. Train Prevalence vs Arm 3 Test PR-AUC:')
    print(f'   Pearson r  = {r_p_prauc:.4f} (p-value = {p_val_p_prauc:.4f})')
    print(f'   Spearman ρ = {rho_s_prauc:.4f} (p-value = {p_val_s_prauc:.4f})')

    # 3. Train prevalence vs Arm 3 Absolute Importance |beta|
    r_p_imp, p_val_p_imp = stats.pearsonr(df['train_prevalence_pct'], df['arm3_abs_importance'])
    rho_s_imp, p_val_s_imp = stats.spearmanr(df['train_prevalence_pct'], df['arm3_abs_importance'])
    print(f'\n3. Train Prevalence vs Arm 3 Absolute Decision Importance (|beta|):')
    print(f'   Pearson r  = {r_p_imp:.4f} (p-value = {p_val_p_imp:.4f})')
    print(f'   Spearman ρ = {rho_s_imp:.4f} (p-value = {p_val_s_imp:.4f})')

    # 4. Arm 3 AUROC vs Arm 3 Absolute Importance |beta|
    r_p_f_imp, p_val_f_imp = stats.pearsonr(df['arm3_test_auroc'], df['arm3_abs_importance'])
    rho_s_f_imp, p_val_s_f_imp = stats.spearmanr(df['arm3_test_auroc'], df['arm3_abs_importance'])
    print(f'\n4. Arm 3 Test AUROC (Fidelity) vs Arm 3 Absolute Importance (|beta|):')
    print(f'   Pearson r  = {r_p_f_imp:.4f} (p-value = {p_val_f_imp:.4f})')
    print(f'   Spearman ρ = {rho_s_f_imp:.4f} (p-value = {p_val_s_f_imp:.4f})')

    # 5. Arm 2 Checks
    r_p_a2, p_val_a2 = stats.pearsonr(df['train_prevalence_pct'], df['arm2_test_auroc'])
    rho_s_a2, p_val_s_a2 = stats.spearmanr(df['train_prevalence_pct'], df['arm2_test_auroc'])
    print(f'\n5. Train Prevalence vs Arm 2 Test AUROC:')
    print(f'   Pearson r  = {r_p_a2:.4f} (p-value = {p_val_a2:.4f})')
    print(f'   Spearman ρ = {rho_s_a2:.4f} (p-value = {p_val_s_a2:.4f})')

    # 2x2 Quadrant Breakdown
    print('\n======================================================')
    print('2x2 DIAGNOSTIC QUADRANT BREAKDOWN (Cutoffs: AUROC >= 0.75, |beta| >= 0.70)')
    print('======================================================')

    def get_quadrant(row):
        high_fid = row['arm3_test_auroc'] >= 0.75
        high_imp = row['arm3_abs_importance'] >= 0.70
        if high_fid and high_imp:
            return 'Q1: High Fidelity & High Importance (The Clinical Anchors)'
        elif not high_fid and high_imp:
            return 'Q2: Low Fidelity & High Importance (The Critical Bottlenecks)'
        elif high_fid and not high_imp:
            return 'Q3: High Fidelity & Low Importance (Specific Markers / Tags)'
        else:
            return 'Q4: Low Fidelity & Low Importance (Sparse / Minor Noise)'

    df['quadrant'] = df.apply(get_quadrant, axis=1)
    df.to_csv(out_csv, index=False)

    for q in sorted(df['quadrant'].unique()):
        sub = df[df['quadrant'] == q]
        print(f'\n=== {q} (Count: {len(sub)}) ===')
        for _, r in sub.iterrows():
            print(f"  * {r['feature']:<34} | AUROC: {r['arm3_test_auroc']:<6} | Train Prev: {r['train_prevalence_pct']:>5}% | Stage2 Coef: {r['arm3_stage2_coef']:>7}")


if __name__ == '__main__':
    main()
