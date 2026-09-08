#!/usr/bin/env python3
"""
Train a baseline TF-IDF + Logistic Regression classifier on labeled-dataset.csv
and compare it against the existing rule-based scorer (prostate-cancer-scorer.py).

Usage:
    python train-baseline-model.py
"""

import argparse
import importlib.util
import re
import sys

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

CONFLICTING_PMIDS = {15774239}  # labeled both positive and negative; excluded from training


def load_scorer():
    """Import score_paper() from prostate-cancer-scorer.py (hyphenated filename)."""
    spec = importlib.util.spec_from_file_location("scorer", "prostate-cancer-scorer.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.score_paper


def load_dataset(path):
    df = pd.read_csv(path)
    df = df[~df['pmid'].isin(CONFLICTING_PMIDS)].reset_index(drop=True)
    df['abstract'] = df['abstract'].fillna('')
    df['title'] = df['title'].fillna('')
    df['text'] = (df['title'] + ' ' + df['abstract']).str.strip()
    df = df[df['text'].str.len() > 0].reset_index(drop=True)
    return df


def train_tfidf_baseline(train_df, test_df):
    vectorizer = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),
        stop_words='english',
        min_df=2,
    )
    X_train = vectorizer.fit_transform(train_df['text'])
    X_test = vectorizer.transform(test_df['text'])

    clf = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
    clf.fit(X_train, train_df['label'])

    probs = clf.predict_proba(X_test)[:, 1]
    preds = clf.predict(X_test)

    top_features = sorted(
        zip(vectorizer.get_feature_names_out(), clf.coef_[0]), key=lambda x: x[1]
    )
    top_negative = top_features[:15]
    top_positive = top_features[-15:][::-1]

    return preds, probs, top_positive, top_negative


def evaluate_rule_based_scorer(test_df, score_paper):
    rows = test_df.to_dict('records')
    scores = np.array([score_paper(r) for r in rows])

    # sweep thresholds to find the best F1 on the test set (informational, not tuned on held-out)
    best_thresh, best_f1 = 0, 0
    for thresh in range(0, 101, 5):
        preds = (scores >= thresh).astype(int)
        f1 = f1_score(test_df['label'], preds, zero_division=0)
        if f1 > best_f1:
            best_f1, best_thresh = f1, thresh

    preds = (scores >= best_thresh).astype(int)
    return scores, preds, best_thresh


def report_section(title):
    return f"\n{'=' * 60}\n{title}\n{'=' * 60}\n"


def main():
    parser = argparse.ArgumentParser(description='Train and evaluate baseline classifier')
    parser.add_argument('--input', default='../data/labeled-dataset.csv')
    parser.add_argument('--output', default='../docs/reports/baseline-model-report.md')
    parser.add_argument('--test_size', type=float, default=0.2)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    score_paper = load_scorer()
    df = load_dataset(args.input)

    train_df, test_df = train_test_split(
        df, test_size=args.test_size, stratify=df['label'], random_state=args.seed
    )

    print(f"Total: {len(df)}  Train: {len(train_df)}  Test: {len(test_df)}")
    print(f"Train label balance: {train_df['label'].value_counts().to_dict()}")
    print(f"Test label balance: {test_df['label'].value_counts().to_dict()}")

    # --- TF-IDF + Logistic Regression baseline ---
    preds, probs, top_pos, top_neg = train_tfidf_baseline(train_df, test_df)
    tfidf_report = classification_report(test_df['label'], preds, target_names=['negative', 'positive'])
    tfidf_cm = confusion_matrix(test_df['label'], preds)
    tfidf_auc = roc_auc_score(test_df['label'], probs)
    tfidf_f1 = f1_score(test_df['label'], preds)

    # --- Rule-based scorer baseline ---
    rb_scores, rb_preds, rb_thresh = evaluate_rule_based_scorer(test_df, score_paper)
    rb_report = classification_report(test_df['label'], rb_preds, target_names=['negative', 'positive'])
    rb_cm = confusion_matrix(test_df['label'], rb_preds)
    rb_f1 = f1_score(test_df['label'], rb_preds)

    # --- Build markdown report ---
    lines = []
    lines.append("# Baseline Model Report\n")
    lines.append(f"Dataset: `{args.input}` ({len(df)} papers after excluding {len(CONFLICTING_PMIDS)} conflicting-label PMID(s))\n")
    lines.append(f"Split: {len(train_df)} train / {len(test_df)} test (stratified, seed={args.seed})\n")

    lines.append(report_section("1. TF-IDF + Logistic Regression"))
    lines.append("```\n" + tfidf_report + "```\n")
    lines.append(f"Confusion matrix (rows=actual, cols=predicted [neg, pos]):\n```\n{tfidf_cm}\n```\n")
    lines.append(f"ROC AUC: {tfidf_auc:.3f}\n")
    lines.append(f"F1 (positive class): {tfidf_f1:.3f}\n")
    lines.append("\nTop positive-indicating terms:\n")
    lines.append("\n".join(f"- {term} ({coef:.3f})" for term, coef in top_pos) + "\n")
    lines.append("\nTop negative-indicating terms:\n")
    lines.append("\n".join(f"- {term} ({coef:.3f})" for term, coef in top_neg) + "\n")

    lines.append(report_section("2. Existing rule-based scorer (prostate-cancer-scorer.py)"))
    lines.append(f"Best threshold found on test set (score >= threshold -> positive): {rb_thresh}\n")
    lines.append("```\n" + rb_report + "```\n")
    lines.append(f"Confusion matrix (rows=actual, cols=predicted [neg, pos]):\n```\n{rb_cm}\n```\n")
    lines.append(f"F1 (positive class): {rb_f1:.3f}\n")

    lines.append(report_section("3. Comparison"))
    lines.append(f"| Model | F1 (positive) |\n|---|---|\n| TF-IDF + Logistic Regression | {tfidf_f1:.3f} |\n| Rule-based scorer (best threshold) | {rb_f1:.3f} |\n")
    if tfidf_f1 > rb_f1:
        lines.append("\nThe learned TF-IDF baseline outperforms the rule-based scorer on this test set.\n")
    else:
        lines.append("\nThe rule-based scorer is competitive with or outperforms the TF-IDF baseline on this test set.\n")

    lines.append(report_section("Notes / caveats"))
    lines.append(
        "- Test set is only ~73 papers (20% of 364); F1 estimates have wide variance. "
        "Treat this as a first signal, not a final benchmark.\n"
        "- The rule-based threshold was chosen by sweeping on the test set itself, "
        "so its reported F1 is optimistic — a proper comparison needs a separate validation set.\n"
        "- Class balance in train/test is imbalanced (negative > positive); "
        "`class_weight='balanced'` was used for the TF-IDF model.\n"
        "- **Possible dataset artifact**: top positive-indicating terms are dominated by "
        "'risk'/'intermediate risk'/'high risk'/'low risk'. This may be genuine signal "
        "(risk stratification is part of the acceptance criteria), but it may also reflect "
        "how the two sets were curated differently — positive papers were sourced by risk "
        "category, negative papers by publication year. Worth checking whether the model is "
        "learning acceptance criteria or just learning 'how this dataset was assembled'.\n"
    )

    report_text = "\n".join(lines)
    with open(args.output, 'w') as f:
        f.write(report_text)

    print(report_text)
    print(f"\nSaved report to {args.output}")


if __name__ == '__main__':
    main()
