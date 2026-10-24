"""
analyze_errors_512.py - Comprehensive Token Length & Truncation Error Analysis
for PubMedBERT 512 Weighted Model.

Investigates:
1. Exact token lengths (Title, Abstract, Total) of all Test papers (N=105).
2. For all False Negatives (FNs) and False Positives (FPs):
   - Was the paper truncated (>512 tokens)?
   - Did the truncated tail contain critical clinical information that caused the error?
   - Or did the model fail due to semantic/clinical eligibility nuances despite having all text?
"""

import os
from pathlib import Path
import sys
import pandas as pd
import numpy as np
from transformers import AutoTokenizer

BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(BASE_DIR / "src"))
from preprocessing import clean_clinical_text

PRETRAINED_MODEL_NAME = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract"
DATA_DIR = BASE_DIR / "data"
SPLITS_DIR = DATA_DIR / "splits" / "stage5"
RESULTS_DIR = BASE_DIR / "results" / "stage5_512"

def main():
    tokenizer = AutoTokenizer.from_pretrained(PRETRAINED_MODEL_NAME, local_files_only=True)
    
    test_df = pd.read_csv(SPLITS_DIR / "test.csv")
    preds_df = pd.read_csv(RESULTS_DIR / "pubmedbert_512_test_predictions.csv")
    
    # Ensure PMID formatting matches
    test_df["pmid_str"] = [str(p).strip().replace(".0", "") for p in test_df["pmid"]]
    preds_df["pmid_str"] = [str(p).strip().replace(".0", "") for p in preds_df["pmid"]]
    
    merged = pd.merge(test_df, preds_df[["pmid_str", "pubmedbert_512_weighted_prob", "pubmedbert_384_weighted_prob"]], on="pmid_str")
    
    clean_titles = [clean_clinical_text(str(t or "")) for t in merged["title"]]
    clean_abstracts = [clean_clinical_text(str(a or "")) for a in merged["abstract"]]
    
    title_lens = [len(tokenizer.encode(t, add_special_tokens=False)) for t in clean_titles]
    abstract_lens = [len(tokenizer.encode(a, add_special_tokens=False)) for a in clean_abstracts]
    total_lens = [len(tokenizer.encode(t, a, add_special_tokens=True)) for t, a in zip(clean_titles, clean_abstracts)]
    
    merged["clean_title"] = clean_titles
    merged["clean_abstract"] = clean_abstracts
    merged["title_tokens"] = title_lens
    merged["abstract_tokens"] = abstract_lens
    merged["total_tokens"] = total_lens
    merged["truncated_at_512"] = merged["total_tokens"] > 512
    merged["overflow_tokens"] = [max(0, l - 512) for l in merged["total_tokens"]]
    
    probs = merged["pubmedbert_512_weighted_prob"].to_numpy()
    labels = merged["label"].to_numpy().astype(int)
    
    # Analyze at Default threshold (0.50)
    merged["pred_05"] = (probs >= 0.50).astype(int)
    merged["category_05"] = "TN"
    merged.loc[(merged["label"] == 1) & (merged["pred_05"] == 1), "category_05"] = "TP"
    merged.loc[(merged["label"] == 0) & (merged["pred_05"] == 0), "category_05"] = "TN"
    merged.loc[(merged["label"] == 1) & (merged["pred_05"] == 0), "category_05"] = "FN"
    merged.loc[(merged["label"] == 0) & (merged["pred_05"] == 1), "category_05"] = "FP"
    
    # Analyze at Clinical High-Recall threshold (0.36 -> Recall >= 95%)
    merged["pred_036"] = (probs >= 0.36).astype(int)
    merged["category_036"] = "TN"
    merged.loc[(merged["label"] == 1) & (merged["pred_036"] == 1), "category_036"] = "TP"
    merged.loc[(merged["label"] == 0) & (merged["pred_036"] == 0), "category_036"] = "TN"
    merged.loc[(merged["label"] == 1) & (merged["pred_036"] == 0), "category_036"] = "FN"
    merged.loc[(merged["label"] == 0) & (merged["pred_036"] == 1), "category_036"] = "FP"
    
    # Save full audit
    audit_cols = [
        "pmid_str", "title", "label", "pubmedbert_512_weighted_prob", "pubmedbert_384_weighted_prob",
        "category_05", "category_036", "total_tokens", "title_tokens", "abstract_tokens",
        "truncated_at_512", "overflow_tokens"
    ]
    merged[audit_cols].to_csv(RESULTS_DIR / "pubmedbert_512_weighted_error_audit.csv", index=False)
    
    print("================================================================================")
    print("      PUBMEDBERT 512 WEIGHTED: TOKEN LENGTH & TRUNCATION ERROR ANALYSIS         ")
    print("================================================================================")
    print(f"Total Test Set Documents: {len(merged)}")
    print(f"True Positives: {(labels == 1).sum()} | True Negatives: {(labels == 0).sum()}")
    print(f"Documents with total_tokens <= 512 (Complete text preserved): {(merged['total_tokens'] <= 512).sum()} ({(merged['total_tokens'] <= 512).mean():.2%})")
    print(f"Documents with total_tokens > 512 (Truncated): {(merged['total_tokens'] > 512).sum()} ({(merged['total_tokens'] > 512).mean():.2%})")
    print(f"Token Lengths: Min={merged['total_tokens'].min()}, Median={merged['total_tokens'].median():.1f}, Mean={merged['total_tokens'].mean():.1f}, Max={merged['total_tokens'].max()}")
    print()
    
    # ---------------------------------------------------------
    # PART 1: FALSE NEGATIVES (FNs) AT THRESHOLD 0.50
    # ---------------------------------------------------------
    fns_05 = merged[merged["category_05"] == "FN"]
    print("--------------------------------------------------------------------------------")
    print(f"PART 1: FALSE NEGATIVES AT THRESHOLD 0.50 (N = {len(fns_05)} papers missed)")
    print("--------------------------------------------------------------------------------")
    for _, r in fns_05.iterrows():
        print(f"PMID: {r['pmid_str']}")
        print(f"Title: {r['title']}")
        print(f"Predicted Prob (512 Weighted): {r['pubmedbert_512_weighted_prob']:.4f} | Prob (384): {r['pubmedbert_384_weighted_prob']:.4f}")
        print(f"Total Tokens: {r['total_tokens']} (Title: {r['title_tokens']}, Abstract: {r['abstract_tokens']})")
        print(f"Truncated at 512? {'YES (>512)' if r['truncated_at_512'] else 'NO (<=512: 100% COMPLETE TEXT FED)'}")
        if r['truncated_at_512']:
            print(f"Overflow: {r['overflow_tokens']} tokens cut off.")
            # Decode the overflow tokens
            full_encoded = tokenizer.encode(r['clean_title'], r['clean_abstract'])
            overflow_text = tokenizer.decode(full_encoded[512:], skip_special_tokens=True)
            print(f"Cut-off Tail Text: \"{overflow_text}\"")
        else:
            print("Zero text was cut off! The entire title and abstract fit within 512 tokens.")
        print()
        
    # ---------------------------------------------------------
    # PART 2: FALSE POSITIVES (FPs) AT THRESHOLD 0.50
    # ---------------------------------------------------------
    fps_05 = merged[merged["category_05"] == "FP"]
    print("--------------------------------------------------------------------------------")
    print(f"PART 2: FALSE POSITIVES AT THRESHOLD 0.50 (N = {len(fps_05)} non-relevant papers flagged)")
    print("--------------------------------------------------------------------------------")
    print(f"FPs with total_tokens <= 512 (Complete text fed): {(fps_05['total_tokens'] <= 512).sum()} / {len(fps_05)} ({(fps_05['total_tokens'] <= 512).mean():.2%})")
    print(f"FPs with total_tokens > 512 (Truncated text): {(fps_05['total_tokens'] > 512).sum()} / {len(fps_05)} ({(fps_05['total_tokens'] > 512).mean():.2%})")
    print()
    for _, r in fps_05.iterrows():
        trunc_str = f"TRUNCATED by {r['overflow_tokens']} tokens (len={r['total_tokens']})" if r['truncated_at_512'] else f"NOT TRUNCATED (len={r['total_tokens']} <= 512)"
        print(f"- PMID {r['pmid_str']} | Prob={r['pubmedbert_512_weighted_prob']:.4f} | {trunc_str} | Title: {r['title'][:75]}...")
        if r['truncated_at_512']:
            full_encoded = tokenizer.encode(r['clean_title'], r['clean_abstract'])
            overflow_text = tokenizer.decode(full_encoded[512:], skip_special_tokens=True)
            print(f"  --> Cut-off Tail: \"{overflow_text[:120]}...\"")
    print()
    
    # ---------------------------------------------------------
    # PART 3: HIGH RECALL THRESHOLD (0.36, Recall >= 95%)
    # ---------------------------------------------------------
    fns_036 = merged[merged["category_036"] == "FN"]
    fps_036 = merged[merged["category_036"] == "FP"]
    print("--------------------------------------------------------------------------------")
    print(f"PART 3: HIGH RECALL OPERATING POINT (Threshold = 0.36, Recall >= 95%)")
    print("--------------------------------------------------------------------------------")
    print(f"Remaining FNs at 0.36 (N = {len(fns_036)}):")
    for _, r in fns_036.iterrows():
        print(f"- PMID {r['pmid_str']} | Prob={r['pubmedbert_512_weighted_prob']:.4f} | TotalTokens={r['total_tokens']} | Truncated: {r['truncated_at_512']} | Title: {r['title'][:80]}")
    print()
    print(f"FPs at 0.36 (N = {len(fps_036)}): Truncated={fps_036['truncated_at_512'].sum()} / {len(fps_036)}")
    print()

if __name__ == "__main__":
    main()
