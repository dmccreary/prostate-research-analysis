"""
preprocessing.py - Text preprocessing and feature extraction for PubMed abstracts.

Key principles:
1. Preserve medical/clinical terminology, acronyms (PSA, EBRT, HDR, LDR, HIFU),
   and risk classification keywords.
2. Safely strip true HTML formatting tags (<sup>, <sub>, <b>, <i>, <p>) while
   preserving mathematical and clinical inequalities (e.g. 'PSA < 15', 'p < 0.001', 'dose >= 72Gy').
3. Normalize unicode quotes, dashes, and whitespace without corrupting scientific symbols.
4. Avoid destructive stemming/lemmatization that damages medical semantics.
5. Provide scikit-learn compatible Transformer pipeline components to prevent
   vocabulary leakage between train and test/validation sets.
"""

import logging
import re
import sys
import unicodedata
from typing import Dict, List, Optional, Tuple, Union

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer

# Safe standard output encoding on Windows terminals
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

logger = logging.getLogger(__name__)

# Regex for HTML formatting tags (e.g. <b>, </i>, <sup>12</sup>, <br/>)
HTML_TAG_REGEX = re.compile(r"</?(?:[a-zA-Z]+[0-9]*)\b[^>]*>", re.IGNORECASE)

# Mapping of problematic Unicode characters to clean ASCII representations
UNICODE_CHAR_MAP = {
    "\u00a0": " ",      # Non-breaking space
    "\u2009": " ",      # Thin space
    "\u200b": "",       # Zero-width space
    "\u2018": "'",      # Left single quote
    "\u2019": "'",      # Right single quote
    "\u201a": "'",      # Single low-9 quotation
    "\u201c": '"',      # Left double quote
    "\u201d": '"',      # Right double quote
    "\u2013": "-",      # En dash
    "\u2014": "-",      # Em dash
    "\u2212": "-",      # Minus sign
    "\u00b1": "+/-",    # Plus-minus sign
    "\u2264": "<=",     # Less than or equal to
    "\u2265": ">=",     # Greater than or equal to
    "\u00b5": "u",      # Micro sign
    "\u03bc": "u",      # Greek mu
    "\u03b1": "alpha",  # Greek alpha
    "\u03b2": "beta",   # Greek beta
    "\u03b3": "gamma",  # Greek gamma
}


def clean_clinical_text(text: Optional[str]) -> str:
    """
    Clean clinical abstract text safely:
    - Replaces nulls with empty string
    - Normalizes Unicode punctuation & characters
    - Strips markup tags while preserving mathematical inequalities
    - Cleans excessive whitespace
    """
    if text is None or not isinstance(text, str):
        return ""

    # Unicode normalization to standard NFKC
    cleaned = unicodedata.normalize("NFKC", text)

    # Custom Unicode replacements
    for char, replacement in UNICODE_CHAR_MAP.items():
        if char in cleaned:
            cleaned = cleaned.replace(char, replacement)

    # Strip true HTML/XML tags (preserves < 15 and > 20 since '< 15' does not match tag regex)
    cleaned = HTML_TAG_REGEX.sub(" ", cleaned)

    # Standardize structured abstract headers (e.g. "BACKGROUND: ... METHODS: ...")
    # Headers are preserved as words so structural context is retained, but colon spacing is standardized
    cleaned = re.sub(r"\b(BACKGROUND|OBJECTIVES?|METHODS?|RESULTS?|CONCLUSIONS?)\s*:\s*", r"\1: ", cleaned, flags=re.IGNORECASE)

    # Collapse multiple whitespaces
    cleaned = re.sub(r"\s+", " ", cleaned).strip()

    return cleaned


class ClinicalTextCleaner(BaseEstimator, TransformerMixin):
    """
    Scikit-learn compatible transformer that applies clinical text cleaning.
    Ensures safe integration into sklearn Pipeline.
    """

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        if hasattr(X, "tolist"):
            X = X.tolist()
        return [clean_clinical_text(text) for text in X]


def get_default_tfidf_vectorizer(
    representation_type: str = "word",
    ngram_range: Tuple[int, int] = (1, 2),
    min_df: int = 2,
    max_df: float = 0.85,
    sublinear_tf: bool = True,
    max_features: Optional[int] = None,
    stop_words: Optional[str] = "english",
) -> TfidfVectorizer:
    """
    Construct a TF-IDF vectorizer configured for clinical text.

    Args:
        representation_type: 'word' for word n-grams, 'char_wb' for character n-grams within word boundaries.
        ngram_range: (min_n, max_n) n-gram range.
        min_df: Minimum document frequency threshold.
        max_df: Maximum document frequency threshold to filter overly frequent terms.
        sublinear_tf: Apply sublinear tf scaling (1 + log(tf)).
        max_features: Optional cap on vocabulary size.
        stop_words: Stopwords strategy ('english' or None).
    """
    if representation_type == "word":
        return TfidfVectorizer(
            analyzer="word",
            ngram_range=ngram_range,
            min_df=min_df,
            max_df=max_df,
            sublinear_tf=sublinear_tf,
            max_features=max_features,
            stop_words=stop_words,
            lowercase=True,
            token_pattern=r"(?u)\b[a-zA-Z0-9_\-\/]+\b",  # Keeps hyphenated medical terms (e.g. high-risk, t1b-2b)
        )
    elif representation_type == "char_wb":
        return TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=ngram_range if ngram_range != (1, 2) else (3, 5),
            min_df=min_df,
            max_df=max_df,
            sublinear_tf=sublinear_tf,
            max_features=max_features,
            lowercase=True,
        )
    else:
        raise ValueError(f"Unknown representation_type: {representation_type}")


if __name__ == "__main__":
    # Test cleaning on clinical edge cases
    samples = [
        "Patient PSA < 15 ng/ml and Gleason <= 6. <sup>125</sup>I brachytherapy dose was \u2265 72Gy.",
        "<b>OBJECTIVE:</b> To evaluate <i>high-risk</i> prostate cancer outcomes \u00b1 5 years follow-up.",
        "Empty abstract check.",
    ]
    cleaner = ClinicalTextCleaner()
    cleaned = cleaner.transform(samples)
    print("=== CLINICAL TEXT PREPROCESSING SAMPLES ===")
    for orig, c in zip(samples, cleaned):
        print(f"Original: {orig}")
        print(f"Cleaned : {c}\n")

    vec = get_default_tfidf_vectorizer(representation_type="word", min_df=1)
    X = vec.fit_transform(cleaned)
    print("Vocabulary sample:", list(vec.vocabulary_.keys())[:15])
    print("Feature matrix shape:", X.shape)
