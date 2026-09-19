"""
rule_engine.py - Clinical rule filters and feature extraction for Stage 3.

Implements the exact exclusion criteria and clinical heuristics established in:
- qwen_classifier.py (lines 42-85)
- docs/selection-criteria.md (PCRSG Guidelines)
- src/prostate-cancer-scorer.py
"""

import re
from typing import Dict, List, Optional, Tuple, Union
import numpy as np

# ── 1. Excluded PubMed Publication Types (Verbatim from qwen_classifier.py) ───
EXCLUDE_TYPES = {
    t.lower()
    for t in [
        "Case Reports",
        "Systematic Review",
        "Meta-Analysis",
        "Observational Study",
        "Retracted Publication",
        "Comment",
        "Practice Guideline",
        "Historical Article",
        "Consensus Statement",
        "Letter",
        "Editorial",
        "Conference Proceedings",
        "News",
        "Guideline",
        "Portrait",
        "Lecture",
        "Biography",
        "Twin Study",
        "Duplicate Publication",
        "Technical Report",
        "Clinical Conference",
        "Introductory Journal Article",
        "Video-Audio Media",
        "Corrected and Republished Article",
        "Patient Education Handout",
        "Interview",
        "Autobiography",
        "Retraction Notice",
    ]
}

# ── 2. Advanced / Metastatic Disease Keyword Filter (Title + Abstract) ────────
# Excludes metastatic or castration-resistant prostate cancer studies (out of scope)
EXCLUDE_KEYWORDS_RE = re.compile(r"\b(?:m?CRPC|m?HRPC|m?CSPC|m?HSPC)\b", re.IGNORECASE)

# ── 3. Title-Only Keyword Filter ──────────────────────────────────────────────
# If "salvage" appears in Title, study evaluates post-failure recurrence, not primary cohort
EXCLUDE_TITLE_KEYWORDS_RE = re.compile(r"\bsalvage", re.IGNORECASE)

# ── 4. Clinical Regex Patterns for Heuristics ─────────────────────────────────
# Sample size regex
SAMPLE_SIZE_RE = re.compile(
    r"(?:n\s*=\s*|n\s+|cohort\s+(?:of\s+)?|total\s+(?:of\s+)?|sample\s+(?:size\s+)?(?:of\s+)?)(\d+)|(\d+)\s+(?:patients?|men|subjects?|participants?)",
    re.IGNORECASE,
)

# Follow-up regex
FOLLOWUP_RE = re.compile(
    r"(?:median|mean)?\s*follow[-\s]?up\s*(?:time\s*)?(?:of\s*)?(\d+(?:\.\d+)?)\s*(?:years?|yrs?|months?|mo)",
    re.IGNORECASE,
)

# Curative modalities regex
MODALITY_RE = re.compile(
    r"\b(?:prostatectomy|radical\s+prostatectomy|robotic|ebrt|external\s+beam|radiation\s+therapy|hypofractionated|sbrt|stereotactic|brachytherapy|seed\s+implant|ldr|hdr|cryotherapy|cryoablation|hifu|ultrasound)\b",
    re.IGNORECASE,
)

# Quantitative endpoints regex
ENDPOINTS_RE = re.compile(
    r"\b(?:brfs|biochemical\s+recurrence|biochemical\s+failure|biochemical\s+relapse|overall\s+survival|\bos\b|metastasis[-\s]free\s+survival|\bmfs\b|cancer[-\s]specific\s+survival|cause[-\s]specific\s+survival|\bcss\b|psa\s+recurrence|psa\s+relapse)\b",
    re.IGNORECASE,
)


def match_excluded_types(pub_types: List[str]) -> List[str]:
    """Returns any publication types matching the exclusion list."""
    if not pub_types:
        return []
    return [t for t in pub_types if str(t).strip().lower() in EXCLUDE_TYPES]


def match_excluded_keywords(text: str) -> List[str]:
    """Returns matched advanced-disease acronyms (mCRPC, etc.)."""
    return sorted(list(set(m.group(0).upper() for m in EXCLUDE_KEYWORDS_RE.finditer(str(text)))))


def match_excluded_title_keywords(title: str) -> List[str]:
    """Returns matched title-only exclusion keywords ('salvage')."""
    return sorted(list(set(m.group(0).lower() for m in EXCLUDE_TITLE_KEYWORDS_RE.finditer(str(title)))))


def evaluate_pre_filter(
    title: str,
    abstract: str,
    pub_types: Optional[List[str]] = None,
) -> Tuple[bool, str]:
    """
    Evaluates the deterministic pre-filter for an article.
    Returns:
        (is_excluded: bool, reason: str)
    If is_excluded is True, the article is deterministically classified as Negative (0).
    """
    title_str = str(title or "")
    abstract_str = str(abstract or "")
    types = pub_types or []

    # Check 1: Excluded publication type
    matched_types = match_excluded_types(types)
    if matched_types:
        return True, f"Excluded publication type: {', '.join(matched_types)}"

    # Check 2: Excluded title keyword ('salvage')
    matched_title = match_excluded_title_keywords(title_str)
    if matched_title:
        return True, f"Excluded title keyword: {', '.join(matched_title)}"

    # Check 3: Advanced-disease keyword (in title or abstract)
    combined_text = f"{title_str} {abstract_str}"
    matched_kw = match_excluded_keywords(combined_text)
    if matched_kw:
        return True, f"Excluded advanced disease keyword: {', '.join(matched_kw)}"

    return False, ""


def extract_clinical_features(
    title: str,
    abstract: str,
    pub_types: Optional[List[str]] = None,
) -> Dict[str, Union[int, float]]:
    """
    Extracts structured domain-rule features from title and abstract text.
    These can be concatenated as tabular features with TF-IDF or BERT embeddings.
    """
    title_str = str(title or "").lower()
    abstract_str = str(abstract or "").lower()
    combined_text = f"{title_str} {abstract_str}"

    # Sample size extraction
    sample_size = None
    for match in SAMPLE_SIZE_RE.finditer(combined_text):
        g1 = match.group(1) or match.group(2)
        if g1:
            try:
                val = int(g1)
                if 10 <= val <= 200000:  # plausible clinical cohort
                    sample_size = val
                    break
            except ValueError:
                pass

    if sample_size is None:
        sample_size_adequate = 1.0  # Assumed met if not stated (per Qwen prompt)
    else:
        sample_size_adequate = 1.0 if sample_size >= 50 else 0.0

    # Follow-up extraction
    followup_years = None
    for match in FOLLOWUP_RE.finditer(combined_text):
        g1 = match.group(1)
        if g1:
            try:
                val = float(g1)
                # Check unit
                matched_str = match.group(0).lower()
                if "mo" in matched_str:
                    val = val / 12.0
                if 0.5 <= val <= 30.0:
                    followup_years = val
                    break
            except ValueError:
                pass

    if followup_years is None:
        followup_adequate = 1.0  # Assumed met if not stated
    else:
        followup_adequate = 1.0 if followup_years >= 5.0 else 0.0

    # Curative modality presence
    has_modality = 1.0 if MODALITY_RE.search(combined_text) else 0.0

    # Survival endpoint presence
    has_endpoint = 1.0 if ENDPOINTS_RE.search(combined_text) else 0.0

    # Pre-filter violation flags
    types = pub_types or []
    has_excluded_type = 1.0 if match_excluded_types(types) else 0.0
    has_salvage_title = 1.0 if match_excluded_title_keywords(title_str) else 0.0
    has_advanced_kw = 1.0 if match_excluded_keywords(combined_text) else 0.0

    return {
        "sample_size_adequate": sample_size_adequate,
        "followup_adequate": followup_adequate,
        "has_modality": has_modality,
        "has_endpoint": has_endpoint,
        "has_excluded_type": has_excluded_type,
        "has_salvage_title": has_salvage_title,
        "has_advanced_kw": has_advanced_kw,
    }


def get_rule_feature_vector(
    title: str,
    abstract: str,
    pub_types: Optional[List[str]] = None,
) -> np.ndarray:
    """Returns a float32 vector of clinical rule features."""
    feats = extract_clinical_features(title, abstract, pub_types)
    return np.array(list(feats.values()), dtype=np.float32)
