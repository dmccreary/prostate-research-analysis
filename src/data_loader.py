"""
data_loader.py - Data loading, quality audit, and splitting for prostate cancer literature.

Handles:
- Locating and loading dataset (supports labeled_dataset.csv and labeled-dataset.csv)
- Quality and leakage auditing
- Removal of conflicting duplicates (PMID 15774239) and empty abstracts (PMID 21056265)
- Stratified train/test partitioning with reproducible seeding and JSON persistence
"""

import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split

# Safe standard output encoding on Windows terminals
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

logger = logging.getLogger(__name__)

# Known conflicting PMIDs in dataset
CONFLICTING_PMIDS = [15774239]


def find_dataset_path(base_dir: Optional[str] = None) -> Path:
    """Locate dataset file in the data/ directory, handling naming variations."""
    if base_dir is None:
        base_dir = Path(__file__).resolve().parent.parent
    else:
        base_dir = Path(base_dir)

    candidates = [
        base_dir / "data" / "labeled-dataset.csv",
        base_dir / "data" / "labeled_dataset.csv",
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(
        f"Could not find labeled dataset in {base_dir / 'data'}. Checked: {[str(c) for c in candidates]}"
    )


def load_raw_dataset(filepath: Optional[str] = None) -> pd.DataFrame:
    """Load raw dataset without modifications."""
    path = Path(filepath) if filepath else find_dataset_path()
    logger.info("Loading raw dataset from %s", path)
    df = pd.read_csv(path)
    return df


def audit_dataset(df: pd.DataFrame) -> Dict:
    """
    Perform deep inspection of raw dataset for data quality, class distribution,
    missing values, duplicates, and leakage sources.
    """
    stats = {}
    stats["num_rows"] = len(df)
    stats["num_cols"] = len(df.columns)
    stats["columns"] = df.columns.tolist()
    stats["dtypes"] = {col: str(dtype) for col, dtype in df.dtypes.items()}
    stats["missing_values"] = df.isnull().sum().to_dict()

    # Label distribution
    label_counts = df["label"].value_counts().to_dict()
    stats["label_distribution"] = label_counts
    stats["num_class_0"] = int(label_counts.get(0, 0))
    stats["num_class_1"] = int(label_counts.get(1, 0))
    stats["class_imbalance_ratio"] = (
        stats["num_class_0"] / stats["num_class_1"] if stats["num_class_1"] > 0 else 0
    )

    # Duplicate rows and PMIDs
    stats["exact_duplicate_rows"] = int(df.duplicated().sum())
    stats["duplicate_pmids_count"] = int(df.duplicated(subset=["pmid"]).sum())
    dup_pmid_rows = df[df.duplicated(subset=["pmid"], keep=False)]
    stats["duplicate_pmids_list"] = dup_pmid_rows["pmid"].unique().tolist()

    # Check for conflicting labels across duplicate PMIDs
    conflicting = []
    for pmid in stats["duplicate_pmids_list"]:
        labels = df[df["pmid"] == pmid]["label"].unique().tolist()
        if len(labels) > 1:
            conflicting.append({"pmid": int(pmid), "conflicting_labels": labels})
    stats["conflicting_label_pmids"] = conflicting

    # Missing/empty abstracts
    empty_abs_mask = df["abstract"].isna() | (df["abstract"].astype(str).str.strip() == "")
    stats["empty_abstracts_count"] = int(empty_abs_mask.sum())
    stats["empty_abstracts_pmids"] = df[empty_abs_mask]["pmid"].tolist()

    # Abstract length statistics (characters and words)
    valid_abstracts = df["abstract"].dropna().astype(str).str.strip()
    char_lens = valid_abstracts.str.len()
    word_lens = valid_abstracts.str.split().apply(len)

    stats["abstract_char_length"] = {
        "mean": float(char_lens.mean()),
        "std": float(char_lens.std()),
        "min": int(char_lens.min()) if len(char_lens) > 0 else 0,
        "p25": float(char_lens.quantile(0.25)) if len(char_lens) > 0 else 0,
        "median": float(char_lens.median()) if len(char_lens) > 0 else 0,
        "p75": float(char_lens.quantile(0.75)) if len(char_lens) > 0 else 0,
        "max": int(char_lens.max()) if len(char_lens) > 0 else 0,
    }

    stats["abstract_word_length"] = {
        "mean": float(word_lens.mean()),
        "std": float(word_lens.std()),
        "min": int(word_lens.min()) if len(word_lens) > 0 else 0,
        "p25": float(word_lens.quantile(0.25)) if len(word_lens) > 0 else 0,
        "median": float(word_lens.median()) if len(word_lens) > 0 else 0,
        "p75": float(word_lens.quantile(0.75)) if len(word_lens) > 0 else 0,
        "max": int(word_lens.max()) if len(word_lens) > 0 else 0,
    }

    # Leakage analysis: check metadata columns
    leakage_indicators = {}
    if "dataset" in df.columns:
        dataset_vs_label = (df["dataset"] == "positive").astype(int) == df["label"]
        leakage_indicators["dataset_column_matches_label_100pct"] = bool(dataset_vs_label.all())

    if "risk_category" in df.columns:
        pos_has_risk = df[df["label"] == 1]["risk_category"].notna().mean()
        neg_has_risk = df[df["label"] == 0]["risk_category"].notna().mean()
        leakage_indicators["pos_has_risk_category_rate"] = float(pos_has_risk)
        leakage_indicators["neg_has_risk_category_rate"] = float(neg_has_risk)

    if "year_group" in df.columns:
        pos_has_yg = df[df["label"] == 1]["year_group"].notna().mean()
        neg_has_yg = df[df["label"] == 0]["year_group"].notna().mean()
        leakage_indicators["pos_has_year_group_rate"] = float(pos_has_yg)
        leakage_indicators["neg_has_year_group_rate"] = float(neg_has_yg)

    stats["leakage_indicators"] = leakage_indicators

    return stats


def clean_dataset(
    df: pd.DataFrame,
    drop_conflicting_pmids: bool = True,
    drop_empty_abstracts: bool = True,
) -> Tuple[pd.DataFrame, Dict]:
    """
    Clean dataset safely without silent deletions.
    Excludes conflicting duplicate PMIDs and empty abstracts.
    Documents all exclusions.
    """
    initial_count = len(df)
    cleaning_log = {
        "initial_rows": initial_count,
        "initial_class_0": int((df["label"] == 0).sum()),
        "initial_class_1": int((df["label"] == 1).sum()),
        "dropped_conflicting_rows": 0,
        "dropped_empty_abstract_rows": 0,
        "excluded_pmids": [],
    }

    cleaned_df = df.copy()

    # 1. Drop conflicting duplicate PMIDs
    if drop_conflicting_pmids:
        conflicting_mask = cleaned_df["pmid"].isin(CONFLICTING_PMIDS)
        dropped_conflicts = int(conflicting_mask.sum())
        if dropped_conflicts > 0:
            cleaned_df = cleaned_df[~conflicting_mask].copy()
            cleaning_log["dropped_conflicting_rows"] = dropped_conflicts
            cleaning_log["excluded_pmids"].extend(CONFLICTING_PMIDS)
            logger.warning(
                "Dropped %d rows with conflicting labels for PMIDs: %s",
                dropped_conflicts,
                CONFLICTING_PMIDS,
            )

    # 2. Drop empty or NaN abstracts
    if drop_empty_abstracts:
        empty_mask = cleaned_df["abstract"].isna() | (
            cleaned_df["abstract"].astype(str).str.strip() == ""
        )
        dropped_empty = int(empty_mask.sum())
        if dropped_empty > 0:
            empty_pmids = cleaned_df[empty_mask]["pmid"].tolist()
            cleaned_df = cleaned_df[~empty_mask].copy()
            cleaning_log["dropped_empty_abstract_rows"] = dropped_empty
            cleaning_log["excluded_pmids"].extend(empty_pmids)
            logger.warning(
                "Dropped %d rows with empty/NaN abstracts for PMIDs: %s",
                dropped_empty,
                empty_pmids,
            )

    cleaned_df = cleaned_df.reset_index(drop=True)

    cleaning_log["final_rows"] = len(cleaned_df)
    cleaning_log["final_class_0"] = int((cleaned_df["label"] == 0).sum())
    cleaning_log["final_class_1"] = int((cleaned_df["label"] == 1).sum())
    cleaning_log["imbalance_ratio"] = (
        cleaning_log["final_class_0"] / cleaning_log["final_class_1"]
        if cleaning_log["final_class_1"] > 0
        else 0
    )

    logger.info(
        "Cleaned dataset: %d -> %d rows (Class 0: %d, Class 1: %d)",
        initial_count,
        len(cleaned_df),
        cleaning_log["final_class_0"],
        cleaning_log["final_class_1"],
    )

    return cleaned_df, cleaning_log


def create_stratified_split(
    df: pd.DataFrame,
    test_size: float = 0.20,
    random_state: int = 42,
    output_dir: Optional[str] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
    """
    Split the dataset into stratified training and held-out test sets.
    Saves split metadata and indices for exact reproducibility.
    """
    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        stratify=df["label"],
        random_state=random_state,
    )
    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    split_meta = {
        "random_state": random_state,
        "test_size": test_size,
        "total_samples": len(df),
        "train_samples": len(train_df),
        "test_samples": len(test_df),
        "train_class_0": int((train_df["label"] == 0).sum()),
        "train_class_1": int((train_df["label"] == 1).sum()),
        "train_pos_rate": float((train_df["label"] == 1).mean()),
        "test_class_0": int((test_df["label"] == 0).sum()),
        "test_class_1": int((test_df["label"] == 1).sum()),
        "test_pos_rate": float((test_df["label"] == 1).mean()),
        "train_pmids": train_df["pmid"].tolist(),
        "test_pmids": test_df["pmid"].tolist(),
    }

    if output_dir:
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        with open(out_path / "train_test_split.json", "w", encoding="utf-8") as f:
            json.dump(split_meta, f, indent=2)
        train_df.to_csv(out_path / "train.csv", index=False)
        test_df.to_csv(out_path / "test.csv", index=False)
        logger.info("Saved split data and metadata to %s", out_path)

    return train_df, test_df, split_meta


def get_cv_folds(
    train_df: pd.DataFrame, n_splits: int = 5, random_state: int = 42
) -> StratifiedKFold:
    """Generate Stratified K-Fold cross-validation splitter on training data."""
    return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    raw_df = load_raw_dataset()
    audit = audit_dataset(raw_df)
    print("=== DATASET AUDIT ===")
    print(f"Total Rows: {audit['num_rows']}, Columns: {audit['num_cols']}")
    print(f"Class Distribution: {audit['label_distribution']}")
    print(f"Imbalance Ratio: {audit['class_imbalance_ratio']:.2f} : 1")
    print(f"Conflicting PMIDs: {audit['conflicting_label_pmids']}")
    print(f"Empty Abstracts: {audit['empty_abstracts_count']}")
    print(f"Leakage Indicators: {audit['leakage_indicators']}")

    clean_df, log = clean_dataset(raw_df)
    print("\n=== CLEANING LOG ===")
    print(log)

    base_splits_dir = Path(__file__).resolve().parent.parent / "data" / "splits"
    train_df, test_df, meta = create_stratified_split(clean_df, output_dir=str(base_splits_dir))
    print("\n=== SPLIT SUMMARY ===")
    print(f"Train: {meta['train_samples']} (Pos: {meta['train_class_1']}, Neg: {meta['train_class_0']})")
    print(f"Test: {meta['test_samples']} (Pos: {meta['test_class_1']}, Neg: {meta['test_class_0']})")
