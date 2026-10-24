"""
split_stage5.py - Generates a new, zero-leakage stratified split for Stage 5.
Uses random_state = 2025 on the updated data/labeled-dataset-v2.csv.
Partitions:
- Sub-Train (64%): 335 samples (180 pos, 155 neg)
- Validation (16%): 84 samples (45 pos, 39 neg)
- Held-Out Test (20%): 105 samples (56 pos, 49 neg)
Total: 524 samples.
"""

import json
import logging
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_PATH = BASE_DIR / "data" / "labeled-dataset-v2.csv"
OUTPUT_DIR = BASE_DIR / "data" / "splits" / "stage5"
SEED = 2025


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"{DATA_PATH} not found.")

    df = pd.read_csv(DATA_PATH)
    logger.info(f"Loaded dataset: {len(df)} rows. Pos: {(df['label']==1).sum()}, Neg: {(df['label']==0).sum()}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Stratified Held-out Test split (105 samples = 20.04%)
    train_full, test_df = train_test_split(
        df,
        test_size=105,
        stratify=df["label"],
        random_state=SEED,
    )

    # 2. Stratified Sub-Train (335 samples) and Validation (84 samples)
    sub_train_df, val_df = train_test_split(
        train_full,
        test_size=84,
        stratify=train_full["label"],
        random_state=SEED,
    )

    sub_train_df = sub_train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    # Zero leakage check
    train_pmids = set(sub_train_df["pmid"].astype(str))
    val_pmids = set(val_df["pmid"].astype(str))
    test_pmids = set(test_df["pmid"].astype(str))

    assert len(train_pmids.intersection(val_pmids)) == 0, "Leakage between train and val"
    assert len(train_pmids.intersection(test_pmids)) == 0, "Leakage between train and test"
    assert len(val_pmids.intersection(test_pmids)) == 0, "Leakage between val and test"

    # Save to disk
    sub_train_df.to_csv(OUTPUT_DIR / "train.csv", index=False)
    val_df.to_csv(OUTPUT_DIR / "val.csv", index=False)
    test_df.to_csv(OUTPUT_DIR / "test.csv", index=False)

    manifest = {
        "seed": SEED,
        "dataset_path": str(DATA_PATH),
        "total_samples": len(df),
        "sub_train_samples": len(sub_train_df),
        "sub_train_pos": int((sub_train_df["label"] == 1).sum()),
        "sub_train_neg": int((sub_train_df["label"] == 0).sum()),
        "val_samples": len(val_df),
        "val_pos": int((val_df["label"] == 1).sum()),
        "val_neg": int((val_df["label"] == 0).sum()),
        "test_samples": len(test_df),
        "test_pos": int((test_df["label"] == 1).sum()),
        "test_neg": int((test_df["label"] == 0).sum()),
        "test_pmids": test_df["pmid"].astype(str).tolist(),
    }

    with open(OUTPUT_DIR / "split_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    logger.info("Stage 5 Split successfully created in %s", OUTPUT_DIR)
    logger.info("Sub-Train: %d (Pos=%d, Neg=%d)", len(sub_train_df), manifest["sub_train_pos"], manifest["sub_train_neg"])
    logger.info("Val: %d (Pos=%d, Neg=%d)", len(val_df), manifest["val_pos"], manifest["val_neg"])
    logger.info("Held-Out Test: %d (Pos=%d, Neg=%d)", len(test_df), manifest["test_pos"], manifest["test_neg"])


if __name__ == "__main__":
    main()
