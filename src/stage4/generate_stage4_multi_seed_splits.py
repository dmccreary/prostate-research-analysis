"""
generate_stage4_multi_seed_splits.py - Generate 5 independent stratified splits for Stage 4 dataset.

Dataset: labeled-dataset-v2.csv (N = 524, 281 positive, 243 negative).
Split specifications per seed:
- Sub-Train (64%): 335 samples (180 positive, 155 negative, 53.73% pos)
- Validation (16%): 84 samples (45 positive, 39 negative, 53.57% pos)
- Held-Out Test (20%): 105 samples (56 positive, 49 negative, 53.33% pos)
Total: 524 samples

Seeds evaluated: [101, 123, 456, 789, 2024]
Strict zero-leakage guarantee across partitions.
"""

import json
import logging
import os
from pathlib import Path
import sys
import pandas as pd
from sklearn.model_selection import train_test_split

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

SEEDS = [101, 123, 456, 789, 2024]
BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATASET_PATH = BASE_DIR / "data" / "labeled-dataset-v2.csv"
OUTPUT_DIR = BASE_DIR / "data" / "splits" / "stage4_multi_seed"


def generate_splits():
    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"Dataset not found at {DATASET_PATH}")

    df = pd.read_csv(DATASET_PATH)
    logger.info(f"Loaded labeled-dataset-v2.csv: {len(df)} samples (Pos: {(df['label'] == 1).sum()}, Neg: {(df['label'] == 0).sum()})")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest = {}

    for seed in SEEDS:
        seed_dir = OUTPUT_DIR / f"split_seed_{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)

        # 1. Stratified Held-out Test split (105 samples, 20.04%)
        train_full, test_df = train_test_split(
            df,
            test_size=105,
            stratify=df["label"],
            random_state=seed,
        )

        # 2. Stratified Sub-Train (335 samples) and Validation (84 samples)
        sub_train_df, val_df = train_test_split(
            train_full,
            test_size=84,
            stratify=train_full["label"],
            random_state=seed,
        )

        sub_train_df = sub_train_df.reset_index(drop=True)
        val_df = val_df.reset_index(drop=True)
        test_df = test_df.reset_index(drop=True)

        # Zero leakage check
        train_pmids = set(sub_train_df["pmid"].astype(str))
        val_pmids = set(val_df["pmid"].astype(str))
        test_pmids = set(test_df["pmid"].astype(str))

        assert len(train_pmids.intersection(val_pmids)) == 0, f"Seed {seed}: Leakage between train and val"
        assert len(train_pmids.intersection(test_pmids)) == 0, f"Seed {seed}: Leakage between train and test"
        assert len(val_pmids.intersection(test_pmids)) == 0, f"Seed {seed}: Leakage between val and test"

        # Save CSVs
        sub_train_df.to_csv(seed_dir / "train.csv", index=False)
        val_df.to_csv(seed_dir / "val.csv", index=False)
        test_df.to_csv(seed_dir / "test.csv", index=False)

        info = {
            "seed": seed,
            "sub_train_size": len(sub_train_df),
            "sub_train_pos": int((sub_train_df["label"] == 1).sum()),
            "sub_train_neg": int((sub_train_df["label"] == 0).sum()),
            "sub_train_pos_rate": float((sub_train_df["label"] == 1).mean()),
            "val_size": len(val_df),
            "val_pos": int((val_df["label"] == 1).sum()),
            "val_neg": int((val_df["label"] == 0).sum()),
            "val_pos_rate": float((val_df["label"] == 1).mean()),
            "test_size": len(test_df),
            "test_pos": int((test_df["label"] == 1).sum()),
            "test_neg": int((test_df["label"] == 0).sum()),
            "test_pos_rate": float((test_df["label"] == 1).mean()),
            "total_size": len(df),
            "test_pmids": test_df["pmid"].astype(str).tolist(),
        }

        with open(seed_dir / "split_info.json", "w", encoding="utf-8") as f:
            json.dump(info, f, indent=2)

        manifest[f"seed_{seed}"] = info
        logger.info(
            f"Seed {seed} generated: Sub-Train={len(sub_train_df)} (Pos={info['sub_train_pos']}, Neg={info['sub_train_neg']}), "
            f"Val={len(val_df)} (Pos={info['val_pos']}, Neg={info['val_neg']}), "
            f"Test={len(test_df)} (Pos={info['test_pos']}, Neg={info['test_neg']})"
        )

    with open(OUTPUT_DIR / "stage4_multi_seed_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    logger.info(f"All 5 splits generated and verified in {OUTPUT_DIR}")


if __name__ == "__main__":
    generate_splits()
