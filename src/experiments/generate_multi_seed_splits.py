"""
generate_multi_seed_splits.py - Generate 5 independent stratified splits for PubMedBERT robustness evaluation.

Original dataset: 361 clinical abstracts (119 positive, 242 negative).
Exact split counts per seed:
- Sub-Train: 230 (76 positive, 154 negative)
- Validation: 58 (19 positive, 39 negative)
- Test: 73 (24 positive, 49 negative)
Total: 361

Target seeds (Option B): [101, 123, 456, 789, 2024]
"""

import json
import logging
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

SEEDS = [101, 123, 456, 789, 2024]
BASE_DIR = Path(__file__).resolve().parent.parent.parent
ORIGINAL_SPLITS_DIR = BASE_DIR / "data" / "splits"
MULTI_SEED_DIR = BASE_DIR / "data" / "splits" / "multi_seed"


def generate_splits():
    train_orig_path = ORIGINAL_SPLITS_DIR / "train.csv"
    test_orig_path = ORIGINAL_SPLITS_DIR / "test.csv"

    if not train_orig_path.exists() or not test_orig_path.exists():
        raise FileNotFoundError(f"Original splits not found at {ORIGINAL_SPLITS_DIR}")

    train_orig = pd.read_csv(train_orig_path)
    test_orig = pd.read_csv(test_orig_path)
    full_df = pd.concat([train_orig, test_orig], ignore_index=True)
    logger.info(f"Loaded full cohort: {len(full_df)} samples (Pos: {(full_df['label'] == 1).sum()}, Neg: {(full_df['label'] == 0).sum()})")

    MULTI_SEED_DIR.mkdir(parents=True, exist_ok=True)

    manifest = {}

    for seed in SEEDS:
        seed_dir = MULTI_SEED_DIR / f"split_seed_{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)

        # 1. Split into Full Train (288) and Test (73)
        train_full, test_df = train_test_split(
            full_df,
            test_size=73,
            stratify=full_df["label"],
            random_state=seed,
        )

        # 2. Split Full Train into Sub-Train (230) and Val (58)
        sub_train_df, val_df = train_test_split(
            train_full,
            test_size=58,
            stratify=train_full["label"],
            random_state=seed,
        )

        sub_train_df = sub_train_df.reset_index(drop=True)
        val_df = val_df.reset_index(drop=True)
        test_df = test_df.reset_index(drop=True)

        # Save CSVs
        sub_train_df.to_csv(seed_dir / "train.csv", index=False)
        val_df.to_csv(seed_dir / "val.csv", index=False)
        test_df.to_csv(seed_dir / "test.csv", index=False)

        info = {
            "seed": seed,
            "train_size": len(sub_train_df),
            "train_pos": int((sub_train_df["label"] == 1).sum()),
            "train_neg": int((sub_train_df["label"] == 0).sum()),
            "val_size": len(val_df),
            "val_pos": int((val_df["label"] == 1).sum()),
            "val_neg": int((val_df["label"] == 0).sum()),
            "test_size": len(test_df),
            "test_pos": int((test_df["label"] == 1).sum()),
            "test_neg": int((test_df["label"] == 0).sum()),
            "total_size": len(sub_train_df) + len(val_df) + len(test_df),
            "test_pmids": test_df["pmid"].astype(str).tolist(),
        }

        with open(seed_dir / "split_info.json", "w", encoding="utf-8") as f:
            json.dump(info, f, indent=2)

        manifest[f"seed_{seed}"] = info

        logger.info(
            f"Seed {seed} generated: Sub-Train={len(sub_train_df)} (Pos={info['train_pos']}), "
            f"Val={len(val_df)} (Pos={info['val_pos']}), "
            f"Test={len(test_df)} (Pos={info['test_pos']})"
        )

    with open(MULTI_SEED_DIR / "multi_seed_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    logger.info("All 5 multi-seed splits successfully generated and persisted.")


if __name__ == "__main__":
    generate_splits()
