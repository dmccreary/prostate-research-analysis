"""
dataset_stage4.py - Data loading, quality audit, splitting, and PyTorch dataset for Stage 4.

Dataset: data/labeled-dataset-v2.csv (N = 524)
Splits:
- Sub-Train: 335 (180 Pos, 155 Neg, 53.73% Pos)
- Validation: 84 (45 Pos, 39 Neg, 53.57% Pos)
- Held-Out Test: 105 (56 Pos, 49 Neg, 53.33% Pos)
Total: 524 (281 Pos, 243 Neg, 53.63% Pos)

Uses random_state = 42 for exact methodological consistency with Stage 3.
Zero overlap across splits (0 duplicate PMIDs).
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer

import sys
BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(BASE_DIR / "src"))
from preprocessing import clean_clinical_text

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

DATASET_PATH = BASE_DIR / "data" / "labeled-dataset-v2.csv"
SPLITS_DIR = BASE_DIR / "data" / "splits" / "stage4"


def create_and_save_stage4_splits(
    dataset_path: Path = DATASET_PATH,
    splits_dir: Path = SPLITS_DIR,
    test_size: float = 0.20,
    val_size: float = 0.20,
    random_state: int = 42,
) -> Dict[str, pd.DataFrame]:
    """
    Creates and persists zero-leakage stratified splits for Stage 4.
    """
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found at {dataset_path}")

    df = pd.read_csv(dataset_path)
    logger.info(f"Loaded labeled-dataset-v2.csv: {len(df)} rows, Pos: {(df['label'] == 1).sum()}, Neg: {(df['label'] == 0).sum()}")

    # 1. Held-Out Test split (20%)
    train_full, test_df = train_test_split(
        df,
        test_size=test_size,
        stratify=df["label"],
        random_state=random_state,
    )

    # 2. Sub-Train and Validation split (20% of train_full)
    sub_train, val_df = train_test_split(
        train_full,
        test_size=val_size,
        stratify=train_full["label"],
        random_state=random_state,
    )

    sub_train = sub_train.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    splits_dir.mkdir(parents=True, exist_ok=True)

    sub_train.to_csv(splits_dir / "train.csv", index=False)
    val_df.to_csv(splits_dir / "val.csv", index=False)
    test_df.to_csv(splits_dir / "test.csv", index=False)

    split_info = {
        "dataset_name": "labeled-dataset-v2.csv",
        "random_state": random_state,
        "total_samples": len(df),
        "total_pos": int((df["label"] == 1).sum()),
        "total_neg": int((df["label"] == 0).sum()),
        "sub_train_samples": len(sub_train),
        "sub_train_pos": int((sub_train["label"] == 1).sum()),
        "sub_train_neg": int((sub_train["label"] == 0).sum()),
        "sub_train_pos_rate": float((sub_train["label"] == 1).mean()),
        "val_samples": len(val_df),
        "val_pos": int((val_df["label"] == 1).sum()),
        "val_neg": int((val_df["label"] == 0).sum()),
        "val_pos_rate": float((val_df["label"] == 1).mean()),
        "test_samples": len(test_df),
        "test_pos": int((test_df["label"] == 1).sum()),
        "test_neg": int((test_df["label"] == 0).sum()),
        "test_pos_rate": float((test_df["label"] == 1).mean()),
        "test_pmids": test_df["pmid"].astype(str).tolist(),
    }

    with open(splits_dir / "stage4_split_info.json", "w", encoding="utf-8") as f:
        json.dump(split_info, f, indent=2)

    logger.info(
        f"Stage 4 Splits Saved: Sub-Train={len(sub_train)} (Pos={split_info['sub_train_pos']}, Neg={split_info['sub_train_neg']}), "
        f"Val={len(val_df)} (Pos={split_info['val_pos']}, Neg={split_info['val_neg']}), "
        f"Test={len(test_df)} (Pos={split_info['test_pos']}, Neg={split_info['test_neg']})"
    )

    return {"train": sub_train, "val": val_df, "test": test_df, "info": split_info}


class DualInputStage4Dataset(Dataset):
    """
    PyTorch Dataset for Title + Abstract dual-sequence input:
    [CLS] Title [SEP] Abstract [SEP]
    """

    def __init__(
        self,
        titles: List[str],
        abstracts: List[str],
        labels: List[int],
        tokenizer,
        max_length: int = 384,
    ):
        self.titles = titles
        self.abstracts = abstracts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        title = clean_clinical_text(str(self.titles[idx] or ""))
        abstract = clean_clinical_text(str(self.abstracts[idx] or ""))
        label = self.labels[idx]

        enc = self.tokenizer(
            text=title,
            text_pair=abstract,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        item = {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "label": torch.tensor(label, dtype=torch.float32),
        }
        if "token_type_ids" in enc:
            item["token_type_ids"] = enc["token_type_ids"].squeeze(0)
        return item


if __name__ == "__main__":
    create_and_save_stage4_splits()
