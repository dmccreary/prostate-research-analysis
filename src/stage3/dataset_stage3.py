"""
dataset_stage3.py - Data loaders, dual-input representation, and feature engineering for Stage 3.

Supports:
1. Exact zero-leakage partitions: Sub-Train (N=230), Val (N=58), Held-Out Test (N=73).
2. Dual-sequence input for PubMedBERT: text_a=Title, text_pair=Abstract.
3. Enriched Title + Abstract TF-IDF representations for Linear SVM and Complement NB.
4. Pre-filter flag and clinical rule feature vector extraction.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from preprocessing import clean_clinical_text
from stage3.rule_engine import evaluate_pre_filter, get_rule_feature_vector

logger = logging.getLogger(__name__)


def get_stage3_splits(
    data_dir: Optional[Path] = None,
    val_ratio: float = 0.20,
    random_state: int = 42,
) -> Dict[str, pd.DataFrame]:
    """
    Loads train.csv and test.csv and splits train.csv into sub-train and val (80/20).
    Injects publication_types and pre-filter evaluation into all splits.
    """
    if data_dir is None:
        base_dir = Path(__file__).resolve().parent.parent.parent
        data_dir = base_dir / "data"
    else:
        data_dir = Path(data_dir)

    train_path = data_dir / "splits" / "train.csv"
    test_path = data_dir / "splits" / "test.csv"
    pt_path = data_dir / "metadata" / "pubmed_publication_types.json"

    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError(f"Missing train/test splits in {data_dir / 'splits'}")

    full_train = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    # Sub-train and Validation split (identical to Stage 2)
    sub_train, val_df = train_test_split(
        full_train,
        test_size=val_ratio,
        stratify=full_train["label"],
        random_state=random_state,
    )
    sub_train = sub_train.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    # Load publication types
    pt_map = {}
    if pt_path.exists():
        with open(pt_path, "r", encoding="utf-8") as f:
            pt_map = json.load(f)

    # Enrich each split with clean text, pub_types, and pre-filter result
    def _enrich_df(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["clean_title"] = [clean_clinical_text(str(t or "")) for t in df["title"]]
        df["clean_abstract"] = [clean_clinical_text(str(a or "")) for a in df["abstract"]]
        # Combined text for TF-IDF: Title given 2x weight by repeating once
        df["combined_text"] = [
            f"{t} {t} {a}".strip() for t, a in zip(df["clean_title"], df["clean_abstract"])
        ]

        # Pub types
        pub_types_col = []
        is_excluded_col = []
        reason_col = []
        rule_vecs = []

        for _, row in df.iterrows():
            p = str(row["pmid"]).strip().replace(".0", "")
            types = pt_map.get(p, [])
            pub_types_col.append(types)

            is_ex, r = evaluate_pre_filter(row["clean_title"], row["clean_abstract"], types)
            is_excluded_col.append(is_ex)
            reason_col.append(r)

            vec = get_rule_feature_vector(row["clean_title"], row["clean_abstract"], types)
            rule_vecs.append(vec)

        df["pub_types"] = pub_types_col
        df["is_pre_filtered"] = is_excluded_col
        df["filter_reason"] = reason_col
        df["rule_vector"] = rule_vecs
        return df

    sub_train = _enrich_df(sub_train)
    val_df = _enrich_df(val_df)
    test_df = _enrich_df(test_df)

    logger.info(
        "Stage 3 Splits Prepared: Sub-Train=%d (Pos=%d), Val=%d (Pos=%d), Test=%d (Pos=%d)",
        len(sub_train),
        (sub_train["label"] == 1).sum(),
        len(val_df),
        (val_df["label"] == 1).sum(),
        len(test_df),
        (test_df["label"] == 1).sum(),
    )
    return {"train": sub_train, "val": val_df, "test": test_df}


class DualInputTransformerDataset(Dataset):
    """
    PyTorch Dataset for dual-sequence input: text_a=Title, text_pair=Abstract.
    Returns input_ids, attention_mask, token_type_ids, rule_vector, and binary label.
    """

    def __init__(
        self,
        titles: List[str],
        abstracts: List[str],
        labels: List[int],
        rule_vectors: List[np.ndarray],
        tokenizer,
        max_length: int = 384,
    ):
        self.titles = titles
        self.abstracts = abstracts
        self.labels = labels
        self.rule_vectors = rule_vectors
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        title = self.titles[idx]
        abstract = self.abstracts[idx]
        label = self.labels[idx]
        rule_vec = self.rule_vectors[idx]

        # Sentence-pair tokenization: [CLS] Title [SEP] Abstract [SEP]
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
            "rule_vector": torch.tensor(rule_vec, dtype=torch.float32),
        }
        if "token_type_ids" in enc:
            item["token_type_ids"] = enc["token_type_ids"].squeeze(0)
        return item


def prepare_stage3_tfidf(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    max_features: int = 3500,
    ngram_range: Tuple[int, int] = (1, 2),
) -> Tuple[TfidfVectorizer, np.ndarray, np.ndarray, np.ndarray]:
    """
    Fits TF-IDF vectorizer strictly on training combined text (Title + Abstract).
    Returns (vec, X_train, X_val, X_test).
    """
    vec = TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        min_df=2,
        sublinear_tf=True,
        stop_words="english",
    )
    X_train = vec.fit_transform(train_df["combined_text"])
    X_val = vec.transform(val_df["combined_text"])
    X_test = vec.transform(test_df["combined_text"])
    return vec, X_train, X_val, X_test
