"""
dataset.py - PyTorch datasets, clinical vocabulary, and data loaders for Stage 2.

Ensures:
1. Zero test set leakage: Vocabulary and TF-IDF are fitted strictly on the training partition.
2. Clinical terminology preservation: Tokenization preserves acronyms, dosages, and hyphenated terms.
3. Dynamic padding with sequence lengths for RNN/LSTM/Attention models.
4. Stratified sub-train / validation splitting from data/splits/train.csv.
5. Exact reuse of held-out test split (data/splits/test.csv).
"""

import logging
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

# Add parent directory to path to import Stage 1 utilities
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from preprocessing import clean_clinical_text

logger = logging.getLogger(__name__)

# Token pattern preserving hyphenated medical terms (e.g. high-risk, t1b-2b, 72gy)
TOKEN_PATTERN = re.compile(r"(?u)\b[a-zA-Z0-9_\-\/]+\b")


class ClinicalVocabulary:
    """
    Vocabulary mapping words to integer token IDs.
    Fitted strictly on training data.
    """

    def __init__(self, min_freq: int = 2):
        self.min_freq = min_freq
        self.pad_token = "<PAD>"
        self.unk_token = "<UNK>"
        self.pad_idx = 0
        self.unk_idx = 1
        self.word2idx = {self.pad_token: self.pad_idx, self.unk_token: self.unk_idx}
        self.idx2word = {self.pad_idx: self.pad_token, self.unk_idx: self.unk_token}
        self.word_counts = {}

    def fit(self, texts: List[str]):
        for text in texts:
            cleaned = clean_clinical_text(text)
            tokens = [t.lower() for t in TOKEN_PATTERN.findall(cleaned)]
            for tok in tokens:
                self.word_counts[tok] = self.word_counts.get(tok, 0) + 1

        for word, count in sorted(self.word_counts.items(), key=lambda x: -x[1]):
            if count >= self.min_freq and word not in self.word2idx:
                idx = len(self.word2idx)
                self.word2idx[word] = idx
                self.idx2word[idx] = word

        logger.info(
            "Built vocabulary: %d total unique tokens -> %d tokens with min_freq>=%d",
            len(self.word_counts),
            len(self.word2idx),
            self.min_freq,
        )
        return self

    def encode(self, text: str, max_len: int = 384) -> Tuple[List[int], int]:
        cleaned = clean_clinical_text(text)
        tokens = [t.lower() for t in TOKEN_PATTERN.findall(cleaned)]
        token_ids = [self.word2idx.get(t, self.unk_idx) for t in tokens[:max_len]]
        length = len(token_ids)
        return token_ids, length

    def __len__(self):
        return len(self.word2idx)


class AbstractSequenceDataset(Dataset):
    """
    Dataset returning token IDs, sequence length, and binary label for recurrent architectures.
    """

    def __init__(self, texts: List[str], labels: List[int], vocab: ClinicalVocabulary, max_len: int = 384):
        self.texts = texts
        self.labels = labels
        self.vocab = vocab
        self.max_len = max_len

        self.encoded_data = []
        for text, label in zip(texts, labels):
            token_ids, length = vocab.encode(text, max_len=max_len)
            self.encoded_data.append((token_ids, length, int(label)))

    def __len__(self):
        return len(self.encoded_data)

    def __getitem__(self, idx):
        return self.encoded_data[idx]


def collate_sequences(batch, pad_idx: int = 0):
    """
    Collate function that dynamically pads sequences to the maximum length in the batch.
    """
    token_ids_list, lengths_list, labels_list = zip(*batch)
    max_len = max(max(lengths_list), 1)

    batch_size = len(batch)
    padded_tokens = torch.full((batch_size, max_len), fill_value=pad_idx, dtype=torch.long)

    for i, seq in enumerate(token_ids_list):
        if len(seq) > 0:
            padded_tokens[i, : len(seq)] = torch.tensor(seq, dtype=torch.long)

    lengths = torch.tensor(lengths_list, dtype=torch.long)
    labels = torch.tensor(labels_list, dtype=torch.float32)

    return padded_tokens, lengths, labels


class TFIDFDataset(Dataset):
    """
    Dataset returning dense TF-IDF vectors and binary label for MLP baseline.
    """

    def __init__(self, X_tfidf, labels: List[int]):
        if hasattr(X_tfidf, "toarray"):
            self.X = torch.tensor(X_tfidf.toarray(), dtype=torch.float32)
        else:
            self.X = torch.tensor(X_tfidf, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.X[idx], self.labels[idx]


def get_neural_data_splits(
    base_dir: Optional[str] = None,
    val_ratio: float = 0.20,
    random_state: int = 42,
) -> Dict[str, pd.DataFrame]:
    """
    Load train.csv and test.csv from Stage 1.
    Splits train.csv into sub-train and validation while preserving exact stratification.
    The test.csv split remains strictly held-out and untouched.
    """
    if base_dir is None:
        base_dir = Path(__file__).resolve().parent.parent.parent
    else:
        base_dir = Path(base_dir)

    splits_dir = base_dir / "data" / "splits"
    train_path = splits_dir / "train.csv"
    test_path = splits_dir / "test.csv"

    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError(f"Missing Stage 1 split files in {splits_dir}.")

    full_train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    # Partition full_train_df into sub_train_df and val_df
    sub_train_df, val_df = train_test_split(
        full_train_df,
        test_size=val_ratio,
        stratify=full_train_df["label"],
        random_state=random_state,
    )
    sub_train_df = sub_train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)

    logger.info(
        "Data partitioned: Sub-Train=%d (Pos=%d, Neg=%d), Val=%d (Pos=%d, Neg=%d), Held-Out Test=%d (Pos=%d, Neg=%d)",
        len(sub_train_df),
        (sub_train_df["label"] == 1).sum(),
        (sub_train_df["label"] == 0).sum(),
        len(val_df),
        (val_df["label"] == 1).sum(),
        (val_df["label"] == 0).sum(),
        len(test_df),
        (test_df["label"] == 1).sum(),
        (test_df["label"] == 0).sum(),
    )

    return {
        "full_train": full_train_df,
        "train": sub_train_df,
        "val": val_df,
        "test": test_df,
    }


def prepare_tfidf_data(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    max_features: int = 3000,
    ngram_range: Tuple[int, int] = (1, 2),
) -> Tuple[TfidfVectorizer, DataLoader, DataLoader, DataLoader]:
    """
    Fits TF-IDF strictly on train_df and returns DataLoaders for MLP.
    """
    vec = TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        min_df=2,
        sublinear_tf=True,
        stop_words="english",
    )
    train_clean = [clean_clinical_text(t) for t in train_df["abstract"]]
    val_clean = [clean_clinical_text(t) for t in val_df["abstract"]]
    test_clean = [clean_clinical_text(t) for t in test_df["abstract"]]

    X_train = vec.fit_transform(train_clean)
    X_val = vec.transform(val_clean)
    X_test = vec.transform(test_clean)

    train_ds = TFIDFDataset(X_train, train_df["label"].tolist())
    val_ds = TFIDFDataset(X_val, val_df["label"].tolist())
    test_ds = TFIDFDataset(X_test, test_df["label"].tolist())

    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False)

    return vec, train_loader, val_loader, test_loader


def prepare_sequence_data(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    min_freq: int = 2,
    max_len: int = 384,
    batch_size: int = 16,
) -> Tuple[ClinicalVocabulary, DataLoader, DataLoader, DataLoader]:
    """
    Builds ClinicalVocabulary strictly on train_df and returns DataLoaders for RNNs/LSTMs.
    """
    vocab = ClinicalVocabulary(min_freq=min_freq)
    vocab.fit(train_df["abstract"].tolist())

    train_ds = AbstractSequenceDataset(train_df["abstract"].tolist(), train_df["label"].tolist(), vocab, max_len=max_len)
    val_ds = AbstractSequenceDataset(val_df["abstract"].tolist(), val_df["label"].tolist(), vocab, max_len=max_len)
    test_ds = AbstractSequenceDataset(test_df["abstract"].tolist(), test_df["label"].tolist(), vocab, max_len=max_len)

    collate_fn = lambda b: collate_sequences(b, pad_idx=vocab.pad_idx)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=batch_size * 2, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_ds, batch_size=batch_size * 2, shuffle=False, collate_fn=collate_fn)

    return vocab, train_loader, val_loader, test_loader


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    splits = get_neural_data_splits()
    print("Testing TF-IDF Data Loader...")
    vec, tr_l, va_l, te_l = prepare_tfidf_data(splits["train"], splits["val"], splits["test"])
    for x, y in tr_l:
        print("TF-IDF batch X shape:", x.shape, "y shape:", y.shape)
        break

    print("Testing Sequence Data Loader...")
    vocab, tr_s, va_s, te_s = prepare_sequence_data(splits["train"], splits["val"], splits["test"])
    for x, lengths, y in tr_s:
        print("Sequence batch X shape:", x.shape, "lengths shape:", lengths.shape, "y shape:", y.shape)
        break
