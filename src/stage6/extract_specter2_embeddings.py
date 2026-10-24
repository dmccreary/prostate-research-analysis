"""
extract_specter2_embeddings.py - Extract and cache frozen SPECTER2 document embeddings.

Model: allenai/specter2_base (frozen, no fine-tuning)
Dataset: Stage 5 partitions (train N=335, val N=84, test N=105)
Input: Title + sep_token + Abstract
Output: 768-dimensional [CLS] embeddings cached to data/embeddings/specter2/
"""

import json
import logging
import os
from pathlib import Path
import random
import sys
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from transformers import AutoModel, AutoTokenizer

# Safe stdout on Windows
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Add src to path
BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(BASE_DIR / "src"))
from preprocessing import clean_clinical_text

# Paths
SPLITS_DIR = BASE_DIR / "data" / "splits" / "stage5"
EMB_DIR = BASE_DIR / "data" / "embeddings" / "specter2"
LOGS_DIR = BASE_DIR / "logs"

EMB_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOGS_DIR / "specter2_embedding_extraction.log", encoding="utf-8", mode="w"),
    ],
)
logger = logging.getLogger(__name__)

MODEL_ID = "allenai/specter2_base"
MAX_LENGTH = 512
BATCH_SIZE = 16
SEED = 42


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def extract_embeddings_for_df(df: pd.DataFrame, tokenizer, model, device: torch.device) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Extracts 768-d CLS embeddings for title + sep_token + abstract."""
    titles = [clean_clinical_text(str(t or "")) for t in df["title"]]
    abstracts = [clean_clinical_text(str(a or "")) for a in df["abstract"]]
    labels = df["label"].to_numpy().astype(int)
    pmids = [str(p).strip().replace(".0", "") for p in df["pmid"]]

    # Format text: title + sep_token + abstract
    input_texts = [f"{t} {tokenizer.sep_token} {a}".strip() for t, a in zip(titles, abstracts)]

    all_embeddings = []
    n_samples = len(input_texts)

    logger.info(f"Extracting SPECTER2 embeddings for {n_samples} articles (batch_size={BATCH_SIZE})...")

    with torch.no_grad():
        for i in range(0, n_samples, BATCH_SIZE):
            batch_texts = input_texts[i : i + BATCH_SIZE]
            inputs = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=MAX_LENGTH,
                return_tensors="pt",
            ).to(device)

            outputs = model(**inputs)
            # CLS token embedding
            cls_emb = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            all_embeddings.append(cls_emb)

    embeddings = np.vstack(all_embeddings).astype(np.float32)
    logger.info(f"Extracted embeddings shape: {embeddings.shape}")
    return embeddings, labels, pmids


def main():
    set_seed()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    train_path = SPLITS_DIR / "train.csv"
    val_path = SPLITS_DIR / "val.csv"
    test_path = SPLITS_DIR / "test.csv"

    if not train_path.exists() or not val_path.exists() or not test_path.exists():
        raise FileNotFoundError(f"Missing Stage 5 splits in {SPLITS_DIR}")

    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    test_df = pd.read_csv(test_path)

    logger.info(f"Loading SPECTER2 model and tokenizer from {MODEL_ID}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModel.from_pretrained(MODEL_ID).to(device)
    model.eval()  # Freeze weights completely

    logger.info("Extracting embeddings for Sub-Train set...")
    X_train, y_train, pmids_train = extract_embeddings_for_df(train_df, tokenizer, model, device)

    logger.info("Extracting embeddings for Validation set...")
    X_val, y_val, pmids_val = extract_embeddings_for_df(val_df, tokenizer, model, device)

    logger.info("Extracting embeddings for Held-Out Test set...")
    X_test, y_test, pmids_test = extract_embeddings_for_df(test_df, tokenizer, model, device)

    # Save npz files
    np.savez_compressed(EMB_DIR / "specter2_train.npz", embeddings=X_train, labels=y_train, pmids=pmids_train)
    np.savez_compressed(EMB_DIR / "specter2_val.npz", embeddings=X_val, labels=y_val, pmids=pmids_val)
    np.savez_compressed(EMB_DIR / "specter2_test.npz", embeddings=X_test, labels=y_test, pmids=pmids_test)

    metadata = {
        "model_id": MODEL_ID,
        "embedding_dim": int(X_train.shape[1]),
        "max_length": MAX_LENGTH,
        "train_samples": int(X_train.shape[0]),
        "val_samples": int(X_val.shape[0]),
        "test_samples": int(X_test.shape[0]),
        "train_pos": int((y_train == 1).sum()),
        "val_pos": int((y_val == 1).sum()),
        "test_pos": int((y_test == 1).sum()),
        "seed": SEED,
        "device": str(device),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    with open(EMB_DIR / "embedding_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"SPECTER2 embeddings successfully extracted and saved to {EMB_DIR}!")


if __name__ == "__main__":
    main()
