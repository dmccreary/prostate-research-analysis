"""
models_classical_stage3.py - Classical ML architectures enriched with Title+Abstract n-grams and domain rules.

Implements:
1. Calibrated Linear SVM (Title + Abstract, +/- Rule features)
2. Complement Naive Bayes (Title + Abstract, +/- Rule features)
"""

import logging
from pathlib import Path
from typing import Dict, Optional, Tuple, Union
import numpy as np
import scipy.sparse as sp
from sklearn.calibration import CalibratedClassifierCV
from sklearn.naive_bayes import ComplementNB
from sklearn.svm import LinearSVC
import joblib

logger = logging.getLogger(__name__)


def train_calibrated_svm(
    X_train: Union[np.ndarray, sp.spmatrix],
    y_train: np.ndarray,
    random_state: int = 42,
    C: float = 0.5,
) -> CalibratedClassifierCV:
    """
    Trains LinearSVC with 5-fold Platt sigmoid calibration.
    """
    base_svc = LinearSVC(C=C, random_state=random_state, max_iter=2000, class_weight="balanced")
    calibrated_svm = CalibratedClassifierCV(estimator=base_svc, method="sigmoid", cv=5)
    calibrated_svm.fit(X_train, y_train)
    return calibrated_svm


def train_complement_nb(
    X_train: Union[np.ndarray, sp.spmatrix],
    y_train: np.ndarray,
    alpha: float = 1.0,
) -> ComplementNB:
    """
    Trains Complement Naive Bayes designed for imbalanced text corpora.
    """
    cnb = ComplementNB(alpha=alpha, norm=True)
    cnb.fit(X_train, y_train)
    return cnb


def combine_tfidf_and_rules(X_tfidf: sp.spmatrix, rule_vectors: np.ndarray) -> sp.csr_matrix:
    """
    Stacks sparse TF-IDF text features with dense domain rule features.
    """
    rule_sparse = sp.csr_matrix(rule_vectors)
    return sp.hstack([X_tfidf, rule_sparse], format="csr")
