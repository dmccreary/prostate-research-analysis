"""
models.py - Classical Machine Learning baselines and hyperparameter search grids.

Models implemented:
1. DummyClassifier: Baseline (most_frequent, stratified)
2. LogisticRegression: L2-regularized with balanced class weights
3. LinearSVC: Support Vector Classifier calibrated via CalibratedClassifierCV (provides proper probabilities)
4. MultinomialNB: Multinomial Naive Bayes
5. ComplementNB: Complement Naive Bayes (tailored for imbalanced text classification)
6. SGDClassifier: Linear model with modified_huber loss (produces calibrated probabilities)
7. RandomForestClassifier: Non-linear tree ensemble baseline for comparison
"""

import logging
import sys
from typing import Any, Dict, List, Optional, Tuple

from sklearn.calibration import CalibratedClassifierCV
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.naive_bayes import ComplementNB, MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from preprocessing import ClinicalTextCleaner, get_default_tfidf_vectorizer

# Safe standard output encoding on Windows terminals
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

logger = logging.getLogger(__name__)


def build_pipeline(
    model_name: str,
    random_state: int = 42,
    representation_type: str = "word",
    custom_clf_params: Optional[Dict[str, Any]] = None,
) -> Pipeline:
    """
    Construct an end-to-end Pipeline: Cleaner -> TfidfVectorizer -> Classifier.
    """
    params = custom_clf_params or {}

    cleaner = ClinicalTextCleaner()
    tfidf = get_default_tfidf_vectorizer(representation_type=representation_type)

    if model_name == "dummy_most_frequent":
        clf = DummyClassifier(strategy="most_frequent", random_state=random_state)

    elif model_name == "dummy_stratified":
        clf = DummyClassifier(strategy="stratified", random_state=random_state)

    elif model_name == "logistic_regression":
        clf = LogisticRegression(
            solver="liblinear",
            class_weight=params.get("class_weight", "balanced"),
            C=params.get("C", 1.0),
            max_iter=1000,
            random_state=random_state,
        )

    elif model_name == "linear_svc":
        base_svc = LinearSVC(
            C=params.get("C", 1.0),
            class_weight=params.get("class_weight", "balanced"),
            random_state=random_state,
            max_iter=5000,
            dual=True,
        )
        # Calibrate LinearSVC using 5-fold CV to provide genuine probability estimates
        # for ROC and PR curves without leaking the test set
        clf = CalibratedClassifierCV(estimator=base_svc, cv=5)

    elif model_name == "multinomial_nb":
        clf = MultinomialNB(alpha=params.get("alpha", 1.0))

    elif model_name == "complement_nb":
        # ComplementNB is specifically designed for imbalanced text collections
        clf = ComplementNB(alpha=params.get("alpha", 1.0), norm=True)

    elif model_name == "sgd_classifier":
        # modified_huber loss provides smooth probability estimates P(y|x)
        clf = SGDClassifier(
            loss="modified_huber",
            penalty=params.get("penalty", "elasticnet"),
            alpha=params.get("alpha", 1e-4),
            l1_ratio=params.get("l1_ratio", 0.15),
            class_weight=params.get("class_weight", "balanced"),
            max_iter=2000,
            random_state=random_state,
        )

    elif model_name == "random_forest":
        clf = RandomForestClassifier(
            n_estimators=params.get("n_estimators", 200),
            max_depth=params.get("max_depth", None),
            class_weight=params.get("class_weight", "balanced_subsample"),
            random_state=random_state,
            n_jobs=-1,
        )

    else:
        raise ValueError(f"Unknown model_name: {model_name}")

    pipeline = Pipeline([
        ("cleaner", cleaner),
        ("tfidf", tfidf),
        ("clf", clf),
    ])

    return pipeline


def get_tuning_param_grids() -> Dict[str, Dict[str, List[Any]]]:
    """
    Returns search grids for hyperparameter optimization using Stratified 5-Fold CV.
    Covers feature representation and classifier parameters.
    """
    grids = {
        "logistic_regression": {
            "tfidf__ngram_range": [(1, 1), (1, 2)],
            "tfidf__min_df": [1, 2, 3],
            "tfidf__sublinear_tf": [True, False],
            "clf__C": [0.1, 0.5, 1.0, 2.0, 5.0, 10.0],
            "clf__class_weight": ["balanced", None],
        },
        "linear_svc": {
            "tfidf__ngram_range": [(1, 1), (1, 2)],
            "tfidf__min_df": [1, 2, 3],
            "clf__estimator__C": [0.1, 0.5, 1.0, 2.0, 5.0],
            "clf__estimator__class_weight": ["balanced", None],
        },
        "multinomial_nb": {
            "tfidf__ngram_range": [(1, 1), (1, 2)],
            "tfidf__min_df": [1, 2],
            "clf__alpha": [0.01, 0.1, 0.5, 1.0, 2.0],
        },
        "complement_nb": {
            "tfidf__ngram_range": [(1, 1), (1, 2)],
            "tfidf__min_df": [1, 2],
            "clf__alpha": [0.01, 0.1, 0.5, 1.0, 2.0],
            "clf__norm": [True, False],
        },
        "sgd_classifier": {
            "tfidf__ngram_range": [(1, 1), (1, 2)],
            "clf__alpha": [1e-4, 5e-4, 1e-3, 5e-3],
            "clf__penalty": ["l2", "elasticnet"],
            "clf__class_weight": ["balanced", None],
        },
        "random_forest": {
            "tfidf__ngram_range": [(1, 1), (1, 2)],
            "clf__n_estimators": [100, 200],
            "clf__max_depth": [None, 15, 30],
            "clf__class_weight": ["balanced", "balanced_subsample"],
        },
    }
    return grids


def get_model_descriptions() -> Dict[str, Dict[str, str]]:
    """Return scientific justification and architectural description for each baseline."""
    return {
        "dummy_most_frequent": {
            "display_name": "Dummy (Most Frequent)",
            "family": "Baseline",
            "justification": "Trivial baseline predicting the majority class (0 / Not Relevant). Establishes minimum accuracy floor that models must significantly beat.",
        },
        "logistic_regression": {
            "display_name": "Logistic Regression",
            "family": "Linear",
            "justification": "Primary linear text classification baseline. Highly effective for high-dimensional sparse TF-IDF vectors, interpretable weights, and well-calibrated posterior probabilities.",
        },
        "linear_svc": {
            "display_name": "Linear SVM (Calibrated)",
            "family": "Support Vector Machine",
            "justification": "Maximizes margin separation in high-dimensional TF-IDF space. Calibrated via 5-fold cross-validation to produce continuous calibrated probabilities and ranking scores.",
        },
        "multinomial_nb": {
            "display_name": "Multinomial Naive Bayes",
            "family": "Probabilistic",
            "justification": "Generative probabilistic baseline computing posterior class probabilities under word conditional independence assumption.",
        },
        "complement_nb": {
            "display_name": "Complement Naive Bayes",
            "family": "Probabilistic",
            "justification": "Specifically engineered for imbalanced text classification by estimating parameters from the complement of each class, correcting standard Multinomial NB parameter bias.",
        },
        "sgd_classifier": {
            "display_name": "SGD Classifier (Modified Huber)",
            "family": "Linear",
            "justification": "Stochastic gradient descent linear classifier with modified Huber loss, resilient to outliers while outputting calibrated probabilities.",
        },
        "random_forest": {
            "display_name": "Random Forest",
            "family": "Tree Ensemble",
            "justification": "Non-linear decision tree ensemble evaluated to assess whether non-linear feature interactions provide benefits over linear sparse text hyperplanes.",
        },
    }


if __name__ == "__main__":
    print("Testing pipeline creation...")
    for model_name in [
        "dummy_most_frequent",
        "logistic_regression",
        "linear_svc",
        "multinomial_nb",
        "complement_nb",
        "sgd_classifier",
        "random_forest",
    ]:
        pipe = build_pipeline(model_name)
        print(f"✓ Pipeline for {model_name} initialized successfully.")
