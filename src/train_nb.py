"""Train Multinomial Naive Bayes and save model + metrics."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
from scipy import sparse
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.naive_bayes import MultinomialNB

from src.features import (
    ARTIFACTS_DIR,
    X_TEST_PATH,
    X_TRAIN_PATH,
    Y_TEST_PATH,
    Y_TRAIN_PATH,
)
from src.preprocess import PROJECT_ROOT

MODEL_PATH = ARTIFACTS_DIR / "naive_bayes.joblib"
METRICS_PATH = ARTIFACTS_DIR / "metrics_nb.json"


def load_features() -> tuple[sparse.spmatrix, sparse.spmatrix, np.ndarray, np.ndarray]:
    for path in (X_TRAIN_PATH, X_TEST_PATH, Y_TRAIN_PATH, Y_TEST_PATH):
        if not path.exists():
            raise FileNotFoundError(
                f"Missing {path.name}. Run `python -m src.features` first."
            )
    X_train = sparse.load_npz(X_TRAIN_PATH)
    X_test = sparse.load_npz(X_TEST_PATH)
    y_train = np.load(Y_TRAIN_PATH)
    y_test = np.load(Y_TEST_PATH)
    return X_train, X_test, y_train, y_test


def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Spam-focused metrics (positive class label = 1)."""
    cm = confusion_matrix(y_true, y_pred).tolist()
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "confusion_matrix": cm,
        "positive_class": "spam (label=1)",
    }


def run() -> None:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    X_train, X_test, y_train, y_test = load_features()

    model = MultinomialNB()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    metrics = evaluate(y_test, y_pred)
    joblib.dump(model, MODEL_PATH)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2) + "\n")

    print("Naive Bayes test metrics (pos_label=spam/1)")
    print(f"  accuracy : {metrics['accuracy']:.4f}")
    print(f"  precision: {metrics['precision']:.4f}")
    print(f"  recall   : {metrics['recall']:.4f}")
    print(f"  f1       : {metrics['f1']:.4f}")
    print(f"  confusion_matrix: {metrics['confusion_matrix']}")
    print(f"Wrote {MODEL_PATH.relative_to(PROJECT_ROOT)}")
    print(f"Wrote {METRICS_PATH.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    run()
