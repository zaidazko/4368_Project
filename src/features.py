"""Fit TF-IDF on training text and save shared feature matrices."""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer

from src.preprocess import PROJECT_ROOT, TEST_PATH, TRAIN_PATH

ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
VECTORIZER_PATH = ARTIFACTS_DIR / "tfidf_vectorizer.joblib"
X_TRAIN_PATH = ARTIFACTS_DIR / "X_train.npz"
X_TEST_PATH = ARTIFACTS_DIR / "X_test.npz"
Y_TRAIN_PATH = ARTIFACTS_DIR / "y_train.npy"
Y_TEST_PATH = ARTIFACTS_DIR / "y_test.npy"


def build_vectorizer() -> TfidfVectorizer:
    return TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
        max_features=10000,
    )


def run() -> None:
    if not TRAIN_PATH.exists() or not TEST_PATH.exists():
        raise FileNotFoundError(
            "Missing train/test CSVs. Run `python -m src.preprocess` first."
        )

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)

    vectorizer = build_vectorizer()
    X_train = vectorizer.fit_transform(train_df["email"])
    X_test = vectorizer.transform(test_df["email"])
    y_train = train_df["label"].to_numpy(dtype=np.int64)
    y_test = test_df["label"].to_numpy(dtype=np.int64)

    joblib.dump(vectorizer, VECTORIZER_PATH)
    sparse.save_npz(X_TRAIN_PATH, X_train)
    sparse.save_npz(X_TEST_PATH, X_test)
    np.save(Y_TRAIN_PATH, y_train)
    np.save(Y_TEST_PATH, y_test)

    print(f"Vocabulary size: {len(vectorizer.vocabulary_)}")
    print(f"X_train shape: {X_train.shape}  X_test shape: {X_test.shape}")
    print(f"Wrote {VECTORIZER_PATH.relative_to(PROJECT_ROOT)}")
    print(f"Wrote {X_TRAIN_PATH.relative_to(PROJECT_ROOT)}, {X_TEST_PATH.relative_to(PROJECT_ROOT)}")
    print(f"Wrote {Y_TRAIN_PATH.relative_to(PROJECT_ROOT)}, {Y_TEST_PATH.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    run()
