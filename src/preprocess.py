"""Load, clean, and split the spam/ham email dataset."""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = PROJECT_ROOT / "data" / "raw" / "spam_or_not_spam.csv"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
CLEANED_PATH = PROCESSED_DIR / "cleaned.csv"
TRAIN_PATH = PROCESSED_DIR / "train.csv"
TEST_PATH = PROCESSED_DIR / "test.csv"

RANDOM_STATE = 42
TEST_SIZE = 0.2


def clean_text(text: str) -> str:
    """Lowercase, strip, keep alphanumeric tokens, collapse whitespace."""
    text = str(text).lower().strip()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def load_and_clean(raw_path: Path = RAW_PATH) -> pd.DataFrame:
    """Load the raw CSV, drop empty emails, and normalize text."""
    df = pd.read_csv(raw_path)
    if "email" not in df.columns or "label" not in df.columns:
        raise ValueError("Expected columns: email, label")

    df = df.dropna(subset=["email"]).copy()
    df["email"] = df["email"].astype(str).map(clean_text)
    df = df[df["email"].str.len() > 0].copy()
    df["label"] = df["label"].astype(int)
    df = df.reset_index(drop=True)
    return df


def make_split(
    df: pd.DataFrame,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Stratified 80/20 train/test split on the spam label."""
    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=df["label"],
    )
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)


def run() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    cleaned = load_and_clean()
    cleaned.to_csv(CLEANED_PATH, index=False)

    train_df, test_df = make_split(cleaned)
    train_df.to_csv(TRAIN_PATH, index=False)
    test_df.to_csv(TEST_PATH, index=False)

    print(f"Cleaned rows: {len(cleaned)} (from raw → drop null/empty emails)")
    print(f"Label counts (cleaned):\n{cleaned['label'].value_counts().to_string()}")
    print(f"Train: {len(train_df)}  Test: {len(test_df)}")
    print(f"Wrote {CLEANED_PATH.relative_to(PROJECT_ROOT)}")
    print(f"Wrote {TRAIN_PATH.relative_to(PROJECT_ROOT)}")
    print(f"Wrote {TEST_PATH.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    run()
