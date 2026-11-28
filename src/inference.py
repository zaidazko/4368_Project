"""Helper functions used by the spam detection app."""

from pathlib import Path

import joblib

from src.preprocess import clean_text


# These files come from the models trained by the group.
MODEL_FILES = {
    "Naive Bayes": "naive_bayes.joblib",
    "SVM": "svm.joblib",
    "Logistic Regression": "logistic_regression.joblib",
}


def available_models(artifacts_dir: Path) -> dict[str, Path]:
    """Only list models that have been saved to the project folder."""
    return {
        name: artifacts_dir / filename
        for name, filename in MODEL_FILES.items()
        if (artifacts_dir / filename).is_file()
    }


def load_files(model_path: str, vectorizer_path: str):
    """Load the selected trained model and shared TF-IDF vectorizer."""
    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)
    return model, vectorizer


def classify_email(email_text: str, model, vectorizer) -> int:
    """Predict 1 for spam or 0 for not spam using the saved model."""
    # Match the preprocessing used to train every model.
    cleaned_text = clean_text(email_text)
    if not cleaned_text:
        raise ValueError("Please enter an email containing some words or numbers.")

    features = vectorizer.transform([cleaned_text])
    return int(model.predict(features)[0])
