
"""Task 2 - Support Vector Machine spam classifier."""

import json
from pathlib import Path

import joblib
import numpy as np
from scipy import sparse
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"


def load_features():
    X_train = sparse.load_npz(ARTIFACTS_DIR / "X_train.npz")
    X_test = sparse.load_npz(ARTIFACTS_DIR / "X_test.npz")

    y_train = np.load(ARTIFACTS_DIR / "y_train.npy")
    y_test = np.load(ARTIFACTS_DIR / "y_test.npy")

    return X_train, X_test, y_train, y_test


def evaluate(y_true, y_pred):
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(
            y_true, y_pred, pos_label=1, zero_division=0
        )),
        "recall": float(recall_score(
            y_true, y_pred, pos_label=1, zero_division=0
        )),
        "f1": float(f1_score(
            y_true, y_pred, pos_label=1, zero_division=0
        )),
        "confusion_matrix": confusion_matrix(
            y_true, y_pred, labels=[0, 1]
        ).tolist(),
        "positive_class": "spam (label=1)",
    }


def run():
    print("Loading TF-IDF features...")

    X_train, X_test, y_train, y_test = load_features()

    print("Training data shape:", X_train.shape)
    print("Testing data shape:", X_test.shape)

    # Initialize and train SVM
    print("\nTraining SVM model...")

    model = LinearSVC(
        C=1.0,
        random_state=42,
        max_iter=10000
    )

    model.fit(X_train, y_train)

    # Predict test labels
    y_pred = model.predict(X_test)

    # Evaluate model
    metrics = evaluate(y_test, y_pred)

    # Save trained model
    model_path = ARTIFACTS_DIR / "svm.joblib"
    joblib.dump(model, model_path)

    # Save evaluation metrics
    metrics_path = ARTIFACTS_DIR / "metrics_svm.json"

    metrics_path.write_text(
        json.dumps(metrics, indent=2) + "\n",
        encoding="utf-8"
    )

    print("\n===== SVM Test Results =====")
    print(f"Accuracy : {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall   : {metrics['recall']:.4f}")
    print(f"F1-score : {metrics['f1']:.4f}")

    print("\nConfusion Matrix:")
    print(np.array(metrics["confusion_matrix"]))

    print("\nModel saved to artifacts/svm.joblib")
    print("Metrics saved to artifacts/metrics_svm.json")


if __name__ == "__main__":
    run()
