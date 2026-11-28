"""Tests for Task 4's email prediction logic."""

import tempfile
import unittest
from pathlib import Path

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

from src.inference import available_models, classify_email, load_files


class EmailClassifierTests(unittest.TestCase):
    def setUp(self):
        # Train a tiny sample model so the tests don't depend on the team's data.
        emails = [
            "claim cash prize today",
            "win free money now",
            "send your account details",
            "hello can we meet tomorrow",
            "please review our class assignment",
            "thanks for your helpful notes",
        ]
        labels = [1, 1, 1, 0, 0, 0]

        self.vectorizer = TfidfVectorizer()
        features = self.vectorizer.fit_transform(emails)
        self.model = MultinomialNB().fit(features, labels)

    def test_prediction_returns_a_label(self):
        prediction = classify_email("Win free money NOW!", self.model, self.vectorizer)
        self.assertIn(prediction, (0, 1))

    def test_empty_messages_are_rejected(self):
        for message in ("", "   ", "!?,."):
            with self.subTest(message=message):
                with self.assertRaises(ValueError):
                    classify_email(message, self.model, self.vectorizer)

    def test_saved_models_can_be_loaded(self):
        with tempfile.TemporaryDirectory() as folder_name:
            folder = Path(folder_name)
            model_path = folder / "naive_bayes.joblib"
            vectorizer_path = folder / "tfidf_vectorizer.joblib"
            joblib.dump(self.model, model_path)
            joblib.dump(self.vectorizer, vectorizer_path)

            self.assertEqual(list(available_models(folder)), ["Naive Bayes"])
            model, vectorizer = load_files(str(model_path), str(vectorizer_path))
            self.assertIn(classify_email("hello there", model, vectorizer), (0, 1))

    def test_project_model_and_vectorizer(self):
        artifacts = Path(__file__).resolve().parents[1] / "artifacts"
        model_path = artifacts / "naive_bayes.joblib"
        vectorizer_path = artifacts / "tfidf_vectorizer.joblib"

        if not model_path.is_file() or not vectorizer_path.is_file():
            self.skipTest("Project model artifacts are not available here.")

        model, vectorizer = load_files(str(model_path), str(vectorizer_path))
        prediction = classify_email("Can we meet after class?", model, vectorizer)
        self.assertIn(prediction, (0, 1))


if __name__ == "__main__":
    unittest.main()
