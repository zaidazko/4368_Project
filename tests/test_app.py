"""Check the actual Streamlit screen and button using Streamlit's test runner."""

import unittest
from pathlib import Path

from streamlit.testing.v1 import AppTest


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class AppTests(unittest.TestCase):
    def test_app_opens_and_predicts(self):
        # The team's Naive Bayes model and TF-IDF vectorizer are committed.
        app = AppTest.from_file(str(PROJECT_ROOT / "app.py"), default_timeout=30)
        app.run()
        self.assertEqual(len(app.exception), 0)
        self.assertGreaterEqual(len(app.selectbox), 1)

        app.text_area[0].set_value("Hi, can we review the project after class?").run()
        app.button[0].click().run()
        self.assertEqual(len(app.exception), 0)
        self.assertEqual(len(app.error), 0)
        self.assertEqual(len(app.success) + len(app.warning), 1)

    def test_empty_email_shows_message(self):
        app = AppTest.from_file(str(PROJECT_ROOT / "app.py"), default_timeout=30)
        app.run()
        self.assertEqual(len(app.exception), 0)
        app.button[0].click().run()
        self.assertEqual(len(app.exception), 0)
        self.assertEqual(len(app.warning), 1)
        self.assertEqual(len(app.error), 0)


if __name__ == "__main__":
    unittest.main()
