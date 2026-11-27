# COSC 4368 — Spam Email Classifier

Team: Ali Shaikh, Zaid Azkoul, Yousif Husni, Rakib Khalil

Spam vs ham classification using TF-IDF features and classical ML models (Naive Bayes, SVM, Logistic Regression).

## Setup

```bash
pip install -r requirements.txt
```

## Task 1 pipeline (dataset → TF-IDF → Naive Bayes)

Run from the project root in order:

```bash
python3 -m src.preprocess   # clean data + stratified train/test split
python3 -m src.features     # fit TF-IDF on train only; save matrices
python3 -m src.train_nb     # train MultinomialNB; save model + metrics
```

### What gets written

| Path | Description |
|------|-------------|
| `data/raw/spam_or_not_spam.csv` | Original dataset |
| `data/processed/cleaned.csv` | Cleaned labeled emails |
| `data/processed/train.csv` / `test.csv` | Stratified 80/20 split (`random_state=42`) |
| `artifacts/tfidf_vectorizer.joblib` | Fitted `TfidfVectorizer` |
| `artifacts/X_train.npz` / `X_test.npz` | Sparse TF-IDF feature matrices |
| `artifacts/y_train.npy` / `y_test.npy` | Labels (`0`=ham, `1`=spam) |
| `artifacts/naive_bayes.joblib` | Trained Multinomial Naive Bayes |
| `artifacts/metrics_nb.json` | Test accuracy, precision, recall, F1, confusion matrix |

Labels: **`0` = not spam (ham)**, **`1` = spam**. Metrics for Naive Bayes use spam as the positive class.

## For teammates (Tasks 2–4)

### Task 2 (SVM) / Task 3 (Logistic Regression)

Reuse the **same** TF-IDF features and split — do not refit the vectorizer on the full dataset:

```python
import joblib
import numpy as np
from scipy import sparse
from sklearn.svm import LinearSVC  # or LogisticRegression

X_train = sparse.load_npz("artifacts/X_train.npz")
X_test = sparse.load_npz("artifacts/X_test.npz")
y_train = np.load("artifacts/y_train.npy")
y_test = np.load("artifacts/y_test.npy")

# Train your model on X_train, y_train; evaluate on X_test, y_test
# Save your model under artifacts/ (e.g. svm.joblib, logistic_regression.joblib)
```

### Task 4 (app / inference)

Load the best model plus the shared vectorizer. New email text must go through the same cleaning used in training (`src.preprocess.clean_text`), then the vectorizer:

```python
import joblib
from src.preprocess import clean_text

vectorizer = joblib.load("artifacts/tfidf_vectorizer.joblib")
model = joblib.load("artifacts/naive_bayes.joblib")  # or best model from Task 2/3

text = clean_text("paste email body here")
X = vectorizer.transform([text])
pred = model.predict(X)[0]  # 0 = ham, 1 = spam
```

## Project layout

```
data/raw/           # original CSV
data/processed/     # cleaned + train/test CSVs
artifacts/          # vectorizer, feature matrices, models, metrics
src/preprocess.py   # cleaning + split
src/features.py     # TF-IDF
src/train_nb.py     # Naive Bayes (Task 1)
```
