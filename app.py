"""Simple Streamlit app for our spam email classifier project."""

from pathlib import Path

import joblib
import streamlit as st

from src.preprocess import clean_text


PROJECT_DIR = Path(__file__).resolve().parent
ARTIFACTS_DIR = PROJECT_DIR / "artifacts"

# The other tasks save their trained models using these filenames.
MODEL_FILES = {
    "Naive Bayes": "naive_bayes.joblib",
    "SVM": "svm.joblib",
    "Logistic Regression": "logistic_regression.joblib",
}

EXAMPLES = {
    "Write my own": "",
    "Spam example": (
        "Congratulations! You won a cash prize. "
        "Reply with your account details to claim it today."
    ),
    "Normal email example": (
        "Hey team, can we meet tomorrow at 3 PM to review our project? "
        "I will send the notes after class."
    ),
}


@st.cache_resource
def load_model_and_vectorizer(model_path: str, vectorizer_path: str):
    """Load the files once instead of reloading them after every click."""
    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)
    return model, vectorizer


def classify_email(email_text: str, model, vectorizer) -> int:
    """Return 1 for spam or 0 for a normal email."""
    # Use the same text cleaning as the training data.
    cleaned_text = clean_text(email_text)
    if not cleaned_text:
        raise ValueError("Please enter an email containing some words or numbers.")

    features = vectorizer.transform([cleaned_text])
    return int(model.predict(features)[0])


st.set_page_config(page_title="Spam Email Classifier", page_icon="📧")
st.title("📧 Spam Email Classifier")
st.write("Paste an email below to see whether our machine learning model thinks it is spam.")

# Only show models that have actually been saved in the artifacts folder.
available_models = {
    name: ARTIFACTS_DIR / filename
    for name, filename in MODEL_FILES.items()
    if (ARTIFACTS_DIR / filename).exists()
}
vectorizer_path = ARTIFACTS_DIR / "tfidf_vectorizer.joblib"

if not vectorizer_path.exists():
    st.error("TF-IDF vectorizer is missing from the artifacts folder.")
    st.stop()

if not available_models:
    st.error("No trained models were found in the artifacts folder.")
    st.stop()

model_names = list(available_models)
default_index = model_names.index("SVM") if "SVM" in model_names else 0
selected_model = st.selectbox("Choose a model", model_names, index=default_index)

example = st.selectbox("Try an example (optional)", list(EXAMPLES))
email_text = st.text_area(
    "Email text",
    value=EXAMPLES[example],
    height=190,
    placeholder="Paste the email message here...",
)

if st.button("Analyze Email", type="primary"):
    if not email_text.strip():
        st.warning("Please enter an email first.")
    else:
        try:
            model, vectorizer = load_model_and_vectorizer(
                str(available_models[selected_model]),
                str(vectorizer_path),
            )
            prediction = classify_email(email_text, model, vectorizer)
        except (OSError, ValueError, ImportError) as error:
            st.error(f"Could not analyze the email: {error}")
        else:
            if prediction == 1:
                st.warning("Prediction: SPAM 🚫")
            elif prediction == 0:
                st.success("Prediction: NOT SPAM ✅")
            else:
                st.error("The model returned an unexpected result.")

with st.expander("How does the app work?"):
    st.write(
        "We clean the email text, turn it into numbers using the team's "
        "TF-IDF vectorizer, and send those numbers to the selected model. "
        "The model returns 1 for spam or 0 for not spam."
    )

st.caption(
    "Class project demo only. Predictions can be wrong, so do not rely "
    "on this app alone to decide whether an email is safe."
)
