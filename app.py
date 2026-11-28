"""Simple Streamlit app for our spam email classifier project."""

from pathlib import Path

import streamlit as st

from src.inference import available_models, classify_email, load_files


PROJECT_DIR = Path(__file__).resolve().parent
ARTIFACTS_DIR = PROJECT_DIR / "artifacts"

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


# Keep the model and vectorizer in memory after they are first loaded.
@st.cache_resource
def load_selected_model(model_path: str, vectorizer_path: str):
    return load_files(model_path, vectorizer_path)


st.set_page_config(page_title="Spam Email Classifier", page_icon="📧")
st.title("📧 Spam Email Classifier")
st.write("Paste an email below to see whether our machine learning model thinks it is spam.")

models = available_models(ARTIFACTS_DIR)
vectorizer_path = ARTIFACTS_DIR / "tfidf_vectorizer.joblib"

if not vectorizer_path.is_file():
    st.error("The TF-IDF vectorizer is missing from the artifacts folder.")
    st.stop()

if not models:
    st.error("No trained models were found in the artifacts folder.")
    st.stop()

model_names = list(models)
default_index = model_names.index("SVM") if "SVM" in model_names else 0
selected_model = st.selectbox("Choose a model", model_names, index=default_index)

example = st.selectbox("Try an example (optional)", list(EXAMPLES))
email_text = st.text_area(
    "Email text",
    value=EXAMPLES[example],
    key=f"email_{example}",
    height=190,
    placeholder="Paste the email message here...",
)

if st.button("Analyze Email", type="primary"):
    if not email_text.strip():
        st.warning("Please enter an email first.")
    else:
        try:
            model, vectorizer = load_selected_model(
                str(models[selected_model]), str(vectorizer_path)
            )
            prediction = classify_email(email_text, model, vectorizer)
            if prediction == 1:
                st.warning("Prediction: SPAM 🚫")
            elif prediction == 0:
                st.success("Prediction: NOT SPAM ✅")
            else:
                st.error("The model returned an unexpected result.")
        except (OSError, ValueError, ImportError, EOFError) as error:
            st.error(f"Could not analyze the email: {error}")

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
