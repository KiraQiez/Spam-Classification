import streamlit as st
import joblib


# -----------------------------
# 1. Load trained models
# -----------------------------
@st.cache_resource
def load_models():
    models = {}

    
    models["Logistic Regression"] = joblib.load("models/logistic_regression_spam.h5")
    models["Naive Bayes"] = joblib.load("models/naive_bayes_spam.h5")
    models["Linear SVM"] = joblib.load("models/linear_svm_spam.h5")

    return models


# -----------------------------
# 2. Streamlit UI
# -----------------------------
def main():
    st.set_page_config(
        page_title="Spam Classifier",
        page_icon="📨",
        layout="centered",
    )

    st.title("📨 Spam Classification App")
    st.write(
        "This app uses a trained machine learning model to classify SMS/email messages "
        "as **ham (not spam)** or **spam**."
    )

    # load all models once
    models = load_models()

    # pick which model to use
    st.sidebar.header("Model Settings")
    model_name = st.sidebar.selectbox(
        "Choose model",
        options=list(models.keys()),
        index=2,  # default to Linear SVM 
    )

    model = models[model_name]

    st.sidebar.write(f"**Current model:** {model_name}")

    st.markdown("---")

    # input message
    st.subheader("Try it out")

    example = st.selectbox(
        "Example messages (optional):",
        [
            "- Choose an example -",
            "You have won a FREE iPhone! Click this link to claim your prize.",
            "Hi, just checking if you're coming to the group meeting later.",
            "URGENT: Your bank account is locked. Visit this site to verify now.",
            "Can you send me the slides from today's lecture?",
        ],
    )

    if example != "- Choose an example -":
        user_text = st.text_area("Message:", value=example, height=120)
    else:
        user_text = st.text_area(
            "Message:",
            placeholder="Type or paste an SMS / email here...",
            height=120,
        )

    if st.button("Classify Message"):
        if not user_text.strip():
            st.warning("Please enter a message first.")
        else:
            # predict label
            label = model.predict([user_text])[0]

            if label == "spam":
                emoji = "🚨"
                color = "red"
                text_label = "SPAM"
            else:
                emoji = "✅"
                color = "green"
                text_label = "HAM (NOT SPAM)"

            st.markdown(
                f"### {emoji} Prediction: "
                f"<span style='color:{color}; font-weight:bold;'>{text_label}</span>",
                unsafe_allow_html=True,
            )

            # show probability only if model supports it
            if hasattr(model, "predict_proba"):
                proba = model.predict_proba([user_text])[0]
                classes = list(model.classes_)
                if "spam" in classes:
                    spam_idx = classes.index("spam")
                    spam_prob = float(proba[spam_idx])
                    st.write(f"**Estimated spam probability:** `{spam_prob:.2f}`")
                    st.progress(spam_prob)
            else:
                st.caption("This model does not provide probability (only decision).")

    st.markdown("---")
    st.caption(
        "Backend: TF-IDF + Logistic Regression / Naive Bayes / Linear SVM trained on SMS spam dataset."
    )


if __name__ == "__main__":
    main()
