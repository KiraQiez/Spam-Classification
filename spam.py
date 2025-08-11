import streamlit as st

st.title("✉️ Spam Classification")
st.write("This app classifies messages as spam or not spam using a pre-trained model.")
message = st.text_area("Enter your message here:", "Type your message...")

# Load pre-trained model
import joblib
model = joblib.load("spam_classifier.pkl")
vectorizer = joblib.load("vectorizer.pkl")
# Predict function
def predict_spam(message):
    message_vec = vectorizer.transform([message])
    prediction = model.predict(message_vec)
    return prediction[0]

# Button to classify
if st.button("Classify"):
    if message.strip() == "":
        st.error("Please enter a message to classify.")
    else:
        result = predict_spam(message)
        if result == 1:
            st.error("This message is classified as **Spam**.")
        else:
            st.success("This message is classified as **Not Spam**.")
# Footer
st.markdown("---")
st.markdown("Made with ❤️ by AkiraDev")