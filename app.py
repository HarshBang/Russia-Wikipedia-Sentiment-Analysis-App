import streamlit as st
import pickle
import numpy as np
import matplotlib.pyplot as plt

# Title and source link
st.title("Applied Artificial Intelligence Mini Project: Russia Wikipedia Sentiment Analysis App")
st.markdown("[Wikipedia Source](https://en.wikipedia.org/wiki/Russia)")

# Load trained model and vectorizer
with open("tfidf_vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

with open("sentiment_model.pkl", "rb") as f:
    model = pickle.load(f)

user_input = st.text_area("Enter a sentence below to analyze its sentiment:")

if st.button("Predict Sentiment"):
    if user_input.strip():
        # Transform input using TF-IDF
        input_vectorized = vectorizer.transform([user_input])
        
        # Predict sentiment
        prediction = model.predict(input_vectorized)[0]
        probabilities = model.predict_proba(input_vectorized)[0]
        
        sentiment_label = "Positive" if prediction == 1 else "Negative"
        
        st.success(f"Predicted Sentiment: {sentiment_label}")
        
        # Bar Chart for Sentiment Distribution
        labels = ["Negative", "Positive"]
        fig, ax = plt.subplots()
        ax.bar(labels, probabilities, color=['red', 'green'])
        ax.set_ylabel("Probability")
        ax.set_title("Sentiment Distribution")
        st.pyplot(fig)
    else:
        st.warning("Please enter some text before predicting.")

# Footer
st.markdown(""" 
     <div style='text-align: center; font-size: 0.8em; color: gray;'>Developed by Harsh Bang (<a href='https://www.linkedin.com/in/harshbang/' target='_blank'>LinkedIn Profile</a>)
    </div>
    """, unsafe_allow_html=True)
