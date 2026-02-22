
import streamlit as st
import re
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import pickle

# Load dataset
df = pd.read_csv('combined_emotion.csv')


# Clean function
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z ]', '', text)
    return text

df['cleaned'] = df['sentence'].apply(clean_text)

# TF-IDF
tfidf = TfidfVectorizer()
X = tfidf.fit_transform(df['cleaned'])
y = df['emotion']

# Train model
model = LogisticRegression(max_iter=2000)
model.fit(X, y)

# Streamlit UI
st.title("Emotion Detection App 😊")
st.write("Enter a sentence and detect emotion")

user_input = st.text_input("Type your sentence here:")

if st.button("Predict"):
    cleaned = clean_text(user_input)
    vector = tfidf.transform([cleaned])
    prediction = model.predict(vector)
    st.success(f"Predicted Emotion: {prediction[0]}")
