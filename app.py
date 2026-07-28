import streamlit as st
import re
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Emotions - Emotion Detection",
    page_icon="🎭",
    layout="wide"
)


# --------------------------------------------------
# CUSTOM CSS
# --------------------------------------------------

st.markdown("""
<style>

.main {
    background-color: #f5f7fb;
}

.title {
    text-align: center;
    font-size: 42px;
    font-weight: bold;
    margin-bottom: 5px;
}

.subtitle {
    text-align: center;
    font-size: 18px;
    color: #666;
    margin-bottom: 30px;
}

.card {
    padding: 25px;
    border-radius: 18px;
    background-color: white;
    box-shadow: 0px 4px 15px rgba(0,0,0,0.08);
    margin-bottom: 20px;
}

.result {
    padding: 25px;
    border-radius: 18px;
    background-color: #eef7ff;
    text-align: center;
    margin-top: 20px;
}

</style>
""", unsafe_allow_html=True)


# --------------------------------------------------
# LOAD DATASET
# --------------------------------------------------

@st.cache_data
def load_data():

    df = pd.read_csv("combined_emotionn.csv")

    return df


df = load_data()


# --------------------------------------------------
# TEXT CLEANING
# --------------------------------------------------

def clean_text(text):

    text = str(text).lower()

    text = re.sub(r'[^a-zA-Z ]', '', text)

    return text


df["cleaned"] = df["sentence"].apply(clean_text)


# --------------------------------------------------
# TRAIN MODEL
# --------------------------------------------------

@st.cache_resource
def train_model(data):

    X = data["cleaned"]
    y = data["emotion"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    tfidf = TfidfVectorizer(
        stop_words="english"
    )

    X_train_tfidf = tfidf.fit_transform(X_train)

    X_test_tfidf = tfidf.transform(X_test)

    model = LogisticRegression(
        max_iter=3000
    )

    model.fit(X_train_tfidf, y_train)

    predictions = model.predict(X_test_tfidf)

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    return tfidf, model, accuracy


tfidf, model, accuracy = train_model(df)


# --------------------------------------------------
# SESSION STATE
# --------------------------------------------------

if "history" not in st.session_state:

    st.session_state.history = []


# --------------------------------------------------
# HEADER
# --------------------------------------------------

st.markdown(
    '<div class="title">🎭 Emotions</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'AI-Based Emotion Detection from Text'
    '</div>',
    unsafe_allow_html=True
)


# --------------------------------------------------
# DASHBOARD METRICS
# --------------------------------------------------

col1, col2, col3 = st.columns(3)


with col1:

    st.metric(
        "📊 Model Accuracy",
        f"{accuracy * 100:.2f}%"
    )


with col2:

    st.metric(
        "📝 Dataset Records",
        len(df)
    )


with col3:

    st.metric(
        "🎭 Emotions",
        df["emotion"].nunique()
    )


st.divider()


# --------------------------------------------------
# PREDICTION SECTION
# --------------------------------------------------

st.markdown(
    '<div class="card">',
    unsafe_allow_html=True
)

st.subheader("🤖 Emotion Prediction")

st.write(
    "Enter a sentence below and the AI model will predict "
    "the emotion expressed in the text."
)

user_input = st.text_area(
    "Enter your text:",
    placeholder="Example: I am very happy today!",
    height=120
)

predict_button = st.button(
    "🔮 Predict Emotion",
    use_container_width=True
)

st.markdown(
    '</div>',
    unsafe_allow_html=True
)


# --------------------------------------------------
# PREDICTION
# --------------------------------------------------

if predict_button:

    if user_input.strip() == "":

        st.warning(
            "⚠️ Please enter some text before predicting."
        )

    else:

        cleaned = clean_text(user_input)

        vector = tfidf.transform([cleaned])

        prediction = model.predict(vector)[0]

        probabilities = model.predict_proba(vector)[0]

        confidence = max(probabilities) * 100

        # Save history
        st.session_state.history.append({
            "Text": user_input,
            "Emotion": prediction,
            "Confidence": f"{confidence:.2f}%"
        })

        # Result
        st.markdown(
            f"""
            <div class="result">

            <h2>🎭 Predicted Emotion</h2>

            <h1>{prediction.upper()}</h1>

            <h3>Confidence: {confidence:.2f}%</h3>

            </div>
            """,
            unsafe_allow_html=True
        )


# --------------------------------------------------
# HISTORY
# --------------------------------------------------

if st.session_state.history:

    st.divider()

    st.subheader("📝 Prediction History")

    history_df = pd.DataFrame(
        st.session_state.history
    )

    st.dataframe(
        history_df,
        use_container_width=True,
        hide_index=True
    )


# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.divider()

st.markdown(
    "<center>🎭 Emotions | AI-Based Emotion Detection System</center>",
    unsafe_allow_html=True
)
