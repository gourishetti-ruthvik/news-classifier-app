# Streamlit app for classifying news articles as fake or real

import streamlit as st
import joblib
import numpy as np
import re
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk

# NLTK downloads
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')

# Load model and TF-IDF vectorizer
model = joblib.load("app/model.pkl")
tfidf = joblib.load("app/tfidf.pkl")

# Preprocessing function
stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word.isalpha() and word not in stop_words]
    return ' '.join(tokens)

# Streamlit page config
st.set_page_config(page_title="News Classifier", page_icon="🧠", layout="centered")

# Toggle for theme
theme = st.sidebar.radio("Choose Theme", ["Light", "Dark"])

# Dynamic CSS with animated gradients and theme switching
theme_css = {
    "Light": """
        <style>
        body {
            background: linear-gradient(-45deg, #e0f7fa, #ffffff, #dfe9f3, #f1f8e9);
            background-size: 600% 600%;
            animation: pulseBG 20s ease infinite;
            color: #000000;
        }
        @keyframes pulseBG {
            0% { background-position: 0% 50%; }
            50% { background-position: 100% 50%; }
            100% { background-position: 0% 50%; }
        }
        .main-title {
            text-align: center;
            font-size: 46px;
            color: #0a3d62;
            font-weight: bold;
            margin-top: 20px;
            transition: all 0.3s ease;
        }
        .main-title:hover {
            color: #0066cc;
            letter-spacing: 1px;
        }
        .subtitle {
            text-align: center;
            font-size: 19px;
            color: #4a4a4a;
            margin-bottom: 35px;
            transition: all 0.3s ease;
        }
        .subtitle:hover {
            color: #0a3d62;
        }
        .stTextArea > label {
            font-weight: bold;
            font-size: 18px;
        }
        .stButton>button {
            background-color: #0066cc;
            color: white;
            font-weight: bold;
            border-radius: 10px;
            padding: 0.6em 2em;
            margin-top: 15px;
            transition: all 0.3s ease;
        }
        .stButton>button:hover {
            background-color: #004c99;
            transform: scale(1.08);
        }
        </style>
    """,
    "Dark": """
        <style>
        body {
            background: linear-gradient(-45deg, #1c1c2b, #2c3e50, #34495e, #22313f);
            background-size: 600% 600%;
            animation: pulseBG 25s ease infinite;
            color: #ffffff;
        }
        @keyframes pulseBG {
            0% { background-position: 0% 50%; }
            50% { background-position: 100% 50%; }
            100% { background-position: 0% 50%; }
        }
        .main-title {
            text-align: center;
            font-size: 46px;
            color: #00d4ff;
            font-weight: bold;
            margin-top: 20px;
            transition: all 0.3s ease;
        }
        .main-title:hover {
            color: #00ffe1;
            letter-spacing: 1px;
        }
        .subtitle {
            text-align: center;
            font-size: 19px;
            color: #ced6e0;
            margin-bottom: 35px;
            transition: all 0.3s ease;
        }
        .subtitle:hover {
            color: #f1f1f1;
        }
        .stTextArea > label {
            font-weight: bold;
            font-size: 18px;
        }
        .stButton>button {
            background-color: #00d4ff;
            color: white;
            font-weight: bold;
            border-radius: 10px;
            padding: 0.6em 2em;
            margin-top: 15px;
            transition: all 0.3s ease;
        }
        .stButton>button:hover {
            background-color: #00aaff;
            transform: scale(1.08);
        }
        </style>
    """
}

# Apply selected theme
st.markdown(theme_css[theme], unsafe_allow_html=True)

# App header
st.markdown('<div class="main-title">🔎 Fake or Real? News Article Classifier</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Powered by Machine Learning and Natural Language Processing</div>', unsafe_allow_html=True)

# Tabs
tab1, tab2 = st.tabs(["📟 Classifier", "📊 About the Model"])

with tab1:
    user_input = st.text_area("Paste your news article below 👇", height=200)

    if user_input:
        word_count = len(user_input.split())
        st.info(f"📝 Word Count: `{word_count}`")
        if word_count < 10:
            st.warning("⚠️ Too short — accuracy might be affected.")

    if st.button("🚀 Classify News"):
        if user_input.strip() == "":
            st.warning("Please enter some news content.")
        else:
            with st.spinner("Analyzing the article... 🤔"):
                cleaned = clean_text(user_input)
                vectorized = tfidf.transform([cleaned])
                prediction = model.predict(vectorized)[0]
                confidence = np.max(model.predict_proba(vectorized)) * 100

                label = "🟢 REAL NEWS" if prediction == 1 else "🔴 FAKE NEWS"

                feature_names = tfidf.get_feature_names_out()
                vector = vectorized.toarray()[0]
                top_indices = vector.argsort()[-5:][::-1]
                keywords = [feature_names[i] for i in top_indices if vector[i] > 0]

                st.markdown("""
                <div style="
                    background: rgba(255, 255, 255, 0.2);
                    backdrop-filter: blur(8px);
                    border-radius: 16px;
                    padding: 30px;
                    margin-top: 30px;
                    box-shadow: 0 8px 30px rgba(0,0,0,0.2);
                ">
                """, unsafe_allow_html=True)

                st.markdown(f"### 🧠 Prediction: {label}")
                st.markdown(f"**Confidence:** `{confidence:.2f}%`")
                st.progress(int(confidence))

                if prediction == 1:
                    st.success("✅ This news seems legitimate.")
                    st.balloons()
                else:
                    st.error("⚠️ This article looks suspicious. Please verify before sharing.")

                if keywords:
                    st.markdown(f"**Top Keywords Detected:** `{' | '.join(keywords)}`")

                st.markdown("</div>", unsafe_allow_html=True)

with tab2:
    st.markdown("""
    ### 🔍 Model Info:
    - **Model Used:** Logistic Regression / Naive Bayes
    - **Vectorizer:** TF-IDF (5000 max features)
    - **Dataset:** [Kaggle: Fake and Real News Dataset](https://www.kaggle.com/clmentbisaillon/fake-and-real-news-dataset)
    - **Accuracy:** ~92%
    - **App Framework:** Streamlit
    """)

# Footer
st.markdown("""
    <hr style="margin-top: 3rem;">
    <p style='text-align: center; font-size: 0.9rem; color: grey;'>
        Developed by <b>Gourishetti Ruthvik</b><br>
        <a href="https://github.com/your-username" target="_blank">GitHub</a> |
        <a href="https://linkedin.com/in/your-profile" target="_blank">LinkedIn</a>
    </p>
""", unsafe_allow_html=True)
