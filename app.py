import streamlit as st
import pickle
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import nltk

# -------------------------
# NLTK downloads
# -------------------------
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# -------------------------
# Load model & vectorizer
# -------------------------
with open('logreg_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('tfidf_vectorizer.pkl', 'rb') as f:
    tfidvect = pickle.load(f)

# -------------------------
# Streamlit config
# -------------------------
st.set_page_config(page_title="Fake News Detection", page_icon="📰")
st.title("Fake News Detection 📰")
st.markdown("A fake news prediction web application using ML algorithms.")

# -------------------------
# Input text & source
# -------------------------
source = st.text_input(
    label="Source (e.g., Reuters, BBC, CNN)",
    placeholder="Enter the news source here"
)

text = st.text_area(
    label="Enter your news text:",
    placeholder="Enter your text to predict whether this is fake or not.",
    height=200
)

st.write(f'You wrote {len(text.split())} words.')

# -------------------------
# Preprocessing function
# -------------------------
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    text = text.lower().strip()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    tokens = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return " ".join(tokens)

# -------------------------
# Prediction function
# -------------------------
def predict(text, source=""):
    combined_text = f"{source} {text}" if source else text
    processed = preprocess_text(combined_text)
    vectorized = tfidvect.transform([processed])
    prediction = model.predict(vectorized)[0]  # 1=FAKE, 0=REAL
    return 'FAKE ⚠️' if prediction == 1 else 'REAL 📰'

# -------------------------
# Button
# -------------------------
if st.button("Predict"):
    if not text.strip():
        st.warning("Please enter some text to predict!")
    else:
        result = predict(text, source)
        st.markdown(f"### Prediction: {result}")
