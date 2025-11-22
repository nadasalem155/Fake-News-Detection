import streamlit as st
import pickle
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import nltk

# -------------------------
# NLTK Downloads
# -------------------------
nltk.download('punkt')
nltk.download('punkt_tab')   # ⭐ مهم جدًا
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
# Streamlit Page Config
# -------------------------
st.set_page_config(
    page_title="Fake News Detector",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------------
# Sidebar (User Guide)
# -------------------------
st.sidebar.title("📘 How to Use")
st.sidebar.markdown("""
Welcome to the **Fake News Detection App**!  
Follow these steps:

1️⃣ **Enter the news source**  
Examples: *Reuters, BBC, CNN...*

2️⃣ **Paste the full news text**  
The more text you provide, the more accurate the prediction becomes.

3️⃣ **Click 'Predict'**  
You'll instantly know if the news is **REAL 📰** or **FAKE ⚠️**.

---

### 💡 Tips for best accuracy:
- Avoid short texts.
- Provide full article when possible.
- Add a trusted source if available.

Enjoy using the app! 😄
""")

# -------------------------
# Main Title
# -------------------------
st.markdown("<h1 style='text-align: center; color:#4A90E2;'>🔍 Fake News Detection</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align: center; color: gray;'>Smart AI system to help you detect misinformation in seconds</h4>", unsafe_allow_html=True)

st.write("")

# -------------------------
# Input Section
# -------------------------
st.markdown("### 🏷️ News Source")
source = st.text_input(
    label=" ",                # ← label fix
    placeholder="Examples: Reuters, BBC, New York Times...",
    label_visibility="collapsed"
)

st.markdown("### ✍️ News Text")
text = st.text_area(
    label=" ",                # ← label fix
    placeholder="Paste the news text here...",
    height=230,
    label_visibility="collapsed"
)

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
# Predict Button
# -------------------------
st.write("")
predict_btn = st.button("🔮 Predict Now")

if predict_btn:
    if not text.strip():
        st.warning("⚠️ Please enter some news text first!")
    else:
        result = predict(text, source)

        if "FAKE" in result:
            st.error(f"### 🚨 Prediction: {result}")
        else:
            st.success(f"### ✅ Prediction: {result}")
