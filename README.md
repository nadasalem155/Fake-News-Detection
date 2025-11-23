# 🔍 Fake News Detection App

Detect whether a news article is **REAL 📰** or **FAKE ⚠️** using Natural Language Processing (NLP) and a Logistic Regression model with TF-IDF vectorization. This project includes a **Jupyter Notebook** for data exploration, preprocessing, and model training, as well as a **Streamlit app** for real-time predictions.  

---

## 📝 Project Notebook

The notebook demonstrates step by step how the model is prepared:

1. **📂 Load Dataset**
   - Combined `fake.csv` and `true.csv` into a single DataFrame
   - Added a `label` column: `Fake=1`, `Real=0`
   - Removed duplicates and irrelevant columns (like `date`)  

2. **🧹 Data Preprocessing**
   - Convert text to lowercase
   - Remove URLs and non-alphabetic characters
   - Tokenize text
   - Remove stopwords
   - Lemmatize words  
   *(Applied on `title`, `subject`, and `text`)*

3. **🖊 Feature Extraction (TF-IDF)**
   - Combined `title`, `subject`, and `text` columns
   - Vectorized text using `TfidfVectorizer(max_features=5000)`

4. **🔄 Handle Imbalance (SMOTE)**
   - Oversampled minority class to balance the dataset
   - Ensured the model learns equally from Real and Fake news

5. **🤖 Model Training**
   - Trained `LogisticRegression` on the oversampled dataset
   - Evaluated performance on test set (~94% accuracy)
   - Saved trained model and TF-IDF vectorizer using `pickle`

---

## 🖥 Streamlit App Usage

The Streamlit app allows **real-time predictions** for any news text:

1. **📰 Enter News Source (Optional)**
   - Examples: Reuters, BBC, CNN…
   - Adding source can slightly improve accuracy, but it’s optional  

2. **✍️ Paste News Text**
   - Full text of the article for better predictions

3. **🔮 Predict Now**
   - Click the button to see the result:
     - `REAL 📰` for genuine news
     - `FAKE ⚠️` for misinformation

---

## ⚙️ Installation

1. Clone the repository:
   ```bash
   git clone <repo_url>
   cd <repo_folder>
Install dependencies:

pip install -r requirements.txt
Run the Streamlit app:

streamlit run app.py
---
## 🗂 File Structure

├── dataset/
│   ├── fake.csv
│   └── true.csv
├── fake_news_detection.ipynb         # Data preprocessing, feature extraction, and model training
├── tfidf_vectorizer.pkl    # Saved TF-IDF vectorizer
├── logreg_model.pkl        # Trained Logistic Regression model
├── app.py                  # Streamlit web application
├── requirements.txt
└── README.md
---
## 📊 Notes
The news source field is optional 📰

Preprocessing ensures the text is clean and ready for the model 🧹

TF-IDF converts text into numeric features for Logistic Regression 📈

SMOTE handles class imbalance 🔄

Streamlit app provides an interactive interface for real-time detection ⚡
---
## 🚀 Example Usage
Input News Source: Reuters (optional)

Input News Text:
"U.S. President signs new bill to improve healthcare system..."

Prediction: REAL 📰

Input News Text:
"Breaking: Celebrity endorses miracle weight loss pill..."

Prediction: FAKE ⚠️
