# 📂 SMS Spam Detection — Practical Coding Worksheet (Python + Machine Learning)

This repository contains a hands-on practical assignment focused on building and evaluating an SMS Spam Detection System using Python and Machine Learning. The worksheet guides you through dataset exploration, preprocessing, feature engineering, model training, and evaluation — along with optional advanced extensions.

---

## 📌 Section B — Practical Coding Tasks (Core ML Workflow)

### ✅ 1️⃣ Dataset Import, Cleaning & Exploration
**Objectives:**
- Load the `spam.csv` dataset
- Understand dataset structure and distribution

**Tasks:**
- Import the dataset using `pandas`
- Display dataset statistics (row count, column count, null values)
- Preview first 5–10 rows
- Identify dataset columns (`text`, `label`, etc.)
- Visualize class distribution using:
  - Bar chart or Pie chart (`matplotlib` / `seaborn`)
- Document observations (e.g., dataset imbalance)

**Expected Skills:**
- pandas
- matplotlib / seaborn

---

### ✍️ 2️⃣ Text Preprocessing Pipeline

**Build a reusable function to:**
- Convert text to lowercase
- Remove punctuation & special characters
- Remove stopwords
- Perform stemming or lemmatization
- Optional: remove numbers, URLs, emojis

**Test Input Example:**
"Congratulations!!! You won $1000 CASH. Claim now!!!"

**Deliverables:**
- Original message
- Intermediate transformations
- Final cleaned output
- Observations on preprocessing effectiveness

**Expected Skills:**
- NLTK / spaCy
- regex

---

### 🔠 3️⃣ Text Vectorization

**Tasks:**
- Apply TF-IDF Vectorizer
- Compare with Bag-of-Words
- Report & analyze:
  - vocabulary size
  - dimensionality of features
  - sample feature weights
  - differences in token importance

**Expected Skills:**
- sklearn.feature_extraction

---

### 🤖 4️⃣ Model Training & Evaluation

**Train at least two models:**
- Naive Bayes
- Logistic Regression
- SVM
- (Optional) Random Forest, XGBoost

**Performance Metrics to Report:**
- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix
- Classification Report

**Discussion Points:**
- Best performing model and justification
- Impact of feature engineering
- Insights from confusion matrix:
  - false positives
  - false negatives

**Expected Skills:**
- train/test split
- sklearn.metrics

---

### 🧍 5️⃣ Custom User Input Prediction

**Implement CLI or Notebook input handling:**
- Accept SMS text from user
- Preprocess using your pipeline
- Vectorize
- Predict as SPAM or HAM
- Return:
  - label
  - probability / confidence score

**Examples to test:**
- promotional offers
- scam-like messages
- informal/personal chats

**Outcome:**
A fully functional text classifier for new/unseen SMS.

---

## ⭐ Section C — Extension / Advanced Tasks (Optional but Recommended)

Choose any ONE (or more):

### 🔹 Replace TF-IDF with Word Embeddings
- Word2Vec / GloVe
- Analyze differences in performance

### 🔹 Deep Learning Model
- LSTM / RNN
- Evaluate using embeddings

### 🔹 Backend API
- Flask or FastAPI
- Endpoint: `/predict`
- Accept raw text → return prediction

### 🔹 UI / Web App
- Streamlit
- Simple & interactive SMS classifier

---

## 📊 Section D — Output, Results & Interpretation

**Write a short report summarizing:**
- Best model (based on F1-score)
- Reasoning behind its performance
- Key features/keywords contributing to spam detection
- Present:
  - at least one confusion matrix
  - FP & FN interpretation

**Custom testing:**
- Evaluate at least 5 custom SMS examples
- Label and justify predictions

**Expected Output:**
- Demonstrated understanding of:
  - evaluation metrics
  - imbalanced dataset considerations
  - model trade-offs

---

## 🧠 Expected Learning Outcomes

By completing this worksheet, you will:

- Understand NLP preprocessing for text classification
- Build vectorizers & ML models
- Perform model evaluation using real-world metrics
- Deploy a functional predictive pipeline
- Gain exposure to advanced ML & deployment tools

---

## 🛠 Recommended Tools & Libraries

- Python 3.x
- pandas
- numpy
- sklearn
- nltk / spaCy
- matplotlib / seaborn
- streamlit / flask / fastapi (optional)
- gensim (for embeddings)

---

## 📁 Suggested Folder Structure






