# 📘 Hands-On Worksheet — SMS Spam Detection (Beginner Friendly)

## 🎯 Project Goal
Build a machine learning model that can classify text messages as **Spam** or **Ham (Not Spam)**.

---

## 🧪 Section A — Dataset Handling & Exploration

### ✅ Task 1 — Import the Dataset
- Load the dataset `spam.csv` using pandas.  
- Display the first 10 rows.

**Expected Code Concepts:**  
`pandas.read_csv`, `head()`

**✏️ Write your observation:**
- How many columns are present?
- What are the labels?

---

### ✅ Task 2 — Clean the Data
Perform the following:
- Remove unwanted columns (if any)
- Rename columns to:
  - `label`
  - `message`

**Expected Concepts:**  
`drop()`, `rename()`

**✏️ Write your observation:**
- What shape did the dataset have before and after cleaning?

---

### ✅ Task 3 — Understand Label Distribution
- Count how many spam and ham messages exist.  
- Create a bar chart or pie chart showing the distribution.

**Expected Concepts:**  
`value_counts()`, `matplotlib`

**✏️ Write your observation:**
- Is the dataset balanced or imbalanced?
- Which label is more common?

---

## 🧪 Section B — Text Preprocessing

### 🔍 Task 4 — Preprocess Messages
Apply:
- lowercase
- remove punctuation
- remove stopwords

**Expected Concepts:**  
`string`, `re`, `nltk`

**✏️ Write your observation:**
- Why is text cleaning important?

---

### 🔍 Task 5 — Convert Text to Numerical Data
- Use Bag-of-Words (`CountVectorizer`) or TF-IDF.

**Expected Concepts:**  
`sklearn.feature_extraction.text`

**✏️ Write your observation:**
- What does vectorization mean?

---

## 🤖 Section C — Model Building

### 🤖 Task 6 — Train a Machine Learning Model
Use any one:
- Naive Bayes
- Logistic Regression

**Expected Concepts:**  
`train_test_split`, `fit()`

**✏️ Observation Questions:**
- What accuracy did your model achieve?
- Training vs Testing accuracy?

---

### 🤖 Task 7 — Evaluate the Model
Calculate:
- accuracy score
- confusion matrix
- classification report

**Expected Concepts:**  
`sklearn.metrics`

**✏️ Write your observation:**
- Which type of error is more common?
  - false positive?
  - false negative?

---

## 🧠 Section D — Prediction & Real-Life Testing

### ✍️ Task 8 — Test the Model
Use sample messages and predict:
- `"You won ₹10,000 click here"`
- `"Hey what's up?"`

**Expected Concepts:**  
`model.predict()`

**✏️ Write your observation:**
- Did the model classify correctly?

---

## 🚀 Bonus Challenges (Optional)

✔ Try a different model (SVM or Random Forest)  
✔ Try Lemmatization  
✔ Build a small GUI using Streamlit  
✔ Test with real SMS from your phone  

---

## 📌 Submission Checklist

Students must submit:
- Notebook (.ipynb)
- Screenshots of:
  - dataset exploration
  - label distribution
  - model evaluation
- Responses to observation questions

---

## ✅ Expected Learning Outcomes

By completing this worksheet, students will learn:
✔ dataset loading & cleaning  
✔ preprocessing text  
✔ converting text to numbers  
✔ training ML models  
✔ understanding evaluation metrics  
✔ testing real messages  

---

