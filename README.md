# 📰 Fake News Classifier

A Natural Language Processing (NLP) project that classifies news articles as **Fake** or **Real** using machine learning. Trained and compared 3 ML models on 40,000+ articles using TF-IDF feature extraction, achieving **98%+ F1 Score**.


---

## 🔍 Overview

Fake news is a growing problem in today's digital world. This project builds a machine learning pipeline that can automatically detect whether a news article is fake or real based on its content.

The project covers the full ML workflow:
- Data loading and merging
- Text preprocessing and cleaning
- Feature extraction using TF-IDF
- Training and comparing 3 classification models
- Evaluation with multiple metrics
- Visual analysis of results

---

## 📂 Dataset

**Source:** [Fake and Real News Dataset – Kaggle](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset)

| File | Description | Size |
|------|-------------|------|
| `Fake.csv` | Fake news articles | ~23,000 articles |
| `True.csv` | Real news articles | ~21,000 articles |

**Label Mapping:**
- `0` → Real News
- `1` → Fake News

---

## 🛠️ Tech Stack

| Tool | Purpose |
|------|---------|
| Python | Core language |
| Pandas | Data loading and manipulation |
| NumPy | Numerical operations |
| Scikit-learn | ML models, TF-IDF, evaluation |
| Matplotlib | Visualizations |
| Seaborn | Heatmaps and styled plots |
| Joblib | Saving and loading models |

---

## ⚙️ How It Works

### Step 1 — Text Preprocessing
- Combined `title` and `text` columns into one `content` field
- Lowercased all text
- Removed URLs, punctuation, numbers, and bracket text
- Removed extra whitespace

### Step 2 — Feature Extraction (TF-IDF)
- Used `TfidfVectorizer` with `max_features=5000`
- Automatically removed English stopwords
- Each article becomes a vector of 5000 numbers

### Step 3 — Model Training
Trained and compared 3 models:
- Logistic Regression
- Naive Bayes (MultinomialNB)
- Random Forest (100 estimators)

### Step 4 — Evaluation
Each model was evaluated using:
- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix

---

## 📊 Results

| Model | Accuracy | Precision | Recall | F1 Score |
|-------|----------|-----------|--------|----------|
| Logistic Regression | 98.64% | 98.86% | 98.24% | 98.55% |
| Naive Bayes | 93.55% | 92.58% | 93.79% | 93.18% |
| Random Forest | 99.75% | 99.87% | 99.61% | 99.74% |

> 🏆 **Best Model: Random Forest** with 99.74% F1 Score

---

## ▶️ How to Run

### 1. Clone the repository
```bash
git clone https://github.com/your-username/fake-news-classifier.git
cd fake-news-classifier
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Download the dataset
Download from [Kaggle](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset) and place `Fake.csv` and `True.csv` inside the `data/` folder.

# Run notebook
jupyter notebook Fake_news_predictor.ipynb

---

## 🧪 Sample Prediction

```python
import  re


def predict_news(text):
    text = text.lower()
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'[^a-z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()

    vector = tfidf.transform([text])
    prediction = lr.predict(vector)[0]
    confidence = lr.predict_proba(vector)[0][prediction] * 100

    label = "🚨 FAKE NEWS" if prediction == 1 else "✅ REAL NEWS"
    print(f"Prediction  : {label}")
    print(f"Confidence  : {confidence:.2f}%")

predict_news("Scientists discover new vaccine that cures all diseases overnight")
# Output: 🚨 FAKE NEWS
```

---

## 📖 What I Learned

- How to preprocess raw text data for ML using Python
- What TF-IDF is and why it works well for text classification
- How to train and compare multiple ML models using Scikit-learn
- Why F1 Score matters more than accuracy for classification problems
- How to read and interpret confusion matrices
- How to save and load ML models using Joblib

---

## 🏷️ Topics
`machine-learning` `fake-news-detection` `nlp` `scikit-learn` `python` `tfidf` `text-classification` `data-science`

---

## 👤 Author
Krishna Gupta
- GitHub:(https://github.com/Krishna28Gupta)
