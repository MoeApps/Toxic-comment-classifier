# 🛡️ Toxic Comment Classifier

> A lightweight NLP project that classifies online comments as **Toxic** or **Non-Toxic** using classical machine learning techniques — no deep learning required.

Built as a university NLP course project demonstrating the full text classification pipeline from raw text to a live Streamlit web application.

---

## 📋 Table of Contents

- [Project Overview](#project-overview)
- [NLP Concepts Used](#nlp-concepts-used)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [How to Train](#how-to-train)
- [How to Evaluate](#how-to-evaluate)
- [How to Run the Streamlit App](#how-to-run-the-streamlit-app)
- [Sample Predictions](#sample-predictions)
- [Screenshots](#screenshots)
- [Future Improvements](#future-improvements)

---

## Project Overview

This system classifies text comments as either **Toxic** (harmful, abusive, or offensive) or **Non-Toxic** (civil and respectful) using a classical NLP pipeline:

```
Raw Text → Preprocessing → TF-IDF Features → Logistic Regression → Prediction
```

The project is fully self-contained, lightweight, and designed to demonstrate core NLP concepts clearly and academically.

**Dataset format** (Jigsaw Toxic Comment Classification):

| column | type | description |
|--------|------|-------------|
| `comment_text` | string | the raw comment |
| `toxic` | int (0/1) | 1 = toxic, 0 = non-toxic |

---

## NLP Concepts Used

### 1. Text Preprocessing (`src/preprocess.py`)

| Step | Library | Purpose |
|------|---------|---------|
| **Lowercasing** | built-in | Normalize case: `Hello` → `hello` |
| **Punctuation Removal** | `re` (regex) | Remove noise: `you're!!` → `youre` |
| **Tokenization** | `nltk` | Split text into tokens: `"hello world"` → `["hello", "world"]` |
| **Stopword Removal** | `nltk.corpus.stopwords` | Drop low-info words: removes `the`, `is`, `a` |
| **Stemming** | `nltk.stem.PorterStemmer` | Rule-based root: `running` → `run` |
| **Lemmatization** | `nltk.stem.WordNetLemmatizer` | Linguistic root: `studies` → `study` |

### 2. Feature Extraction — TF-IDF

**TF-IDF** (Term Frequency – Inverse Document Frequency) converts text into numerical vectors:

- **TF** (Term Frequency): how often a word appears in a document
- **IDF** (Inverse Document Frequency): penalizes words common across all documents (low information value)
- **N-grams**: captures two-word phrases (`"shut up"`, `"get lost"`) alongside single words

### 3. Classification — Logistic Regression

Logistic Regression predicts the probability of toxicity using the sigmoid function. It is well-suited for TF-IDF features because:

- Handles high-dimensional sparse matrices efficiently
- Outputs calibrated probability scores (useful for confidence)
- Built-in L2 regularization prevents overfitting
- Fast training and inference

### 4. Evaluation Metrics

| Metric | Formula | Meaning |
|--------|---------|---------|
| **Accuracy** | (TP+TN)/Total | Overall correct predictions |
| **Precision** | TP/(TP+FP) | Quality of positive predictions |
| **Recall** | TP/(TP+FN) | Coverage of actual positives |
| **F1-Score** | 2·P·R/(P+R) | Harmonic mean of Precision & Recall |
| **ROC-AUC** | Area under ROC curve | Ranking/discrimination ability |

> In content moderation, **Recall** is especially important — missing toxic content (false negatives) is more costly than flagging a safe comment (false positive).

---

## Project Structure

```
toxic-comment-classifier/
│
├── data/
│   └── train.csv              ← Dataset (comment_text, toxic columns)
│
├── models/
│   ├── toxic_model.pkl        ← Trained Logistic Regression model
│   ├── tfidf_vectorizer.pkl   ← Fitted TF-IDF vectorizer
│   ├── confusion_matrix.png   ← Generated after evaluate.py
│   └── roc_curve.png          ← Generated after evaluate.py
│
├── nltk_data/                 ← Bundled NLTK corpora (offline support)
│   ├── corpora/stopwords/
│   └── tokenizers/punkt_tab/
│
├── src/
│   ├── preprocess.py          ← NLP preprocessing pipeline
│   ├── train.py               ← Model training script
│   ├── evaluate.py            ← Evaluation + visualizations
│   └── predict.py             ← Inference module (used by app.py)
│
├── app.py                     ← Streamlit web interface
├── notebook.ipynb             ← Jupyter notebook walkthrough
├── requirements.txt           ← Python dependencies
└── README.md
```

---

## Installation

### Prerequisites

- Python 3.8 or higher
- pip

### Steps

```bash
# 1. Clone or download the project
cd toxic-comment-classifier

# 2. Install all dependencies
pip install -r requirements.txt
```

That's it! No external data downloads required — all NLTK data is bundled.

---

## How to Train

```bash
python src/train.py
```

This will:
1. Load `data/train.csv`
2. Preprocess all comments using the NLP pipeline
3. Split into 80% train / 20% test
4. Extract TF-IDF features
5. Train a Logistic Regression classifier
6. Print accuracy and classification report
7. Save `models/toxic_model.pkl` and `models/tfidf_vectorizer.pkl`

**Expected output:**
```
=======================================================
   Toxic Comment Classifier — Training Pipeline
=======================================================
[1/5] Loading dataset from: data/train.csv
      Loaded 550 rows ...
[2/5] Preprocessing text ...
[3/5] Extracting TF-IDF features...
      Feature matrix shape (train): (440, 403)
[4/5] Training Logistic Regression classifier...
[5/5] Saving model and vectorizer...

Training complete!
```

---

## How to Evaluate

```bash
python src/evaluate.py
```

Generates:
- Full metrics report (Accuracy, Precision, Recall, F1, ROC-AUC)
- `models/confusion_matrix.png`
- `models/roc_curve.png`

---

## How to Run the Streamlit App

```bash
streamlit run app.py
```

Then open your browser at **http://localhost:8501**

The app provides:
- A text area to enter any comment
- A "Classify Comment" button
- Result box showing **Toxic / Non-Toxic**
- A confidence percentage bar
- Technical details (probabilities, model info)
- Sample comment shortcuts

---

## Sample Predictions

| Comment | Prediction | Confidence |
|---------|-----------|------------|
| *"You are such an idiot, nobody cares about your opinion."* | 🔴 Toxic | 85.2% |
| *"Go kill yourself, worthless piece of garbage."* | 🔴 Toxic | 87.3% |
| *"You filthy animal, you do not deserve to breathe."* | 🔴 Toxic | 82.7% |
| *"I really enjoyed reading this article, very informative."* | 🟢 Non-Toxic | 84.9% |
| *"This is a great community, everyone is so helpful."* | 🟢 Non-Toxic | 84.0% |
| *"Could you please elaborate on that point?"* | 🟢 Non-Toxic | 83.9% |
| *"I disagree with your approach, but I understand the reasoning."* | 🟢 Non-Toxic | 78.4% |

---

## Screenshots

> _Run the app and replace these placeholders with your own screenshots_

### Streamlit Interface
```
[Screenshot: app.py running in browser — paste screenshot here]
```

### Toxic Comment Result
```
[Screenshot: red result card with confidence bar — paste here]
```

### Non-Toxic Comment Result
```
[Screenshot: green result card with confidence bar — paste here]
```

### Confusion Matrix
```
[Screenshot: models/confusion_matrix.png — paste here]
```

---

## Future Improvements

| Improvement | Description |
|-------------|-------------|
| **Richer dataset** | Train on the full Kaggle Jigsaw dataset (~160k rows) for higher generalization |
| **Multi-label classification** | Predict subcategories: severe toxic, obscene, threat, insult, identity hate |
| **Better features** | Add character n-grams, word embeddings (Word2Vec, FastText) |
| **Ensemble model** | Combine Logistic Regression with Naive Bayes or SVM |
| **SMOTE oversampling** | Handle class imbalance more rigorously |
| **Explainability** | Show which words contributed most to the toxic prediction (LIME/SHAP) |
| **API endpoint** | Wrap predict.py in a FastAPI REST endpoint |
| **Language support** | Extend preprocessing and model to support multilingual comments |

---

## Academic References

1. Joachims, T. (1998). *Text categorization with Support Vector Machines.* ECML.
2. Salton, G., & Buckley, C. (1988). *Term-weighting approaches in automatic text retrieval.* Information Processing & Management.
3. Bird, S., Klein, E., & Loper, E. (2009). *Natural Language Processing with Python.* O'Reilly.
4. Pedregosa et al. (2011). *Scikit-learn: Machine Learning in Python.* JMLR.
5. Jigsaw/Conversation AI. (2018). *Toxic Comment Classification Challenge.* Kaggle.

---

*University NLP Course Project — Classical Text Classification Pipeline*
# Toxic-comment-classifier
# Toxic-comment-classifier
# Toxic-comment-classifier
# Toxic-comment-classifier
