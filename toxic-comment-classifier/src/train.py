"""
train.py
========
Model Training Script for Toxic Comment Classifier

Academic Concepts Demonstrated:
- Feature Extraction with TF-IDF (Term Frequency – Inverse Document Frequency)
- Train/Test Split for unbiased model evaluation
- Logistic Regression for binary text classification
- Model Persistence with pickle

Run this script first to produce the saved model and vectorizer:
    python src/train.py
"""

import os
import pickle
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

# Add parent directory to path so we can import from src/
import sys

# Windows consoles default to the cp1252 codec, which cannot encode the
# Unicode arrows/box-drawing characters used in the progress messages below.
# Reconfigure stdout to UTF-8 so printing never crashes the training run.
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.preprocess import preprocess

# ── Path configuration ────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH  = os.path.join(BASE_DIR, "src", "train.csv")
MODEL_DIR  = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(MODEL_DIR, "toxic_model.pkl")
TFIDF_PATH = os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl")

os.makedirs(MODEL_DIR, exist_ok=True)


def load_data(path: str) -> pd.DataFrame:
    """
    Load dataset from CSV.
    Expected columns: comment_text, toxic
    """
    print(f"[1/5] Loading dataset from: {path}")
    df = pd.read_csv(path)
    print(f"      Loaded {len(df)} rows | Class distribution:")
    print(df["toxic"].value_counts().to_string())
    return df


def preprocess_data(df: pd.DataFrame) -> pd.Series:
    """
    Apply the full NLP preprocessing pipeline to every comment.
    This can take a few seconds for large datasets.
    """
    print("[2/5] Preprocessing text (lowercase → clean → tokenize → lemmatize)...")
    # fillna/astype guards against any blank or non-string cell in the dataset,
    # which would otherwise crash this long-running preprocessing pass.
    cleaned = df["comment_text"].fillna("").astype(str).apply(preprocess)
    print(f"      Sample raw:     {df['comment_text'].iloc[0][:60]}...")
    print(f"      Sample cleaned: {cleaned.iloc[0][:60]}...")
    return cleaned


def extract_features(texts_train, texts_test):
    """
    TF-IDF Feature Extraction
    --------------------------
    TF-IDF converts raw text into a numerical matrix where:
      - TF  (Term Frequency)         : how often a word appears in a document
      - IDF (Inverse Document Freq.) : penalizes words common across all documents

    Parameters:
        max_features : vocabulary size limit (top N most informative terms)
        ngram_range  : (1,2) means we use single words AND two-word phrases
        sublinear_tf : apply log(1 + tf) to dampen extreme term counts

    Returns a fitted vectorizer and transformed sparse matrices.
    """
    print("[3/5] Extracting TF-IDF features...")
    vectorizer = TfidfVectorizer(
        max_features=10_000,   # keep top 10k terms
        ngram_range=(1, 2),    # unigrams + bigrams for richer features
        sublinear_tf=True,     # apply log normalization on term frequencies
        min_df=2,              # ignore terms appearing in fewer than 2 docs
    )
    X_train = vectorizer.fit_transform(texts_train)   # fit on train only!
    X_test  = vectorizer.transform(texts_test)        # transform test with same vocab
    print(f"      Feature matrix shape (train): {X_train.shape}")
    print(f"      Feature matrix shape (test) : {X_test.shape}")
    return vectorizer, X_train, X_test


def train_model(X_train, y_train) -> LogisticRegression:
    """
    Logistic Regression Classifier
    --------------------------------
    Logistic Regression is well-suited for text classification because:
      - It works well with high-dimensional sparse TF-IDF features
      - It outputs calibrated probabilities (useful for confidence scores)
      - It is fast, interpretable, and academically well-understood
      - It avoids overfitting with built-in L2 regularization (C parameter)

    C=1.0         : regularization strength (lower = stronger regularization)
    max_iter      : maximum iterations for solver convergence
    solver        : 'lbfgs' is efficient for smaller datasets
    class_weight  : 'balanced' compensates for the ~10% toxic / 90% non-toxic
                    class imbalance, so the model actually flags toxic comments
                    instead of defaulting to the majority class.
    """
    print("[4/5] Training Logistic Regression classifier...")
    model = LogisticRegression(
        C=1.0,
        max_iter=1000,
        solver="lbfgs",
        random_state=42,
        class_weight="balanced",
    )
    model.fit(X_train, y_train)
    print("      Training complete.")
    return model


def save_artifacts(model, vectorizer):
    """
    Save the trained model and vectorizer using pickle for later use in the app.
    Both files are required by predict.py and app.py.
    """
    print("[5/5] Saving model and vectorizer...")
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)
    with open(TFIDF_PATH, "wb") as f:
        pickle.dump(vectorizer, f)
    print(f"      Model saved     → {MODEL_PATH}")
    print(f"      Vectorizer saved → {TFIDF_PATH}")


def main():
    print("=" * 55)
    print("   Toxic Comment Classifier — Training Pipeline")
    print("=" * 55)

    # Step 1: Load data
    df = load_data(DATA_PATH)

    # Step 2: Preprocess text
    df["cleaned"] = preprocess_data(df)

    # Step 3: Train/test split (80% train, 20% test)
    # stratify=y ensures both splits have same class ratio
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        df["cleaned"], df["toxic"],
        test_size=0.2,
        random_state=42,
        stratify=df["toxic"],
    )
    print(f"\n      Train size: {len(X_train_raw)} | Test size: {len(X_test_raw)}")

    # Step 4: Feature extraction
    vectorizer, X_train, X_test = extract_features(X_train_raw, X_test_raw)

    # Step 5: Train model
    model = train_model(X_train, y_train)

    # Step 6: Quick evaluation summary
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\n── Quick Evaluation on Test Set ───────────────────────")
    print(f"   Accuracy: {acc:.4f} ({acc*100:.2f}%)")
    print()
    print(classification_report(y_test, y_pred, target_names=["Non-Toxic", "Toxic"]))

    # Step 7: Save artifacts
    save_artifacts(model, vectorizer)

    print("\n Training complete! Run the Streamlit app with:")
    print("   streamlit run app.py\n")


if __name__ == "__main__":
    main()
