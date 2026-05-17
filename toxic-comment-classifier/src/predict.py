"""
predict.py
==========
Inference / Prediction Module for Toxic Comment Classifier

Loads the saved model and vectorizer and exposes a simple predict() function
that can be called from app.py (Streamlit) or from the command line.

Usage (CLI):
    python src/predict.py "This is an amazing article!"
    python src/predict.py "You are such an idiot!"
"""

import os
import sys
import pickle

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.preprocess import preprocess

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "toxic_model.pkl")
TFIDF_PATH = os.path.join(BASE_DIR, "models", "tfidf_vectorizer.pkl")

# ── Module-level cache: load once, reuse across calls ─────────────────────────
_model      = None
_vectorizer = None


def _load_artifacts():
    """
    Lazy-load model and vectorizer from disk.
    Uses module-level variables so they are only loaded once per process.
    """
    global _model, _vectorizer
    if _model is None or _vectorizer is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                "Trained model not found.\n"
                "Please run:  python src/train.py\n"
                "before using predict.py or app.py."
            )
        with open(MODEL_PATH, "rb") as f:
            _model = pickle.load(f)
        with open(TFIDF_PATH, "rb") as f:
            _vectorizer = pickle.load(f)


def predict(text: str) -> dict:
    """
    Predict whether a comment is toxic.

    Pipeline:
        1. Preprocess raw text (lowercase → clean → lemmatize)
        2. Transform with TF-IDF vectorizer
        3. Run Logistic Regression inference
        4. Return label and confidence probability

    Parameters:
        text : raw comment string from the user

    Returns:
        dict with keys:
            label      : "Toxic" or "Non-Toxic"
            confidence : float 0-100 (probability of the predicted class)
            toxic_prob : float 0-1 (raw probability of being toxic)
    """
    _load_artifacts()

    # Preprocess the input text using the same pipeline used during training
    cleaned  = preprocess(text)

    # Transform into TF-IDF feature vector (same vocabulary as training)
    features = _vectorizer.transform([cleaned])

    # Get predicted class (0 = Non-Toxic, 1 = Toxic)
    pred_class = _model.predict(features)[0]

    # Get probability for both classes: [P(Non-Toxic), P(Toxic)]
    proba      = _model.predict_proba(features)[0]

    label      = "Toxic" if pred_class == 1 else "Non-Toxic"
    toxic_prob = float(proba[1])

    # Confidence = probability of the *predicted* class
    confidence = toxic_prob if pred_class == 1 else float(proba[0])

    return {
        "label":      label,
        "confidence": round(confidence * 100, 2),
        "toxic_prob": round(toxic_prob, 4),
    }


# ── CLI entry point ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Predefined sample comments if no argument is given
    samples = [
        "You are such an idiot, nobody cares about your stupid opinion.",
        "I really enjoyed reading this article, very informative.",
        "Go kill yourself, worthless piece of garbage.",
        "This is a great community, everyone is so helpful here.",
        "You filthy animal, you do not deserve to breathe.",
        "Could you please elaborate on that point a bit more?",
    ]

    # Use command-line argument if provided
    if len(sys.argv) > 1:
        samples = [" ".join(sys.argv[1:])]

    print("\n" + "=" * 60)
    print("        Toxic Comment Classifier — Sample Predictions")
    print("=" * 60)

    for comment in samples:
        result = predict(comment)
        icon   = "🔴" if result["label"] == "Toxic" else "🟢"
        print(f"\n  Comment   : {comment[:70]}{'...' if len(comment)>70 else ''}")
        print(f"  Prediction: {icon} {result['label']}")
        print(f"  Confidence: {result['confidence']}%")
        print(f"  Toxic Prob: {result['toxic_prob']}")
    print()
