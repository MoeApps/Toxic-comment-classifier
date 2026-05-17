"""
evaluate.py
===========
Model Evaluation Script for Toxic Comment Classifier

Academic Concepts Demonstrated:
- Accuracy  : fraction of total predictions that are correct
- Precision : of all predicted positives, how many are truly positive?
- Recall    : of all actual positives, how many did we catch?
- F1-Score  : harmonic mean of precision and recall (balances both)
- Confusion Matrix : table showing TP, TN, FP, FN counts

Run after training:
    python src/evaluate.py
"""

import os
import sys
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")   # non-interactive backend (no GUI window needed)

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    roc_curve,
)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.preprocess import preprocess

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH  = os.path.join(BASE_DIR, "data",   "train.csv")
MODEL_PATH = os.path.join(BASE_DIR, "models", "toxic_model.pkl")
TFIDF_PATH = os.path.join(BASE_DIR, "models", "tfidf_vectorizer.pkl")
EVAL_DIR   = os.path.join(BASE_DIR, "models")   # save plots alongside models


def load_artifacts():
    """Load the pre-trained model and TF-IDF vectorizer from disk."""
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            "Model not found. Run 'python src/train.py' first."
        )
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    with open(TFIDF_PATH, "rb") as f:
        vectorizer = pickle.load(f)
    return model, vectorizer


def prepare_test_set():
    """
    Reproduce the same 80/20 split used during training.
    Using the same random_state=42 guarantees the same test rows.
    """
    df = pd.read_csv(DATA_PATH)
    df["cleaned"] = df["comment_text"].apply(preprocess)
    _, X_test_raw, _, y_test = train_test_split(
        df["cleaned"], df["toxic"],
        test_size=0.2, random_state=42, stratify=df["toxic"],
    )
    return X_test_raw, y_test


def print_metrics(y_true, y_pred, y_prob):
    """
    Print a formatted table of all evaluation metrics.

    Metric Explanations:
    ┌──────────────┬────────────────────────────────────────────────────┐
    │ Accuracy     │ (TP+TN)/(TP+TN+FP+FN) — overall correct rate      │
    │ Precision    │ TP/(TP+FP)            — quality of positive preds  │
    │ Recall       │ TP/(TP+FN)            — coverage of true positives │
    │ F1-Score     │ 2*(P*R)/(P+R)         — balance of P and R         │
    │ ROC-AUC      │ area under ROC curve  — ranking ability            │
    └──────────────┴────────────────────────────────────────────────────┘
    """
    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec  = recall_score(y_true, y_pred)
    f1   = f1_score(y_true, y_pred)
    auc  = roc_auc_score(y_true, y_prob)

    print("\n" + "=" * 50)
    print("        EVALUATION METRICS REPORT")
    print("=" * 50)
    print(f"  Accuracy      : {acc:.4f}  ({acc*100:.2f}%)")
    print(f"  Precision     : {prec:.4f}")
    print(f"  Recall        : {rec:.4f}")
    print(f"  F1-Score      : {f1:.4f}")
    print(f"  ROC-AUC Score : {auc:.4f}")
    print("=" * 50)
    print("\n  Full Classification Report:")
    print(classification_report(y_true, y_pred, target_names=["Non-Toxic", "Toxic"]))


def plot_confusion_matrix(y_true, y_pred):
    """
    Confusion Matrix Visualization
    --------------------------------
    A confusion matrix shows 4 outcomes:
      TP (True Positive)  : correctly predicted toxic
      TN (True Negative)  : correctly predicted non-toxic
      FP (False Positive) : non-toxic predicted as toxic
      FN (False Negative) : toxic predicted as non-toxic

    In content moderation, FN (missed toxicity) is more costly than FP.
    """
    cm = confusion_matrix(y_true, y_pred)
    labels = ["Non-Toxic", "Toxic"]

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.colorbar(im, ax=ax)

    ax.set(
        xticks=range(2), yticks=range(2),
        xticklabels=labels, yticklabels=labels,
        xlabel="Predicted Label",
        ylabel="True Label",
        title="Confusion Matrix — Toxic Comment Classifier",
    )

    # Annotate each cell with count and percentage
    total = cm.sum()
    for i in range(2):
        for j in range(2):
            count = cm[i, j]
            pct   = 100 * count / total
            color = "white" if count > cm.max() / 2 else "black"
            ax.text(j, i, f"{count}\n({pct:.1f}%)",
                    ha="center", va="center", color=color, fontsize=12, fontweight="bold")

    # Label the four quadrants
    quad_labels = {(0,0): "TN", (0,1): "FP", (1,0): "FN", (1,1): "TP"}
    for (r, c), lbl in quad_labels.items():
        ax.text(c + 0.35, r - 0.35, lbl, ha="right", va="top",
                color="gray", fontsize=9, style="italic")

    plt.tight_layout()
    save_path = os.path.join(EVAL_DIR, "confusion_matrix.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"\n  Confusion matrix saved → {save_path}")
    plt.close()
    return save_path


def plot_roc_curve(y_true, y_prob):
    """
    ROC Curve (Receiver Operating Characteristic)
    -----------------------------------------------
    Shows the trade-off between True Positive Rate and False Positive Rate
    at different classification thresholds.
    AUC = 1.0 → perfect classifier | AUC = 0.5 → random baseline.
    """
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, color="steelblue", lw=2, label=f"ROC Curve (AUC = {auc:.3f})")
    ax.plot([0, 1], [0, 1], color="gray", lw=1, linestyle="--", label="Random Baseline")
    ax.fill_between(fpr, tpr, alpha=0.1, color="steelblue")
    ax.set(xlabel="False Positive Rate", ylabel="True Positive Rate",
           title="ROC Curve — Toxic Comment Classifier",
           xlim=[0, 1], ylim=[0, 1.02])
    ax.legend(loc="lower right")
    plt.tight_layout()
    save_path = os.path.join(EVAL_DIR, "roc_curve.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"  ROC curve saved      → {save_path}")
    plt.close()
    return save_path


def main():
    print("=" * 50)
    print("   Toxic Comment Classifier — Evaluation")
    print("=" * 50)

    # Load trained artifacts
    print("\n[1/4] Loading model and vectorizer...")
    model, vectorizer = load_artifacts()

    # Prepare test data
    print("[2/4] Preparing test set...")
    X_test_raw, y_test = prepare_test_set()
    X_test = vectorizer.transform(X_test_raw)

    # Generate predictions
    print("[3/4] Running predictions...")
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]   # probability of class 1 (toxic)

    # Print metrics
    print_metrics(y_test, y_pred, y_prob)

    # Save plots
    print("[4/4] Generating evaluation plots...")
    plot_confusion_matrix(y_test, y_pred)
    plot_roc_curve(y_test, y_prob)

    print("\n Evaluation complete!\n")


if __name__ == "__main__":
    main()
