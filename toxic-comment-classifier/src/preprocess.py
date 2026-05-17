"""
preprocess.py
=============
NLP Preprocessing Pipeline for Toxic Comment Classifier

Academic Concepts Demonstrated:
- Tokenization: splitting raw text into individual word units
- Lowercasing: text normalization to reduce vocabulary size
- Punctuation Removal: removing noise that does not carry semantic meaning
- Stopword Removal: filtering common words (the, is, at) that add little meaning
- Stemming: reducing words to their root form (e.g., "running" → "run")
- Lemmatization: reducing words to their dictionary base form using linguistic rules

Libraries Used:
- nltk: Natural Language Toolkit, the standard Python NLP library
- re:   Python built-in regex library for pattern matching and text cleaning
"""

import re
import os
import nltk

# ── Point NLTK to bundled data directory first ────────────────────────────────
# The project ships with a local copy of required NLTK corpora so the system
# works offline without needing 'nltk.download()' to reach the internet.
_BUNDLE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "nltk_data")
if os.path.isdir(_BUNDLE):
    nltk.data.path.insert(0, _BUNDLE)

# Fallback: try downloading quietly if not already present
# Silenced — expected to fail in offline/restricted environments;
# the project bundles the required data in the nltk_data/ folder.
import warnings as _warnings
with _warnings.catch_warnings():
    _warnings.simplefilter("ignore")
    for _pkg in ("punkt", "punkt_tab", "stopwords", "wordnet", "omw-1.4"):
        try:
            nltk.download(_pkg, quiet=True, raise_on_error=False)
        except Exception:
            pass

from nltk.corpus import stopwords as _sw_corpus
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

# ── WordNet lemmatizer: use if available, fall back to stemmer ─────────────────
try:
    from nltk.stem import WordNetLemmatizer
    _lemmatizer = WordNetLemmatizer()
    # Verify it works
    _lemmatizer.lemmatize("running")
    _LEMMATIZER_AVAILABLE = True
except Exception:
    _LEMMATIZER_AVAILABLE = False

# ── Initialize NLP tools ──────────────────────────────────────────────────────
stemmer = PorterStemmer()   # Stemmer: fast, no corpus needed

# ── Load stopwords ─────────────────────────────────────────────────────────────
# Try NLTK corpus first; fall back to a comprehensive built-in list
try:
    STOP_WORDS = set(_sw_corpus.words("english"))
except Exception:
    STOP_WORDS = {
        "i","me","my","myself","we","our","ours","ourselves","you","your","yours",
        "yourself","yourselves","he","him","his","himself","she","her","hers",
        "herself","it","its","itself","they","them","their","theirs","themselves",
        "what","which","who","whom","this","that","these","those","am","is","are",
        "was","were","be","been","being","have","has","had","having","do","does",
        "did","doing","a","an","the","and","but","if","or","because","as","until",
        "while","of","at","by","for","with","about","against","between","into",
        "through","during","before","after","above","below","to","from","up","down",
        "in","out","on","off","over","under","again","further","then","once","here",
        "there","when","where","why","how","all","both","each","few","more","most",
        "other","some","such","no","nor","not","only","own","same","so","than",
        "too","very","s","t","can","will","just","don","should","now","d","ll",
        "m","re","ve","y","ain","aren","couldn","didn","doesn","hadn","hasn",
        "haven","isn","ma","mightn","mustn","needn","shan","shouldn","wasn",
        "weren","won","wouldn",
    }


def lowercase(text: str) -> str:
    """
    Step 1 – Lowercasing
    --------------------
    Convert all characters to lowercase so that 'Hello' and 'hello'
    are treated as the same token. This reduces vocabulary size and
    prevents duplicate features.
    """
    return text.lower()


def remove_punctuation(text: str) -> str:
    """
    Step 2 – Punctuation & Special Character Removal
    -------------------------------------------------
    Use a regular expression to keep only alphabetic characters and spaces.
    Punctuation marks like '!', '?', and '@' rarely carry classification
    information in a bag-of-words model, so we strip them.

    Regex pattern: [^a-z\\s]
      ^a-z  → not a lowercase letter
      \\s   → not a whitespace character
    """
    return re.sub(r"[^a-z\s]", "", text)


def tokenize(text: str) -> list:
    """
    Step 3 – Tokenization
    ---------------------
    Split the cleaned text string into a list of individual word tokens.
    NLTK's word_tokenize handles edge cases better than a simple split().

    Example:
        "hello world" → ["hello", "world"]
    """
    return word_tokenize(text)


def remove_stopwords(tokens: list) -> list:
    """
    Step 4 – Stopword Removal
    -------------------------
    Remove high-frequency, low-information words such as:
    'the', 'is', 'at', 'which', 'on', 'a', 'an', 'in', ...

    Removing stopwords reduces noise and keeps features meaningful
    for classification. We only keep tokens with length > 1.
    """
    return [t for t in tokens if t not in STOP_WORDS and len(t) > 1]


def stem(tokens: list) -> list:
    """
    Step 5 – Stemming (Porter Stemmer)
    ------------------------------------
    Reduce each token to its root/stem form using rule-based suffix stripping.
    Stemming is fast but can produce non-dictionary words.

    Examples:
        "running"  → "run"
        "studies"  → "studi"
        "jumped"   → "jump"
    """
    return [stemmer.stem(t) for t in tokens]


def lemmatize(tokens: list) -> list:
    """
    Step 6 – Lemmatization (WordNet Lemmatizer)
    --------------------------------------------
    Reduce each token to its canonical dictionary form (lemma).
    Lemmatization is linguistically smarter than stemming.

    Examples:
        "running"  → "run"
        "better"   → "good"
        "studies"  → "study"

    Falls back to PorterStemmer if WordNet corpus is unavailable.
    """
    if _LEMMATIZER_AVAILABLE:
        return [_lemmatizer.lemmatize(t) for t in tokens]
    # Graceful fallback: use stemmer (still demonstrates the concept)
    return [stemmer.stem(t) for t in tokens]


def preprocess(text: str, use_stemming: bool = False) -> str:
    """
    Full Preprocessing Pipeline
    ---------------------------
    Applies all NLP steps in sequence and returns a single cleaned string.

    Steps:
        1. Lowercase
        2. Remove punctuation
        3. Tokenize
        4. Remove stopwords
        5. Lemmatize  (default)  OR  Stem  (if use_stemming=True)
        6. Rejoin tokens into a string for TF-IDF

    Parameters:
        text         : raw comment string
        use_stemming : if True, apply stemming instead of lemmatization

    Returns:
        Preprocessed string ready for feature extraction.
    """
    text   = lowercase(text)
    text   = remove_punctuation(text)
    tokens = tokenize(text)
    tokens = remove_stopwords(tokens)
    tokens = stem(tokens) if use_stemming else lemmatize(tokens)
    return " ".join(tokens)


# ── Quick smoke-test when run directly ────────────────────────────────────────
if __name__ == "__main__":
    sample = "You're such an IDIOT!! Nobody likes your stupid comments, get lost!!!"
    print("Original :", sample)
    print("Processed:", preprocess(sample))
    print()
    sample2 = "I really enjoyed reading this thoughtful article. Thanks for sharing!"
    print("Original :", sample2)
    print("Processed:", preprocess(sample2))
