"""
app.py
======
Streamlit Web Interface for the Toxic Comment Classifier

Run with:
    streamlit run app.py

Requires trained model:
    python src/train.py   (run this first)
"""

import os
import sys
import streamlit as st

# Make src/ importable when running from project root
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.predict import predict

# ── Page configuration ─────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Toxic Comment Classifier",
    page_icon="🛡️",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ── Custom CSS styling ─────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,500;9..144,600;9..144,900&family=Public+Sans:wght@400;500;600;700&family=JetBrains+Mono:wght@500;700&display=swap');

    :root {
        --bg:         #14120d;
        --bg-2:       #211d15;
        --ink:        #ece4d2;
        --ink-soft:   #968c74;
        --rule:       #3d3728;
        --red:        #e23b2b;
        --red-deep:   #4d0f08;
        --green:      #36a866;
        --green-deep: #08301c;
        --amber:      #e7a93a;
        --cream:      #faf6e8;
    }

    /* Base typography */
    html, body, [class*="css"], .stApp {
        font-family: 'Public Sans', sans-serif;
    }

    /* Dark paper background with a faint dot-grid texture */
    .stApp {
        background-color: var(--bg);
        background-image: radial-gradient(rgba(255,255,255,0.045) 1px, transparent 1px);
        background-size: 24px 24px;
    }

    /* Hide Streamlit chrome */
    #MainMenu, footer, header,
    [data-testid="stToolbar"], [data-testid="stDecoration"] { visibility: hidden; }

    /* Main container */
    .main .block-container {
        max-width: 680px;
        padding-top: 2.2rem;
        padding-bottom: 3.5rem;
        color: var(--ink);
    }

    /* ── Masthead ──────────────────────────────────────────────── */
    .masthead { margin-bottom: 1.9rem; }
    .masthead__kicker {
        display: flex;
        align-items: center;
        gap: 0.55rem;
        font-size: 0.74rem;
        font-weight: 700;
        letter-spacing: 0.16em;
        text-transform: uppercase;
        color: var(--red);
        margin-bottom: 0.85rem;
    }
    .masthead__kicker::before {
        content: "";
        width: 11px; height: 11px;
        background: var(--red);
        display: inline-block;
    }
    .masthead__title {
        font-family: 'Fraunces', serif;
        font-weight: 900;
        font-size: 3.35rem;
        line-height: 0.94;
        letter-spacing: -0.02em;
        color: var(--ink);
        margin: 0 0 0.85rem 0;
    }
    .masthead__lede {
        font-size: 1.02rem;
        line-height: 1.55;
        color: var(--ink-soft);
        max-width: 33rem;
        margin: 0;
    }
    .specs {
        margin-top: 1.25rem;
        padding-top: 0.7rem;
        border-top: 2px solid var(--ink);
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.72rem;
        font-weight: 500;
        letter-spacing: 0.04em;
        text-transform: uppercase;
        color: var(--ink-soft);
    }

    /* ── Section labels ───────────────────────────────────────────── */
    .section-label {
        font-size: 0.76rem;
        font-weight: 700;
        letter-spacing: 0.13em;
        text-transform: uppercase;
        color: var(--ink);
        margin: 1.7rem 0 0.55rem 0;
    }
    .section-label span { color: var(--ink-soft); font-weight: 500; }

    /* ── Textarea ─────────────────────────────────────────────────── */
    .stTextArea textarea {
        font-family: 'Public Sans', sans-serif !important;
        font-size: 1rem !important;
        border-radius: 0 !important;
        border: 2px solid var(--ink) !important;
        background: var(--bg-2) !important;
        color: var(--ink) !important;
        padding: 0.95rem 1.05rem !important;
        line-height: 1.6 !important;
        resize: vertical !important;
        box-shadow: 4px 4px 0 var(--rule) !important;
    }
    .stTextArea textarea:focus {
        border-color: var(--amber) !important;
        box-shadow: 4px 4px 0 var(--amber) !important;
    }
    .stTextArea textarea::placeholder { color: #6b6453 !important; }

    /* ── Buttons ──────────────────────────────────────────────────── */
    /* Sample chips (secondary) */
    .stButton button[kind="secondary"] {
        background: var(--bg-2) !important;
        color: var(--ink) !important;
        border: 2px solid var(--ink) !important;
        border-radius: 0 !important;
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 0.8rem !important;
        font-weight: 700 !important;
        letter-spacing: 0.03em !important;
        padding: 0.45rem 0.6rem !important;
        width: 100% !important;
        transition: background 0.12s, color 0.12s !important;
    }
    .stButton button[kind="secondary"]:hover {
        background: var(--amber) !important;
        color: var(--bg) !important;
        border-color: var(--ink) !important;
    }
    .stButton button[kind="secondary"]:focus { box-shadow: none !important; }

    /* Classify button (primary) */
    .stButton button[kind="primary"] {
        background: var(--amber) !important;
        color: var(--bg) !important;
        border: 2px solid var(--ink) !important;
        border-radius: 0 !important;
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 0.95rem !important;
        font-weight: 700 !important;
        letter-spacing: 0.06em !important;
        text-transform: uppercase !important;
        padding: 0.8rem 1.8rem !important;
        width: 100% !important;
        box-shadow: 5px 5px 0 var(--red) !important;
        transition: transform 0.1s, box-shadow 0.1s, background 0.1s !important;
    }
    .stButton button[kind="primary"]:hover {
        background: var(--red) !important;
        color: var(--cream) !important;
        border-color: var(--ink) !important;
        transform: translate(2px, 2px) !important;
        box-shadow: 3px 3px 0 var(--ink) !important;
    }
    .stButton button[kind="primary"]:active {
        transform: translate(5px, 5px) !important;
        box-shadow: 0 0 0 var(--ink) !important;
    }

    /* ── Verdict cards ────────────────────────────────────────────── */
    .verdict {
        border: 3px solid var(--ink);
        padding: 1.5rem 1.6rem;
        margin-top: 1.3rem;
    }
    .verdict--toxic { background: var(--red);   box-shadow: 7px 7px 0 var(--red-deep);   }
    .verdict--clean { background: var(--green); box-shadow: 7px 7px 0 var(--green-deep); }
    .verdict__tag {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 0.18em;
        text-transform: uppercase;
        color: rgba(250,246,232,0.78);
        margin-bottom: 0.25rem;
    }
    .verdict__label {
        font-family: 'Fraunces', serif;
        font-size: 2.45rem;
        font-weight: 900;
        line-height: 1;
        letter-spacing: -0.02em;
        color: var(--cream);
        margin-bottom: 0.5rem;
    }
    .verdict__sub {
        font-size: 0.95rem;
        line-height: 1.5;
        color: rgba(250,246,232,0.93);
        margin: 0;
    }

    /* Confidence meter */
    .conf {
        margin-top: 1.15rem;
        padding-top: 1rem;
        border-top: 2px solid rgba(250,246,232,0.38);
    }
    .conf__top {
        display: flex;
        justify-content: space-between;
        align-items: baseline;
        margin-bottom: 0.5rem;
    }
    .conf__name {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.74rem;
        font-weight: 500;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: rgba(250,246,232,0.88);
    }
    .conf__pct {
        font-family: 'Fraunces', serif;
        font-size: 1.7rem;
        font-weight: 900;
        color: var(--cream);
    }
    .conf__track {
        height: 14px;
        background: rgba(0,0,0,0.34);
        border: 2px solid var(--cream);
    }
    .conf__fill { height: 100%; background: var(--cream); }

    /* ── Note panel ───────────────────────────────────────────────── */
    .note {
        background: var(--bg-2);
        border: 2px solid var(--ink);
        border-left: 8px solid var(--amber);
        padding: 1rem 1.2rem;
        margin-top: 1.9rem;
        font-size: 0.9rem;
        line-height: 1.65;
        color: var(--ink-soft);
    }
    .note strong { color: var(--ink); }
    .note .note__head {
        display: block;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.74rem;
        font-weight: 700;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        color: var(--ink);
        margin-bottom: 0.35rem;
    }

    /* ── Expander ─────────────────────────────────────────────────── */
    [data-testid="stExpander"] details {
        border: 2px solid var(--ink) !important;
        border-radius: 0 !important;
        background: var(--bg-2) !important;
    }
    [data-testid="stExpander"] summary {
        font-family: 'JetBrains Mono', monospace !important;
        font-weight: 700 !important;
        font-size: 0.82rem !important;
        letter-spacing: 0.05em !important;
        color: var(--ink) !important;
    }
    [data-testid="stExpander"] table {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.84rem;
    }
    [data-testid="stExpander"] th,
    [data-testid="stExpander"] td {
        border-color: var(--rule) !important;
        color: var(--ink) !important;
    }

    /* Divider */
    .rule {
        border: none;
        border-top: 2px solid var(--ink);
        margin: 1.7rem 0 0.2rem 0;
    }
</style>
""", unsafe_allow_html=True)


# ── Masthead ───────────────────────────────────────────────────────────────────
st.markdown("""
<div class="masthead">
    <div class="masthead__kicker">Content Safety &nbsp;·&nbsp; NLP Demo</div>
    <h1 class="masthead__title">Toxic Comment<br>Classifier</h1>
    <p class="masthead__lede">
        Paste any comment below to check whether it reads as toxic or civil.
        Predictions come from a logistic-regression model trained on labeled
        comment data — not a rule list.
    </p>
    <div class="specs">
        TF-IDF Features &nbsp;/&nbsp; Logistic Regression &nbsp;/&nbsp;
        NLTK Preprocessing &nbsp;/&nbsp; Binary Classifier
    </div>
</div>
""", unsafe_allow_html=True)


# ── Sample comment shortcuts ────────────────────────────────────────────────────
SAMPLES = {
    "Toxic":      "You are such an idiot, nobody cares about your stupid opinion. Get lost!",
    "Civil":      "I really enjoyed reading this article. Very informative and well written!",
    "Borderline": "I disagree with your approach, but I understand your reasoning.",
}

# The comment lives in session_state so that loading a sample and then clicking
# Classify does not wipe the textarea (Streamlit reruns the whole script).
if "comment_box" not in st.session_state:
    st.session_state["comment_box"] = ""

st.markdown(
    '<p class="section-label">Load an example &nbsp;<span>— or write your own below</span></p>',
    unsafe_allow_html=True,
)
cols = st.columns(len(SAMPLES))
for col, (label, text) in zip(cols, SAMPLES.items()):
    with col:
        if st.button(label, key=f"sample_{label}"):
            # Set the value *before* the textarea is created on this rerun.
            st.session_state["comment_box"] = text

# ── Text input area ─────────────────────────────────────────────────────────────
st.markdown('<p class="section-label">Comment to classify</p>', unsafe_allow_html=True)
st.text_area(
    label="Comment to classify",
    key="comment_box",
    height=130,
    placeholder="Type or paste any comment here, then run the classifier…",
    label_visibility="collapsed",
)
comment = st.session_state["comment_box"]

# ── Classify button ─────────────────────────────────────────────────────────────
classify_clicked = st.button("Classify Comment", type="primary")

# ── Prediction result ───────────────────────────────────────────────────────────
if classify_clicked:
    if not comment.strip():
        st.warning("Please enter a comment before classifying.")
    else:
        with st.spinner("Analyzing comment…"):
            try:
                result = predict(comment)
            except FileNotFoundError as e:
                st.error(f"{e}")
                st.stop()

        label      = result["label"]
        confidence = result["confidence"]
        toxic_prob = result["toxic_prob"]
        is_toxic   = label == "Toxic"

        card_class = "verdict--toxic" if is_toxic else "verdict--clean"
        tag_text   = "Verdict — Flagged" if is_toxic else "Verdict — Cleared"
        big_label  = "Toxic" if is_toxic else "Civil"
        sub_text   = (
            "This comment contains language the model reads as harmful or offensive."
            if is_toxic else
            "This comment reads as respectful — no toxic language detected."
        )

        st.markdown(f"""
        <div class="verdict {card_class}">
            <div class="verdict__tag">{tag_text}</div>
            <div class="verdict__label">{big_label}</div>
            <p class="verdict__sub">{sub_text}</p>
            <div class="conf">
                <div class="conf__top">
                    <span class="conf__name">Model confidence</span>
                    <span class="conf__pct">{confidence}%</span>
                </div>
                <div class="conf__track">
                    <div class="conf__fill" style="width:{confidence}%"></div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Extra detail expander
        with st.expander("Technical details"):
            st.markdown(f"""
| Metric | Value |
|---|---|
| **Predicted Label** | `{label}` |
| **Confidence Score** | `{confidence}%` |
| **P(Toxic)** | `{toxic_prob}` |
| **P(Non-Toxic)** | `{round(1 - toxic_prob, 4)}` |
| **Model** | Logistic Regression (L2, balanced) |
| **Features** | TF-IDF (unigrams + bigrams, top 10k) |
            """)


# ── How it works ────────────────────────────────────────────────────────────────
st.markdown('<hr class="rule">', unsafe_allow_html=True)
st.markdown("""
<div class="note">
    <span class="note__head">How it works</span>
    Your comment is preprocessed with <strong>NLTK</strong> (lowercase → punctuation
    removal → stopword removal → lemmatization), converted into a numerical
    representation using <strong>TF-IDF</strong>, and classified by a
    <strong>Logistic Regression</strong> model trained on labeled comment data.
    The confidence score is the model's predicted probability for the output class.
</div>
""", unsafe_allow_html=True)
