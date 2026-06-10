"""
app.py — SmartSummarizer: AI-Powered Text Summarization & Keyword Extraction

Main Streamlit application entry point.
Provides a web interface for summarizing text and extracting keywords
using multiple NLP techniques.
"""

import os
import streamlit as st

from summarizer import summarize
from keyword_extractor import (
    extract_keywords_tfidf,
    extract_keywords_rake,
)
from utils import calculate_text_stats, generate_word_cloud, format_compression_ratio

# ──────────────────────────────────────────────
# Page Configuration
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="SmartSummarizer — Text Summarization & Keyword Extraction",
    page_icon="📝",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────
# Custom CSS for Premium Styling
# ──────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    /* ── Global ── */
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        background-color: #0a0f1e;
    }

    /* ── Hide Deploy button & hamburger menu ── */
    .stDeployButton,
    #MainMenu {
        display: none !important;
    }

    /* ── Header ── */
    .app-header {
        text-align: center;
        padding: 2rem 0 1.5rem;
    }
    .app-header h1 {
        font-size: 2.6rem;
        font-weight: 800;
        background: linear-gradient(90deg, #00d2ff 0%, #0072ff 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.4rem;
        letter-spacing: -0.5px;
    }
    .app-header p {
        color: #7a8baa;
        font-size: 1.05rem;
        margin-top: 0;
        font-weight: 400;
    }

    /* ── Metric Cards ── */
    .metric-card {
        background: linear-gradient(145deg, #0f1729 0%, #111d35 100%);
        border: 1px solid #1e2d4a;
        border-radius: 14px;
        padding: 1.3rem;
        text-align: center;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .metric-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 10px 30px rgba(0, 114, 255, 0.15);
        border-color: #0072ff44;
    }
    .metric-value {
        font-size: 1.9rem;
        font-weight: 700;
        background: linear-gradient(90deg, #00d2ff, #0072ff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .metric-label {
        font-size: 0.82rem;
        color: #5a6a88;
        margin-top: 0.3rem;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    /* ── Section Headers ── */
    .section-header {
        font-size: 1.2rem;
        font-weight: 700;
        color: #e0eaff;
        margin: 1.5rem 0 0.75rem;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #0072ff55;
        display: inline-block;
    }

    /* ── Summary Box ── */
    .summary-box {
        background: linear-gradient(145deg, #0f1729 0%, #0a1628 100%);
        border-left: 4px solid #00d2ff;
        border-radius: 0 14px 14px 0;
        padding: 1.4rem 1.6rem;
        font-size: 1.05rem;
        line-height: 1.8;
        color: #c8d8f0;
    }

    /* ── Keyword Tags ── */
    .keyword-container {
        display: flex;
        flex-wrap: wrap;
        gap: 0.5rem;
        margin-top: 0.5rem;
    }
    .keyword-tag {
        background: linear-gradient(135deg, #0072ff 0%, #00d2ff 100%);
        color: white;
        padding: 0.4rem 1rem;
        border-radius: 20px;
        font-size: 0.84rem;
        font-weight: 600;
        display: inline-flex;
        align-items: center;
        gap: 0.35rem;
        transition: transform 0.15s ease, box-shadow 0.15s ease;
        box-shadow: 0 2px 8px rgba(0,114,255,0.3);
    }
    .keyword-tag:hover {
        transform: scale(1.06);
        box-shadow: 0 4px 14px rgba(0,210,255,0.4);
    }
    .keyword-score {
        background: rgba(255,255,255,0.2);
        padding: 0.15rem 0.45rem;
        border-radius: 10px;
        font-size: 0.75rem;
    }

    /* ── Compression Badge ── */
    .compression-badge {
        background: linear-gradient(135deg, #00c6a7 0%, #0072ff 100%);
        color: white;
        padding: 0.6rem 1.3rem;
        border-radius: 25px;
        font-size: 1rem;
        font-weight: 600;
        display: inline-block;
        margin-top: 0.5rem;
        box-shadow: 0 3px 12px rgba(0,198,167,0.3);
    }

    /* ── Info Box ── */
    .info-box {
        background: #0f1729;
        border: 1px solid #1e2d4a;
        border-radius: 12px;
        padding: 1rem 1.25rem;
        font-size: 0.9rem;
        color: #7a8baa;
        line-height: 1.6;
    }

    /* ── Sidebar Styling ── */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0a0f1e 0%, #0d1530 100%);
        border-right: 1px solid #1e2d4a;
    }
    [data-testid="stSidebar"] .block-container {
        padding-top: 2rem;
    }
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] span {
        color: #a0b4d0 !important;
    }

    /* ── Sentence Score Table ── */
    .score-row {
        display: flex;
        align-items: flex-start;
        gap: 0.75rem;
        padding: 0.65rem 0;
        border-bottom: 1px solid #1e2d4a;
    }
    .score-badge {
        min-width: 52px;
        background: linear-gradient(135deg, #0072ff, #00d2ff);
        color: white;
        text-align: center;
        padding: 0.25rem 0.5rem;
        border-radius: 7px;
        font-size: 0.8rem;
        font-weight: 700;
    }
    .score-text {
        font-size: 0.9rem;
        color: #8a9bbf;
        line-height: 1.5;
    }
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────
# Load Sample Texts
# ──────────────────────────────────────────────
def load_sample_texts() -> dict[str, str]:
    """Load all .txt files from assets/sample_texts/ directory."""
    samples = {}
    sample_dir = os.path.join(os.path.dirname(__file__), "assets", "sample_texts")
    if os.path.exists(sample_dir):
        for filename in sorted(os.listdir(sample_dir)):
            if filename.endswith(".txt"):
                filepath = os.path.join(sample_dir, filename)
                with open(filepath, "r", encoding="utf-8") as f:
                    name = filename.replace("_", " ").replace(".txt", "").title()
                    samples[name] = f.read()
    return samples


# ──────────────────────────────────────────────
# App Header
# ──────────────────────────────────────────────
st.markdown("""
<div class="app-header">
    <h1>📝 SmartSummarizer</h1>
    <p>AI-Powered Text Summarization & Keyword Extraction</p>
</div>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────
# Sidebar — Input Panel
# ──────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Controls")
    st.markdown("---")

    # Sample text loader
    sample_texts = load_sample_texts()
    if sample_texts:
        st.markdown("##### 📂 Load Sample Text")
        sample_choice = st.selectbox(
            "Choose a sample article:",
            options=["— Select —"] + list(sample_texts.keys()),
            label_visibility="collapsed",
        )
    else:
        sample_choice = "— Select —"

    st.markdown("---")

    # Summary length slider
    st.markdown("##### 📏 Summary Length")
    num_sentences = st.slider(
        "Number of sentences in summary:",
        min_value=1,
        max_value=15,
        value=3,
        step=1,
        help="Choose how many sentences should be included in the summary.",
    )

    # Keywords count slider
    st.markdown("##### 🔑 Number of Keywords")
    num_keywords = st.slider(
        "Keywords to extract:",
        min_value=5,
        max_value=20,
        value=10,
        step=1,
        help="Choose how many keywords to extract from the text.",
    )

    # Keyword method selector
    st.markdown("##### 🧪 Keyword Extraction Method")
    keyword_method = st.radio(
        "Select method:",
        options=["TF-IDF", "RAKE"],
        index=0,
        help="TF-IDF: statistical importance | RAKE: multi-word phrases",
        label_visibility="collapsed",
    )

    st.markdown("---")

    # Summarize button
    summarize_btn = st.button(
        "🚀 Summarize Now",
        use_container_width=True,
        type="primary",
    )

    # st.markdown("---")
    # st.markdown(
    #     '<div class="info-box">'
    #     "<strong>💡 How it works:</strong><br>"
    #     "1. Text is cleaned & tokenized<br>"
    #     "2. Stopwords are removed<br>"
    #     "3. Words are lemmatized<br>"
    #     "4. Sentences scored by word frequency<br>"
    #     "5. Top sentences selected as summary<br>"
    #     "6. Keywords extracted via TF-IDF / RAKE"
    #     "</div>",
    #     unsafe_allow_html=True,
    # )


# ──────────────────────────────────────────────
# Main Panel — Text Input
# ──────────────────────────────────────────────

# Determine initial text value
default_text = ""
if sample_choice != "— Select —" and sample_choice in sample_texts:
    default_text = sample_texts[sample_choice]

input_text = st.text_area(
    "📄 Paste your text below (news article, research paper, assignment, etc.):",
    value=default_text,
    height=250,
    placeholder="Paste or type any long text here... The app will summarize it and extract keywords.",
)


# ──────────────────────────────────────────────
# Processing & Output
# ──────────────────────────────────────────────
if summarize_btn:
    if not input_text or len(input_text.strip()) < 50:
        st.warning("⚠️ Please enter at least a few sentences of text to summarize.")
    else:
        with st.spinner("🔄 Analyzing text with NLP pipeline..."):

            # ── Text Stats ──
            stats = calculate_text_stats(input_text)

            # ── Summarization ──
            result = summarize(input_text, num_sentences=num_sentences)

            # ── Keyword Extraction ──
            if keyword_method == "TF-IDF":
                keywords = extract_keywords_tfidf(input_text, num_keywords=num_keywords)
            else:
                keywords = extract_keywords_rake(input_text, num_keywords=num_keywords)

        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # OUTPUT SECTION
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

        st.markdown("---")

        # ── 1. Original Text Stats ──
        st.markdown('<div class="section-header">📊 Original Text Statistics</div>', unsafe_allow_html=True)

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.markdown(
                f'<div class="metric-card">'
                f'<div class="metric-value">{stats["word_count"]}</div>'
                f'<div class="metric-label">Total Words</div>'
                f"</div>",
                unsafe_allow_html=True,
            )
        with col2:
            st.markdown(
                f'<div class="metric-card">'
                f'<div class="metric-value">{stats["sentence_count"]}</div>'
                f'<div class="metric-label">Sentences</div>'
                f"</div>",
                unsafe_allow_html=True,
            )
        with col3:
            st.markdown(
                f'<div class="metric-card">'
                f'<div class="metric-value">{stats["reading_time_display"]}</div>'
                f'<div class="metric-label">Reading Time</div>'
                f"</div>",
                unsafe_allow_html=True,
            )
        with col4:
            st.markdown(
                f'<div class="metric-card">'
                f'<div class="metric-value">{stats["avg_sentence_length"]}</div>'
                f'<div class="metric-label">Avg Words/Sentence</div>'
                f"</div>",
                unsafe_allow_html=True,
            )

        st.markdown("<br>", unsafe_allow_html=True)

        # ── 2. Summary Section ──
        st.markdown('<div class="section-header">📝 Generated Summary</div>', unsafe_allow_html=True)

        if result["summary"]:
            st.markdown(
                f'<div class="summary-box">{result["summary"]}</div>',
                unsafe_allow_html=True,
            )

            # Compression ratio
            st.markdown(
                f'<div class="compression-badge">'
                f'{format_compression_ratio(result["compression_ratio"])} '
                f'({result["original_sentence_count"]} → {result["summary_sentence_count"]} sentences)'
                f"</div>",
                unsafe_allow_html=True,
            )

            # Summary reading time
            summary_stats = calculate_text_stats(result["summary"])
            st.markdown(
                f"⏱️ **Summary reading time:** {summary_stats['reading_time_display']} "
                f"(Original: {stats['reading_time_display']})",
            )
        else:
            st.info("Could not generate a summary. Please provide more text.")

        st.markdown("<br>", unsafe_allow_html=True)

        # ── 3. Keywords Section ──
        st.markdown(
            f'<div class="section-header">🔑 Extracted Keywords ({keyword_method})</div>',
            unsafe_allow_html=True,
        )

        if keywords:
            tags_html = '<div class="keyword-container">'
            for word, score in keywords:
                tags_html += (
                    f'<span class="keyword-tag">'
                    f"{word} "
                    f'<span class="keyword-score">{score}</span>'
                    f"</span>"
                )
            tags_html += "</div>"
            st.markdown(tags_html, unsafe_allow_html=True)
        else:
            st.info("No keywords extracted. Please provide more text.")

        st.markdown("<br>", unsafe_allow_html=True)

        # ── 4. Word Cloud ──
        # st.markdown('<div class="section-header">☁️ Word Cloud</div>', unsafe_allow_html=True)

        # fig = generate_word_cloud(keywords)
        # st.pyplot(fig)

        # st.markdown("<br>", unsafe_allow_html=True)

        # ── 5. Sentence Scores (Expandable) ──
        with st.expander("📈 View All Sentence Scores (NLP Scoring Details)", expanded=False):
            st.markdown(
                "Each sentence is scored by summing the normalized word frequencies "
                "of its constituent words. Higher score = more important sentence."
            )
            st.markdown("")

            for idx, (sentence, score) in enumerate(result["scored_sentences"], 1):
                is_selected = sentence.strip() in result["summary"]
                marker = "✅" if is_selected else ""
                st.markdown(
                    f'<div class="score-row">'
                    f'<span class="score-badge">{score}</span>'
                    f'<span class="score-text">{marker} {sentence}</span>'
                    f"</div>",
                    unsafe_allow_html=True,
                )

        # ── 6. NLP Pipeline Explanation ──
#         with st.expander("🧠 NLP Pipeline — How It Works Behind the Scenes", expanded=False):
#             st.markdown("""
# ### Extractive Summarization Algorithm

# ```
# Input Text
#     ↓
# Step 1: Sentence Tokenization        → split into list of sentences
#     ↓
# Step 2: Word Tokenization            → split each sentence into words
#     ↓
# Step 3: Remove Stopwords             → filter out "the", "is", "a", etc.
#     ↓
# Step 4: Lemmatization                → reduce words to root form
#     ↓
# Step 5: Word Frequency Calculation   → count how often each word appears
#     ↓
# Step 6: Normalize Frequencies        → divide by max frequency (0 to 1 scale)
#     ↓
# Step 7: Sentence Scoring             → sum word frequencies for each sentence
#     ↓
# Step 8: Select Top N Sentences       → rank and pick highest-scoring ones
#     ↓
# Step 9: Reconstruct in Order         → preserve original sentence order
#     ↓
# Output: Summary
# ```

# ### Keyword Extraction (TF-IDF)

# ```
# Input Text
#     ↓
# Preprocess (tokenize, remove stopwords, lemmatize)
#     ↓
# Calculate TF  →  how often a word appears in THIS text
#     ↓
# Calculate IDF →  how rare the word is across general language
#     ↓
# TF-IDF Score  =  TF × IDF
#     ↓
# Rank words by score
#     ↓
# Output: Top N Keywords
# ```

# ### Libraries Used

# | NLP Technique | Library | Where Used |
# |---|---|---|
# | Tokenization | `nltk.tokenize` | Preprocessing |
# | Stopword Removal | `nltk.corpus.stopwords` | Preprocessing |
# | Lemmatization | `nltk.stem.WordNetLemmatizer` | Preprocessing |
# | Word Frequency Analysis | Pure Python + NLTK | Sentence scoring |
# | Extractive Summarization | Custom algorithm (NLTK) | Summary generation |
# | TF-IDF Keywords | `sklearn.TfidfVectorizer` | Keyword extraction |
# | RAKE Keywords | `rake-nltk` | Keyphrase extraction |
# | Word Cloud | `wordcloud` library | Visual output |
#             """)


# ──────────────────────────────────────────────
# Footer
# ──────────────────────────────────────────────
st.markdown("---")
st.markdown(
    '<p style="text-align:center; color:#555; font-size:0.85rem;">'
    "SmartSummarizer — NLP Course Project | Built with Streamlit & NLTK | Python 3.x"
    "</p>",
    unsafe_allow_html=True,
)
