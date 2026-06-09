# 📝 SmartSummarizer — AI-Powered Text Summarization & Keyword Extraction App

A Streamlit-based NLP application that helps students, readers, and professionals quickly understand long texts by generating concise summaries and extracting the most important keywords.

---

## 🎯 Problem Statement

Students and professionals face an **information overload** problem:
- News articles are too long to read fully
- Research papers take hours to process
- Assignment readings pile up quickly

**Solution:** SmartSummarizer takes any long text and instantly returns a clean summary and key topics.

---

## 🧠 NLP Techniques Used

| # | Technique | Purpose |
|---|-----------|---------|
| 1 | **Extractive Text Summarization** | Picks the most important sentences from the original text |
| 2 | **TF-IDF Keyword Extraction** | Identifies the most relevant terms using statistical scoring |
| 3 | **RAKE Keyword Extraction** | Extracts multi-word keyphrases |
| 4 | **Text Preprocessing** | Tokenization, stopword removal, lemmatization |
| 5 | **Sentence Scoring** | Ranks sentences by importance using word frequency |

---

## 🛠️ Tech Stack

| Layer | Tool |
|-------|------|
| **UI / Frontend** | Streamlit |
| **NLP Library** | NLTK (Natural Language Toolkit) |
| **Keyword Extraction** | `rake-nltk`, `scikit-learn` TF-IDF |
| **Visualization** | `wordcloud`, `matplotlib` |
| **Language** | Python 3.x |

---

## 📁 Project Structure

```
NLP-Project/
│
├── app.py                  # Main Streamlit app entry point
├── summarizer.py           # Extractive summarization logic
├── keyword_extractor.py    # Keyword & keyphrase extraction
├── preprocessor.py         # Text cleaning and preprocessing
├── utils.py                # Helper functions (word cloud, stats)
│
├── requirements.txt        # Python dependencies
├── README.md               # This file
│
└── assets/
    └── sample_texts/       # Sample articles for demo
```

---

## 🚀 How to Run

### Prerequisites
- Python 3.8 or higher

### Setup & Launch

```bash
# Step 1: Install dependencies
pip install -r requirements.txt

# Step 2: Download NLTK data (automatic on first run, or run manually)
python -c "import nltk; nltk.download('punkt_tab'); nltk.download('stopwords'); nltk.download('wordnet')"

# Step 3: Launch the Streamlit app
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`.

---

## 🖥️ App Features

### Sidebar (Input Panel)
- **Sample Text Loader** — pre-loaded articles for quick demo
- **Summary Length Slider** — choose 1–15 sentences
- **Keywords Count Slider** — choose 5–20 keywords
- **Extraction Method** — TF-IDF, RAKE, or Frequency
- **"Summarize Now" Button** — triggers the NLP pipeline

### Main Panel (Output)
1. **Original Text Stats** — word count, sentence count, reading time
2. **Generated Summary** — extracted top sentences
3. **Compression Ratio** — shows how much text was reduced
4. **Keywords** — displayed as color-coded tags with scores
5. **Word Cloud** — visual keyword cloud
6. **Sentence Scores** — expandable view of all sentence rankings
7. **NLP Pipeline Explanation** — step-by-step algorithm breakdown

---

## 🌐 Demo Flow

1. Open the Streamlit app in browser
2. Paste a sample news article (or use the built-in sample)
3. Set summary length to **3 sentences**
4. Set keywords to **10**
5. Click **"Summarize Now"**
6. View: summary, keywords, word cloud, compression ratio
7. Expand the NLP pipeline section to explain the algorithms

---

## 📊 NLP Techniques Summary

| NLP Technique | Library Used | Where Used |
|---|---|---|
| Tokenization | `nltk.tokenize` | Preprocessing |
| Stopword Removal | `nltk.corpus.stopwords` | Preprocessing |
| Lemmatization | `nltk.stem.WordNetLemmatizer` | Preprocessing |
| Word Frequency Analysis | Pure Python + NLTK | Sentence scoring |
| Extractive Summarization | Custom algorithm (NLTK) | Summary generation |
| TF-IDF Keywords | `sklearn.TfidfVectorizer` | Keyword extraction |
| RAKE Keywords | `rake-nltk` | Keyphrase extraction |
| Word Cloud | `wordcloud` library | Visual output |

---

*Project for NLP Course | Framework: Streamlit | Language: Python*
