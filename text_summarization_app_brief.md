# 📝 Text Summarization & Keyword Extraction App
### NLP Project Brief — Streamlit Application

---

## 🎯 Project Overview

**Project Title:** SmartSummarizer — AI-Powered Text Summarization & Keyword Extraction App  
**Framework:** Streamlit  
**Purpose:** Help students, readers, and professionals quickly understand long texts — news articles, research papers, assignments — by generating concise summaries and extracting the most important keywords.

---

## 🔴 Real-World Problem Being Solved

Students and professionals face an information overload problem every day:
- News articles are too long to read fully
- Research papers take hours to process
- Assignment readings pile up quickly

**Solution:** A smart app that takes any long text (or URL/PDF) as input and instantly returns:
1. A clean, readable **summary**
2. A set of **keywords** that capture the core topics

---

## 🧠 NLP Techniques Used (Minimum 2 — Rule Satisfied)

| # | Technique | Purpose |
|---|-----------|---------|
| 1 | **Extractive Text Summarization** | Picks the most important sentences directly from the original text |
| 2 | **Keyword Extraction (TF-IDF / RAKE)** | Identifies the most relevant and frequent terms in the text |
| 3 | **Text Preprocessing (Bonus)** | Tokenization, stopword removal, lemmatization before analysis |
| 4 | **Sentence Scoring (Bonus)** | Ranks sentences by importance using word frequency scoring |

---

## 📚 NLP Topics Covered by This Application

### 1. Text Preprocessing
- **Tokenization** — splitting text into words and sentences
- **Stopword Removal** — removing common words like "the", "is", "and"
- **Lemmatization / Stemming** — reducing words to their base form (e.g., "running" → "run")
- **Punctuation & Noise Removal** — cleaning raw input text

### 2. Text Summarization
- **Extractive Summarization** — scores each sentence and selects the top N most important ones
- **Word Frequency Method** — calculates how often meaningful words appear and uses that to score sentences
- **Sentence Ranking** — sorts sentences by score, picks the best ones

### 3. Keyword Extraction
- **TF-IDF (Term Frequency – Inverse Document Frequency)** — finds words that are important in this text but not common everywhere
- **RAKE (Rapid Automatic Keyword Extraction)** — extracts multi-word keyword phrases
- **Frequency-Based Extraction** — simply picks the most frequent non-stopword terms

### 4. Core NLP Concepts Demonstrated
- Bag of Words (BoW) model understanding
- N-grams (for multi-word keywords)
- Corpus and vocabulary concepts
- Natural Language Toolkit (NLTK) usage

---

## 🛠️ Tech Stack

| Layer | Tool |
|-------|------|
| **UI / Frontend** | Streamlit |
| **NLP Library** | NLTK (Natural Language Toolkit) |
| **Keyword Extraction** | `rake-nltk` or `sklearn` TF-IDF |
| **Text Processing** | NLTK tokenizers, stopwords corpus |
| **Visualization** | `wordcloud`, `matplotlib` (keyword cloud display) |
| **Language** | Python 3.x |

---

## 📁 Project Structure

```
text-summarizer-app/
│
├── app.py                  # Main Streamlit app entry point
├── summarizer.py           # Extractive summarization logic
├── keyword_extractor.py    # Keyword & keyphrase extraction logic
├── preprocessor.py         # Text cleaning and preprocessing
├── utils.py                # Helper functions (word cloud, display, etc.)
│
├── requirements.txt        # All Python dependencies
├── README.md               # Project documentation
│
└── assets/
    └── sample_texts/       # Sample articles for demo/testing
```

---

## 🖥️ App Features & UI Layout

### Sidebar (Input Panel)
- Text input area (paste any article or paragraph)
- Slider: **Summary Length** — choose how many sentences (3, 5, 7, 10)
- Slider: **Number of Keywords** — choose how many keywords to extract (5–20)
- Button: **"Summarize Now"**

### Main Panel (Output)
1. **Original Text Stats** — word count, sentence count, reading time estimate
2. **Summary Section** — clean summarized text with highlighted sentences
3. **Keywords Section** — displayed as tags/badges
4. **Word Cloud** — visual keyword cloud (matplotlib/wordcloud)
5. **Compression Ratio** — shows how much the text was reduced (e.g., "Reduced by 72%")

---

## ⚙️ Core Algorithm — Extractive Summarization (Step-by-Step)

```
Input Text
    ↓
Step 1: Sentence Tokenization        → split into list of sentences
    ↓
Step 2: Word Tokenization            → split each sentence into words
    ↓
Step 3: Remove Stopwords             → filter out "the", "is", "a", etc.
    ↓
Step 4: Lemmatization                → reduce words to root form
    ↓
Step 5: Word Frequency Calculation   → count how often each word appears
    ↓
Step 6: Normalize Frequencies        → divide by max frequency (0 to 1 scale)
    ↓
Step 7: Sentence Scoring             → sum word frequencies for each sentence
    ↓
Step 8: Select Top N Sentences       → rank and pick highest-scoring ones
    ↓
Step 9: Reconstruct in Order         → preserve original sentence order
    ↓
Output: Summary
```

---

## ⚙️ Core Algorithm — Keyword Extraction (TF-IDF Approach)

```
Input Text
    ↓
Preprocess (tokenize, remove stopwords, lemmatize)
    ↓
Calculate TF  →  how often a word appears in THIS text
    ↓
Calculate IDF →  how rare the word is across general language
    ↓
TF-IDF Score  =  TF × IDF
    ↓
Rank words by score
    ↓
Output: Top N Keywords
```

---

## 📦 Python Dependencies (requirements.txt)

```
streamlit
nltk
rake-nltk
scikit-learn
matplotlib
wordcloud
pandas
```

---

## 🚀 How to Run the App

```bash
# Step 1: Install dependencies
pip install -r requirements.txt

# Step 2: Download NLTK data (run once)
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet')"

# Step 3: Launch the Streamlit app
streamlit run app.py
```

---

## 🎓 NLP Techniques Summary Table (For Teacher)

| NLP Technique | Library Used | Where Used in App |
|---------------|-------------|-------------------|
| Tokenization | `nltk.tokenize` | Preprocessing step |
| Stopword Removal | `nltk.corpus.stopwords` | Preprocessing step |
| Lemmatization | `nltk.stem.WordNetLemmatizer` | Preprocessing step |
| Word Frequency Analysis | Pure Python + NLTK | Sentence scoring |
| Extractive Summarization | Custom algorithm (NLTK-based) | Summary generation |
| TF-IDF Keyword Extraction | `sklearn.TfidfVectorizer` | Keyword extraction |
| RAKE Keyword Extraction | `rake-nltk` | Keyword phrases |
| Word Cloud Visualization | `wordcloud` library | Visual output |

---

## ✅ Project Requirements Checklist

- [x] Real-world problem identified ✔ (information overload / long text reading)
- [x] At least 2 NLP techniques ✔ (Summarization + Keyword Extraction)
- [x] Demoable ✔ (Streamlit web app, runs locally or on Streamlit Cloud)
- [x] Built with Streamlit ✔
- [x] Uses NLTK (open-source NLP library) ✔

---

## 🌐 Demo Flow (For Presentation)

1. Open the Streamlit app in browser
2. Paste a sample news article (e.g., from BBC or Dawn News)
3. Set summary length to 3 sentences
4. Set keywords to 10
5. Click **"Summarize Now"**
6. Show the output: summary, keywords, word cloud, compression ratio
7. Explain each NLP step that happened behind the scenes

---

*Project for NLP Course | Framework: Streamlit | Language: Python*
