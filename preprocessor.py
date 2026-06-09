"""
preprocessor.py — Text Cleaning and Preprocessing Module

NLP Techniques Used:
  - Tokenization (sentence + word level)
  - Stopword Removal
  - Lemmatization
  - Punctuation & Noise Removal
"""

import re
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


def ensure_nltk_data():
    """Download required NLTK data packages if not already present."""
    packages = [
        ("tokenizers/punkt_tab", "punkt_tab"),
        ("corpora/stopwords", "stopwords"),
        ("corpora/wordnet", "wordnet"),
    ]
    for path, name in packages:
        try:
            nltk.data.find(path)
        except LookupError:
            nltk.download(name, quiet=True)


# Ensure NLTK data is available on import
ensure_nltk_data()


def clean_text(text: str) -> str:
    """
    Remove noise from raw text input:
      - Extra whitespace
      - Special characters (keeping sentence-ending punctuation)
      - URLs
      - Email addresses
    """
    # Remove URLs
    text = re.sub(r"https?://\S+|www\.\S+", "", text)
    # Remove email addresses
    text = re.sub(r"\S+@\S+\.\S+", "", text)
    # Replace multiple whitespace / newlines with a single space
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def tokenize_sentences(text: str) -> list[str]:
    """Split text into a list of sentences using NLTK punkt tokenizer."""
    return sent_tokenize(text)


def tokenize_words(text: str) -> list[str]:
    """Split text into a list of word tokens."""
    return word_tokenize(text)


def remove_stopwords(words: list[str]) -> list[str]:
    """Remove English stopwords and non-alphabetic tokens."""
    stop_words = set(stopwords.words("english"))
    return [w for w in words if w.lower() not in stop_words and w.isalpha()]


def lemmatize_words(words: list[str]) -> list[str]:
    """Reduce words to their base/root form using WordNet lemmatizer."""
    lemmatizer = WordNetLemmatizer()
    return [lemmatizer.lemmatize(w.lower()) for w in words]


def preprocess(text: str) -> dict:
    """
    Full preprocessing pipeline.

    Returns a dict with:
      - cleaned_text: noise-removed text
      - sentences: list of sentence strings
      - words: raw word tokens
      - filtered_words: stopwords removed
      - lemmatized_words: lemmatized tokens
    """
    cleaned = clean_text(text)
    sentences = tokenize_sentences(cleaned)
    words = tokenize_words(cleaned)
    filtered = remove_stopwords(words)
    lemmatized = lemmatize_words(filtered)

    return {
        "cleaned_text": cleaned,
        "sentences": sentences,
        "words": words,
        "filtered_words": filtered,
        "lemmatized_words": lemmatized,
    }
