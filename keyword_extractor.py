"""
keyword_extractor.py — Keyword & Keyphrase Extraction Module

NLP Techniques:
  1. TF-IDF (Term Frequency – Inverse Document Frequency) via scikit-learn
  2. RAKE (Rapid Automatic Keyword Extraction) via rake-nltk
  3. Frequency-Based Extraction (simple word frequency ranking)
"""

from sklearn.feature_extraction.text import TfidfVectorizer
from rake_nltk import Rake

from preprocessor import preprocess


def extract_keywords_tfidf(text: str, num_keywords: int = 10) -> list[tuple[str, float]]:
    """
    Extract keywords using TF-IDF scoring.

    How it works:
      - TF (Term Frequency): how often a word appears in THIS text
      - IDF (Inverse Document Frequency): how rare the word is in general
      - TF-IDF = TF × IDF → high score = important AND relatively unique

    We split the text into sentences and treat each sentence as a "document"
    so that TF-IDF can calculate meaningful IDF values across sentence-documents.

    Args:
        text: Raw input text.
        num_keywords: Number of top keywords to return.

    Returns:
        List of (keyword, score) tuples sorted by score descending.
    """
    data = preprocess(text)
    sentences = data["sentences"]

    if not sentences:
        return []

    # Use sentences as individual "documents" for IDF calculation
    vectorizer = TfidfVectorizer(
        stop_words="english",
        max_features=500,
        ngram_range=(1, 2),  # Include unigrams and bigrams
    )

    try:
        tfidf_matrix = vectorizer.fit_transform(sentences)
    except ValueError:
        return []

    feature_names = vectorizer.get_feature_names_out()

    # Sum TF-IDF scores across all sentences for each word
    scores = tfidf_matrix.sum(axis=0).A1
    word_scores = list(zip(feature_names, scores))
    word_scores.sort(key=lambda x: x[1], reverse=True)

    return [(word, round(score, 4)) for word, score in word_scores[:num_keywords]]


def extract_keywords_rake(text: str, num_keywords: int = 10) -> list[tuple[str, float]]:
    """
    Extract keyword phrases using RAKE algorithm.

    RAKE identifies multi-word keyphrases by:
      1. Splitting text at stopwords and punctuation
      2. Scoring candidate phrases based on word co-occurrence
      3. Ranking by composite score

    Args:
        text: Raw input text.
        num_keywords: Number of top keyphrases to return.

    Returns:
        List of (keyphrase, score) tuples sorted by score descending.
    """
    rake = Rake()
    rake.extract_keywords_from_text(text)

    # get_ranked_phrases_with_scores returns (score, phrase) tuples
    ranked = rake.get_ranked_phrases_with_scores()

    return [
        (phrase, round(score, 4))
        for score, phrase in ranked[:num_keywords]
    ]


def extract_keywords_frequency(text: str, num_keywords: int = 10) -> list[tuple[str, int]]:
    """
    Extract keywords using simple frequency-based ranking.

    Picks the most frequent non-stopword, lemmatized terms.

    Args:
        text: Raw input text.
        num_keywords: Number of top keywords to return.

    Returns:
        List of (keyword, frequency) tuples sorted by frequency descending.
    """
    data = preprocess(text)
    words = data["lemmatized_words"]

    freq: dict[str, int] = {}
    for w in words:
        freq[w] = freq.get(w, 0) + 1

    sorted_words = sorted(freq.items(), key=lambda x: x[1], reverse=True)
    return sorted_words[:num_keywords]
