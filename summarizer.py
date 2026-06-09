"""
summarizer.py — Extractive Text Summarization Module

Algorithm (from brief):
  1. Sentence Tokenization
  2. Word Tokenization
  3. Remove Stopwords
  4. Lemmatization
  5. Word Frequency Calculation
  6. Normalize Frequencies (0–1 scale)
  7. Sentence Scoring (sum of word frequencies per sentence)
  8. Select Top N Sentences (highest scoring)
  9. Reconstruct in Original Order

NLP Technique: Extractive Summarization using Word Frequency Scoring
"""

from preprocessor import preprocess, tokenize_words, remove_stopwords, lemmatize_words


def _build_word_frequencies(lemmatized_words: list[str]) -> dict[str, float]:
    """
    Step 5 & 6: Calculate raw word frequencies and normalize by max frequency.
    Returns a dict mapping each word to its normalized frequency (0–1).
    """
    freq: dict[str, int] = {}
    for word in lemmatized_words:
        freq[word] = freq.get(word, 0) + 1

    if not freq:
        return {}

    max_freq = max(freq.values())
    return {word: count / max_freq for word, count in freq.items()}


def _score_sentences(
    sentences: list[str], word_frequencies: dict[str, float]
) -> dict[int, float]:
    """
    Step 7: Score each sentence by summing the normalized word frequencies
    of its constituent words (after preprocessing each sentence individually).
    Returns a dict mapping sentence index → score.
    """
    scores: dict[int, float] = {}

    for idx, sentence in enumerate(sentences):
        words = tokenize_words(sentence)
        filtered = remove_stopwords(words)
        lemmatized = lemmatize_words(filtered)

        score = sum(word_frequencies.get(w, 0.0) for w in lemmatized)

        # Normalize by sentence length to avoid bias toward very long sentences
        if lemmatized:
            score /= len(lemmatized)

        scores[idx] = score

    return scores


def summarize(text: str, num_sentences: int = 3) -> dict:
    """
    Perform extractive summarization on the input text.

    Args:
        text: The raw input text to summarize.
        num_sentences: Number of sentences to include in the summary.

    Returns:
        A dict with:
          - summary: the summarized text string
          - scored_sentences: list of (sentence, score) tuples sorted by score desc
          - selected_indices: indices of sentences chosen for the summary
          - original_sentence_count: total sentences in original text
          - summary_sentence_count: sentences in the summary
          - compression_ratio: percentage of text reduced
    """
    data = preprocess(text)
    sentences = data["sentences"]
    lemmatized_words = data["lemmatized_words"]

    if not sentences:
        return {
            "summary": "",
            "scored_sentences": [],
            "selected_indices": [],
            "original_sentence_count": 0,
            "summary_sentence_count": 0,
            "compression_ratio": 0.0,
        }

    # Clamp num_sentences to available range
    num_sentences = min(num_sentences, len(sentences))

    # Step 5–6: Build normalized word frequencies
    word_frequencies = _build_word_frequencies(lemmatized_words)

    # Step 7: Score each sentence
    sentence_scores = _score_sentences(sentences, word_frequencies)

    # Step 8: Select top N sentence indices by score
    ranked_indices = sorted(sentence_scores, key=sentence_scores.get, reverse=True)
    selected_indices = sorted(ranked_indices[:num_sentences])  # Step 9: preserve order

    # Reconstruct summary
    summary = " ".join(sentences[i] for i in selected_indices)

    # Build scored sentences list for display
    scored = [
        (sentences[i], round(sentence_scores[i], 4))
        for i in sorted(sentence_scores, key=sentence_scores.get, reverse=True)
    ]

    # Compression ratio
    original_word_count = len(data["words"])
    summary_word_count = len(summary.split())
    compression = (
        round((1 - summary_word_count / original_word_count) * 100, 1)
        if original_word_count > 0
        else 0.0
    )

    return {
        "summary": summary,
        "scored_sentences": scored,
        "selected_indices": selected_indices,
        "original_sentence_count": len(sentences),
        "summary_sentence_count": num_sentences,
        "compression_ratio": compression,
    }
