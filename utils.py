"""
utils.py — Helper Functions

Includes:
  - Word cloud generation
  - Reading time estimation
  - Text statistics calculation
  - Display formatting helpers
"""

import math
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from preprocessor import preprocess


# Average adult reading speed (words per minute)
AVERAGE_WPM = 200


def calculate_text_stats(text: str) -> dict:
    """
    Calculate statistics about the input text.

    Returns:
        Dict with:
          - word_count: total number of words
          - sentence_count: total number of sentences
          - reading_time_seconds: estimated reading time in seconds
          - reading_time_display: human-readable reading time string
          - avg_sentence_length: average words per sentence
    """
    data = preprocess(text)
    word_count = len(data["words"])
    sentence_count = len(data["sentences"])
    reading_seconds = math.ceil((word_count / AVERAGE_WPM) * 60)

    if reading_seconds < 60:
        display = f"{reading_seconds} sec"
    else:
        minutes = reading_seconds // 60
        seconds = reading_seconds % 60
        display = f"{minutes} min {seconds} sec" if seconds else f"{minutes} min"

    avg_len = round(word_count / sentence_count, 1) if sentence_count else 0

    return {
        "word_count": word_count,
        "sentence_count": sentence_count,
        "reading_time_seconds": reading_seconds,
        "reading_time_display": display,
        "avg_sentence_length": avg_len,
    }


def generate_word_cloud(keywords: list[tuple[str, float]], width: int = 800, height: int = 400):
    """
    Generate a matplotlib Figure containing a word cloud from keyword scores.

    Args:
        keywords: List of (word, score) tuples.
        width: Image width in pixels.
        height: Image height in pixels.

    Returns:
        matplotlib Figure object.
    """
    if not keywords:
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.text(0.5, 0.5, "No keywords to display", ha="center", va="center",
                fontsize=16, color="#888888")
        ax.set_axis_off()
        return fig

    # Build frequency dict for WordCloud
    word_freq = {word: float(score) for word, score in keywords}

    wc = WordCloud(
        width=width,
        height=height,
        background_color="#0E1117",
        colormap="cool",
        max_words=50,
        prefer_horizontal=0.7,
        min_font_size=10,
        max_font_size=120,
        relative_scaling=0.5,
        contour_width=0,
    ).generate_from_frequencies(word_freq)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wc, interpolation="bilinear")
    ax.set_axis_off()
    fig.patch.set_facecolor("#0E1117")
    plt.tight_layout(pad=0)
    return fig


def format_compression_ratio(ratio: float) -> str:
    """Format compression ratio as a display string."""
    if ratio > 0:
        return f"🗜️ Reduced by {ratio}%"
    return "No compression"
