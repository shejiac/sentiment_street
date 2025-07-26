# This file contains utility functions for data cleaning and text analysis, which are used in various pipelines.
# Data Cleaning libs
import re
import contractions
import emoji
import nltk
from nltk.corpus import stopwords

# Set up stopwords; only download if not already available
try:
    STOPWORDS = set(stopwords.words("english"))
except LookupError:
    nltk.download("stopwords")
    STOPWORDS = set(stopwords.words("english"))


# ==== FILTER JUNK TEXT ==== #
# Taken from topic modeling/classifier.py
def is_junk_comment(text, min_word_count=4):
    """Check if a comment is junk based on several rules:
    1. Too short (less than min_word_count words)
    2. Only stopwords
    3. Contains only special characters or gibberish
    4. Repeated characters (e.g., "loooool", "?????", "!!!!")
    5. Contains only emojis or emoticons
    6. Mostly non-alphabetic (e.g., links, symbols)
    7. One or two words repeated many times (e.g., "yes yes yes yes yes")

    Args:
        text (str): The text to check.
        min_word_count (int): Minimum number of words to consider the text valid.
    Returns:
        bool: True if the text is junk, False otherwise."""

    if not isinstance(text, str) or not text.strip():
        return True

    tokens = text.split()

    # Rule 1: Too short
    if len(tokens) < min_word_count:
        return True

    # Rule 2: Only stopwords
    if all(word.lower() in STOPWORDS for word in tokens):
        return True

    # Rule 3: Contains only special characters or gibberish
    if re.fullmatch(r"[^a-zA-Z0-9\s]+", text):
        return True

    # Rule 4: Repeated characters (e.g., "loooool", "?????", "!!!!")
    if re.fullmatch(r"(.)\1{4,}", text.replace(" ", "")):
        return True

    # Rule 5: Contains only emojis or emoticons
    if re.fullmatch(r"[\U00010000-\U0010ffff\W\s]+", text):
        return True

    # Rule 6: Mostly non-alphabetic (e.g., links, symbols)
    num_alpha = sum(c.isalpha() for c in text)
    num_total = len(text)
    if num_alpha / max(num_total, 1) < 0.3:
        return True

    # Rule 7: One or two words repeated many times (e.g., "yes yes yes yes yes")
    if len(set(tokens)) <= 2 and len(tokens) > 5:
        return True

    return False


# ==== CLEAN TEXT FUNCTION ==== #
def clean_text(text):
    """General data cleaning function for subreddit texts, including post's title, body and comments

    Args:
        text (str): The text to clean.
    Returns:
        str: Cleaned text."""
    # If text is empty, leave it be
    if text == "":
        return ""

    # Replace multiple whitespaces with just one
    text = re.sub(r"\s+", " ", text)

    # Convert all emojis to textual representation
    text = emoji.demojize(text)

    # Replace URLs with tag <URL>
    text = re.sub(
        r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+",
        "<URL>",
        text,
    )

    # Expand contractions in the text
    text = contractions.fix(text)

    return text
