"""Helper functions for the email workflow system.

This module keeps small, single-purpose utilities used by the
classification pipeline (text cleaning, keyword extraction, and
mapping a category to a priority and recommended action).
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Dict, List


# A short list of English stop words. Kept inline so the project has
# no extra NLTK download step. Good enough for short email bodies.
STOP_WORDS = {
    "a", "an", "the", "and", "or", "but", "if", "while", "with", "to",
    "of", "in", "on", "for", "at", "by", "from", "is", "am", "are",
    "was", "were", "be", "been", "being", "i", "you", "he", "she",
    "it", "we", "they", "me", "him", "her", "us", "them", "my",
    "your", "our", "their", "this", "that", "these", "those", "as",
    "so", "do", "does", "did", "have", "has", "had", "will", "would",
    "can", "could", "should", "may", "might", "just", "not", "no",
    "yes", "please", "hi", "hello", "dear", "regards", "thanks",
    "thank", "team", "sir", "madam",
}


# Mapping from a predicted category to its business priority and the
# action a downstream system should take. Defined in one place so the
# rules are easy to audit and change.
CATEGORY_RULES: Dict[str, Dict[str, str]] = {
    "complaint": {
        "priority": "high",
        "recommended_action": "escalate to support team",
    },
    "inquiry": {
        "priority": "medium",
        "recommended_action": "send auto response",
    },
    "feedback": {
        "priority": "low",
        "recommended_action": "store for review",
    },
}


def preprocess_email_text(email_text: str) -> str:
    """Normalize an email body for downstream processing.

    Steps:
      * lowercase the text
      * strip URLs and email addresses (they add noise, not signal)
      * remove punctuation and digits
      * collapse repeated whitespace

    Args:
        email_text: Raw email body submitted by the caller.

    Returns:
        A cleaned, lowercase string safe to feed into the classifier.
    """
    text = email_text.lower()

    # Drop URLs and email addresses before stripping punctuation,
    # otherwise they leave behind meaningless fragments.
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"\S+@\S+", " ", text)

    # Keep only letters and spaces; everything else becomes a space.
    text = re.sub(r"[^a-z\s]", " ", text)

    # Collapse runs of whitespace into a single space.
    text = re.sub(r"\s+", " ", text).strip()
    return text


def extract_keywords(cleaned_text: str, top_n: int = 5) -> List[str]:
    """Return the most frequent meaningful tokens in the cleaned text.

    Stop words and very short tokens are ignored so the result is more
    informative than a raw word count.

    Args:
        cleaned_text: Output of `preprocess_email_text`.
        top_n: Maximum number of keywords to return.

    Returns:
        A list of up to `top_n` keywords ordered by frequency.
    """
    if not cleaned_text:
        return []

    tokens = [
        token
        for token in cleaned_text.split()
        if token not in STOP_WORDS and len(token) > 2
    ]
    if not tokens:
        return []

    most_common = Counter(tokens).most_common(top_n)
    return [token for token, _ in most_common]


def build_decision(category: str) -> Dict[str, str]:
    """Translate a category into a priority and recommended action.

    Args:
        category: One of the supported categories.

    Returns:
        A dict with `category`, `priority`, and `recommended_action`.

    Raises:
        ValueError: If the category is not recognized.
    """
    rule = CATEGORY_RULES.get(category)
    if rule is None:
        raise ValueError(f"Unknown category: {category}")

    return {
        "category": category,
        "priority": rule["priority"],
        "recommended_action": rule["recommended_action"],
    }

