"""Core workflow: preprocessing, classification, and decision making.

The classifier is a small scikit-learn pipeline (TF-IDF + Logistic
Regression) trained on a handful of labelled examples at startup.
A keyword-based fallback is used if the model is ever unavailable,
which keeps the API resilient.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Tuple

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from utils import (
    CATEGORY_RULES,
    build_decision,
    extract_keywords,
    preprocess_email_text,
)


logger = logging.getLogger(__name__)


"""Small, hand-curated training set. In a real project this would come
from labelled production data, but for an internship-scale demo a
few clear examples per class are enough to get sensible behaviour."""

TRAINING_DATA: List[Tuple[str, str]] = [
    # complaints
    ("This is unacceptable, my order never arrived and no one is responding.", "complaint"),
    ("I am very disappointed with the service, please refund my money immediately.", "complaint"),
    ("The product broke after one day, I want to return it and get my money back.", "complaint"),
    ("Worst experience ever, your support team has been ignoring my emails for a week.", "complaint"),
    ("I have been charged twice for the same order and nobody is helping me.", "complaint"),
    ("This bug has cost us a full day of work, we need an urgent fix.", "complaint"),

    # inquiries
    ("Hi, could you please tell me the price of the enterprise plan?", "inquiry"),
    ("I would like to know if your service supports single sign on.", "inquiry"),
    ("Can you share more information about the upcoming features?", "inquiry"),
    ("How do I reset my password and update my billing email?", "inquiry"),
    ("Is there a student discount available for the annual plan?", "inquiry"),
    ("What are the office hours for your support team in Europe?", "inquiry"),

    # feedback
    ("Just wanted to say the new dashboard looks great, very clean.", "feedback"),
    ("Loving the recent updates, the app feels much faster now.", "feedback"),
    ("Great job on the redesign, my team finds it much easier to use.", "feedback"),
    ("The onboarding flow was smooth and friendly, well done.", "feedback"),
    ("Nice work on the latest release, the dark mode is beautiful.", "feedback"),
    ("Thanks for the quick fix yesterday, the new version is much more stable.", "feedback"),
]


"""Simple keyword rules used as a safety net if the ML model fails."""
KEYWORD_RULES: Dict[str, List[str]] = {
    "complaint": [
        "refund", "broken", "disappointed", "unacceptable", "complaint",
        "angry", "worst", "terrible", "issue", "problem", "bug", "charged",
    ],
    "inquiry": [
        "price", "pricing", "how", "what", "when", "where", "could",
        "would", "information", "details", "support", "help", "question",
    ],
    "feedback": [
        "great", "love", "loving", "awesome", "thanks", "nice", "good",
        "amazing", "feedback", "well done", "beautiful", "smooth",
    ],
}


class EmailWorkflow:
    """Encapsulates the full Input -> Process -> Decision -> Output flow.

    The model is trained once when an instance is created so subsequent
    requests are fast. Keeping it in a class also makes the workflow
    easy to swap or mock in tests.
    """

    def __init__(self) -> None:
        self._pipeline = self._train_pipeline()

    @staticmethod
    def _train_pipeline() -> Pipeline:
        """Train a small TF-IDF + Logistic Regression pipeline."""
        texts = [text for text, _ in TRAINING_DATA]
        labels = [label for _, label in TRAINING_DATA]

        pipeline = Pipeline(
            steps=[
                (
                    "tfidf",
                    TfidfVectorizer(
                        lowercase=True,
                        ngram_range=(1, 2),
                        min_df=1,
                    ),
                ),
                (
                    "classifier",
                    LogisticRegression(max_iter=1000, class_weight="balanced"),
                ),
            ]
        )
        pipeline.fit(texts, labels)
        logger.info("Email classifier trained on %d examples", len(texts))
        return pipeline

    def classify_email(self, cleaned_text: str) -> str:
        """Predict the category of a cleaned email body.

        Falls back to keyword matching if the model raises an error
        (e.g. an empty input slipped through validation).
        """
        try:
            prediction = self._pipeline.predict([cleaned_text])[0]
            return str(prediction)
        except Exception:  # pragma: no cover - defensive fallback
            logger.exception("Model prediction failed; using keyword fallback")
            return self._keyword_fallback(cleaned_text)

    @staticmethod
    def _keyword_fallback(cleaned_text: str) -> str:
        """Pick a category by counting keyword matches per class."""
        scores = {category: 0 for category in CATEGORY_RULES}
        for category, keywords in KEYWORD_RULES.items():
            for keyword in keywords:
                if keyword in cleaned_text:
                    scores[category] += 1

        """Default to "inquiry" when nothing matches: it is the most
        common neutral case and triggers a safe auto-response."""
        best_category = max(scores, key=scores.get)
        if scores[best_category] == 0:
            return "inquiry"
        return best_category

    def generate_response(self, email_text: str) -> Dict[str, object]:
        """Run the full workflow and return the structured response.

        Args:
            email_text: Raw email body from the API request.

        Returns:
            A dict with the category, priority, recommended action, and
            the keywords that were extracted from the email.
        """
        cleaned_text = preprocess_email_text(email_text)
        if not cleaned_text:

            """Caught by the route handler and turned into a 400."""

            raise ValueError("Email text is empty after preprocessing")

        category = self.classify_email(cleaned_text)
        decision = build_decision(category)
        decision["keywords"] = extract_keywords(cleaned_text)
        return decision
