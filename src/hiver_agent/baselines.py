"""Two baselines the real pipeline has to beat to be worth its complexity.

Trivial: majority-class intent, one canned reply, never escalates. This is
the floor — if the LLM pipeline doesn't clear this by a wide margin on intent
accuracy, nothing else about it matters.

Simple: TF-IDF + logistic regression intent classifier (trained on the
processed thread pool, held out from the golden set), a template reply per
intent, and a fixed-confidence-threshold escalation rule. No LLM, no
retrieval grounding. This is the bar a "why not just ship something cheap
and dumb" reviewer would ask about.
"""
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from . import config
from .intents import ALWAYS_ESCALATE
from .weak_labels import heuristic_bucket

_MAJORITY_INTENT = "order_status_delivery"  # the largest bucket in the taxonomy sample
_MAJORITY_TEMPLATE = (
    "Thanks for reaching out — we're looking into this and will follow up "
    "with an update shortly."
)

_TEMPLATES = {
    "order_status_delivery": "Sorry for the delay! Could you share your order ID so we can check the latest tracking status?",
    "refund_return_cancellation": "We can help with that. Please share your order ID so we can start the refund/return process.",
    "billing_payment_issue": "Sorry about the charge confusion — please share your order ID so we can review the billing details.",
    "account_security": "For your security, please don't share account details here — contact us via the link in our bio to secure your account.",
    "product_quality_defect": "Sorry to hear the item arrived damaged/incorrect. Please share your order ID so we can arrange a replacement or refund.",
    "app_website_technical": "Sorry about that! Could you share your device/browser and what happens when the issue occurs?",
    "general_inquiry": "Happy to help — could you share a bit more detail on what you're trying to do?",
    "compliment_positive_feedback": "Thank you so much for the kind words — we'll pass it along to the team!",
    "complaint_negative_other": "We're sorry to hear that. Could you share more detail so we can look into it?",
    "spam_unrelated": "Thanks for reaching out! This doesn't look like a support request — let us know if we can help with an order.",
}


class TrivialBaseline:
    """Always predicts the majority intent, always the same reply, never escalates."""

    name = "trivial"

    def process(self, message: str) -> dict:
        return {
            "intent": _MAJORITY_INTENT,
            "confidence": None,
            "reply_draft": _MAJORITY_TEMPLATE,
            "decision": "auto_handle",
            "reason": "trivial baseline never escalates",
        }


class SimpleBaseline:
    """TF-IDF + logistic regression classifier, template reply, confidence-threshold escalation."""

    name = "simple"

    def __init__(self, escalate_confidence_threshold: float = 0.5):
        self.threshold = escalate_confidence_threshold
        self.vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
        self.clf = LogisticRegression(max_iter=1000)

    def fit(self, texts: list[str], labels: list[str]):
        X = self.vectorizer.fit_transform(texts)
        self.clf.fit(X, labels)
        return self

    def process(self, message: str) -> dict:
        X = self.vectorizer.transform([message])
        proba = self.clf.predict_proba(X)[0]
        idx = proba.argmax()
        intent = self.clf.classes_[idx]
        confidence = float(proba[idx])

        if intent in ALWAYS_ESCALATE:
            decision, reason = "escalate", f"intent '{intent}' is always escalated regardless of confidence"
        elif confidence < self.threshold:
            decision, reason = "escalate", f"classifier confidence {confidence:.2f} below threshold {self.threshold}"
        else:
            decision, reason = "auto_handle", f"classifier confidence {confidence:.2f} met threshold"

        return {
            "intent": intent,
            "confidence": confidence,
            "reply_draft": _TEMPLATES.get(intent, _MAJORITY_TEMPLATE),
            "decision": decision,
            "reason": reason,
        }


def train_simple_baseline(pool: pd.DataFrame, exclude_ids: set = frozenset()) -> SimpleBaseline:
    """Trains on keyword-heuristic weak labels, not the golden set — the
    golden set is eval-only, and hand-labeling the full 20k pool defeats the
    point of a "simple, cheap" baseline. Rows the heuristic can't bucket
    ("unmatched") are dropped from training rather than guessed at."""
    pool = pool[~pool["customer_tweet_id"].isin(exclude_ids)].copy()
    pool["weak_intent"] = pool["customer_text"].map(heuristic_bucket)
    pool = pool[pool["weak_intent"] != "unmatched"]

    baseline = SimpleBaseline()
    baseline.fit(pool["customer_text"].tolist(), pool["weak_intent"].tolist())
    return baseline
