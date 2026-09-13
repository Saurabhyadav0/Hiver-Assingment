"""Cheap keyword heuristic for intent, used in two places that both need a
free (no-LLM, no-human) label: stratifying the golden-set sample so it isn't
90% "where's my order" (see eval/sample_golden.py), and training the simple
TF-IDF baseline's labels (see baselines.py). This is deliberately not used
to grade anything — it's weak supervision, not ground truth.
"""

KEYWORDS = {
    "order_status_delivery": ["deliver", "shipping", "shipped", "track", "arrive", "package", "parcel", "late", "out for delivery"],
    "refund_return_cancellation": ["refund", "return", "cancel", "money back"],
    "billing_payment_issue": ["charge", "charged", "billed", "payment", "price", "mrp", "overcharg"],
    "account_security": ["hack", "compromise", "security code", "sign in", "signin", "login", "password", "unauthorized", "fraud"],
    "product_quality_defect": ["broken", "defect", "damaged", "wrong item", "wrong size", "missing item", "fake", "counterfeit"],
    "app_website_technical": ["app ", "website", "glitch", "bug", "form ", "link ", "error", "crash", "freeze"],
    "compliment_positive_feedback": ["thank", "thanks", "great service", "appreciate", "awesome", "kudos"],
    "complaint_negative_other": ["worst", "sucks", "terrible", "unacceptable", "ridiculous", "liar", "stupid", "pathetic"],
}


def heuristic_bucket(text: str) -> str:
    t = text.lower()
    for name, words in KEYWORDS.items():
        if any(w in t for w in words):
            return name
    return "unmatched"  # general_inquiry / spam_unrelated / ambiguous cases live here
