"""Automated metrics comparing a system's predictions against the golden set.
Split from llm_judge.py on purpose: these are cheap, deterministic, and don't
need an LLM call, so they should never be blocked by API quota/cost.
"""
from sklearn.metrics import f1_score, precision_recall_fscore_support


def intent_metrics(y_true: list[str], y_pred: list[str]) -> dict:
    accuracy = sum(t == p for t, p in zip(y_true, y_pred)) / len(y_true)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    return {"intent_accuracy": accuracy, "intent_macro_f1": macro_f1}


def escalation_metrics(y_true: list[bool], y_pred: list[bool]) -> dict:
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", pos_label=True, zero_division=0
    )
    return {"escalate_precision": precision, "escalate_recall": recall, "escalate_f1": f1}


def grounding_overlap(reply: str, retrieved: list[dict]) -> float:
    """Cheap proxy for "did the reply actually use the retrieved context":
    word-level overlap between the reply and the best-matching historical
    reply. Not a substitute for the LLM judge's grounding score — just a
    fast, free sanity signal to compute over every row."""
    if not retrieved:
        return 0.0
    reply_words = set(reply.lower().split())
    best = max(
        retrieved,
        key=lambda r: len(reply_words & set(r["brand_text"].lower().split())),
    )
    ref_words = set(best["brand_text"].lower().split())
    if not ref_words:
        return 0.0
    return len(reply_words & ref_words) / len(ref_words)
