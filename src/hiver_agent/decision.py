"""Auto-handle vs escalate, with a stated reason. Rule-based on purpose: the
assignment explicitly asks for a *stated reason*, and a rule-based decision
gives a reason that's actually traceable to a signal, instead of an LLM
post-hoc-rationalizing a decision it made for opaque reasons.
"""
from .intents import ALWAYS_ESCALATE

CONFIDENCE_ESCALATE_THRESHOLD = 0.6
MIN_GROUNDING_SIMILARITY = 0.5

_ANGER_MARKERS = [
    "stupid", "liar", "sucks", "unacceptable", "ridiculous", "pathetic",
    "worst", "scam", "furious", "disgusted", "lawsuit", "lawyer", "sue",
]


def _looks_angry(message: str) -> bool:
    t = message.lower()
    return any(marker in t for marker in _ANGER_MARKERS)


def decide(message: str, intent: str, confidence: float, retrieved: list[dict], reply_grounded: bool) -> dict:
    """Returns {"decision": "auto_handle" | "escalate", "reason": str}.
    Checked in order of how serious the failure mode is if we get it wrong."""

    if intent in ALWAYS_ESCALATE:
        return {
            "decision": "escalate",
            "reason": f"intent '{intent}' involves money or account security — always routed to a human regardless of confidence",
        }

    if confidence < CONFIDENCE_ESCALATE_THRESHOLD:
        return {
            "decision": "escalate",
            "reason": f"classifier confidence {confidence:.2f} is below the {CONFIDENCE_ESCALATE_THRESHOLD} threshold",
        }

    best_similarity = max((r["similarity"] for r in retrieved), default=0.0)
    if not reply_grounded or best_similarity < MIN_GROUNDING_SIMILARITY:
        return {
            "decision": "escalate",
            "reason": f"no sufficiently similar historical resolution found (best similarity {best_similarity:.2f}) — drafted reply isn't grounded",
        }

    if _looks_angry(message):
        return {
            "decision": "escalate",
            "reason": "message contains strong negative-sentiment language — routed to a human to de-escalate",
        }

    return {
        "decision": "auto_handle",
        "reason": f"high classifier confidence ({confidence:.2f}), grounded reply (similarity {best_similarity:.2f}), no anger markers",
    }
