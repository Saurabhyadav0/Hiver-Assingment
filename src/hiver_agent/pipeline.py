"""The actual agent: classify -> retrieve -> draft -> decide, glued together."""
from . import config
from .classify import classify
from .decision import decide
from .draft_reply import draft_reply
from .retrieval import RetrievalIndex, build_index

_index: RetrievalIndex = None


def _get_index() -> RetrievalIndex:
    global _index
    if _index is None:
        _index = build_index()
    return _index


def process(message: str, index: RetrievalIndex = None) -> dict:
    index = index or _get_index()

    classification = classify(message)
    intent, confidence = classification["intent"], classification["confidence"]

    retrieved = index.search(message, k=config.RETRIEVAL_TOP_K)
    draft = draft_reply(message, intent, retrieved)

    decision = decide(
        message=message,
        intent=intent,
        confidence=confidence,
        retrieved=retrieved,
        reply_grounded=draft["grounded"],
    )

    return {
        "message": message,
        "intent": intent,
        "confidence": confidence,
        "reply_draft": draft["reply"],
        "reply_grounded": draft["grounded"],
        "retrieved": retrieved,
        "decision": decision["decision"],
        "reason": decision["reason"],
    }
