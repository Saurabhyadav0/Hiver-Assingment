"""Zero/few-shot intent classification against the taxonomy in intents.py."""
from . import config
from .intents import INTENT_NAMES, INTENTS
from .llm import generate_json

_SCHEMA = {
    "type": "object",
    "properties": {
        "intent": {"type": "string", "enum": INTENT_NAMES},
        "confidence": {"type": "number"},
        "rationale": {"type": "string"},
    },
    "required": ["intent", "confidence", "rationale"],
}


def _taxonomy_block() -> str:
    lines = []
    for name, spec in INTENTS.items():
        examples = " | ".join(f'"{e}"' for e in spec["examples"])
        lines.append(f"- {name}: {spec['description']}\n  examples: {examples}")
    return "\n".join(lines)


_PROMPT_TEMPLATE = """You are classifying a customer support tweet sent to Amazon's \
support account (AmazonHelp) into exactly one intent.

Intents:
{taxonomy}

Message: "{message}"

Pick the single best-fitting intent from the list above (use the exact name), \
a confidence from 0 to 1, and a one-sentence rationale."""


def classify(message: str) -> dict:
    prompt = _PROMPT_TEMPLATE.format(taxonomy=_taxonomy_block(), message=message)
    result = generate_json(config.CLASSIFY_MODEL, prompt, _SCHEMA)
    if isinstance(result, str):
        import json

        result = json.loads(result)
    if result["intent"] not in INTENT_NAMES:
        result["intent"] = "complaint_negative_other"
        result["confidence"] = min(result.get("confidence", 0.0), 0.3)
    return result
