"""Draft a reply grounded in how AmazonHelp has historically handled similar
messages, using the retrieved (customer_msg, brand_reply) pairs from
retrieval.py as few-shot style/content grounding — not a template, but not
free-floating generation either.
"""
from . import config
from .llm import generate_json

_SCHEMA = {
    "type": "object",
    "properties": {
        "reply": {"type": "string"},
        "grounded": {"type": "boolean"},
    },
    "required": ["reply", "grounded"],
}

_PROMPT_TEMPLATE = """You are drafting a public Twitter reply as AmazonHelp, Amazon's customer \
support account. Match the brand's real voice from the examples below: short, \
polite, asks for an order ID or specifics rather than resolving on the spot, \
no over-promising, no exclamation-mark enthusiasm.

Here is how AmazonHelp has replied to similar past messages:
{examples}

New customer message: "{message}"
Detected intent: {intent}

Draft a reply in AmazonHelp's voice. Set "grounded" to true only if the \
examples above actually informed your reply's approach; false if none of \
them were relevant and you had to draft from general knowledge instead."""


def _examples_block(retrieved: list[dict]) -> str:
    if not retrieved:
        return "(no similar past examples found)"
    lines = []
    for r in retrieved:
        lines.append(f'- customer: "{r["customer_text"]}"\n  AmazonHelp: "{r["brand_text"]}"')
    return "\n".join(lines)


def draft_reply(message: str, intent: str, retrieved: list[dict]) -> dict:
    prompt = _PROMPT_TEMPLATE.format(
        examples=_examples_block(retrieved), message=message, intent=intent
    )
    result = generate_json(config.DRAFT_MODEL, prompt, _SCHEMA)
    if isinstance(result, str):
        import json

        result = json.loads(result)
    return result
