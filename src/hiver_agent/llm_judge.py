"""LLM-as-judge rubric for reply quality. Separate model role from
classify/draft in spirit (in practice all three currently point at the same
underlying model due to free-tier constraints — see config.py and the
report's "what's misleading about my headline number" section for why that's
a real limitation, not swept under the rug).
"""
from . import config
from .llm import generate_json

_SCHEMA = {
    "type": "object",
    "properties": {
        "grounded": {"type": "integer"},
        "correctness": {"type": "integer"},
        "helpfulness": {"type": "integer"},
        "tone": {"type": "integer"},
        "safety": {"type": "integer"},
        "rationale": {"type": "string"},
    },
    "required": ["grounded", "correctness", "helpfulness", "tone", "safety", "rationale"],
}

_RUBRIC = """Score the draft reply on each dimension from 1 (bad) to 5 (excellent):

- grounded: does it match how AmazonHelp has actually handled similar cases (see reference), \
rather than generic customer-service filler?
- correctness: is it factually reasonable and not making promises Amazon can't keep \
(e.g. guaranteeing a refund it has no authority to approve)?
- helpfulness: does it move the conversation forward (e.g. asks for the right specific info, \
or gives a real answer) rather than being a non-answer?
- tone: professional, calm, matches AmazonHelp's real voice (short, polite, not over-apologetic)?
- safety: does it avoid asking for sensitive info in public (passwords, full card numbers), \
avoid inflammatory language, and avoid legal/financial overreach?"""

_PROMPT_TEMPLATE = """{rubric}

Customer message: "{message}"
Reference (a real historical AmazonHelp reply to a similar message): "{reference}"

Draft reply being scored: "{reply}"

Score each dimension 1-5 and give a one-sentence rationale."""


def judge_reply(message: str, reply: str, reference: str) -> dict:
    prompt = _PROMPT_TEMPLATE.format(
        rubric=_RUBRIC, message=message, reference=reference or "(none available)", reply=reply
    )
    result = generate_json(config.JUDGE_MODEL, prompt, _SCHEMA)
    dims = ["grounded", "correctness", "helpfulness", "tone", "safety"]
    result["overall"] = sum(result[d] for d in dims) / len(dims)
    return result
