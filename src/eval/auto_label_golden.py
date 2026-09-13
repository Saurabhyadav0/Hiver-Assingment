"""Draft true_intent / true_escalate / escalate_reason for the golden set via
an LLM pass that is deliberately NOT the same call as the production
classifier or the production decision logic — using the classifier's own
suggested_intent as ground truth would make the intent-accuracy metric
circular (grading the classifier against its own output), and using
decision.py's rule engine to generate true_escalate would do the same to the
escalation metric. Both would produce a headline number that looks great and
means nothing.

IMPORTANT — this is still LLM-generated, not human-labeled. The assignment
asks for a hand-labeled golden set "you built yourself," and says you'll be
asked to explain your labeling live. Treat this script's output as a first
draft: spot-check a real sample (a few from every intent, not just the ones
that look right) before treating any number derived from it as your
headline result. See decision_log.md and report/REPORT.md for how this
limitation is disclosed.
"""
import pandas as pd

from ..hiver_agent import config
from ..hiver_agent.intents import INTENT_NAMES, INTENTS
from ..hiver_agent.llm import generate_json

PATH = config.DATA_PROCESSED / "golden_candidates.csv"

_INTENT_SCHEMA = {
    "type": "object",
    "properties": {
        "intent": {"type": "string", "enum": INTENT_NAMES},
        "reasoning": {"type": "string"},
    },
    "required": ["intent", "reasoning"],
}

_ESCALATE_SCHEMA = {
    "type": "object",
    "properties": {
        "escalate": {"type": "boolean"},
        "reason": {"type": "string"},
    },
    "required": ["escalate", "reason"],
}

_INTENT_PROMPT = """You are an experienced human annotator building a labeled dataset for a \
customer-support intent taxonomy. You were NOT the one who built this taxonomy or any \
classifier — you're independently reading each message fresh and deciding which single \
category fits best. Think about what the customer is actually trying to accomplish, not \
just keywords.

Taxonomy:
{taxonomy}

Message to label: "{message}"

First reason briefly about what the customer wants, then commit to exactly one intent name \
from the list above."""

_ESCALATE_PROMPT = """You are a senior customer-support operations lead at Amazon, deciding \
whether a specific incoming customer message is safe to hand to a fully-automated AI agent \
for an immediate public reply, with NO human reviewing it first. You have not seen any draft \
reply — judge based on the message and its risk alone.

Escalate to a human (do not auto-handle) if: it involves money, account security, legal \
threats, or safety; the customer is clearly angry and needs de-escalation rather than a \
templated response; the message is ambiguous enough that a wrong automated reply would make \
things worse; or you're simply not confident a canned/generated reply is adequate.

Customer message: "{message}"
Intent category: {intent}

Decide true (escalate to a human) or false (safe to auto-handle), with a one-sentence reason."""


def _taxonomy_block() -> str:
    return "\n".join(f"- {name}: {spec['description']}" for name, spec in INTENTS.items())


def label_intent(message: str) -> dict:
    prompt = _INTENT_PROMPT.format(taxonomy=_taxonomy_block(), message=message)
    return generate_json(config.CLASSIFY_MODEL, prompt, _INTENT_SCHEMA)


def label_escalate(message: str, intent: str) -> dict:
    prompt = _ESCALATE_PROMPT.format(message=message, intent=intent)
    return generate_json(config.CLASSIFY_MODEL, prompt, _ESCALATE_SCHEMA)


def main(limit: int = None):
    df = pd.read_csv(PATH, keep_default_na=False)
    df["true_intent"] = df["true_intent"].astype(object)
    df["true_escalate"] = df["true_escalate"].astype(object)
    df["escalate_reason"] = df["escalate_reason"].astype(object)

    todo = [i for i in range(len(df)) if not df.at[i, "true_intent"]]
    if limit:
        todo = todo[:limit]

    for n, i in enumerate(todo):
        message = df.at[i, "customer_text"]
        try:
            intent_result = label_intent(message)
            intent = intent_result["intent"]
            escalate_result = label_escalate(message, intent)

            df.at[i, "true_intent"] = intent
            df.at[i, "true_escalate"] = escalate_result["escalate"]
            df.at[i, "escalate_reason"] = escalate_result["reason"]
            agree = "agrees" if df.at[i, "suggested_intent"] == intent else "DISAGREES"
            print(f"{n + 1}/{len(todo)} row={i} intent={intent} ({agree} w/ classifier) escalate={escalate_result['escalate']}", flush=True)
        except Exception as e:
            print(f"{n + 1}/{len(todo)} row={i} ERROR: {e}", flush=True)
        if (n + 1) % 10 == 0:
            df.to_csv(PATH, index=False)

    df.to_csv(PATH, index=False)
    remaining = sum(1 for x in df["true_intent"] if not x)
    print(f"Done this batch -> {PATH} ({remaining} still unlabeled)")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    main(limit=args.limit)
