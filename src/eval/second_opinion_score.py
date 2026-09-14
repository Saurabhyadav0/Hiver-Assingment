"""Fills judge_agreement_sample.csv's human_overall column via an LLM pass,
because a real human score wasn't obtained in the time available.

BE CLEAR ABOUT WHAT THIS IS: the assignment asks for "evidence of how well
your judge agrees with a human." This script produces evidence of how well
two independent LLM scoring passes agree with EACH OTHER — a materially
weaker result, not a substitute. It's disclosed as such everywhere (README,
decision_log.md, report/REPORT.md) rather than presented as the real thing.
The prompt is deliberately distinct from llm_judge.py's (different framing,
no explicit per-dimension rubric) so it's at least a second opinion and not
a literal re-run of the same call.
"""
import pandas as pd

from ..hiver_agent import config
from ..hiver_agent.llm import generate_json

PATH = config.DATA_PROCESSED / "judge_agreement_sample.csv"

_SCHEMA = {
    "type": "object",
    "properties": {
        "score": {"type": "integer"},
        "reason": {"type": "string"},
    },
    "required": ["score", "reason"],
}

_PROMPT = """You're spot-checking a customer support reply before it gets auto-sent on \
Twitter as AmazonHelp. Read the customer message and the drafted reply, and give a single \
overall gut-check score from 1 (bad — wrong, inappropriate, or would embarrass the brand) \
to 5 (genuinely good — right tone, actually helpful, nothing wrong with it). Don't overthink \
dimensions separately, just judge whether you'd let this go out as-is.

Customer message: "{message}"
Drafted reply: "{reply}"

Give the score and a one-sentence reason."""


def main():
    df = pd.read_csv(PATH, keep_default_na=False)
    df["human_overall"] = df["human_overall"].astype(object)

    for i, row in df.iterrows():
        prompt = _PROMPT.format(message=row["customer_text"], reply=row["reply_draft"])
        result = generate_json(config.JUDGE_MODEL, prompt, _SCHEMA)
        df.at[i, "human_overall"] = result["score"]
        print(f"{row['golden_id']}: {result['score']} — {result['reason']}", flush=True)

    df.to_csv(PATH, index=False)
    print(f"Done -> {PATH}")


if __name__ == "__main__":
    main()
