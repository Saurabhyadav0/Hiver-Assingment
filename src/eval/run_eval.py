"""Run the trivial baseline, simple baseline, and the real pipeline over the
hand-labeled golden set, and produce one comparison table. This is the
script the README points at for the "reproduce our headline results in
under 15 minutes" claim — every LLM call it makes is cache-hit (see
hiver_agent/llm.py) as long as data/processed/llm_cache/ is present, so
reproduction doesn't depend on API quota or even having a working API key.
"""
import json

import pandas as pd

from ..hiver_agent import config
from ..hiver_agent.baselines import SimpleBaseline, TrivialBaseline, train_simple_baseline
from ..hiver_agent.classify import classify
from ..hiver_agent.decision import decide
from ..hiver_agent.draft_reply import draft_reply
from ..hiver_agent.llm_judge import judge_reply
from ..hiver_agent.retrieval import build_index
from . import metrics

GOLDEN_PATH = config.DATA_PROCESSED / "golden_candidates.csv"
RESULTS_PATH = config.DATA_PROCESSED / "eval_results.csv"
SUMMARY_PATH = config.DATA_PROCESSED / "eval_summary.json"


def load_golden() -> pd.DataFrame:
    df = pd.read_csv(GOLDEN_PATH, keep_default_na=False)
    labeled = df[(df["true_intent"] != "") & (df["true_escalate"] != "")]
    if len(labeled) < len(df):
        print(f"Warning: only {len(labeled)}/{len(df)} golden rows are hand-labeled; "
              f"evaluating on the labeled subset only.")
    labeled = labeled.copy()
    labeled["true_escalate"] = labeled["true_escalate"].astype(str).str.lower().isin(["true", "1", "yes"])
    return labeled


def run_pipeline_row(message: str, reference_reply: str, index) -> dict:
    classification = classify(message)
    intent, confidence = classification["intent"], classification["confidence"]
    retrieved = index.search(message, k=config.RETRIEVAL_TOP_K)
    draft = draft_reply(message, intent, retrieved)
    decision = decide(message, intent, confidence, retrieved, draft["grounded"])
    judged = judge_reply(message, draft["reply"], reference_reply)
    return {
        "intent": intent,
        "confidence": confidence,
        "reply": draft["reply"],
        "grounded": draft["grounded"],
        "decision": decision["decision"] == "escalate",
        "reason": decision["reason"],
        "judge_overall": judged["overall"],
        "grounding_overlap": metrics.grounding_overlap(draft["reply"], retrieved),
    }


def run_baseline_row(baseline, message: str, reference_reply: str) -> dict:
    r = baseline.process(message)
    return {
        "intent": r["intent"],
        "reply": r["reply_draft"],
        "decision": r["decision"] == "escalate",
        "reason": r["reason"],
        "grounding_overlap": metrics.grounding_overlap(r["reply_draft"], []),
    }


def _load_partial_results() -> dict:
    """golden_id -> row dict, from a previous run that got cut short (e.g. by
    hitting the daily quota mid-loop) — every LLM call is cached anyway, but
    without this the *script* still had to redo finished rows on retry, and
    on this project's quota, retrying a finished row can 429 before it even
    gets to a not-yet-cached row."""
    if not RESULTS_PATH.exists():
        return {}
    prev = pd.read_csv(RESULTS_PATH, keep_default_na=False)
    return {row["golden_id"]: row.to_dict() for _, row in prev.iterrows()}


def main():
    golden = load_golden()
    if len(golden) == 0:
        raise SystemExit(
            "No hand-labeled golden rows found. Fill in true_intent/true_escalate "
            f"in {GOLDEN_PATH} before running eval — see decision_log.md #12."
        )
    pool = pd.read_parquet(config.THREADS_PARQUET)

    index = build_index(exclude_ids=set(golden["customer_tweet_id"]))
    trivial = TrivialBaseline()
    simple = train_simple_baseline(pool, exclude_ids=set(golden["customer_tweet_id"]))

    done = _load_partial_results()
    rows = list(done.values())
    todo = [g for _, g in golden.iterrows() if g["golden_id"] not in done]
    if done:
        print(f"Resuming: {len(done)}/{len(golden)} rows already in {RESULTS_PATH}")

    for n, g in enumerate(todo):
        pipeline_out = run_pipeline_row(g["customer_text"], g["brand_text"], index)
        trivial_out = run_baseline_row(trivial, g["customer_text"], g["brand_text"])
        simple_out = run_baseline_row(simple, g["customer_text"], g["brand_text"])
        rows.append({
            "golden_id": g["golden_id"],
            "true_intent": g["true_intent"],
            "true_escalate": g["true_escalate"],
            **{f"pipeline_{k}": v for k, v in pipeline_out.items()},
            **{f"trivial_{k}": v for k, v in trivial_out.items()},
            **{f"simple_{k}": v for k, v in simple_out.items()},
        })
        print(f"{len(rows)}/{len(golden)} row={g['golden_id']} done", flush=True)
        if (n + 1) % 5 == 0:
            pd.DataFrame(rows).to_csv(RESULTS_PATH, index=False)

    results = pd.DataFrame(rows)
    results.to_csv(RESULTS_PATH, index=False)

    summary = {}
    for system in ["pipeline", "trivial", "simple"]:
        y_true_intent = results["true_intent"].tolist()
        y_pred_intent = results[f"{system}_intent"].tolist()
        y_true_escalate = results["true_escalate"].tolist()
        y_pred_escalate = results[f"{system}_decision"].tolist()

        m = metrics.intent_metrics(y_true_intent, y_pred_intent)
        m.update(metrics.escalation_metrics(y_true_escalate, y_pred_escalate))
        m["mean_grounding_overlap"] = float(results[f"{system}_grounding_overlap"].mean())
        if f"{system}_judge_overall" in results.columns:
            m["mean_judge_overall"] = float(results[f"{system}_judge_overall"].mean())
        summary[system] = m

    SUMMARY_PATH.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"\nPer-row results -> {RESULTS_PATH}\nSummary -> {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
