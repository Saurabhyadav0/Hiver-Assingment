"""Build the candidate pool for the golden evaluation set.

Stratified, not random: a pure random sample of 20k AmazonHelp threads would
be ~30% "where's my order" and starve the rarer intents (account security,
billing) that matter most for the escalate/auto-handle decision. We bucket
every candidate with a cheap keyword heuristic first (same one used to sanity
-check the taxonomy — see decision_log.md), then sample a capped number per
bucket plus a slice of "unmatched" (the hard/ambiguous cases) so the golden
set actually exercises every intent and isn't dominated by the easy majority
class.

This script only produces candidates + a heuristic guess. The actual
true_intent / true_escalate / escalate_reason columns are left blank (or
pre-filled by classify_golden_candidates.py as a *suggestion* to review, never
as a substitute for a human decision) — labeling is a separate, human step.
"""
import random

import pandas as pd

from ..hiver_agent import config
from ..hiver_agent.weak_labels import heuristic_bucket


def sample_candidates(df: pd.DataFrame, target_total: int = 220, seed: int = 42) -> pd.DataFrame:
    df = df.copy()
    df["heuristic_bucket"] = df["customer_text"].map(heuristic_bucket)

    buckets = df["heuristic_bucket"].unique().tolist()
    per_bucket_cap = max(target_total // len(buckets), 10)

    rng = random.Random(seed)
    parts = []
    for b in buckets:
        pool = df[df["heuristic_bucket"] == b]
        n = min(per_bucket_cap, len(pool))
        parts.append(pool.sample(n=n, random_state=seed))

    sampled = pd.concat(parts).drop_duplicates(subset=["customer_tweet_id"])
    if len(sampled) > target_total:
        sampled = sampled.sample(n=target_total, random_state=seed)

    sampled = sampled.sample(frac=1, random_state=seed).reset_index(drop=True)  # shuffle bucket order out
    sampled.insert(0, "golden_id", [f"g{i:04d}" for i in range(len(sampled))])
    return sampled


def main():
    df = pd.read_parquet(config.THREADS_PARQUET)
    sampled = sample_candidates(df)

    for col in ["true_intent", "true_escalate", "escalate_reason", "suggested_intent", "suggested_confidence"]:
        sampled[col] = ""

    out_path = config.DATA_PROCESSED / "golden_candidates.csv"
    cols = [
        "golden_id", "customer_text", "brand_text", "heuristic_bucket",
        "suggested_intent", "suggested_confidence", "true_intent",
        "true_escalate", "escalate_reason",
    ]
    sampled[cols].to_csv(out_path, index=False)
    print(f"Wrote {len(sampled)} candidates -> {out_path}")
    print(sampled["heuristic_bucket"].value_counts())


if __name__ == "__main__":
    main()
