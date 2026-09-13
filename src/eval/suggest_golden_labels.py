"""Pre-fill suggested_intent/suggested_confidence on the golden candidate CSV
using the real classifier, so the human labeler (you) is reviewing/correcting
suggestions instead of typing 200+ labels from scratch. These are explicitly
*suggestions*, kept in separate columns from true_intent — you still decide
and can override every single one. See decision_log.md for why this is fine
(and why it isn't the same as letting the model grade its own homework: the
judge-agreement check in judge_agreement.py is a harder, separate check).

Run in small batches via --limit rather than one long-lived process — plain
print+flush, no progress-bar library, so it behaves the same whether piped,
redirected, or backgrounded.
"""
import argparse
import time

import pandas as pd

from ..hiver_agent import config
from ..hiver_agent.classify import classify

PATH = config.DATA_PROCESSED / "golden_candidates.csv"


def main(limit: int = None):
    df = pd.read_csv(PATH, keep_default_na=False)
    # pandas 3.x enforces column dtype on .at[] writes; these columns start
    # as all-empty strings (inferred dtype=str) but need to hold floats too.
    # This is also the real explanation for the earlier multi-minute "stalls":
    # a ValueError here, raised in the main thread after a worker thread
    # returned, propagated while other threads were still mid-flight, so the
    # ThreadPoolExecutor context manager blocked waiting for them to finish
    # before the crash surfaced — looking exactly like a hang, not a crash.
    df["suggested_confidence"] = df["suggested_confidence"].astype(object)
    df["suggested_intent"] = df["suggested_intent"].astype(object)
    todo = [i for i in range(len(df)) if not df.at[i, "suggested_intent"]]
    if limit:
        todo = todo[:limit]

    for n, i in enumerate(todo):
        t0 = time.time()
        try:
            r = classify(df.at[i, "customer_text"])
            df.at[i, "suggested_intent"] = r["intent"]
            df.at[i, "suggested_confidence"] = r["confidence"]
            print(f"{n + 1}/{len(todo)} row={i} {r['intent']} ({time.time() - t0:.1f}s)", flush=True)
        except Exception as e:
            df.at[i, "suggested_intent"] = f"ERROR: {e}"
            print(f"{n + 1}/{len(todo)} row={i} ERROR ({time.time() - t0:.1f}s): {e}", flush=True)
        if (n + 1) % 10 == 0:
            df.to_csv(PATH, index=False)

    df.to_csv(PATH, index=False)
    remaining = sum(1 for x in df["suggested_intent"] if not x)
    print(f"Done this batch -> {PATH} ({remaining} still unlabeled)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    main(limit=args.limit)
