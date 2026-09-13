"""Pre-fill suggested_intent/suggested_confidence on the golden candidate CSV
using the real classifier, so the human labeler (you) is reviewing/correcting
suggestions instead of typing 200+ labels from scratch. These are explicitly
*suggestions*, kept in separate columns from true_intent — you still decide
and can override every single one. See decision_log.md for why this is fine
(and why it isn't the same as letting the model grade its own homework: the
judge-agreement check in judge_agreement.py is a harder, separate check).
"""
import pandas as pd
from tqdm import tqdm

from ..hiver_agent import config
from ..hiver_agent.classify import classify

PATH = config.DATA_PROCESSED / "golden_candidates.csv"


def main():
    df = pd.read_csv(PATH, keep_default_na=False)
    for i in tqdm(range(len(df))):
        try:
            r = classify(df.at[i, "customer_text"])
            df.at[i, "suggested_intent"] = r["intent"]
            df.at[i, "suggested_confidence"] = r["confidence"]
        except Exception as e:
            df.at[i, "suggested_intent"] = f"ERROR: {e}"
        if i % 20 == 0:
            df.to_csv(PATH, index=False)  # checkpoint in case of a long-running failure
    df.to_csv(PATH, index=False)
    print(f"Done -> {PATH}")


if __name__ == "__main__":
    main()
