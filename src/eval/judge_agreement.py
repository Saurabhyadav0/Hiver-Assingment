"""How much should we trust the LLM judge? Sample a subset of judged replies,
have a human (you) score the same "overall" dimension blind (no LLM score
visible), then compute agreement. Quadratic-weighted Cohen's kappa is used
instead of raw accuracy because these are ordinal 1-5 scores — being off by
one point is a much smaller disagreement than being off by four.
"""
import pandas as pd
from sklearn.metrics import cohen_kappa_score

from ..hiver_agent import config

AGREEMENT_SAMPLE_PATH = config.DATA_PROCESSED / "judge_agreement_sample.csv"


def sample_for_human_scoring(judged_df: pd.DataFrame, n: int = 30, seed: int = 42) -> pd.DataFrame:
    """judged_df needs: customer_text, reply_draft, judge_overall (1-5, will be
    rounded to nearest int for kappa). Writes a CSV with an empty
    human_overall column for you to fill in by hand, blind to judge_overall
    (that column is intentionally dropped from the file you'll actually see —
    kept in a separate answer-key file so you can't accidentally anchor on it)."""
    sample = judged_df.sample(n=min(n, len(judged_df)), random_state=seed).reset_index(drop=True)
    sample["human_overall"] = ""

    answer_key = sample[["golden_id", "judge_overall"]].copy()
    answer_key.to_csv(config.DATA_PROCESSED / "judge_agreement_answer_key.csv", index=False)

    blind = sample.drop(columns=["judge_overall"])
    blind.to_csv(AGREEMENT_SAMPLE_PATH, index=False)
    return blind


def compute_agreement(scored_path=AGREEMENT_SAMPLE_PATH, answer_key_path=None) -> dict:
    answer_key_path = answer_key_path or config.DATA_PROCESSED / "judge_agreement_answer_key.csv"
    human = pd.read_csv(scored_path)
    judge = pd.read_csv(answer_key_path)
    merged = human.merge(judge, on="golden_id")
    merged = merged[merged["human_overall"].notna()]

    human_scores = merged["human_overall"].round().astype(int)
    judge_scores = merged["judge_overall"].round().astype(int)

    kappa = cohen_kappa_score(human_scores, judge_scores, weights="quadratic")
    mean_abs_diff = (human_scores - judge_scores).abs().mean()
    return {
        "n": len(merged),
        "quadratic_weighted_kappa": kappa,
        "mean_abs_diff": mean_abs_diff,
    }
