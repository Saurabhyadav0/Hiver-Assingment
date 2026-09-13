import pandas as pd

from src.eval.judge_agreement import compute_agreement, sample_for_human_scoring


def test_sample_drops_judge_score_from_blind_file(tmp_path, monkeypatch):
    from src.eval import judge_agreement as ja

    monkeypatch.setattr(ja.config, "DATA_PROCESSED", tmp_path)
    monkeypatch.setattr(ja, "AGREEMENT_SAMPLE_PATH", tmp_path / "sample.csv")

    df = pd.DataFrame(
        {
            "golden_id": [f"g{i}" for i in range(10)],
            "customer_text": [f"msg {i}" for i in range(10)],
            "reply_draft": [f"reply {i}" for i in range(10)],
            "judge_overall": [3.0] * 10,
        }
    )
    blind = sample_for_human_scoring(df, n=5)

    assert "judge_overall" not in blind.columns
    assert "human_overall" in blind.columns
    assert (tmp_path / "judge_agreement_answer_key.csv").exists()


def test_compute_agreement_perfect_match(tmp_path):
    scored = tmp_path / "scored.csv"
    key = tmp_path / "key.csv"
    pd.DataFrame({"golden_id": ["a", "b", "c"], "human_overall": [4, 3, 5]}).to_csv(scored, index=False)
    pd.DataFrame({"golden_id": ["a", "b", "c"], "judge_overall": [4, 3, 5]}).to_csv(key, index=False)

    result = compute_agreement(scored_path=scored, answer_key_path=key)

    assert result["n"] == 3
    assert result["quadratic_weighted_kappa"] == 1.0
    assert result["mean_abs_diff"] == 0.0
