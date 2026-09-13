import pandas as pd

from src.eval import run_eval


def _fake_index():
    class FakeIndex:
        def search(self, message, k=3):
            return [{"customer_text": "similar past msg", "brand_text": "please share your order id", "similarity": 0.8}]

    return FakeIndex()


def test_run_pipeline_row_shapes_output(monkeypatch):
    monkeypatch.setattr(run_eval, "classify", lambda msg: {"intent": "order_status_delivery", "confidence": 0.9})
    monkeypatch.setattr(run_eval, "draft_reply", lambda msg, intent, retrieved: {"reply": "please share your order id", "grounded": True})
    monkeypatch.setattr(run_eval, "judge_reply", lambda msg, reply, ref: {"overall": 4.2})

    result = run_eval.run_pipeline_row("where is my order", "please share your order id", _fake_index())

    assert result["intent"] == "order_status_delivery"
    assert result["decision"] is False  # high confidence + grounded + no anger -> auto_handle
    assert result["judge_overall"] == 4.2
    assert 0.0 <= result["grounding_overlap"] <= 1.0


def test_load_golden_filters_to_labeled_rows(tmp_path, monkeypatch):
    path = tmp_path / "golden.csv"
    pd.DataFrame({
        "golden_id": ["g0", "g1"],
        "customer_tweet_id": [1, 2],
        "customer_text": ["a", "b"],
        "brand_text": ["ra", "rb"],
        "true_intent": ["order_status_delivery", ""],
        "true_escalate": ["False", ""],
    }).to_csv(path, index=False)
    monkeypatch.setattr(run_eval, "GOLDEN_PATH", path)

    result = run_eval.load_golden()

    assert len(result) == 1
    assert result.iloc[0]["golden_id"] == "g0"
    assert result.iloc[0]["true_escalate"] == False  # noqa: E712


def test_main_exits_clearly_when_nothing_labeled(tmp_path, monkeypatch):
    import pytest

    path = tmp_path / "golden.csv"
    pd.DataFrame({
        "golden_id": ["g0"],
        "customer_tweet_id": [1],
        "customer_text": ["a"],
        "brand_text": ["ra"],
        "true_intent": [""],
        "true_escalate": [""],
    }).to_csv(path, index=False)
    monkeypatch.setattr(run_eval, "GOLDEN_PATH", path)

    with pytest.raises(SystemExit, match="No hand-labeled golden rows"):
        run_eval.main()
