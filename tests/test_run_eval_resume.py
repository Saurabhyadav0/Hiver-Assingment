import pandas as pd

from src.eval import run_eval


def test_load_partial_results_keys_by_golden_id(tmp_path, monkeypatch):
    path = tmp_path / "results.csv"
    pd.DataFrame({
        "golden_id": ["g0", "g1"],
        "true_intent": ["order_status_delivery", "billing_payment_issue"],
        "pipeline_intent": ["order_status_delivery", "billing_payment_issue"],
    }).to_csv(path, index=False)
    monkeypatch.setattr(run_eval, "RESULTS_PATH", path)

    result = run_eval._load_partial_results()

    assert set(result.keys()) == {"g0", "g1"}
    assert result["g0"]["pipeline_intent"] == "order_status_delivery"


def test_load_partial_results_empty_when_no_file(tmp_path, monkeypatch):
    monkeypatch.setattr(run_eval, "RESULTS_PATH", tmp_path / "does_not_exist.csv")

    assert run_eval._load_partial_results() == {}


def test_load_partial_results_empty_when_file_is_zero_bytes(tmp_path, monkeypatch):
    path = tmp_path / "empty.csv"
    path.write_text("")
    monkeypatch.setattr(run_eval, "RESULTS_PATH", path)

    assert run_eval._load_partial_results() == {}
