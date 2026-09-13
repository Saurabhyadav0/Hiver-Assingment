import pandas as pd

from src.hiver_agent.baselines import TrivialBaseline, train_simple_baseline


def test_trivial_baseline_never_escalates():
    result = TrivialBaseline().process("anything at all")
    assert result["decision"] == "auto_handle"


def _toy_pool():
    return pd.DataFrame(
        [
            {"customer_tweet_id": 1, "customer_text": "where is my package, it hasn't shipped"},
            {"customer_tweet_id": 2, "customer_text": "my package was late again, still no tracking"},
            {"customer_tweet_id": 3, "customer_text": "why was I charged twice for this order"},
            {"customer_tweet_id": 4, "customer_text": "I was overcharged on my payment"},
            {"customer_tweet_id": 5, "customer_text": "thanks so much, great service as always"},
            {"customer_tweet_id": 6, "customer_text": "thank you, appreciate the quick help"},
        ]
    )


def test_simple_baseline_always_escalates_security_and_billing():
    baseline = train_simple_baseline(_toy_pool())
    result = baseline.process("why was I charged twice")
    assert result["intent"] == "billing_payment_issue"
    assert result["decision"] == "escalate"


def test_simple_baseline_excludes_golden_ids_from_training():
    pool = _toy_pool()
    baseline = train_simple_baseline(pool, exclude_ids={1, 2})
    assert set(baseline.clf.classes_) == {"billing_payment_issue", "compliment_positive_feedback"}
