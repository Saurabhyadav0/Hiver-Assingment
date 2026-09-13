from src.eval.metrics import escalation_metrics, grounding_overlap, intent_metrics


def test_intent_metrics_perfect_match():
    m = intent_metrics(["a", "b", "a"], ["a", "b", "a"])
    assert m["intent_accuracy"] == 1.0
    assert m["intent_macro_f1"] == 1.0


def test_intent_metrics_partial_match():
    m = intent_metrics(["a", "b", "a", "b"], ["a", "a", "a", "b"])
    assert m["intent_accuracy"] == 0.75


def test_escalation_metrics_all_correct():
    m = escalation_metrics([True, False, True], [True, False, True])
    assert m["escalate_precision"] == 1.0
    assert m["escalate_recall"] == 1.0


def test_escalation_metrics_missed_escalation():
    m = escalation_metrics([True, True, False], [False, True, False])
    assert m["escalate_recall"] == 0.5
    assert m["escalate_precision"] == 1.0


def test_grounding_overlap_high_for_near_identical_text():
    retrieved = [{"brand_text": "please share your order id so we can help"}]
    score = grounding_overlap("please share your order id so we can look into it", retrieved)
    assert score > 0.5


def test_grounding_overlap_zero_for_empty_retrieval():
    assert grounding_overlap("some reply", []) == 0.0
