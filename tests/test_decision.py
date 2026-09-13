from src.hiver_agent.decision import decide


def test_always_escalate_intent_overrides_everything():
    r = decide("my card was charged twice", "billing_payment_issue", confidence=0.99,
               retrieved=[{"similarity": 0.9}], reply_grounded=True)
    assert r["decision"] == "escalate"
    assert "billing_payment_issue" in r["reason"]


def test_low_confidence_escalates():
    r = decide("hmm not sure what's going on", "general_inquiry", confidence=0.3,
               retrieved=[{"similarity": 0.9}], reply_grounded=True)
    assert r["decision"] == "escalate"
    assert "confidence" in r["reason"]


def test_ungrounded_reply_escalates():
    r = decide("where is my order", "order_status_delivery", confidence=0.9,
               retrieved=[], reply_grounded=False)
    assert r["decision"] == "escalate"
    assert "grounded" in r["reason"] or "historical" in r["reason"]


def test_angry_message_escalates_even_when_confident_and_grounded():
    r = decide("this is unacceptable, worst service ever", "order_status_delivery",
               confidence=0.9, retrieved=[{"similarity": 0.9}], reply_grounded=True)
    assert r["decision"] == "escalate"
    assert "negative" in r["reason"] or "anger" in r["reason"]


def test_clean_case_auto_handles():
    r = decide("where is my order, tracking says delayed", "order_status_delivery",
               confidence=0.9, retrieved=[{"similarity": 0.85}], reply_grounded=True)
    assert r["decision"] == "auto_handle"
