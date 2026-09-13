from src.eval.auto_label_golden import label_escalate, label_intent


def test_label_intent_calls_llm_with_taxonomy_and_message(monkeypatch):
    captured = {}

    def fake_generate_json(model, prompt, schema):
        captured["prompt"] = prompt
        captured["schema"] = schema
        return {"intent": "order_status_delivery", "reasoning": "asking about delivery"}

    monkeypatch.setattr("src.eval.auto_label_golden.generate_json", fake_generate_json)

    result = label_intent("where is my order")

    assert result["intent"] == "order_status_delivery"
    assert "where is my order" in captured["prompt"]
    assert "order_status_delivery" in captured["prompt"]  # taxonomy included


def test_label_escalate_does_not_reuse_decision_rules(monkeypatch):
    captured = {}

    def fake_generate_json(model, prompt, schema):
        captured["prompt"] = prompt
        return {"escalate": True, "reason": "involves account security"}

    monkeypatch.setattr("src.eval.auto_label_golden.generate_json", fake_generate_json)

    result = label_escalate("my account was hacked", "account_security")

    assert result["escalate"] is True
    assert "account_security" in captured["prompt"]
    assert "senior customer-support operations lead" in captured["prompt"]
