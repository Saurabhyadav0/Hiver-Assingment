from src.hiver_agent.intents import ALWAYS_ESCALATE, INTENT_NAMES, INTENTS


def test_taxonomy_size_in_range():
    assert 6 <= len(INTENT_NAMES) <= 10


def test_every_intent_has_description_and_examples():
    for name, spec in INTENTS.items():
        assert spec["description"]
        assert len(spec["examples"]) >= 2


def test_always_escalate_is_subset_of_intents():
    assert ALWAYS_ESCALATE <= set(INTENT_NAMES)
