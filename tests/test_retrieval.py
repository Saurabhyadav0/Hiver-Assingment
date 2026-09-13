import pandas as pd

from src.hiver_agent.retrieval import RetrievalIndex


def test_search_returns_top_k_by_similarity():
    pool = pd.DataFrame(
        [
            {"customer_text": "where is my package", "brand_text": "checking tracking now"},
            {"customer_text": "I want a refund for my order", "brand_text": "sorry, let's process that refund"},
            {"customer_text": "my account got hacked", "brand_text": "let's secure your account"},
        ]
    )
    index = RetrievalIndex(pool)

    results = index.search("my order hasn't shipped yet, any tracking update?", k=2)

    assert len(results) == 2
    assert results[0]["similarity"] >= results[1]["similarity"]
    returned = {r["customer_text"] for r in results}
    assert "my account got hacked" not in returned  # least relevant of the three
