import pandas as pd

from src.hiver_agent.data_prep import build_brand_pairs, clean_text, is_english


def test_clean_text_strips_mentions_and_links():
    assert clean_text("@acme hi https://x.co/y  there") == "hi there"


def test_is_english_filters_non_ascii():
    assert is_english("thanks for the help")
    assert not is_english("ご質問ありがとうございます")


def test_build_brand_pairs_links_reply_to_parent():
    df = pd.DataFrame(
        [
            {
                "tweet_id": 1,
                "author_id": "cust1",
                "inbound": True,
                "created_at": "t1",
                "text": "@acme my order is late",
                "in_response_to_tweet_id": None,
            },
            {
                "tweet_id": 2,
                "author_id": "acme",
                "inbound": False,
                "created_at": "t2",
                "text": "@cust1 sorry, can you share the order id?",
                "in_response_to_tweet_id": 1,
            },
            # reply to a reply (author is a customer) should be skipped
            {
                "tweet_id": 3,
                "author_id": "acme",
                "inbound": False,
                "created_at": "t3",
                "text": "orphaned reply",
                "in_response_to_tweet_id": 999,
            },
        ]
    )

    pairs = build_brand_pairs(df, "acme")

    assert len(pairs) == 1
    assert pairs.iloc[0]["customer_text"] == "my order is late"
    assert pairs.iloc[0]["brand_text"] == "sorry, can you share the order id?"
