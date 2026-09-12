"""Turn the raw twcs.csv export into (customer_message, brand_reply) pairs for one brand.

The Kaggle file has one row per tweet: tweet_id, author_id, inbound,
created_at, text, response_tweet_id, in_response_to_tweet_id. `inbound=True`
means a customer wrote it; brand support accounts show up as author_id with
inbound=False.
"""
import re

import pandas as pd

from . import config

USECOLS = [
    "tweet_id",
    "author_id",
    "inbound",
    "created_at",
    "text",
    "in_response_to_tweet_id",
]


def load_raw(path=config.RAW_CSV) -> pd.DataFrame:
    df = pd.read_csv(path, usecols=USECOLS, dtype={"tweet_id": "int64"})
    df["in_response_to_tweet_id"] = pd.to_numeric(
        df["in_response_to_tweet_id"], errors="coerce"
    )
    return df


def top_brand_candidates(df: pd.DataFrame, n: int = 20) -> pd.Series:
    """Support accounts, ranked by how many replies they sent. Use this to pick a brand."""
    brands = df[df["inbound"] == False]  # noqa: E712
    return brands["author_id"].value_counts().head(n)


def clean_text(text: str) -> str:
    text = re.sub(r"@\w+", "", text)  # strip @mentions (both directions)
    text = re.sub(r"https?://\S+", "", text)
    return re.sub(r"\s+", " ", text).strip()


_ENGLISH_STOPWORDS = {
    "the", "i", "you", "to", "a", "is", "and", "my", "it", "for", "on", "in",
    "of", "this", "have", "was", "with", "me", "your", "not", "be", "that",
    "but", "are", "at", "just", "can", "please", "why", "no", "do", "if",
    "still", "has", "been", "will", "would", "so", "get", "order", "thanks",
}


def is_english(text: str, min_ascii_ratio: float = 0.9) -> bool:
    """ASCII-script filter (catches Japanese/Arabic/etc.) plus an English
    stopword check (catches French/German/Spanish, which are ASCII-heavy but
    not English) — a handful of non-English threads would otherwise force a
    multilingual taxonomy and judge for one brand's worth of data."""
    if not text:
        return False
    if sum(c.isascii() for c in text) / len(text) < min_ascii_ratio:
        return False
    words = re.findall(r"[a-zA-Z']+", text.lower())
    if len(words) < 4:
        return True  # too short to judge reliably, let it through
    hits = sum(1 for w in words if w in _ENGLISH_STOPWORDS)
    return hits / len(words) >= 0.15


def build_brand_pairs(df: pd.DataFrame, brand_handle: str) -> pd.DataFrame:
    """One row per (customer opening message, brand's reply) for the given brand."""
    by_id = df.set_index("tweet_id")

    replies = df[(df["inbound"] == False) & (df["author_id"] == brand_handle)]  # noqa: E712
    replies = replies.dropna(subset=["in_response_to_tweet_id"])

    rows = []
    for _, reply in replies.iterrows():
        parent_id = int(reply["in_response_to_tweet_id"])
        parent = by_id.loc[parent_id] if parent_id in by_id.index else None
        if parent is None or not bool(parent["inbound"]):
            continue  # only keep replies that answer an actual customer tweet
        rows.append(
            {
                "customer_tweet_id": parent_id,
                "brand_tweet_id": int(reply["tweet_id"]),
                "customer_text": clean_text(str(parent["text"])),
                "brand_text": clean_text(str(reply["text"])),
                "created_at": reply["created_at"],
            }
        )

    pairs = pd.DataFrame(rows)
    pairs = pairs[pairs["customer_text"].str.len() > 0]
    pairs = pairs[pairs["customer_text"].map(is_english)]
    pairs = pairs.drop_duplicates(subset=["customer_tweet_id", "brand_tweet_id"])
    return pairs.reset_index(drop=True)


def main(brand_handle: str = None, max_threads: int = config.MAX_THREADS):
    brand_handle = brand_handle or config.BRAND_HANDLE
    if not brand_handle:
        raise SystemExit("Set BRAND_HANDLE in config.py or pass --brand")

    df = load_raw()
    pairs = build_brand_pairs(df, brand_handle)
    if len(pairs) > max_threads:
        pairs = pairs.sample(n=max_threads, random_state=42).reset_index(drop=True)

    config.DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    pairs.to_parquet(config.THREADS_PARQUET, index=False)
    print(f"Wrote {len(pairs)} pairs for {brand_handle} -> {config.THREADS_PARQUET}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--brand", default=None)
    parser.add_argument("--list-brands", action="store_true")
    args = parser.parse_args()

    if args.list_brands:
        print(top_brand_candidates(load_raw()))
    else:
        main(brand_handle=args.brand)
