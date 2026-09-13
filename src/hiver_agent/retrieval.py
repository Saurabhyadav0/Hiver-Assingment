"""Retrieve similar historically-resolved (customer_msg, brand_reply) pairs to
ground a new reply draft. Embeddings run locally (sentence-transformers) —
see config.py for why: embedding thousands of pool rows through a rate-limited
API would take hours and burn quota needed for classify/draft/judge.
"""
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

from . import config

_model = None


def _embedder() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(config.EMBEDDING_MODEL)
    return _model


class RetrievalIndex:
    def __init__(self, pool: pd.DataFrame):
        self.pool = pool.reset_index(drop=True)
        embeddings = _embedder().encode(
            self.pool["customer_text"].tolist(), normalize_embeddings=True, show_progress_bar=False
        )
        self.embeddings = np.asarray(embeddings)

    def search(self, message: str, k: int = config.RETRIEVAL_TOP_K):
        query = _embedder().encode([message], normalize_embeddings=True, show_progress_bar=False)[0]
        sims = self.embeddings @ query
        top_idx = np.argsort(-sims)[:k]
        return [
            {
                "customer_text": self.pool.at[i, "customer_text"],
                "brand_text": self.pool.at[i, "brand_text"],
                "similarity": float(sims[i]),
            }
            for i in top_idx
        ]


def build_index(exclude_ids: set = frozenset()) -> RetrievalIndex:
    """Pool is the full processed thread set minus whatever's held out for
    the golden set (so grounding never leaks the eval answer back to itself),
    capped at RETRIEVAL_POOL_SIZE for speed."""
    df = pd.read_parquet(config.THREADS_PARQUET)
    if exclude_ids:
        df = df[~df["customer_tweet_id"].isin(exclude_ids)]
    if len(df) > config.RETRIEVAL_POOL_SIZE:
        df = df.sample(n=config.RETRIEVAL_POOL_SIZE, random_state=42)
    return RetrievalIndex(df)
