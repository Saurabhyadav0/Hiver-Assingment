from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"

RAW_CSV = DATA_RAW / "twcs.csv"

# Picked after comparing volume, reply diversity, and boilerplate rate across
# the top support accounts (see decision_log.md) — AmazonHelp had the most
# volume, the lowest "please DM us" rate (1%), and 91% unique reply text,
# meaning it actually resolves things publicly instead of punting to DMs.
BRAND_HANDLE = "AmazonHelp"

THREADS_PARQUET = DATA_PROCESSED / "threads.parquet"

# Caps so the whole pipeline reproduces in well under 15 minutes on a laptop.
MAX_THREADS = 20_000
RETRIEVAL_POOL_SIZE = 5_000

# Switched from OpenAI to Gemini mid-build (no OpenAI billing credits available).
# Free-tier Gemini quota turned out to be the real constraint: Pro models get
# ZERO free-tier requests (confirmed via a live 429 naming
# "GenerateRequestsPerDayPerProjectPerModel-FreeTier", limit 0), and
# "gemini-flash-latest" (currently gemini-3.8-flash) caps at ~5 req/min.
# "gemini-flash-lite-latest" tolerated ~15 req/min before throttling, so it's
# used for classify/draft AND judge. Using one model for both generation and
# judging is a known bias risk (self-preference) — see report's "what's
# misleading about my headline number" section.
CLASSIFY_MODEL = "gemini-flash-lite-latest"
DRAFT_MODEL = "gemini-flash-lite-latest"
JUDGE_MODEL = "gemini-flash-lite-latest"

# Retrieval embeddings run locally (sentence-transformers) instead of through
# the Gemini embeddings API — embedding the multi-thousand-row retrieval pool
# at ~15 req/min would take hours and burn quota needed for classify/draft/judge.
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

RETRIEVAL_TOP_K = 3
