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

CLASSIFY_MODEL = "gpt-4o-mini"
DRAFT_MODEL = "gpt-4o-mini"
JUDGE_MODEL = "gpt-4o"
EMBEDDING_MODEL = "text-embedding-3-small"

RETRIEVAL_TOP_K = 3
