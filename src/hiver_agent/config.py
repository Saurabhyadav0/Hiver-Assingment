from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"

RAW_CSV = DATA_RAW / "twcs.csv"

# Set after inspecting the data (Part 1). Twitter handle of the support account.
BRAND_HANDLE = None

THREADS_PARQUET = DATA_PROCESSED / "threads.parquet"

# Caps so the whole pipeline reproduces in well under 15 minutes on a laptop.
MAX_THREADS = 20_000
RETRIEVAL_POOL_SIZE = 5_000

CLASSIFY_MODEL = "gpt-4o-mini"
DRAFT_MODEL = "gpt-4o-mini"
JUDGE_MODEL = "gpt-4o"
EMBEDDING_MODEL = "text-embedding-3-small"

RETRIEVAL_TOP_K = 3
