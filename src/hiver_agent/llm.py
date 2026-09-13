"""Thin shared wrapper around the Gemini client so every module doesn't
re-implement API-key loading, rate limiting, retry, and caching logic.
"""
import collections
import hashlib
import json
import threading
import time
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from tenacity import retry, stop_after_attempt, wait_exponential

from . import config

load_dotenv()

_client = None

# Free-tier Gemini flash-lite tolerates ~15 req/min before 429ing; stay under
# that with margin rather than relying on retries alone for every call.
_MAX_CALLS_PER_MINUTE = 12
_call_times: collections.deque = collections.deque()
_lock = threading.Lock()

# Free tier also caps at 500 requests/day *per model* (discovered mid-build —
# see decision_log.md). The eval harness alone needs ~600 calls (classify +
# draft + judge over ~200 golden rows), so every call is cached to disk keyed
# by (model, prompt): a re-run of the harness costs zero quota, and if a day's
# quota runs out mid-run, resuming tomorrow just fills in what's missing
# instead of redoing completed work. This is also what makes the README's
# "reproduce in 15 minutes" claim honest regardless of API rate limits.
_CACHE_DIR = config.DATA_PROCESSED / "llm_cache"


def _cache_path(model: str, prompt: str) -> Path:
    key = hashlib.sha256(f"{model}\n{prompt}".encode()).hexdigest()
    return _CACHE_DIR / f"{key}.json"


def _throttle():
    with _lock:
        now = time.monotonic()
        while _call_times and now - _call_times[0] > 60:
            _call_times.popleft()
        if len(_call_times) >= _MAX_CALLS_PER_MINUTE:
            time.sleep(60 - (now - _call_times[0]) + 0.5)
        _call_times.append(time.monotonic())


def client() -> genai.Client:
    global _client
    if _client is None:
        # Without an explicit timeout, a stalled connection can hang a worker
        # thread indefinitely instead of failing into the retry logic below.
        _client = genai.Client(http_options={"timeout": 30_000})
    return _client


@retry(stop=stop_after_attempt(4), wait=wait_exponential(multiplier=1, min=5, max=30))
def _call_gemini(model: str, prompt: str, schema: dict):
    _throttle()
    resp = client().models.generate_content(
        model=model,
        contents=prompt,
        config={
            "response_mime_type": "application/json",
            "response_schema": schema,
        },
    )
    return resp.parsed if resp.parsed is not None else resp.text


def generate_json(model: str, prompt: str, schema: dict, use_cache: bool = True):
    """Call Gemini asking for a response matching a JSON schema, with
    disk caching (see _CACHE_DIR above) and retries for transient errors."""
    cache_file = _cache_path(model, prompt)
    if use_cache and cache_file.exists():
        return json.loads(cache_file.read_text())

    result = _call_gemini(model, prompt, schema)
    if isinstance(result, str):
        result = json.loads(result)

    if use_cache:
        _CACHE_DIR.mkdir(parents=True, exist_ok=True)
        cache_file.write_text(json.dumps(result))
    return result
