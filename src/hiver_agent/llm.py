"""Thin shared wrapper around the Gemini client so every module doesn't
re-implement API-key loading, rate limiting, and retry logic."""
import collections
import threading
import time

from dotenv import load_dotenv
from google import genai
from tenacity import retry, stop_after_attempt, wait_exponential

load_dotenv()

_client = None

# Free-tier Gemini flash-lite tolerates ~15 req/min before 429ing; stay under
# that with margin rather than relying on retries alone for every call. Each
# call also has ~7-14s of model latency, so callers doing bulk work (see
# eval/suggest_golden_labels.py) should parallelize with threads — this lock
# makes the throttle safe to share across them.
_MAX_CALLS_PER_MINUTE = 12
_call_times: collections.deque = collections.deque()
_lock = threading.Lock()


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
def generate_json(model: str, prompt: str, schema: dict):
    """Call Gemini asking for a response matching a JSON schema, with retries
    for transient rate limits / 5xx."""
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
