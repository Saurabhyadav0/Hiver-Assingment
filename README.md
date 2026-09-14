# Hiver take-home: AI support agent for AmazonHelp

An AI support agent for AmazonHelp (the brand chosen from the Kaggle
"Customer Support on Twitter" dataset — see `decision_log.md` #1 for why)
that: classifies an incoming customer message into one of 10 intents,
drafts a reply grounded in how AmazonHelp has historically resolved similar
messages, and decides whether to auto-handle or escalate to a human, with a
stated reason.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # add your GOOGLE_API_KEY (aistudio.google.com/apikey)
```

An API key is only required to *generate new* results. Everything already
computed (the golden-set evaluation) is cached to disk at
`data/processed/llm_cache/` and committed to the repo — see below.

## Reproduce the headline results in under 15 minutes

```bash
python -m pytest tests/ -q                 # ~15s, 31 unit tests
python -m src.eval.run_eval                # ~1-2 min, reads from cache
```

Headline numbers (216 golden examples — see the important caveat below on
how ground truth was produced):

| Metric | Pipeline | Simple baseline | Trivial baseline |
|---|---|---|---|
| Intent accuracy | 85.6% | 45.8% | 13.4% |
| Intent macro-F1 | 78.2% | 39.6% | 2.4% |
| Escalation F1 | 60.3% | 66.7% | 0% |
| Mean LLM-judge score (1-5) | 4.27 | n/a | n/a |

The pipeline beats both baselines on intent classification. It does *not*
beat the simple baseline on escalation F1 — see `report/REPORT.md`'s failure
analysis for why (short version: the always-escalate intent list is too
narrow).

`run_eval.py` reads `data/processed/golden_candidates.csv` (the golden set)
and `data/processed/threads.parquet` (the processed AmazonHelp thread pool,
both already committed), runs the trivial baseline, the simple (TF-IDF)
baseline, and the real pipeline over every golden example, and writes:

- `data/processed/eval_results.csv` — per-example predictions for all three systems
- `data/processed/eval_summary.json` — the headline metrics table

Every classify/draft/judge call the real pipeline makes is a cache hit
against `data/processed/llm_cache/` (keyed by model+prompt), so this doesn't
call the Gemini API and doesn't need a working `GOOGLE_API_KEY` at all for
the numbers already in this repo. Retrieval embeddings run locally
(sentence-transformers) regardless.

To run the agent on a new message live (this does need an API key):

```bash
python3 -c "from src.hiver_agent.pipeline import process; import json; print(json.dumps(process('where is my order'), indent=2, default=str))"
```

## Rebuilding from scratch (optional — not needed to reproduce headline numbers)

The full dataset (twcs.csv, ~3M tweets, 516MB) isn't in this repo. To
regenerate `threads.parquet` yourself:

1. Download `twcs.csv` from [kaggle.com/datasets/thoughtvector/customer-support-on-twitter](https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter) and place it at `data/raw/twcs.csv`.
2. `python -m src.hiver_agent.data_prep` — parses the CSV, reconstructs (customer message, brand reply) pairs for AmazonHelp, filters to English, subsamples to 20K pairs. Takes ~20s.
3. `python -m src.eval.sample_golden` — regenerates the golden-set candidate sampling (this will produce a *different* sample than the committed one unless you keep the same seed and input data).

## Repo layout

```
src/hiver_agent/     the actual agent: data prep, intents, classify, retrieval,
                     draft_reply, decision, pipeline, baselines
src/eval/            golden-set sampling/labeling tools, metrics, LLM judge
                     agreement check, run_eval.py orchestration
data/processed/      threads.parquet, golden_candidates.csv, llm_cache/, eval outputs
tests/               31 unit tests, mostly logic-only (no API calls)
decision_log.md      17 non-obvious decisions and why
report/REPORT.md     problem framing, real results, failure analysis, next steps
```

## Known limitations (see `report/REPORT.md` for the full write-up)

- **The golden set's `true_intent`/`true_escalate` are an LLM draft, not
  hand-labeled by a human.** This is the single most important caveat in
  this repo — the assignment asks for hand-labeled ground truth, and time
  constraints (compounded by hitting the Gemini free-tier daily quota twice)
  meant that didn't happen. `auto_label_golden.py` uses two prompts
  deliberately distinct from the production classify/decision code as a
  less-circular stand-in; see `decision_log.md` #12 and the report's "what's
  misleading about my headline number" section.
- Classify, draft, and judge all currently use the same underlying Gemini
  model (free-tier constraints — Pro models get zero free-tier requests).
  This is a real self-preference bias risk for the judge scores.
- **The judge-vs-human agreement number (kappa = 0.29) is LLM-vs-LLM, not
  real human agreement** — a genuine human score wasn't obtained in time.
  See `decision_log.md` #17 and the report's dedicated section on this.
