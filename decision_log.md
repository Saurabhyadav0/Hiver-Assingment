# Decision log

Non-obvious decisions made while building this, and why.

1. **Brand: AmazonHelp**, not AppleSupport/Uber_Support/SpotifyCares. Compared all four on reply volume, "please DM us" deflection rate, and reply-text diversity. AmazonHelp had the most volume (169K raw replies), the lowest DM-deflection rate (1% vs 31-53% for the others), and 91% unique reply text — meaning it actually resolves things in public replies instead of punting everything to DMs, which matters when the whole point is grounding replies in historical resolutions.

2. **Dropped non-English threads** (~6-7% of AmazonHelp traffic) rather than building a multilingual taxonomy/judge. First pass used an ASCII-ratio filter, which missed French/German/Spanish (also mostly ASCII); added an English-stopword-density check on top after re-sampling and spotting the leak.

3. **10 intents, not fewer.** Ran a keyword-heuristic coverage check over 500 random messages before finalizing the taxonomy — order/delivery issues dominate (~29%+, undercounted by keywords) but there's a real long tail (compliments, pure venting, off-topic/spam) that would get force-fit into the wrong bucket with fewer categories.

4. **billing_payment_issue and account_security are always-escalate**, regardless of classifier confidence. Financial and account-security exposure is too high to risk a template reply, even a high-confidence one.

5. **Switched LLM provider from OpenAI to Gemini mid-build** — the OpenAI account had zero billing credits. Not a technical decision, a resourcing one.

6. **Classify, draft, and judge all use the same Gemini model** (`gemini-flash-lite-latest`). Not the original plan — Pro models get literally zero free-tier requests (confirmed via a live 429 naming `GenerateRequestsPerDayPerProjectPerModel-FreeTier` with `limit: 0`), and `gemini-flash-latest` caps around 5 req/min vs flash-lite's ~15. Using one model as both generator and judge is a known bias risk (self-preference) — flagged explicitly in the report rather than hidden.

7. **Retrieval embeddings run locally** (sentence-transformers `all-MiniLM-L6-v2`), not through the Gemini embeddings API. Embedding the multi-thousand-row retrieval pool at ~15 req/min would take hours and burn quota needed for classify/draft/judge; local embeddings are free and took 29s for 5,000 pairs.

8. **Every LLM call is cached to disk**, keyed by (model, prompt). Discovered mid-build that free tier caps at 500 requests/day *per model*, and the eval harness alone needs ~600 calls. Caching means a re-run costs zero quota, a day's-quota exhaustion just means picking up tomorrow instead of redoing work, and — deliberately — a grader can reproduce the headline numbers without their own API key at all, since the cache for the golden set is committed to the repo.

9. **Golden-set candidates are stratified by a keyword heuristic**, not randomly sampled. A pure random sample of the 20K processed threads would be ~30% "where's my order" and starve the rarer intents (account security, billing) that matter most for the escalate/auto-handle decision. Same heuristic is reused (not reimplemented) to train the simple baseline's weak labels — see `weak_labels.py`.

10. **The simple baseline can only ever predict 8 of the 10 intents.** Its training labels come from the same keyword heuristic used for golden-set stratification, and that heuristic can't distinguish general_inquiry or spam_unrelated from "unmatched" — so those rows are dropped from training rather than guessed at. This is a real, honest limitation of a "cheap" baseline, not a bug to paper over.

11. **The auto-handle/escalate decision is rule-based, not another LLM call.** The assignment explicitly asks for a *stated reason*. A rule gives a reason that's traceable to an actual signal (always-escalate intent, low confidence, weak retrieval grounding, anger markers); an LLM asked to justify its own decision tends to produce a plausible-sounding rationalization rather than the real cause.

12. **Golden-set `true_intent`/`true_escalate` are an independent LLM draft, not hand-labeled** — a real deviation from the plan, made under time pressure, and disclosed rather than hidden. The original design (`suggest_golden_labels.py`) pre-filled a `suggested_intent` column via the production classifier for a human to review into `true_intent`. That review didn't happen in time. Copying `suggested_intent` straight into `true_intent` was considered and rejected: it's the *same function* being evaluated, so intent-accuracy would be circular (~100%, meaningless). Instead `auto_label_golden.py` uses two prompts deliberately distinct from the production code — an "independent annotator" framing for intent, a "senior ops lead judging risk with no draft reply visible" framing for escalate — as a less-circular but still LLM-only stand-in. This is flagged prominently in the report's "what's misleading about my headline number" section: it is not the hand-labeled golden set the assignment asks for, and any accuracy number from it should be read as "agreement between two LLM passes," not "agreement with ground truth."

13. **Judge-vs-human agreement uses quadratic-weighted Cohen's kappa**, not raw accuracy or Pearson correlation. These are ordinal 1-5 scores where being off by one point is a much smaller disagreement than being off by four — unweighted accuracy would treat both misses identically.

14. **Retrieval pool explicitly excludes golden-set customer_tweet_ids.** Without this, the pipeline being evaluated could retrieve the exact historical reply to a golden-set message as "grounding," which would inflate reply-quality scores by leaking the answer back into the system being tested.

15. **`gemini-flash-lite-latest` over dated model names** (e.g. `gemini-2.5-flash`) for anything other than the underlying resolved model in error messages. Google rotates which dated model an alias points to; pinning a dated name that later gets deprecated (as happened live during this build — 404s for `gemini-2.5-flash` and `gemini-2.5-flash-lite`, both "no longer available to new users") breaks reproducibility worse than tracking a moving alias does.

16. **Chose to wait for tomorrow's quota reset over enabling billing**, after hitting the 500/day free-tier cap twice in one session (once during golden-set pre-labeling, once during the independent auto-labeling pass). This was a deliberate tradeoff, not a default: billing was offered and declined both times. Consequence: the golden set's LLM-drafted labels and the full `run_eval.py` numbers were incomplete at the time of this decision, finished the next day once quota refreshed.
