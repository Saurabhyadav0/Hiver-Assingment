# Report: AI support agent for AmazonHelp

## Problem framing

**What "good" means for this brand.** AmazonHelp handles an extremely high
volume of public replies across a long tail of issue types — from routine
"where's my order" questions to account-security incidents. Given that
volume, "good" here means: (1) never auto-send a reply on anything touching
money, account security, or an angry customer — the cost of a wrong
automated reply in those cases (an unauthorized-looking response, a
mishandled security concern) is much higher than the cost of an unnecessary
escalation; (2) for the large routine bucket (order status, general
questions), a reply that's actually grounded in how AmazonHelp has handled
similar cases before, not generic customer-service filler; (3) every
auto/escalate decision has to come with a reason a human can audit, not a
black-box confidence score.

**What I chose not to build:**
- **Multi-turn conversation state.** Each decision is made on a single
  customer message, not a full thread history. Real support conversations
  are multi-turn; handling that well would need thread-level context
  tracking, which was out of scope given the time budget.
- **Multilingual support.** ~6-7% of AmazonHelp traffic is non-English;
  dropped rather than building a second taxonomy/judge for it (see
  `decision_log.md` #2).
- **Actually posting to Twitter.** This is a decision-and-draft system, not
  a deployed bot — no posting, no rate-limiting against Twitter's API, no
  handling of Twitter-specific formatting beyond what's already in the data.
- **PII redaction beyond basic mention-stripping.** Customer order IDs,
  emails, and phone numbers that appear in the raw tweets are not redacted
  in the retrieval pool or drafts. A production system would need this.
- **A learned confidence calibration.** The escalate threshold on classifier
  confidence (0.6) is a reasonable-looking constant, not something tuned
  against held-out data — there wasn't a large enough labeled set to
  calibrate it properly (see the golden-set caveat below).

## Results vs. baselines

*(TODO once `run_eval.py` runs against a real golden set — see caveat below.
Table will be: intent accuracy/macro-F1, escalation precision/recall/F1,
mean grounding overlap, mean LLM-judge score, for trivial baseline / simple
baseline / full pipeline.)*

## Failure analysis: top 5 failure modes

*(TODO — to be filled in from `data/processed/eval_results.csv` once real
eval numbers exist, with actual message/reply examples per failure mode.)*

Candidates I expect to see, based on manual spot-checks during development:
1. **Ambiguous multi-intent messages** (e.g. a message that's both a
   complaint and a refund request) — the classifier has to pick one intent,
   and the wrong pick cascades into the wrong template/grounding.
2. **Sarcasm/irony read as literal** — "great, ANOTHER late package" scored
   by keyword-based anger detection may or may not catch this; needs
   checking against real examples.
3. **Retrieval grounding on superficially similar but substantively
   different cases** — e.g. two "where's my order" messages that are
   actually about very different underlying problems (address issue vs.
   carrier delay vs. never-shipped).
4. **Over-escalation on the always-escalate intents** — by design,
   *every* billing/security message escalates regardless of how trivial it
   actually is, which is deliberately conservative but will show up as a
   real cost in the escalation-rate numbers.
5. **Non-English or heavily abbreviated/typo-laden messages** slipping past
   the language filter and confusing the classifier.

## What is misleading about my headline number

This section is unusually important for this build, because of how the
golden set actually got made:

- **The golden set is not truly hand-labeled.** Time pressure (compounded by
  hitting the Gemini free-tier daily quota twice) meant `true_intent` and
  `true_escalate` come from an independent LLM labeling pass
  (`auto_label_golden.py`), not a human. Any intent-accuracy or
  escalation-precision number computed against it measures **agreement
  between two LLM passes**, not agreement with ground truth. See
  `decision_log.md` #12 for exactly what was done and why the more obvious
  shortcut (reusing the classifier's own suggestions) was rejected as
  circular.
- **Classify, draft, and judge share one underlying model** (free-tier
  constraint — see `decision_log.md` #6). The LLM judge scoring the
  pipeline's own replies is not an independent judge in the usual sense;
  self-preference bias is a real risk, not a hypothetical one.
- **The retrieval pool and the golden set come from the same 20K-pair
  subsample**, which itself was capped for reproducibility speed, not
  because it's representative of AmazonHelp's full traffic. Headline numbers
  describe performance on this subsample, not on AmazonHelp support traffic
  in general.
- **The simple baseline can't predict every intent** (see `decision_log.md`
  #10), so its comparison numbers against the full pipeline overstate the
  pipeline's relative advantage on the intents the baseline structurally
  can't reach.
- **The escalation policy is deliberately conservative** (two intents always
  escalate, confidence threshold is a guess, anger keywords are a small
  hardcoded list) — a high escalation rate looks "safe" but also means a lot
  of the auto-handle rate the system could plausibly achieve is left on the
  table by design, not because the underlying components can't do better.

## What I'd do next with one more week

1. **Actually hand-label the golden set.** This is the single biggest gap —
   everything downstream of it (accuracy numbers, the report's headline
   claims) is only as good as this, and right now it's LLM-drafted.
2. **Get a genuinely independent judge model** (different provider/family
   from the generator) once budget allows, and re-run the judge-agreement
   check against real human scores.
3. **Multi-turn context** — most real support interactions aren't
   one-shot; incorporating thread history into classification and drafting
   would likely change both accuracy and the right escalation policy.
4. **Calibrate the confidence threshold and anger-keyword list** against
   the (real) golden set instead of using reasonable-looking constants.
5. **PII handling** — redact order IDs/emails/phone numbers before they
   reach the retrieval pool or the LLM calls at all.
6. **Expand the simple baseline's coverage** to all 10 intents (better weak
   labels, or a small amount of active learning) so the baseline comparison
   isn't structurally unfair to it.

## Golden-set sampling and labeling methodology

- **Sampling**: stratified by a keyword heuristic (see `weak_labels.py`)
  over the 20K-pair processed AmazonHelp thread sample, capped per bucket,
  so all 10 intents are represented instead of the sample being dominated by
  the majority "where's my order" class. 216 candidates total.
- **Labeling**: see the caveat above — `true_intent`/`true_escalate` are an
  LLM draft (`auto_label_golden.py`), not hand-labeled by a human, due to
  time constraints compounded by hitting the Gemini free-tier daily quota
  twice in one session. This is disclosed, not hidden, and is the report's
  single most important caveat.

## Citations / borrowed material

- Dataset: Kaggle "Customer Support on Twitter" (thoughtvector/customer-support-on-twitter).
- Embedding model: `sentence-transformers/all-MiniLM-L6-v2` (open weights, used as-is for retrieval).
- No code borrowed from external repos/tutorials beyond standard library/framework usage (pandas, scikit-learn, google-genai, sentence-transformers).
