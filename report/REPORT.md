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

Evaluated on all 216 golden examples (`data/processed/eval_summary.json`):

| Metric | Pipeline | Simple (TF-IDF) | Trivial |
|---|---|---|---|
| Intent accuracy | **85.6%** | 45.8% | 13.4% |
| Intent macro-F1 | **78.2%** | 39.6% | 2.4% |
| Escalation precision | 78.7% | 74.8% | 0% |
| Escalation recall | 49.0% | **60.1%** | 0% |
| Escalation F1 | 60.3% | **66.7%** | 0% |
| Mean grounding overlap | 53.5% | 0% | 0% |
| Mean LLM-judge score (1-5) | 4.27 | n/a | n/a |

The pipeline clearly beats both baselines on intent classification — a
25-point macro-F1 lead over the TF-IDF baseline, a 72-point lead over always-
guessing the majority class. That's the expected result and not especially
interesting on its own.

The more interesting, less flattering result: **the simple baseline's
escalation F1 (66.7%) beats the pipeline's (60.3%)**, driven entirely by
recall (60.1% vs 49.0%) — the pipeline under-escalates relative to a dumb
TF-IDF classifier with a confidence threshold. See failure mode #1 below;
this isn't noise, it's a real gap in the always-escalate intent list.

Per-intent, the pipeline's classifier is strong on the well-represented
intents (account_security 0.92 F1, product_quality_defect 0.95,
compliment_positive_feedback 0.97) and weakest on general_inquiry (0.70 F1,
mostly confused with app_website_technical) and spam_unrelated (0/1 — see
the misleading-number section, this is a sample-size artifact, not a real
failure).

## Failure analysis: top 5 failure modes

**1. The always-escalate intent list is too narrow — 73/216 messages
(34%) that a careful reviewer would flag got auto-handled anyway.**
`decision.py` only force-escalates `billing_payment_issue` and
`account_security`. But plenty of `product_quality_defect` and
`order_status_delivery` messages describe things just as risky — fraud,
wrong address, a business account's ₹10K order — and the classifier's
confidence and retrieval grounding being *high* for these doesn't make
auto-sending a reply safe:
   > "Hi order number is 171-4216090-2817924 ordered wallet yesterday but
   > there is no wallet in it. Got e[mpty box]" — high confidence (0.95),
   > well-grounded (0.67 similarity) → auto-handled. A human reviewer
   > flagged this as escalate-worthy (possible fraud/misship).

   Hypothesis: confidence and grounding-similarity measure "can I produce a
   plausible-sounding reply," not "is this actually low-stakes." They're the
   wrong signals for this specific risk category.

**2. Retrieval grounding can copy an inappropriate template from a
superficially similar case.** 12/216 pipeline replies (5.6%) contain some
version of "please don't share your order details/phone number, we
consider that personal information" — copied from historical AmazonHelp
replies to people who over-shared, even when the *current* customer's
message includes an order number as legitimate context, not oversharing:
   > Customer: "reg. Ord 404-7091504-9405917. You should have alerted me
   > regarding the deduction to my Amazon Pay Balance..."
   > Drafted reply: "Please don't provide your order details as we consider
   > it personal information..." (judge score: 2.0/5, the lowest in the set)

   This is retrieval doing exactly what it's supposed to (finding a
   textually similar historical exchange) while missing that similarity in
   surface form doesn't imply similarity in what's actually needed.

**3. Intent confusion at the general_inquiry / app_website_technical
boundary** (weakest per-intent F1 at 0.70 and 0.83 respectively):
   > "Why there are no viewing rights for Suits Season 6 episode 10
   > onwards???" — true: app_website_technical, predicted: general_inquiry.
   > "I have filled the form and I am available now." — true:
   > general_inquiry, predicted: app_website_technical.

   Hypothesis: both intents cover "something about the app/site isn't doing
   what I expect," and the taxonomy's dividing line (a technical bug vs. a
   how-do-I question) isn't always clear from a single decontextualized
   tweet — this may need thread context (what prompted this message) to
   resolve reliably.

**4. Escalation over-triggers on messages with no good retrieval match,
even when the content is benign.** 19/216 cases were escalated where a
human reviewer said auto-handle was fine — several purely because
retrieval similarity fell under the 0.5 threshold on ordinary questions:
   > "can you buy Prime as a gift in the UK yet? Thanks" — escalated for
   > "no sufficiently similar historical resolution found," despite the
   > drafted reply being factually reasonable.

   Hypothesis: the 20K-pair retrieval pool, while large, doesn't have deep
   coverage of less-common but perfectly ordinary questions — the grounding
   threshold conflates "rare question" with "risky question."

**5. Very short, low-context messages get force-fit into a plausible-sounding
but under-determined intent.** "Plz update" was labeled complaint_negative_
other by the reviewer but classified as order_status_delivery by the
pipeline — both are defensible reads of two words with no thread context.
This is a genuine limit of single-message (not thread-aware) classification,
called out as an explicit scope cut in Problem Framing above, showing up
concretely here.

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
  table by design, not because the underlying components can't do better. In
  practice it's actually *not* conservative enough in the right places — see
  failure mode #1: the pipeline's escalation recall (49.0%) is lower than
  the dumb TF-IDF baseline's (60.1%), because the always-escalate list is
  too narrow, not because the pipeline is somehow smarter about safe cases.
- **85.6% intent accuracy hides a macro-F1 of 78.2%**, an 8-point gap driven
  almost entirely by `spam_unrelated`, which has exactly one example in the
  golden set (F1 = 0 on that one miss) and `general_inquiry` (0.70 F1, real
  confusion with app_website_technical — see failure mode #3). A single
  golden-set example for an entire intent category is a sampling artifact,
  not a real signal that the pipeline can't handle spam — it just means this
  216-example set can't measure that category at all.
- **The judge's 4.27/5 average reply-quality score has a real, findable
  failure hiding inside its own distribution**: 12/216 replies (5.6%) contain
  an inappropriate "please don't share personal information" scold copied
  from retrieval (failure mode #2), and the judge still scored the rest of
  that set 4.03/5 on average — a single bad sentence in an otherwise
  plausible-sounding reply doesn't tank the score much, so a good mean
  hides a real, systemic pattern.

## What I'd do next with one more week

1. **Actually hand-label the golden set.** This is the single biggest gap —
   everything downstream of it (accuracy numbers, the report's headline
   claims) is only as good as this, and right now it's LLM-drafted.
2. **Fix the escalation recall gap directly (failure mode #1).** Add a risk
   classifier — or just a broader keyword/intent list (fraud, wrong item,
   wrong address, business-account order value) — instead of relying on
   confidence and grounding similarity as risk proxies, since the data shows
   they aren't good ones for this purpose.
3. **Post-filter or re-prompt against the "don't share personal info"
   template-copy failure (#2)** — a cheap fix would be detecting when a
   retrieved reply is itself a privacy-scold and suppressing it as a
   grounding source unless the current message actually overshares.
4. **Get a genuinely independent judge model** (different provider/family
   from the generator) once budget allows, and re-run the judge-agreement
   check against real human scores.
5. **Multi-turn context** — most real support interactions aren't
   one-shot; incorporating thread history into classification and drafting
   would likely help both failure mode #3 (general_inquiry/app_technical
   confusion) and #5 (short, under-determined messages like "Plz update").
6. **Calibrate the confidence and grounding-similarity thresholds** against
   the (real) golden set instead of using reasonable-looking constants —
   failure mode #4 suggests the 0.5 grounding threshold is too strict for
   ordinary-but-uncommon questions.
7. **Expand the simple baseline's coverage** to all 10 intents (better weak
   labels, or a small amount of active learning) so the baseline comparison
   isn't structurally unfair to it.
8. **Get more than 1 golden example for spam_unrelated** and other thin
   categories — 216 examples across 10 intents means some categories can't
   be measured reliably at all.

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
