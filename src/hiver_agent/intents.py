"""Intent taxonomy for AmazonHelp, derived from eyeballing ~500 real customer
messages (see decision_log.md for the sampling note). Order/delivery issues
dominate the traffic; the rest is a long tail that still needs its own bucket
so the classifier doesn't force everything into "order status".
"""

INTENTS = {
    "order_status_delivery": {
        "description": "Asking where an order/package is, or complaining it's late, missing, or mis-delivered.",
        "examples": [
            "hey! Order was due by 8pm, it's now 9:30pm and nothing. Not sure what to do?",
            "curious why the new echo i preordered a month ago didn't ship 2 days ago like it was supposed to?",
        ],
    },
    "refund_return_cancellation": {
        "description": "Wants a refund, wants to return an item, or wants to cancel an order.",
        "examples": [
            "I accidentally bought season 1 of the Dresden Files in my sleep, how would I go about getting a refund?",
            "having lots of problems with returns not being picked up for days, or at all.",
        ],
    },
    "billing_payment_issue": {
        "description": "Disputes a charge — wrong amount, double charge, price mismatch, payment method problem.",
        "examples": [
            "why have I been charged 1 pound twice for no reason?",
            "How is this even possible. Why am I paying more than MRP price",
        ],
    },
    "account_security": {
        "description": "Account compromised, unauthorized access, login/security-code problems, privacy concerns.",
        "examples": [
            "my account has been compromised, I see some russian email address registered with my phone no.",
            "I found a major bug in ur signin system..reported few months bk...still not reslvd.Its a serious privacy violation.",
        ],
    },
    "product_quality_defect": {
        "description": "Item received is wrong, damaged, defective, or not as described.",
        "examples": [
            "Finally received the defected and broken product with totally broken outer box. Refund requested.",
            "Frustration is opening an package and realising you ordered the wrong size phonecase",
        ],
    },
    "app_website_technical": {
        "description": "A bug or usability problem in the app/website itself, not tied to account security.",
        "examples": [
            "The PC app is horrible- takes forever to load and then freeze up right away :(",
            "couldn't complete the form also...how pathetic is that!!",
        ],
    },
    "general_inquiry": {
        "description": "A how-to, policy, or informational question not tied to an existing problem.",
        "examples": [
            "Can I request Kindle edition for a book? If I can do that, how long would it take?",
            "can we got support for gaana on echo device in india",
        ],
    },
    "compliment_positive_feedback": {
        "description": "Praise, thanks, or positive sentiment with no request attached.",
        "examples": [
            "Its an absolute satisfying customer care , I hv ever experienced . amazed . kudos",
            "thanks for your 1.5 day delivery to a Non-Prime customer. Saved my day on her birthday!!!",
        ],
    },
    "complaint_negative_other": {
        "description": "Venting or frustration without one specific actionable request, or a complaint that doesn't fit the categories above.",
        "examples": [
            "well done you're officially stupid. I leave you simple instructions... and now my parcel has been stolen. cheers",
            "Amazon is a liar",
        ],
    },
    "spam_unrelated": {
        "description": "Off-topic content: contests, business pitches, unrelated chatter, phone numbers dropped with no context.",
        "examples": [
            "i have no any special degree i have business idia only then please help how to join for only business idea.",
            "I want to participate in this contest, please send the details as soon as possible. Amazon Is Love and Trust for me",
        ],
    },
}

INTENT_NAMES = list(INTENTS.keys())

# Intents where auto-handling is never appropriate regardless of classifier
# confidence — legal/financial/security exposure is too high for a template
# reply. Used by decision.py.
ALWAYS_ESCALATE = {"account_security", "billing_payment_issue"}
