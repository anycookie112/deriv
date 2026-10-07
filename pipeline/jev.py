"""TypeSafe Jev judgments, each one API call, plus offline fallbacks with the same return shapes.

- classify_intent:     Choice over the intent labels
- select_articles:     one Noul per retrieved candidate (acts as the reranker)
- detect_safety_flags: Nouls for ticket-level risks
- verify_reply:        Nouls checking the drafted reply against its articles
"""

from typing import get_args

from typesafe_sdk import Choice, JSONContent, Noul, NoulCriteria, Question, TypeSafeClient

from .cache import cached
from .schemas import Intent, SafetyFlag


def certainty(p: float) -> float:
    """How decided a Noul answer is, 0..1: 0 at p=0.5 (coin flip), 1 at p=0 or p=1."""
    return abs(2 * p - 1)


def ask(client: TypeSafeClient, state: JSONContent, questions: dict[str, Question]) -> dict[str, dict]:
    """One cached System One call. Returns answers as plain dicts keyed by question name."""
    request = {"state": state, "questions": {k: q.model_dump(mode="json") for k, q in questions.items()}}
    return cached(
        "jev",
        request,
        lambda: {
            k: a.model_dump(mode="json") for k, a in client.system_one(state=state, questions=questions).answers.items()
        },
    )


# --- Intent ---------------------------------------------------------------------------

INTENT_QUESTION = Choice(
    instructions="What is the customer's main intent in this support ticket?",
    criteria={
        "deposit_issue": "Money added to the account is missing, delayed or failed",
        "password_reset": "Forgot password or not receiving the reset email",
        "withdrawal_issue": "Taking money out is declined, delayed or under review",
        "account_lock": "Account locked or blocked after too many sign-in attempts",
        "trading_advice_request": "Asks for market predictions, which asset to buy, or how to make profit",
        "other": "None of the above",
    },
)


def classify_intent(state: JSONContent, client: TypeSafeClient) -> tuple[Intent, float]:
    """Classify a ticket's intent. Returns (intent, confidence in 0..1)."""
    answer = ask(client, state, {"intent": INTENT_QUESTION})["intent"]
    if answer["choice"] not in get_args(Intent):
        raise ValueError(f"Unexpected intent label from Jev: {answer['choice']!r}")
    return answer["choice"], answer["confidence"]


# --- Article relevance ------------------------------------------------------------------

ARTICLE_CRITERIA = NoulCriteria(
    true="The article gives guidance a support agent can use to answer this ticket",
    false="The article is about a different topic or only loosely related",
)


def select_articles(
    state: JSONContent, candidates: list[dict], client: TypeSafeClient, top_k: int = 3, min_score: float = 0.5
) -> tuple[list[tuple[str, float]], dict[str, float]]:
    """Judge which retrieved candidates are relevant to the ticket.

    Returns (selected, scores): up to `top_k` (article_id, relevance) pairs scoring at least
    `min_score`, best first, and the relevance of every candidate. An empty selection means
    nothing is relevant enough to ground a reply.
    """
    if not candidates:
        return [], {}
    full_state = {
        "ticket": state,
        "articles": {a["article_id"]: {"title": a["title"], "body": a["body"]} for a in candidates},
    }
    questions = {
        a["article_id"]: Noul(
            instructions=f"Is the KB article `articles.{a['article_id']}` relevant to answering `ticket`?",
            criteria=ARTICLE_CRITERIA,
        )
        for a in candidates
    }
    scores = {aid: a["noul"] for aid, a in ask(client, full_state, questions).items()}
    ranked = sorted(scores.items(), key=lambda x: (-x[1], x[0]))
    return [(aid, s) for aid, s in ranked if s >= min_score][:top_k], scores


# --- Safety flags -----------------------------------------------------------------------

# Flags judged from the ticket text. insufficient_grounding and low_confidence depend on
# retrieval and generation results, so they are set when assembling the result.
SAFETY_QUESTIONS: dict[SafetyFlag, Noul] = {
    "advice_request": Noul(
        instructions="Does the customer ask for trading or investment advice?",
        criteria=NoulCriteria(
            true="Asks for market predictions, which asset to buy or sell, or how to make a profit",
            false="Asks about using the platform, payments or their account",
        ),
    ),
    "account_specific_request": Noul(
        instructions="Does answering require looking up this customer's own account or transaction records?",
        criteria=NoulCriteria(
            true="Needs details only available from their account, e.g. why a specific payment was declined",
            false="General guidance or troubleshooting steps can answer it",
        ),
    ),
    "policy_sensitive": Noul(
        instructions="Does the ticket involve a compliance, verification or security review decision?",
        criteria=NoulCriteria(
            true="Involves a withdrawal review, verification or compliance check, or account restrictions",
            false="Routine troubleshooting the customer can follow, such as waiting out a temporary sign-in lock",
        ),
    ),
}


def detect_safety_flags(
    state: JSONContent, client: TypeSafeClient, threshold: float = 0.5
) -> tuple[list[SafetyFlag], dict[SafetyFlag, float]]:
    """Flag risky aspects of a ticket.

    Returns (flags, probabilities): the flags scoring at least `threshold`, and every question's probability.
    """
    probs = {flag: a["noul"] for flag, a in ask(client, state, SAFETY_QUESTIONS).items()}
    return [flag for flag in SAFETY_QUESTIONS if probs[flag] >= threshold], probs


# --- Reply verification -----------------------------------------------------------------

VERIFY_QUESTIONS = {
    "supported": Noul(
        instructions="Is every statement in `reply` supported by `articles`?",
        criteria=NoulCriteria(
            true="Each statement appears in or directly follows from the articles",
            false="The reply states facts, promises, timings or account details the articles don't contain",
        ),
    ),
    "refuses_advice": Noul(
        instructions="Does `reply` decline to give trading advice, market predictions or asset recommendations?",
        criteria=NoulCriteria(
            true="Politely declines and does not suggest any asset, prediction or profit strategy",
            false="Gives or hints at a prediction, recommendation or way to make a profit, or doesn't decline",
        ),
    ),
}


def verify_reply(reply: str, articles: list[dict], client: TypeSafeClient) -> dict[str, float]:
    """Check the drafted reply. Returns probabilities for `supported` and `refuses_advice`."""
    state = {"reply": reply, "articles": [{"title": a["title"], "body": a["body"]} for a in articles]}
    answers = ask(client, state, VERIFY_QUESTIONS)
    return {k: a["noul"] for k, a in answers.items()}


# --- Offline fallbacks (no TYPESAFE_API_KEY) ----------------------------------------------

ARTICLE_INTENT: dict[str, Intent] = {
    "A1": "deposit_issue",
    "A2": "password_reset",
    "A3": "withdrawal_issue",
    "A4": "account_lock",
    "A5": "trading_advice_request",
}


def fallback_classify(
    candidates: list[dict], min_similarity: float = 0.6
) -> tuple[Intent, float, list[tuple[str, float]]]:
    """Offline: the article with the highest cosine similarity to the ticket decides the intent.

    Returns (intent, similarity, matched article as [(article_id, similarity)]). Below
    `min_similarity` nothing matches and the intent is `other`.
    """
    if not candidates:
        return "other", 0.0, []
    best = max(candidates, key=lambda c: c["vector_score"])
    score = best["vector_score"]
    if score < min_similarity or best["article_id"] not in ARTICLE_INTENT:
        return "other", score, []
    return ARTICLE_INTENT[best["article_id"]], score, [(best["article_id"], score)]


def fallback_safety_flags(intent: Intent) -> list[SafetyFlag]:
    return {
        "trading_advice_request": ["advice_request"],
        "withdrawal_issue": ["account_specific_request", "policy_sensitive"],
    }.get(intent, [])
