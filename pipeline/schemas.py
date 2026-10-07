"""Output schemas: the LLM's reply draft and the final per-ticket result."""

from collections.abc import Sequence
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

Intent = Literal[
    "deposit_issue",
    "password_reset",
    "withdrawal_issue",
    "account_lock",
    "trading_advice_request",
    "other",
]

SafetyFlag = Literal[
    "advice_request",
    "account_specific_request",
    "policy_sensitive",
    "insufficient_grounding",
    "low_confidence",
]


class ReplyDraft(BaseModel):
    """What the LLM produces for a ticket."""

    model_config = ConfigDict(extra="forbid")  # additionalProperties: false, required by strict JSON schema

    reply_draft: str = Field(description="Reply to the customer, using only the provided articles")
    grounded: bool = Field(description="True if every statement in the reply is supported by the provided articles")
    needs_human_escalation: bool = Field(
        description="True if a human agent should review: the articles don't cover the question, "
        "or it needs account-specific checks or a decision support can't make"
    )


class TicketResult(BaseModel):
    """One entry in results.json."""

    ticket_id: str
    intent: Intent
    retrieved_articles: list[str] = Field(max_length=3)
    reply_draft: str = Field(min_length=1)
    grounded: bool
    confidence: float = Field(ge=0.0, le=1.0)
    needs_human_escalation: bool
    safety_flags: list[SafetyFlag]


ESCALATING_FLAGS: tuple[SafetyFlag, ...] = ("insufficient_grounding", "low_confidence", "policy_sensitive")


def build_result(
    ticket_id: str,
    intent: Intent,
    intent_confidence: float,
    articles: list[tuple[str, float]],
    reply: ReplyDraft,
    safety_flags: list[SafetyFlag],
    uncertain: Sequence[str] = (),
    min_confidence: float = 0.6,
) -> tuple[TicketResult, list[str]]:
    """Combine the judgments and the reply into a validated result, plus the escalation reasons.

    `articles` are (article_id, relevance) pairs, best first. `uncertain` describes judgments
    too uncertain to act on; any of them adds `low_confidence` and escalates.
    Confidence is the weaker of intent certainty and the best article's relevance.
    Escalation is the LLM's call, but code also forces it when grounding or
    confidence is weak, or the ticket is policy-sensitive.
    """
    confidence = min(intent_confidence, articles[0][1]) if articles else 0.0
    grounded = reply.grounded and bool(articles)
    flags = list(safety_flags)
    if not grounded:
        flags.append("insufficient_grounding")
    if confidence < min_confidence or uncertain:
        flags.append("low_confidence")
    flags = list(dict.fromkeys(flags))

    reasons = [f"flag: {f}" for f in flags if f in ESCALATING_FLAGS]
    reasons += [f"uncertain judgment: {u}" for u in uncertain]
    if reply.needs_human_escalation:
        reasons.insert(0, "reply model requested human review")
    result = TicketResult(
        ticket_id=ticket_id,
        intent=intent,
        retrieved_articles=[aid for aid, _ in articles],
        reply_draft=reply.reply_draft,
        grounded=grounded,
        confidence=round(confidence, 4),
        needs_human_escalation=bool(reasons),
        safety_flags=flags,
    )
    return result, reasons
