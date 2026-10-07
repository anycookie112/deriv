"""Grounded reply generation: Groq when GROQ_API_KEY is set, a template fallback otherwise."""

import os

from groq import Groq

from .cache import cached
from .schemas import Intent, ReplyDraft

MODEL = "openai/gpt-oss-120b"  # supports strict JSON-schema output on Groq
SEED = 7

SYSTEM_PROMPT = """You draft replies to customer support tickets for a digital trading platform.

Use only the help-center articles provided with the ticket. Do not answer beyond them.
- Answer the customer's question directly, in a few short sentences, addressed to the customer.
- Never state or guess account-specific facts (balances, transaction status, why their specific request was declined).
- Never promise payment, deposit or withdrawal outcomes or timings.
- If the customer asks for trading or investment advice, market predictions or ways to make profit, politely \
decline and point them to general educational resources.
- If the articles don't cover the question, say a support specialist will follow up rather than guessing.

Set `grounded` to true only if every statement in your reply is supported by the articles.
Set `needs_human_escalation` to true if the articles don't fully cover the question, or it needs \
account-specific checks or a decision support can't make (e.g. a compliance or verification review)."""


def groq_client() -> Groq | None:
    """A Groq client if GROQ_API_KEY is set, else None (use the offline fallback)."""
    return Groq() if os.environ.get("GROQ_API_KEY") else None


def format_articles(articles: list[dict]) -> str:
    return "\n".join(
        f'<article id="{a["article_id"]}" title="{a["title"]}">\n{a["body"]}\n</article>' for a in articles
    )


def generate_reply(message: str, intent: Intent, articles: list[dict], client: Groq | None = None) -> ReplyDraft:
    """Draft a reply grounded in `articles` (the KB articles selected for this ticket).

    Uses Groq when `client` is given (see `groq_client()`), otherwise a template fallback.
    """
    if client is None:
        return fallback_reply(intent, articles)

    request = dict(
        model=MODEL,
        temperature=0,
        seed=SEED,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"<articles>\n{format_articles(articles) or 'No relevant articles found.'}\n</articles>\n\n"
                f'<ticket intent="{intent}">\n{message}\n</ticket>',
            },
        ],
        response_format={
            "type": "json_schema",
            "json_schema": {"name": "reply_draft", "strict": True, "schema": ReplyDraft.model_json_schema()},
        },
    )
    content = cached("groq", request, lambda: client.chat.completions.create(**request).choices[0].message.content)
    return ReplyDraft.model_validate_json(content)


def holding_reply(intent: Intent) -> ReplyDraft:
    """Reply for a ticket queued until the online pipeline can classify it.

    Suspected trading advice requests are refused straight away; refusing is safe even if unsure.
    """
    if intent == "trading_advice_request":
        return fallback_reply(intent, [])
    return ReplyDraft(
        reply_draft="Thanks for reaching out. We've received your request and a support specialist "
        "will follow up shortly.",
        grounded=False,
        needs_human_escalation=True,
    )


def fallback_reply(intent: Intent, articles: list[dict]) -> ReplyDraft:
    """Offline reply built only from article text. Conservative: escalates when unsure."""
    if intent == "trading_advice_request":
        return ReplyDraft(
            reply_draft="Thanks for reaching out. We're not able to provide market predictions, profit "
            "guarantees or asset recommendations. Our general educational resources are a good place "
            "to learn more about how trading works.",
            grounded=True,
            needs_human_escalation=False,
        )
    if not articles:
        return ReplyDraft(
            reply_draft="Thanks for reaching out. We've passed your question to a support specialist, "
            "who will follow up with you.",
            grounded=False,
            needs_human_escalation=True,
        )
    top = articles[0]
    return ReplyDraft(
        reply_draft=f"Thanks for reaching out. Here is what our help center says about "
        f"{top['title'].lower()}: {top['body']}",
        grounded=True,
        needs_human_escalation=False,
    )
