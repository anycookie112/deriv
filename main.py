"""Support ticket pipeline.

LOAD_DATA -> INDEX_KB -> CLASSIFY_INTENT -> RETRIEVE_CONTEXT -> GENERATE_REPLY -> VALIDATE_OUTPUT -> WRITE_RESULTS

    python main.py              # uses Jev / Groq when TYPESAFE_API_KEY / GROQ_API_KEY are set
    python main.py --offline    # no API keys needed: confident tickets are answered, the rest queued

Offline, a ticket is classified by its cosine similarity to the KB articles. Only matches at or
above --offline-threshold are answered; the others get a holding reply, are escalated, and are
written to the queue file. The next online run classifies them with the full pipeline and clears
them from the queue.
"""

import argparse
import json
import os
import sys
from contextlib import nullcontext
from pathlib import Path

from dotenv import load_dotenv
from typesafe_sdk import TypeSafeClient

from pipeline import cache, jev
from pipeline.embed_kb import load_articles
from pipeline.reply import fallback_reply, generate_reply, groq_client, holding_reply
from pipeline.retrieval import HybridRetriever, load_or_build_index
from pipeline.schemas import ReplyDraft, TicketResult, build_result

JEV_MODE = "TypeSafe Jev"
OFFLINE_THRESHOLD = 0.8


def stage(name: str) -> None:
    print(f"[{name}]", file=sys.stderr)


# --- LOAD_DATA ----------------------------------------------------------------------------


def load_tickets(path: Path) -> list[dict]:
    tickets = json.loads(path.read_text(encoding="utf-8"))
    for t in tickets:
        if not isinstance(t.get("id"), str) or not isinstance(t.get("message"), str) or not t["message"].strip():
            raise ValueError(f"Invalid ticket (needs string `id` and non-empty `message`): {t!r}")
    ids = [t["id"] for t in tickets]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate ticket ids in tickets file")
    return tickets


# --- Pipeline -----------------------------------------------------------------------------


def run_pipeline(
    tickets: list[dict],
    kb: list[dict],
    index_path: Path,
    jev_client: TypeSafeClient | None,
    llm,
    offline_threshold: float = OFFLINE_THRESHOLD,
) -> tuple[list[TicketResult], list[dict]]:
    """Run every stage for all tickets. Returns the validated results and per-ticket debug records."""
    kb_by_id = {a["article_id"]: a for a in kb}
    records = [{"ticket": t, "state": {"message": t["message"], "language": t.get("language", "en")}} for t in tickets]

    stage("INDEX_KB")
    retriever = HybridRetriever(load_or_build_index(kb, index_path))
    for r in records:
        # Deterministic hybrid search: top-3 vector + top-3 BM25, merged.
        r["candidates"] = retriever.retrieve(r["ticket"]["message"])

    stage("CLASSIFY_INTENT")
    for r in records:
        if jev_client:
            r["intent"], r["intent_confidence"] = jev.classify_intent(r["state"], jev_client)
            r["flags"], r["flag_probs"] = jev.detect_safety_flags(r["state"], jev_client)
        else:
            r["intent"], r["intent_confidence"], r["offline_match"] = jev.fallback_classify(r["candidates"])
            r["flags"], r["flag_probs"] = jev.fallback_safety_flags(r["intent"]), {}
            # Only confident matches are answered offline; the rest wait for the online pipeline.
            r["queued"] = r["intent"] == "other" or r["intent_confidence"] < offline_threshold
        if r["intent"] == "trading_advice_request" and "advice_request" not in r["flags"]:
            r["flags"].insert(0, "advice_request")

    stage("RETRIEVE_CONTEXT")
    for r in records:
        if jev_client:
            # Jev judges each candidate's relevance; this replaces a separate reranker.
            r["selected"], r["article_probs"] = jev.select_articles(r["state"], r["candidates"], jev_client)
        else:
            r["selected"], r["article_probs"] = r["offline_match"], {}
        r["uncertain"] = uncertain_judgments(r)
        if r.get("queued"):
            r["uncertain"].append(
                f"offline similarity {r['intent_confidence']:.2f} below {offline_threshold}; queued for online pipeline"
            )

    stage("GENERATE_REPLY")
    for r in records:
        if r.get("queued"):
            r["reply"] = holding_reply(r["intent"])
            continue
        articles = [kb_by_id[aid] for aid, _ in r["selected"]]
        r["reply"] = generate_reply(r["ticket"]["message"], r["intent"], articles, llm)

    stage("VALIDATE_OUTPUT")
    results = []
    for r in records:
        articles = [kb_by_id[aid] for aid, _ in r["selected"]]
        r["reply"], r["checks"] = check_reply(r, articles, jev_client, llm is not None)
        result, r["reasons"] = build_result(
            r["ticket"]["id"],
            r["intent"],
            r["intent_confidence"],
            r["selected"],
            r["reply"],
            r["flags"],
            r["uncertain"],
        )
        problems = validate_result(result, set(kb_by_id))
        if problems:
            raise ValueError(f"Result for {result.ticket_id} failed validation: {problems}")
        r["result"] = result
        results.append(result)
    return results, records


JUDGMENT_CONFIDENCE_THRESHOLD = 0.5


def uncertain_judgments(record: dict) -> list[str]:
    """Judgments too uncertain to act on: intent confidence, or Noul certainty (see jev.certainty),
    below JUDGMENT_CONFIDENCE_THRESHOLD. Any of these escalates the ticket."""
    t = JUDGMENT_CONFIDENCE_THRESHOLD
    reasons = []
    if record["intent_confidence"] < t:
        reasons.append(f"intent confidence {record['intent_confidence']:.2f}")
    for aid, p in record["article_probs"].items():
        if jev.certainty(p) < t:
            reasons.append(f"article {aid} relevance uncertain (p={p:.2f})")
    for flag, p in record["flag_probs"].items():
        if jev.certainty(p) < t:
            reasons.append(f"safety question {flag} uncertain (p={p:.2f})")
    return reasons


def is_advice_request(record: dict) -> bool:
    return record["intent"] == "trading_advice_request" or "advice_request" in record["flags"]


def check_reply(record: dict, articles: list[dict], jev_client, used_llm: bool) -> tuple[ReplyDraft, list[str]]:
    """Verify the drafted reply; fix or downgrade it when checks fail. Returns (reply, notes)."""
    reply, notes = record["reply"], []
    if record.get("queued"):
        notes.append("holding reply while queued")
        return reply, notes
    if not used_llm:
        notes.append("template reply (no LLM key)")
        return reply, notes

    if jev_client:
        verdict = jev.verify_reply(reply.reply_draft, articles, jev_client)
        notes.append(f"Jev: supported by articles {verdict['supported']:.2f}")
        if is_advice_request(record):
            notes.append(f"Jev: refuses advice {verdict['refuses_advice']:.2f}")
            if verdict["refuses_advice"] < 0.5:
                notes.append("reply did not clearly refuse advice; replaced with standard refusal")
                return fallback_reply("trading_advice_request", articles), notes
        if reply.grounded and verdict["supported"] < 0.5:
            notes.append("reply has claims not supported by the articles; marked ungrounded")
            reply = reply.model_copy(update={"grounded": False})
    elif is_advice_request(record):
        notes.append("advice request without Jev verification; using standard refusal")
        return fallback_reply("trading_advice_request", articles), notes
    return reply, notes


def validate_result(result: TicketResult, known_ids: set[str]) -> list[str]:
    """Checks beyond the Pydantic schema. Returns a list of problems (empty if valid)."""
    problems = []
    unknown = [aid for aid in result.retrieved_articles if aid not in known_ids]
    if unknown:
        problems.append(f"unknown article ids {unknown}")
    if result.intent == "trading_advice_request" and "advice_request" not in result.safety_flags:
        problems.append("trading advice request not flagged")
    if not result.grounded and not (result.needs_human_escalation or "insufficient_grounding" in result.safety_flags):
        problems.append("ungrounded reply without flag or escalation")
    return problems


# --- WRITE_RESULTS ------------------------------------------------------------------------


def write_results(results: list[TicketResult], path: Path) -> None:
    path.write_text(
        json.dumps([r.model_dump() for r in results], indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def write_debug_report(records: list[dict], kb: list[dict], modes: dict[str, str], path: Path) -> None:
    titles = {a["article_id"]: a["title"] for a in kb}
    lines = ["# Debug report", "", f"- Judgments: {modes['judgments']}", f"- Replies: {modes['replies']}", ""]
    for r in records:
        res: TicketResult = r["result"]
        escalated = (
            "yes, " + "; ".join(r["reasons"])
            if res.needs_human_escalation
            else "no, grounded with sufficient confidence and no policy flags"
        )
        lines += [
            f"## {res.ticket_id}",
            "",
            f"> {r['ticket']['message']}",
            "",
            f"- **Intent:** `{res.intent}` (classifier confidence {r['intent_confidence']:.2f})",
            "- **Search candidates:** "
            + ", ".join(
                f"{c['article_id']} (vector {c['vector_score']}, keyword {c['keyword_score']})" for c in r["candidates"]
            ),
            "- **Retrieved:** "
            + (", ".join(f"{aid} {titles[aid]} (relevance {s:.2f})" for aid, s in r["selected"]) or "none"),
            f"- **Confidence:** {res.confidence}  **Grounded:** {res.grounded}",
            f"- **Safety flags:** {', '.join(res.safety_flags) or 'none'}",
            f"- **Reply checks:** {'; '.join(r['checks']) or 'none'}",
            f"- **Queued for online pipeline:** {'yes' if r.get('queued') else 'no'}",
            f"- **Escalated:** {escalated}",
            "",
        ]
    path.write_text("\n".join(lines), encoding="utf-8")


# --- Entry point --------------------------------------------------------------------------


def run(
    tickets_path: Path, kb_path: Path, index_path: Path, offline: bool, offline_threshold: float = OFFLINE_THRESHOLD
) -> tuple[list[TicketResult], list[dict], list[dict], dict[str, str]]:
    """Load data and run the pipeline. Returns (results, debug records, kb articles, modes).

    Runs offline automatically when the API keys are missing from the environment / .env.
    """
    load_dotenv()
    stage("LOAD_DATA")
    tickets, kb = load_tickets(tickets_path), load_articles(kb_path)

    use_jev = not offline and bool(os.environ.get("TYPESAFE_API_KEY"))
    llm = None if offline else groq_client()
    modes = {
        "judgments": JEV_MODE
        if use_jev
        else f"offline (cosine similarity >= {offline_threshold} answered, the rest queued)",
        "replies": "Groq" if llm else "offline template",
    }
    print(f"judgments: {modes['judgments']} | replies: {modes['replies']}", file=sys.stderr)

    with TypeSafeClient() if use_jev else nullcontext() as jev_client:
        results, records = run_pipeline(tickets, kb, index_path, jev_client, llm, offline_threshold)
    return results, records, kb, modes


def update_queue(path: Path, records: list[dict], online: bool) -> str:
    """Offline: write queued tickets to `path`. Online: remove the tickets just processed from it."""
    if not online:
        queue = [
            {**r["ticket"], "similarity": r["intent_confidence"], "best_guess_intent": r["intent"]}
            for r in records
            if r.get("queued")
        ]
        path.write_text(json.dumps(queue, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        return f"{len(queue)} ticket(s) queued for the online pipeline in {path}"
    if not path.exists():
        return "queue empty"
    processed = {r["ticket"]["id"] for r in records}
    queue = json.loads(path.read_text(encoding="utf-8"))
    remaining = [t for t in queue if t["id"] not in processed]
    if remaining:
        path.write_text(json.dumps(remaining, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    else:
        path.unlink()
    return f"processed {len(queue) - len(remaining)} queued ticket(s), {len(remaining)} still queued"


def main() -> None:
    parser = argparse.ArgumentParser(description="Process support tickets into grounded reply drafts.")
    parser.add_argument("--tickets", type=Path, default=Path("tickets.json"))
    parser.add_argument("--kb", type=Path, default=Path("data/kb_articles.json"))
    parser.add_argument("--index", type=Path, default=Path("data/kb_index.json"))
    parser.add_argument("--out", type=Path, default=Path("results.json"))
    parser.add_argument("--debug", type=Path, default=Path("debug_report.md"))
    parser.add_argument("--queue", type=Path, default=Path("queue.json"))
    parser.add_argument(
        "--offline", action="store_true", help="force offline mode (automatic when API keys are missing)"
    )
    parser.add_argument(
        "--offline-threshold",
        type=float,
        default=OFFLINE_THRESHOLD,
        help="minimum cosine similarity to answer a ticket offline; others are queued",
    )
    parser.add_argument("--no-cache", action="store_true", help="call the models even for cached requests")
    args = parser.parse_args()

    cache.enabled = not args.no_cache
    results, records, kb, modes = run(args.tickets, args.kb, args.index, args.offline, args.offline_threshold)

    stage("WRITE_RESULTS")
    write_results(results, args.out)
    write_debug_report(records, kb, modes, args.debug)
    queue_status = update_queue(args.queue, records, online=modes["judgments"] == JEV_MODE)
    escalated = sum(r.needs_human_escalation for r in results)
    print(
        f"Wrote {len(results)} results to {args.out} ({escalated} escalated), debug report to {args.debug}; "
        f"{queue_status}",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
