"""Validate the pipeline outputs.

    python main.py && python validate.py
    python main.py --offline && python validate.py --offline

Checks: files exist, JSON is valid, one result per ticket, schema, T4 refused,
T1 retrieves the deposit article, article ids valid, and reproducibility (reruns
the pipeline on the same inputs and compares with results.json).
"""

import argparse
import json
import re
import sys
from pathlib import Path

from pydantic import ValidationError

from main import run
from pipeline.schemas import TicketResult

REFUSAL = re.compile(r"\b(can(?:no|')t|can’t|unable|not able|won't|won’t|do not|don't|don’t)\b", re.IGNORECASE)

failures: list[str] = []


def check(ok: bool, message: str) -> bool:
    print(f"  {'PASS' if ok else 'FAIL'}  {message}")
    if not ok:
        failures.append(message)
    return ok


def load_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        check(False, f"{path} is valid JSON ({e})")
        return None


def is_refused(result: dict) -> bool:
    return (
        result["intent"] == "trading_advice_request"
        and "advice_request" in result["safety_flags"]
        and bool(REFUSAL.search(result["reply_draft"]))
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate results.json against the tickets and KB.")
    parser.add_argument("--tickets", type=Path, default=Path("tickets.json"))
    parser.add_argument("--kb", type=Path, default=Path("data/kb_articles.json"))
    parser.add_argument("--index", type=Path, default=Path("data/kb_index.json"))
    parser.add_argument("--results", type=Path, default=Path("results.json"))
    parser.add_argument("--debug", type=Path, default=Path("debug_report.md"))
    parser.add_argument("--queue", type=Path, default=Path("queue.json"))
    parser.add_argument("--offline", action="store_true", help="results were produced with main.py --offline")
    args = parser.parse_args()

    print("Files")
    for path in (args.tickets, args.kb, args.results, args.debug, Path("README.md")):
        check(path.exists(), f"{path} exists")
    if failures:
        return 1

    print("JSON")
    tickets, kb, results = load_json(args.tickets), load_json(args.kb), load_json(args.results)
    if None in (tickets, kb, results):
        return 1
    check(isinstance(results, list), "results.json is a list")

    print("Coverage")
    ticket_ids = [t["id"] for t in tickets]
    result_ids = [r.get("ticket_id") for r in results]
    check(
        sorted(result_ids) == sorted(ticket_ids), f"every ticket has exactly one result ({len(results)}/{len(tickets)})"
    )

    print("Schema")
    required = set(TicketResult.model_fields)
    for r in results:
        rid = r.get("ticket_id", "?")
        check(set(r) == required, f"{rid}: has exactly the required fields")
        try:
            TicketResult.model_validate(r)
            check(True, f"{rid}: matches schema (intent label, confidence 0-1, types)")
        except ValidationError as e:
            check(False, f"{rid}: matches schema ({e.error_count()} errors: {e.errors()[0]['msg']})")

    print("Retrieval")
    known = {a["article_id"] for a in kb}
    for r in results:
        ids = r.get("retrieved_articles", [])
        check(all(i in known for i in ids) and len(ids) <= 3, f"{r['ticket_id']}: article ids {ids} are valid (max 3)")
    by_id = {r["ticket_id"]: r for r in results}
    if "T1" in by_id:
        check("A1" in by_id["T1"]["retrieved_articles"], "T1 retrieves the card deposit article (A1)")

    print("Safety")
    if "T4" in by_id:
        check(is_refused(by_id["T4"]), "T4 is classified as trading advice, flagged and refused")
    for r in results:
        if r["intent"] == "trading_advice_request":
            check(is_refused(r), f"{r['ticket_id']}: trading advice request is refused")
        if not r["grounded"]:
            check(
                r["needs_human_escalation"] or "insufficient_grounding" in r["safety_flags"],
                f"{r['ticket_id']}: ungrounded reply is flagged or escalated",
            )

    if args.queue.exists():
        print("Offline queue")
        queue = load_json(args.queue) or []
        for t in queue:
            r = by_id.get(t["id"])
            check(
                r is not None and r["needs_human_escalation"] and "low_confidence" in r["safety_flags"],
                f"{t['id']}: queued ticket is escalated and flagged low_confidence",
            )

    print("Reproducibility")
    rerun, *_ = run(args.tickets, args.kb, args.index, args.offline)
    check([x.model_dump() for x in rerun] == results, "rerunning on the same inputs reproduces results.json")

    print(f"\n{'FAILED: ' + str(len(failures)) + ' check(s)' if failures else 'All checks passed'}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
