"""On-disk cache of model responses, keyed by the exact request.

Hosted models aren't bit-for-bit deterministic (Groq varies even at temperature 0),
so reruns on the same inputs reuse the stored response. Any change to the request
(ticket text, prompt, questions, model) produces a new key and a fresh call.
"""

import hashlib
import json
from collections.abc import Callable
from pathlib import Path

CACHE_DIR = Path(".cache")
enabled = True


def cached(namespace: str, request: dict, call: Callable[[], dict | list | str]) -> dict | list | str:
    """Return the stored response for `request`, or run `call()` and store its JSON-able result."""
    if not enabled:
        return call()
    key = hashlib.sha256(json.dumps(request, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    path = CACHE_DIR / namespace / f"{key}.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    result = call()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result
