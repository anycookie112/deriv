# Support ticket pipeline

Classifies support tickets, retrieves relevant help-center articles, and drafts a grounded reply
with confidence, escalation and safety flags. Each result is validated with Pydantic before it is written.

## Approach

1. **Index the documents.** The KB articles are embedded into a local vector index.
2. **Jev judges the ticket.** TypeSafe's Jev model handles every judgment the reply depends on:
   - **intent**, one of the six required labels;
   - **retrieved articles**: keyword (BM25) and vector search each return their top 3, giving up to 6 candidates.
     Jev then scores each candidate's relevance and the best 3 (at most) are kept, so Jev acts as the reranker;
   - **confidence**, from Jev's intent confidence and article relevance;
   - **safety flags**: advice request, account-specific request, policy-sensitive.
3. **The LLM writes the reply** (Groq), using only the articles Jev selected and the intent it classified.
4. **Validate the output.** Jev checks that the reply is supported by its articles and, for advice requests, that it
   refuses. Each result is then validated against the Pydantic schema.
5. **Write the results.** Only results with the correct structure are written to `results.json`.

**Escalation.** Low confidence from Jev escalates a ticket to a human: a weak intent classification, an uncertain
article relevance or no relevant article (low retrieval score), or an uncertain safety judgment.

**Offline.** Without API keys, a ticket is classified by cosine similarity to the KB articles. Only high-confidence
matches (similarity ≥ 0.8) are answered; the others are queued for the full pipeline once it runs online again.

Design notes: [docs/design-notes.md](docs/design-notes.md). Challenge brief: [docs/challenge.md](docs/challenge.md).

## Run

Requires Python 3.13 and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
uv run main.py          # writes results.json and debug_report.md
uv run validate.py      # checks the outputs
```

API keys are read from `.env` (both optional). Copy the example and fill in your keys:

```bash
cp .env.example .env
```

```
TYPESAFE_API_KEY=...   # Jev: intent, article relevance, safety flags, reply checks
GROQ_API_KEY=...       # reply drafting
```

### Offline mode

When the keys are missing from `.env` (or with `--offline`), the pipeline runs with no secrets:

```bash
uv run main.py --offline && uv run validate.py --offline
```

- Each ticket is classified by its cosine similarity to the KB articles: the closest article decides the intent.
- Only tickets with similarity ≥ 0.8 (`--offline-threshold`) are answered, with a reply built from that article.
- The rest get a holding reply ("a specialist will follow up"), are escalated with `low_confidence`, and are written to
  `queue.json`. Suspected trading advice requests are refused straight away.
- The next online run classifies every ticket with the full pipeline and removes them from `queue.json`.

The first run downloads the local embedding model (`BAAI/bge-small-en-v1.5`, ~70 MB); after that it works without
internet access.

## Pipeline

```
LOAD_DATA -> INDEX_KB -> CLASSIFY_INTENT -> RETRIEVE_CONTEXT -> GENERATE_REPLY -> VALIDATE_OUTPUT -> WRITE_RESULTS
```

| Stage | With keys | Offline fallback |
|---|---|---|
| INDEX_KB | Embeds `data/kb_articles.json` into `data/kb_index.json` (rebuilt when the KB changes); hybrid search = top-3 vector + top-3 BM25, merged by reciprocal rank fusion | same |
| CLASSIFY_INTENT | Jev Choice over the six intent labels; Jev Nouls for `advice_request`, `account_specific_request`, `policy_sensitive` | intent of the article with the highest cosine similarity (`other` below 0.6); flags by intent; queued below 0.8 |
| RETRIEVE_CONTEXT | Jev judges each search candidate's relevance (acts as the reranker); keeps up to 3 scoring ≥ 0.5 | that closest article |
| GENERATE_REPLY | Groq `openai/gpt-oss-120b`, strict JSON schema; prompt contains only the selected articles and forbids answering beyond them | template built from the article text; holding reply if queued |
| VALIDATE_OUTPUT | Jev checks the reply is supported by the articles and, for advice requests, that it refuses; then Pydantic + rule checks | Pydantic + rule checks; advice requests always get the standard refusal |
| WRITE_RESULTS | `results.json`, `debug_report.md` | same |

Retrieval only uses the ticket text and is fully separate from generation.

### Confidence and escalation

- `confidence` = the lower of the intent classifier's confidence and the top article's relevance (0 when nothing relevant was found).
- `insufficient_grounding` is added when no article is relevant, or the reply isn't supported by its articles.
- `low_confidence` is added when confidence < 0.6, or when any Jev judgment is too uncertain to act on:
  - intent confidence < 0.5;
  - an article relevance or safety question with certainty < 0.5. These are yes/no probabilities `p`, so
    certainty is `|2p − 1|`, i.e. `p` between 0.25 and 0.75;
  - offline: cosine similarity < 0.8 (the ticket is queued).
- `needs_human_escalation` is true if the reply model asks for review, or the ticket has `insufficient_grounding`,
  `low_confidence` or `policy_sensitive`.

`debug_report.md` lists, per ticket, the predicted intent, search candidates, retrieved article titles, reply checks and why it was or wasn't escalated.

### Reproducibility

Retrieval and the offline fallbacks are deterministic. Hosted model output is not (Groq varies even at temperature 0),
so Jev and Groq responses are cached in `.cache/`, keyed by the exact request. A rerun on the same inputs returns identical
results; any change to a ticket, prompt or question makes a fresh call. Use `--no-cache` to force fresh calls.

## Files

```
main.py                 pipeline stages and entry point
validate.py             output validation (files, JSON, coverage, schema, T4 refusal, T1 retrieval, article ids, reproducibility)
tickets.json            input tickets
data/
  kb_articles.json      help-center articles
  kb_index.json         embedding index (generated)
pipeline/
  schemas.py            Pydantic models for the reply draft and the result, plus result assembly
  retrieval.py          embedding index and hybrid vector + BM25 search
  jev.py                TypeSafe Jev judgments and their offline fallbacks
  reply.py              Groq reply generation and template fallback
  embed_kb.py           builds the embedding index (python -m pipeline.embed_kb; main.py does this automatically)
  cache.py              request-keyed model response cache
docs/
  challenge.md          challenge brief
  design-notes.md       design notes
results.json            output (generated)
debug_report.md         per-ticket explainability report (generated)
queue.json              tickets waiting for the online pipeline (generated offline, cleared online)
```

Lint and format with `uvx ruff check . && uvx ruff format .`.
