"""Embed KB articles into a vector space and store them in a JSON index.

Usage:
    python -m pipeline.embed_kb [--input data/kb_articles.json] [--output data/kb_index.json]

main.py also rebuilds the index automatically when it is missing or stale.

Retrieval later should embed queries with the same model via
`TextEmbedding(model).query_embed(...)` and score with a dot product
(vectors are L2-normalised, so dot product == cosine similarity).
"""

import argparse
import json
from pathlib import Path

import numpy as np
from fastembed import TextEmbedding

DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"


def load_articles(path: Path) -> list[dict]:
    articles = json.loads(path.read_text(encoding="utf-8"))
    for article in articles:
        missing = {"article_id", "title", "body"} - article.keys()
        if missing:
            raise ValueError(f"Article {article!r} is missing fields: {sorted(missing)}")
    return articles


def article_text(article: dict) -> str:
    # Title carries strong topical signal, so embed it alongside the body.
    return f"{article['title']}\n{article['body']}"


def embed_articles(articles: list[dict], model_name: str) -> np.ndarray:
    model = TextEmbedding(model_name)
    vectors = np.array(list(model.passage_embed([article_text(a) for a in articles])))
    return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)


def build_index(articles: list[dict], vectors: np.ndarray, model_name: str) -> dict:
    return {
        "model": model_name,
        "dim": int(vectors.shape[1]),
        "normalized": True,
        "articles": [
            {
                "article_id": a["article_id"],
                "title": a["title"],
                "body": a["body"],
                "embedding": [round(float(x), 6) for x in vec],
            }
            for a, vec in zip(articles, vectors, strict=True)
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--input", type=Path, default=Path("data/kb_articles.json"))
    parser.add_argument("--output", type=Path, default=Path("data/kb_index.json"))
    parser.add_argument("--model", default=DEFAULT_MODEL)
    args = parser.parse_args()

    articles = load_articles(args.input)
    vectors = embed_articles(articles, args.model)
    index = build_index(articles, vectors, args.model)
    args.output.write_text(json.dumps(index, indent=2), encoding="utf-8")
    print(f"Embedded {len(articles)} articles ({index['dim']}-d, {args.model}) -> {args.output}")


if __name__ == "__main__":
    main()
