"""Hybrid KB retrieval: vector search over the embedding index plus BM25 keyword search.

Both are deterministic. Candidates from each are merged and then judged by Jev
(or the offline fallback) to pick the 1-3 articles used for the reply.
"""

import json
import re
from pathlib import Path

import numpy as np
from fastembed import TextEmbedding
from rank_bm25 import BM25Okapi

from .embed_kb import DEFAULT_MODEL, article_text, build_index, embed_articles

STOPWORDS = set(
    "a an and are as at be but by can do does for from has have how i if in is it its may me my "
    "no not of on or our so than that the their them then there they this to too up was we what "
    "when which why will with you your".split()
)


def tokenize(text: str) -> list[str]:
    """Lowercase word tokens without stopwords, with a light suffix strip (deposited/deposits -> deposit)."""
    tokens = [t for t in re.findall(r"[a-z]+", text.lower()) if t not in STOPWORDS]
    return [re.sub(r"(ing|ed|es|s)$", "", t) if len(t) > 4 else t for t in tokens]


def load_or_build_index(articles: list[dict], path: Path, model_name: str = DEFAULT_MODEL) -> dict:
    """Load the embedding index, rebuilding it if missing or out of date with the KB articles."""
    if path.exists():
        index = json.loads(path.read_text(encoding="utf-8"))
        indexed = [(a["article_id"], a["title"], a["body"]) for a in index["articles"]]
        if index["model"] == model_name and indexed == [(a["article_id"], a["title"], a["body"]) for a in articles]:
            return index
    index = build_index(articles, embed_articles(articles, model_name), model_name)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(index, indent=2), encoding="utf-8")
    return index


class HybridRetriever:
    def __init__(self, index: dict):
        self.articles = [{k: a[k] for k in ("article_id", "title", "body")} for a in index["articles"]]
        self.matrix = np.array([a["embedding"] for a in index["articles"]])
        self.model = TextEmbedding(index["model"])
        self.bm25 = BM25Okapi([tokenize(article_text(a)) for a in self.articles])

    def vector_search(self, query: str, k: int) -> list[tuple[int, float]]:
        q = np.array(next(iter(self.model.query_embed([query]))))
        scores = self.matrix @ (q / np.linalg.norm(q))
        return top_k(scores, k)

    def keyword_search(self, query: str, k: int) -> list[tuple[int, float]]:
        scores = self.bm25.get_scores(tokenize(query))
        return [(i, s) for i, s in top_k(scores, k) if s > 0]

    def retrieve(self, query: str, k: int = 3) -> list[dict]:
        """Top-k from each search, merged by reciprocal rank fusion.

        Returns candidate articles with their `vector_score`, `keyword_score` and `rrf` score,
        best first.
        """
        vec, kw = self.vector_search(query, k), self.keyword_search(query, k)
        vec_scores = dict(vec)
        kw_scores = dict(kw)
        rrf: dict[int, float] = {}
        for ranking in (vec, kw):
            for rank, (i, _) in enumerate(ranking):
                rrf[i] = rrf.get(i, 0.0) + 1 / (60 + rank + 1)
        order = sorted(rrf, key=lambda i: (-rrf[i], self.articles[i]["article_id"]))
        return [
            {
                **self.articles[i],
                "vector_score": round(float(vec_scores.get(i, 0.0)), 4),
                "keyword_score": round(float(kw_scores.get(i, 0.0)), 4),
                "rrf": round(rrf[i], 6),
            }
            for i in order
        ]


def top_k(scores: np.ndarray, k: int) -> list[tuple[int, float]]:
    # Stable sort so ties break by article order: deterministic results.
    order = np.argsort(-scores, kind="stable")[:k]
    return [(int(i), float(scores[i])) for i in order]
