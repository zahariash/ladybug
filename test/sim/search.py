"""Checks of full-text and vector search results.

Full-text results are checked against the model when the index has no stemmer, and against an
index built from scratch over the same docs when it stems, which catches maintenance bugs
without modelling the stemmer. Vector results must be the exact k nearest up to EXACT_LIMIT
embedded docs, and reach RECALL of them beyond that.
"""

from __future__ import annotations

import math

from model import canonical
from session import Session
from strategies import STOPWORDS

EXACT_LIMIT = 200
RECALL = 0.9
# Beyond this many docs, building a fresh full-text index for every check is too slow.
FRESH_INDEX_LIMIT = 3000


def create_statement(index: str, option: str, name: str | None = None) -> str:
    name = name or index
    if index == "doc_fts":
        return f"CALL CREATE_FTS_INDEX('Doc', '{name}', ['text'], stemmer := '{option}')"
    return f"CALL CREATE_VECTOR_INDEX('Doc', '{name}', 'emb', metric := '{option}')"


def drop_statement(index: str, name: str | None = None) -> str:
    function = "DROP_FTS_INDEX" if index == "doc_fts" else "DROP_VECTOR_INDEX"
    return f"CALL {function}('Doc', '{name or index}')"


def fts_query(index: str) -> str:
    return f"CALL QUERY_FTS_INDEX('Doc', '{index}', $q) RETURN node.id"


def rank(metric: str, a, b) -> float:
    """How far b is from a under the metric; smaller is nearer."""
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    if metric == "dotproduct":
        return -dot
    if metric == "cosine":
        return 1 - dot / (math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b)))
    l2sq = sum((x - y) ** 2 for x, y in zip(a, b, strict=True))
    return math.sqrt(l2sq) if metric == "l2" else l2sq


def unstemmed_match(query: list[str], words: list[str] | None) -> bool:
    """Whether an index without a stemmer returns a doc with `words` for `query`."""
    tokens = {w.lower() for w in words or []} - set(STOPWORDS)
    return any(w.lower() in tokens for w in query)


class SearchChecks(Session):
    def check_fts(self, query: list[str]) -> None:
        params = dict(q=" ".join(query))
        stemmer = self.model.doc_indexes["doc_fts"]
        if stemmer == "none":
            docs = self.model.docs.items()
            expected = [(id,) for id, doc in docs if unstemmed_match(query, doc.words)]
            self.check(fts_query("doc_fts"), params, expected)
            return
        if len(self.model.docs) > FRESH_INDEX_LIMIT:
            return
        self.ok(create_statement("doc_fts", stemmer, "doc_fts_fresh"))
        try:
            # A freshly prepared statement: the driver's cached one would still refer to the
            # index of the same name the previous check dropped.
            fresh = self.engine.execute_prepared(fts_query("doc_fts_fresh"), params)[0]
            expected = [canonical(row) for row in fresh]
        finally:
            self.ok(drop_statement("doc_fts", "doc_fts_fresh"))
        self.check(fts_query("doc_fts"), params, expected)

    def check_vector(self, emb, k: int) -> None:
        metric = self.model.doc_indexes["doc_vec"]
        embedded = {id: doc.emb for id, doc in self.model.docs.items() if doc.emb is not None}
        best = sorted(rank(metric, emb, e) for e in embedded.values())[:k]
        exact = len(embedded) <= EXACT_LIMIT
        if not exact and metric != "cosine" and "vector_recall" in self.options.skipped:
            return
        query = "CALL QUERY_VECTOR_INDEX('Doc', 'doc_vec', $q, $k) RETURN node.id"
        for conn in (0, 0, 1):
            ids = [id for (id,) in self.rows(query, dict(q=list(emb), k=k), conn)]
            found = sorted(rank(metric, emb, embedded[id]) for id in ids)
            assert len(found) == len(best), (metric, emb, k, found, best)
            if exact:
                close = all(abs(a - b) < 1e-4 for a, b in zip(found, best, strict=True))
                assert close, (metric, emb, k, found, best)
            else:
                near = sum(f <= best[-1] + 1e-4 for f in found)
                assert near >= max(1, math.floor(RECALL * len(best))), (metric, emb, k, found, best)
