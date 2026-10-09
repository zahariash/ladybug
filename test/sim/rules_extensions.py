"""Rules for the Doc table and its full-text (fts) and vector (HNSW) indexes, run when
--sim-extensions loads the extensions."""

from __future__ import annotations

import math

import hypothesis.strategies as st
from hypothesis.stateful import rule
from session import Session, with_extensions
from strategies import WORDS, doc_words, embeddings, small_ids

INDEXES = {
    "doc_fts": (
        "CALL CREATE_FTS_INDEX('Doc', 'doc_fts', ['text'])",
        "CALL DROP_FTS_INDEX('Doc', 'doc_fts')",
    ),
    "doc_vec": (
        "CALL CREATE_VECTOR_INDEX('Doc', 'doc_vec', 'emb')",
        "CALL DROP_VECTOR_INDEX('Doc', 'doc_vec')",
    ),
}


def cosine_distance(a, b) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    return 1 - dot / (math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b)))


class ExtensionRules(Session):
    @with_extensions("docs")
    @rule(id=small_ids, words=doc_words, emb=embeddings)
    def insert_doc(self, id, words, emb):
        params = dict(id=id, text=" ".join(words), emb=list(emb))
        query = "CREATE (:Doc {id: $id, text: $text, emb: $emb})"
        if id in self.model.docs:
            self.fails(query, params)
            return
        self.ok(query, params)
        self.model.docs[id] = (words, emb)

    @with_extensions("docs")
    @rule(id=small_ids, words=doc_words)
    def update_doc_text(self, id, words):
        self.ok("MATCH (d:Doc {id: $id}) SET d.text = $text", dict(id=id, text=" ".join(words)))
        if id in self.model.docs:
            self.model.docs[id] = (words, self.model.docs[id][1])

    @with_extensions("docs")
    @rule(id=small_ids, emb=embeddings)
    def update_doc_emb(self, id, emb):
        self.ok("MATCH (d:Doc {id: $id}) SET d.emb = $emb", dict(id=id, emb=list(emb)))
        if id in self.model.docs:
            self.model.docs[id] = (self.model.docs[id][0], emb)

    @with_extensions("docs")
    @rule(id=small_ids)
    def delete_doc(self, id):
        self.ok("MATCH (d:Doc {id: $id}) DELETE d", dict(id=id))
        self.model.docs.pop(id, None)

    @with_extensions("doc_indexes")
    @rule(index=st.sampled_from(sorted(INDEXES)))
    def create_doc_index(self, index):
        if index in self.model.doc_indexes:
            self.fails(INDEXES[index][0])
            return
        self.ok(INDEXES[index][0])
        self.model.doc_indexes.add(index)

    @with_extensions("doc_indexes")
    @rule(index=st.sampled_from(sorted(INDEXES)))
    def drop_doc_index(self, index):
        if index not in self.model.doc_indexes:
            self.fails(INDEXES[index][1])
            return
        self.ok(INDEXES[index][1])
        self.model.doc_indexes.discard(index)

    @with_extensions("docs")
    @rule(
        word=st.sampled_from(WORDS),
        other=st.sampled_from(WORDS),
        emb=embeddings,
        k=st.integers(1, 6),
    )
    def search_matches_model(self, word, other, emb, k):
        """Full-text queries return the docs with any of the words; vector queries return the k
        smallest cosine distances."""
        if "doc_fts" in self.model.doc_indexes:
            for words in ([word], [word, other]):
                expected = [
                    (id,) for id, (text, _) in self.model.docs.items() if set(words) & set(text)
                ]
                query = "CALL QUERY_FTS_INDEX('Doc', 'doc_fts', $q) RETURN node.id"
                self.check(query, dict(q=" ".join(words)), expected)
        if "doc_vec" in self.model.doc_indexes:
            expected = sorted(cosine_distance(emb, e) for _, e in self.model.docs.values())[:k]
            query = "CALL QUERY_VECTOR_INDEX('Doc', 'doc_vec', $q, $k) RETURN distance"
            for conn in (0, 0, 1):
                actual = sorted(d for (d,) in self.rows(query, dict(q=list(emb), k=k), conn))
                assert len(actual) == len(expected) and all(
                    abs(a - b) < 1e-4 for a, b in zip(actual, expected, strict=True)
                ), (emb, k, actual, expected)
