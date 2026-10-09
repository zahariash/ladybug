"""Rules for the Doc table and its full-text (fts) and vector (HNSW) indexes, run when
--sim-extensions loads the extensions.

Full-text results are checked against the model when the index has no stemmer, and against an
index built from scratch over the same docs when it stems, which catches maintenance bugs
without modelling the stemmer. Vector results must be the exact k nearest up to EXACT_LIMIT
embedded docs, and reach RECALL of them beyond that.
"""

from __future__ import annotations

import math
import os
import time

import hypothesis.strategies as st
from hypothesis import assume
from hypothesis.stateful import rule
from session import Session, with_extensions
from strategies import ALL_WORDS, STOPWORDS, WORDS, doc_words, embeddings, small_ids

STEMMERS = ["none", "english"]
METRICS = ["cosine", "l2", "l2sq", "dotproduct"]
EXACT_LIMIT = 200
# Beyond this many docs, building a fresh full-text index for every check is too slow.
FRESH_INDEX_LIMIT = 3000
RECALL = 0.9
MAX_DOCS = 30000


def create_statement(index: str, option: str, name: str | None = None) -> str:
    name = name or index
    if index == "doc_fts":
        return f"CALL CREATE_FTS_INDEX('Doc', '{name}', ['text'], stemmer := '{option}')"
    return f"CALL CREATE_VECTOR_INDEX('Doc', '{name}', 'emb', metric := '{option}')"


def drop_statement(index: str, name: str | None = None) -> str:
    function = "DROP_FTS_INDEX" if index == "doc_fts" else "DROP_VECTOR_INDEX"
    return f"CALL {function}('Doc', '{name or index}')"


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


def fts_query(index: str) -> str:
    return f"CALL QUERY_FTS_INDEX('Doc', '{index}', $q) RETURN node.id"


class ExtensionRules(Session):
    @with_extensions("docs")
    @rule(id=small_ids, words=doc_words, emb=embeddings)
    def insert_doc(self, id, words, emb):
        params = dict(id=id, text=words and " ".join(words), emb=emb and list(emb))
        query = "CREATE (:Doc {id: $id, text: $text, emb: $emb})"
        if id in self.model.docs:
            self.fails(query, params)
            return
        self.ok(query, params)
        self.model.docs[id] = (words, emb)

    @with_extensions("docs")
    @rule(id=small_ids, words=doc_words)
    def update_doc_text(self, id, words):
        text = words and " ".join(words)
        self.ok("MATCH (d:Doc {id: $id}) SET d.text = $text", dict(id=id, text=text))
        if id in self.model.docs:
            self.model.docs[id] = (words, self.model.docs[id][1])

    @with_extensions("docs")
    @rule(id=small_ids, emb=embeddings)
    def update_doc_emb(self, id, emb):
        self.ok("MATCH (d:Doc {id: $id}) SET d.emb = $emb", dict(id=id, emb=emb and list(emb)))
        if id in self.model.docs:
            self.model.docs[id] = (self.model.docs[id][0], emb)

    @with_extensions("docs")
    @rule(id=small_ids)
    def delete_doc(self, id):
        self.ok("MATCH (d:Doc {id: $id}) DELETE d", dict(id=id))
        self.model.docs.pop(id, None)

    @with_extensions("bulk_docs")
    @rule(count=st.integers(500, 6000), rng=st.randoms(use_true_random=False))
    def bulk_docs(self, count, rng):
        """COPYs docs with random words and embeddings, some of them NULL, into Doc and every
        index on it."""
        assume(len(self.model.docs) + count <= MAX_DOCS)
        ids = self.model.take_bulk_ids(count)
        csv = os.path.join(self.dir, f"docs{ids.start}.csv")
        added = {}
        with open(csv, "w") as f:
            for id in ids:
                words = None if rng.random() < 0.1 else rng.choices(ALL_WORDS, k=rng.randint(1, 5))
                emb = None
                if rng.random() >= 0.1:
                    emb = tuple(rng.randint(-3, 3) for _ in range(3))
                    emb = emb if any(emb) else (1, 0, 0)
                added[id] = (words, emb)
                text = f'"{" ".join(words)}"' if words else ""
                vector = f'"[{",".join(map(str, emb))}]"' if emb else ""
                f.write(f"{id},{text},{vector}\n")
        self.ok(f'COPY Doc FROM "{csv}"')
        self.model.docs.update(added)

    @with_extensions("doc_indexes")
    @rule(index=st.sampled_from(["doc_fts", "doc_vec"]), data=st.data())
    def create_doc_index(self, index, data):
        option = data.draw(self.index_options(index))
        if index in self.model.doc_indexes:
            self.fails(create_statement(index, option))
            return
        self.ok(create_statement(index, option))
        self.model.doc_indexes[index] = option

    @with_extensions("drop_doc_index")
    @rule(index=st.sampled_from(["doc_fts", "doc_vec"]))
    def drop_doc_index(self, index):
        if index not in self.model.doc_indexes:
            self.fails(drop_statement(index))
            return
        self.ok(drop_statement(index))
        del self.model.doc_indexes[index]

    def index_options(self, index: str):
        if index == "doc_fts":
            return st.sampled_from(STEMMERS)
        return st.sampled_from([m for m in METRICS if m not in self.options.skipped])

    @with_extensions("crash_index")
    @rule(index=st.sampled_from(["doc_fts", "doc_vec"]), delay=st.floats(0, 0.3), data=st.data())
    def crash_during_index_build(self, index, delay, data):
        """Kills the engine while it builds an index; afterwards the index exists and is
        correct, or does not exist."""
        assume(index not in self.model.doc_indexes and len(self.model.docs) >= 1000)
        option = data.draw(self.index_options(index))
        self.engine.start(create_statement(index, option))
        time.sleep(delay)
        self.engine.kill()
        self.reopen_engine()
        indexes = self.rows("CALL show_indexes() RETURN table_name, index_name")
        if ("Doc", index) in indexes:
            self.model.doc_indexes[index] = option

    @with_extensions("docs")
    @rule(
        word=st.sampled_from(ALL_WORDS),
        other=st.sampled_from(ALL_WORDS),
        emb=embeddings.filter(bool),
        k=st.integers(1, 10),
    )
    def search_matches_model(self, word, other, emb, k):
        if "doc_fts" in self.model.doc_indexes:
            for query in ([word], [word, other]):
                self.check_fts(query)
        if "doc_vec" in self.model.doc_indexes:
            self.check_vector(emb, k)

    def check_fts(self, query: list[str]) -> None:
        params = dict(q=" ".join(query))
        stemmer = self.model.doc_indexes["doc_fts"]
        if stemmer == "none":
            docs = self.model.docs.items()
            expected = [(id,) for id, (words, _) in docs if unstemmed_match(query, words)]
            self.check(fts_query("doc_fts"), params, expected)
            return
        if len(self.model.docs) > FRESH_INDEX_LIMIT:
            return
        self.ok(create_statement("doc_fts", stemmer, "doc_fts_fresh"))
        try:
            expected = self.rows(fts_query("doc_fts_fresh"), params)
        finally:
            self.ok(drop_statement("doc_fts", "doc_fts_fresh"))
        self.check(fts_query("doc_fts"), params, expected)

    def check_vector(self, emb, k: int) -> None:
        metric = self.model.doc_indexes["doc_vec"]
        embedded = {id: e for id, (_, e) in self.model.docs.items() if e is not None}
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
                assert near >= RECALL * len(best), (metric, emb, k, found, best)

    @with_extensions("search_race")
    @rule(
        ops=st.lists(
            st.tuples(
                st.sampled_from(["set", "delete"]), small_ids, st.lists(st.sampled_from(WORDS))
            ),
            min_size=1,
            max_size=30,
        ),
        word=st.sampled_from(WORDS),
        num_readers=st.integers(2, 4),
    )
    def search_race(self, ops, word, num_readers):
        """Text updates race full-text searches; every search must see a committed state."""
        assume(self.model.doc_indexes.get("doc_fts") == "none")
        docs = dict(self.model.docs)

        def state() -> tuple:
            matching = [id for id, (words, _) in docs.items() if unstemmed_match([word], words)]
            return (tuple(sorted((id,) for id in matching)),)

        states, writes = [state()], []
        for op, id, words in ops:
            text = " ".join(words) or None
            if op == "delete":
                writes.append(("MATCH (d:Doc {id: $id}) DELETE d", dict(id=id)))
                docs.pop(id, None)
            elif id in docs:
                writes.append(("MATCH (d:Doc {id: $id}) SET d.text = $t", dict(id=id, t=text)))
                docs[id] = (words or None, docs[id][1])
            else:
                writes.append(("CREATE (:Doc {id: $id, text: $t})", dict(id=id, t=text)))
                docs[id] = (words or None, None)
            states.append(state())
        query = f"CALL QUERY_FTS_INDEX('Doc', 'doc_fts', '{word}') RETURN node.id"
        self.check_race(writes, [query], states, num_readers)
        self.model.docs = docs
