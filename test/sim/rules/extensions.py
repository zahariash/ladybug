"""Rules for the Doc table and its full-text (fts) and vector (HNSW) indexes, run when
--sim-extensions loads the extensions. The checks they call are in search.py."""

from __future__ import annotations

import os

import hypothesis.strategies as st
from hypothesis.stateful import precondition, rule
from model import Doc
from search import create_statement, drop_statement, unstemmed_match
from session import DUPLICATE_KEY, Session, gated, with_extension
from strategies import ALL_WORDS, WORDS, doc_words, embeddings, small_ids

STEMMERS = ["none", "english"]
METRICS = ["cosine", "l2", "l2sq", "dotproduct"]
INDEXES = {"doc_fts": "fts", "doc_vec": "vector"}  # index -> the extension that provides it
MAX_DOCS = 30000
gated(*METRICS, "vector_recall")


class ExtensionRules(Session):
    def available_indexes(self) -> list[str]:
        return [index for index, extension in INDEXES.items() if extension in self.options.loaded]

    def index_options(self, index: str):
        if index == "doc_fts":
            return st.sampled_from(STEMMERS)
        return st.sampled_from([m for m in METRICS if m not in self.options.skipped])

    @with_extension("docs")
    @rule(id=small_ids, words=doc_words, emb=embeddings)
    def insert_doc(self, id, words, emb):
        doc = Doc(words, emb)
        params = dict(id=id, text=doc.text, emb=emb and list(emb))
        query = "CREATE (:Doc {id: $id, text: $text, emb: $emb})"
        if id in self.model.docs:
            self.fails(query, params, DUPLICATE_KEY)
            return
        self.ok(query, params)
        self.model.docs[id] = doc

    @with_extension("docs")
    @rule(id=small_ids, words=doc_words)
    def update_doc_text(self, id, words):
        text = Doc(words, None).text
        self.ok("MATCH (d:Doc {id: $id}) SET d.text = $text", dict(id=id, text=text))
        if id in self.model.docs:
            self.model.docs[id] = self.model.docs[id]._replace(words=words)

    @with_extension("docs")
    @rule(id=small_ids, emb=embeddings)
    def update_doc_emb(self, id, emb):
        self.ok("MATCH (d:Doc {id: $id}) SET d.emb = $emb", dict(id=id, emb=emb and list(emb)))
        if id in self.model.docs:
            self.model.docs[id] = self.model.docs[id]._replace(emb=emb)

    @with_extension("docs")
    @rule(id=small_ids)
    def delete_doc(self, id):
        self.ok("MATCH (d:Doc {id: $id}) DELETE d", dict(id=id))
        self.model.docs.pop(id, None)

    @with_extension("bulk_docs")
    @precondition(lambda self: len(self.model.docs) <= MAX_DOCS - 6000)
    @rule(count=st.integers(500, 6000), rng=st.randoms(use_true_random=False))
    def bulk_docs(self, count, rng):
        """COPYs docs with random words and embeddings, some of them NULL, into Doc and every
        index on it."""
        ids = self.model.take_bulk_ids(count)
        csv = os.path.join(self.files, f"docs{ids.start}.csv")
        added = {}
        with open(csv, "w") as f:
            for id in ids:
                words = None if rng.random() < 0.1 else rng.choices(ALL_WORDS, k=rng.randint(1, 5))
                emb = None
                if rng.random() >= 0.1:
                    emb = tuple(rng.randint(-3, 3) for _ in range(3))
                    emb = emb if any(emb) else (1, 0, 0)
                added[id] = Doc(words, emb)
                text = f'"{added[id].text}"' if words else ""
                vector = f'"[{",".join(map(str, emb))}]"' if emb else ""
                f.write(f"{id},{text},{vector}\n")
        self.ok(f'COPY Doc FROM "{csv}"')
        self.model.docs.update(added)

    @with_extension("doc_indexes")
    @rule(data=st.data())
    def create_doc_index(self, data):
        index = data.draw(st.sampled_from(self.available_indexes()))
        option = data.draw(self.index_options(index))
        if index in self.model.doc_indexes:
            self.fails(create_statement(index, option), None, "already exists")
            return
        self.ok(create_statement(index, option))
        self.model.doc_indexes[index] = option

    @with_extension("drop_doc_index")
    @rule(data=st.data())
    def drop_doc_index(self, data):
        index = data.draw(st.sampled_from(self.available_indexes()))
        if index not in self.model.doc_indexes:
            self.fails(drop_statement(index), None, "doesn't have an index")
            return
        self.ok(drop_statement(index))
        del self.model.doc_indexes[index]

    @with_extension("crash_index")
    @precondition(lambda self: len(self.model.docs) >= 1000)
    @rule(delay=st.floats(0.01, 0.3), data=st.data())
    def crash_during_index_build(self, delay, data):
        """Kills the engine while it builds an index; afterwards the index exists and is
        correct, or does not exist."""
        missing = [i for i in self.available_indexes() if i not in self.model.doc_indexes]
        if not missing:
            return
        index = data.draw(st.sampled_from(missing))
        option = data.draw(self.index_options(index))
        self.crash_during_statement(create_statement(index, option), delay)
        indexes = self.rows("CALL show_indexes() RETURN table_name, index_name")
        if ("Doc", index) in indexes:
            self.model.doc_indexes[index] = option

    @with_extension("docs")
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

    @with_extension("search_race", "fts")
    @precondition(lambda self: self.model.doc_indexes.get("doc_fts") == "none")
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
        docs = dict(self.model.docs)

        def state() -> tuple:
            matching = [id for id, doc in docs.items() if unstemmed_match([word], doc.words)]
            return (tuple(sorted((id,) for id in matching)),)

        states, writes = [state()], []
        for op, id, words in ops:
            text = Doc(words, None).text
            if op == "delete":
                writes.append(("MATCH (d:Doc {id: $id}) DELETE d", dict(id=id)))
                docs.pop(id, None)
            elif id in docs:
                writes.append(("MATCH (d:Doc {id: $id}) SET d.text = $t", dict(id=id, t=text)))
                docs[id] = docs[id]._replace(words=words or None)
            else:
                writes.append(("CREATE (:Doc {id: $id, text: $t})", dict(id=id, t=text)))
                docs[id] = Doc(words or None, None)
            states.append(state())
        query = f"CALL QUERY_FTS_INDEX('Doc', 'doc_fts', '{word}') RETURN node.id"
        self.check_race(writes, [query], states, num_readers)
        self.model.docs = docs
