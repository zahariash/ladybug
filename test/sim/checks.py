"""Checks of the data, the catalog and a set of queries against the model."""

from __future__ import annotations

from collections import Counter

import hypothesis.strategies as st
from hypothesis.stateful import invariant, precondition, rule
from session import Session, diff
from strategies import clusters, small_ids

USER_TABLES = {"Person", "Knows", "L"}


def is_internal_table(name: str) -> bool:
    """Partitions of L, and the tables FTS and HNSW indexes keep."""
    return name.startswith(("L_p", "_")) or name[0].isdigit()


class Checks(Session):
    @invariant()
    def data_matches_model(self):
        if not self.engine:
            return
        expected = self.model.person_rows()
        actual = self.person_rows()
        assert actual == expected, diff(actual, expected)
        query = "MATCH (a:Person)-[k:Knows]->(b:Person) RETURN a.id, b.id, k.since"
        actual = Counter(self.rows(query))
        assert actual == self.model.knows, diff(actual, self.model.knows)
        expected = Counter(self.model.partitioned.items())
        actual = Counter(self.rows("MATCH (n:L) RETURN n.id, n.cluster"))
        assert actual == expected, diff(actual, expected)
        if self.options.extensions:
            docs = self.model.docs.items()
            expected = Counter((id, " ".join(words)) for id, (words, _) in docs)
            actual = Counter(self.rows("MATCH (d:Doc) RETURN d.id, d.text"))
            assert actual == expected, diff(actual, expected)

    @invariant()
    def catalog_matches_model(self):
        if not self.engine:
            return
        names = self.rows("CALL show_tables() RETURN name")
        tables = {name for (name,) in names if not is_internal_table(name)}
        expected = USER_TABLES | {f"T{i}" for i in self.model.tables}
        if self.options.extensions:
            expected.add("Doc")
        assert tables == expected, tables
        for i, values in self.model.tables.items():
            total = sum(values) if values else None
            self.check(f"MATCH (t:T{i}) RETURN count(*), sum(t.v)", {}, [(len(values), total)])
        self.check_macros()
        indexes = self.rows("CALL show_indexes() RETURN table_name, index_name")
        person = {name for table, name in indexes if table == "Person"}
        assert person == ({self.model.pk_index} - {None}), person
        if self.options.extensions:
            docs = {name for table, name in indexes if table == "Doc"} - {"_PK"}
            assert docs == self.model.doc_indexes, docs

    def check_macros(self) -> None:
        macros = {name.lower() for (name,) in self.rows("CALL show_macros() RETURN name")}
        # The FTS index owns an internal `<table id>_doc_fts_tokenize` macro, which
        # DROP_FTS_INDEX leaves behind.
        fts = {macro for macro in macros if macro.endswith("_doc_fts_tokenize")}
        assert ("doc_fts" in self.model.doc_indexes) <= len(fts) <= 1, fts
        assert macros - fts == {f"m{i}" for i in self.model.macros}, macros
        for i, k in self.model.macros.items():
            self.check(f"RETURN m{i}(10)", {}, [(10 + k,)])

    @precondition(lambda self: self.engine is not None)
    @rule(x=st.integers(-200, 200), id=small_ids, limit=st.integers(1, 20), cluster=clusters)
    def queries_match_model(self, x, id, limit, cluster):
        persons, knows = self.model.persons, self.model.knows
        ages = [p["age"] for p in persons.values() if p["age"] is not None]
        self.check(
            "MATCH (p:Person) WHERE p.age > $x RETURN count(*)",
            dict(x=x),
            [(sum(a > x for a in ages),)],
        )
        self.check(
            "MATCH (p:Person) WHERE p.age IS NOT NULL RETURN min(p.age), max(p.age), count(*)",
            {},
            [(min(ages), max(ages), len(ages)) if ages else (None, None, 0)],
        )
        person = persons.get(id)
        self.check(
            "MATCH (p:Person {id: $id}) RETURN p.name, p.age",
            dict(id=id),
            [(person["name"], person["age"])] if person else [],
        )
        self.check(
            "MATCH (p:Person) WHERE p.id IN [$id, $id + 1] RETURN p.id",
            dict(id=id),
            [(i,) for i in (id, id + 1) if i in persons],
        )
        top = sorted((p["age"], k) for k, p in persons.items() if p["age"] is not None)[:limit]
        self.check(
            "MATCH (p:Person) WHERE p.age IS NOT NULL "
            "RETURN p.age, p.id ORDER BY p.age, p.id LIMIT $l",
            dict(l=limit),
            top,
            ordered=True,
        )
        degrees = Counter()
        for (src, _, _), n in knows.items():
            degrees[src] += n
        self.check(
            "MATCH (a:Person)-[:Knows]->(:Person) RETURN a.id, count(*)",
            {},
            list(degrees.items()),
        )
        two_hop = sum(
            n1 * n2 for (_, b, _), n1 in knows.items() for (c, _, _), n2 in knows.items() if b == c
        )
        self.check(
            "MATCH (:Person)-[:Knows]->(:Person)-[:Knows]->(:Person) RETURN count(*)",
            {},
            [(two_hop,)],
        )
        self.check(
            "MATCH (n:L) WHERE n.cluster = $c RETURN count(*)",
            dict(c=cluster),
            [(sum(c == cluster for c in self.model.partitioned.values()),)],
        )
        for column, default in self.model.columns.items():
            if column.startswith("c"):
                self.check(
                    f"MATCH (p:Person) WHERE p.{column} = $d RETURN count(*)",
                    dict(d=default),
                    [(sum(p[column] == default for p in persons.values()),)],
                )
