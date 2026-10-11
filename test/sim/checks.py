"""Checks of the data, the catalog and a set of queries against the model."""

from __future__ import annotations

from collections import Counter
from contextlib import contextmanager

import hypothesis.strategies as st
from hypothesis.stateful import invariant, precondition, rule
from model import Model
from session import Session, diff
from strategies import clusters, small_ids

USER_TABLES = {"Person", "Knows", "L"}


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
        if self.model.knows_w is not None:
            # Every edge, old or new, carries the default of the added property.
            total = sum(self.model.knows.values())
            query = "MATCH (:Person)-[k:Knows]->(:Person) RETURN k.w, count(*)"
            actual = self.rows(query)
            assert actual == ([(self.model.knows_w, total)] if total else []), actual
        expected = Counter(self.model.partitioned.items())
        actual = Counter(self.rows("MATCH (n:L) RETURN n.id, n.cluster"))
        assert actual == expected, diff(actual, expected)
        for i, values in self.model.tables.items():
            # SERIAL ids must be unique; values are compared as a multiset.
            rows = self.rows(f"MATCH (t:T{i}) RETURN t.id, t.v")
            assert len({id for id, _ in rows}) == len(rows), rows
            assert Counter(v for _, v in rows) == Counter(values), (i, rows, values)
        if self.options.loaded:
            expected = Counter(
                (id, doc.text, doc.emb and tuple(map(float, doc.emb)))
                for id, doc in self.model.docs.items()
            )
            actual = Counter(self.rows("MATCH (d:Doc) RETURN d.id, d.text, d.emb"))
            assert actual == expected, diff(actual, expected)

    @invariant()
    def catalog_matches_model(self):
        if not self.engine:
            return
        # show_tables leaves out the tables FTS and HNSW indexes keep, but lists partitions.
        names = {name for (name,) in self.rows("CALL show_tables() RETURN name")}
        tables = {name for name in names if not name.startswith("L_p")}
        expected = USER_TABLES | {f"T{i}" for i in self.model.tables}
        if self.options.loaded:
            expected.add("Doc")
        assert tables == expected, tables
        self.check_macros()
        indexes = self.rows("CALL show_indexes() RETURN table_name, index_name")
        person = {name for table, name in indexes if table == "Person"}
        assert person == ({self.model.pk_index} - {None}), person
        if self.options.loaded:
            docs = {name for table, name in indexes if table == "Doc"} - {"_PK"}
            assert docs == set(self.model.doc_indexes), docs

    def check_macros(self) -> None:
        macros = {name.lower() for (name,) in self.rows("CALL show_macros() RETURN name")}
        # Each FTS index owns an internal `<table id>_<index>_tokenize` macro, which
        # DROP_FTS_INDEX leaves behind.
        fts = {macro for macro in macros if macro.endswith("_tokenize")}
        if "doc_fts" in self.model.doc_indexes:
            assert any(macro.endswith("_doc_fts_tokenize") for macro in fts), fts
        assert macros - fts == {f"m{i}" for i in self.model.macros}, macros
        for i, k in self.model.macros.items():
            self.check(f"RETURN m{i}(10)", {}, [(10 + k,)])

    @precondition(lambda self: self.engine is not None)
    @rule(
        x=st.integers(-200, 200),
        id=small_ids,
        limit=st.integers(1, 20),
        cluster=clusters,
        optimizer=st.booleans(),
    )
    def queries_match_model(self, x, id, limit, cluster, optimizer):
        """The query battery on the committed state, with the optimizer on or off."""
        with self.optimizer(optimizer, 0, 1):
            self.check_view(self.model, 0, x, id, limit, cluster)

    @contextmanager
    def optimizer(self, enabled: bool, *conns: int):
        """Runs the block with the plan optimizer of `conns` on or off."""
        if not enabled:
            for conn in conns:
                self.ok("CALL enable_plan_optimizer=false", conn=conn)
        yield
        if not enabled:
            for conn in conns:
                self.ok("CALL enable_plan_optimizer=true", conn=conn)

    def check_view(
        self,
        view: Model,
        conn: int,
        x: int,
        id: int,
        limit: int,
        cluster: int,
        alone=False,
    ) -> None:
        """Checks what `conn` sees against `view`, through access paths that each decide
        visibility on their own: scans, primary-key lookups, count fast paths, pushed-down
        filters, both edge directions, variable-length paths and OPTIONAL MATCH."""

        def check(query, params, expected, ordered=False):
            self.check(query, params, expected, ordered, conn, alone)

        persons, knows = view.persons, view.knows
        actual = self.person_rows(view, conn)
        assert actual == view.person_rows(), diff(actual, view.person_rows())
        edges = Counter(
            self.rows(
                "MATCH (a:Person)-[k:Knows]->(b:Person) RETURN a.id, b.id, k.since",
                None,
                conn,
            )
        )
        assert edges == knows, diff(edges, knows)
        ages = [p["age"] for p in persons.values() if p["age"] is not None]
        check("MATCH (p:Person) RETURN count(*)", {}, [(len(persons),)])
        check(
            "MATCH (p:Person) WHERE p.age > $x RETURN count(*)",
            dict(x=x),
            [(sum(a > x for a in ages),)],
        )
        check(
            "MATCH (p:Person) WHERE p.age > $x AND p.age <= $x + 50 RETURN count(*)",
            dict(x=x),
            [(sum(x < a <= x + 50 for a in ages),)],
        )
        check(
            "MATCH (p:Person) WHERE p.age IS NOT NULL RETURN min(p.age), max(p.age), count(*)",
            {},
            [(min(ages), max(ages), len(ages)) if ages else (None, None, 0)],
        )
        person = persons.get(id)
        check(
            "MATCH (p:Person {id: $id}) RETURN p.name, p.age",
            dict(id=id),
            [(person["name"], person["age"])] if person else [],
        )
        check(
            "MATCH (p:Person) WHERE p.id IN [$id, $id + 1] RETURN p.id",
            dict(id=id),
            [(i,) for i in (id, id + 1) if i in persons],
        )
        top = sorted((p["age"], k) for k, p in persons.items() if p["age"] is not None)[:limit]
        check(
            "MATCH (p:Person) WHERE p.age IS NOT NULL "
            "RETURN p.age, p.id ORDER BY p.age, p.id LIMIT $l",
            dict(l=limit),
            top,
            ordered=True,
        )
        degrees = Counter()
        for (src, _, _), n in knows.items():
            degrees[src] += n
        check(
            "MATCH (a:Person)-[:Knows]->(:Person) RETURN a.id, count(*)",
            {},
            list(degrees.items()),
        )
        check(
            "MATCH (a:Person) OPTIONAL MATCH (a)-[:Knows]->(b:Person) RETURN a.id, count(b)",
            {},
            [(i, degrees[i]) for i in persons],
        )
        check(
            "MATCH (b:Person)<-[k:Knows]-(a:Person) RETURN a.id, b.id, k.since",
            {},
            list(knows.elements()),
        )
        txn = self.txns.get(conn)
        if not (txn and txn.kind == "read" and "snapshot_rel_count" in self.options.skipped):
            check("MATCH ()-[k:Knows]->() RETURN count(*)", {}, [(sum(knows.values()),)])
        two_hop = sum(
            n1 * n2 for (_, b, _), n1 in knows.items() for (c, _, _), n2 in knows.items() if b == c
        )
        check(
            "MATCH (:Person)-[:Knows]->(:Person)-[:Knows]->(:Person) RETURN count(*)",
            {},
            [(two_hop,)],
        )
        # Variable-length patterns repeat edges (walk semantics), as the two-hop count does.
        check(
            "MATCH (:Person)-[:Knows*1..2]->(:Person) RETURN count(*)",
            {},
            [(sum(knows.values()) + two_hop,)],
        )
        check(
            "MATCH (n:L) WHERE n.cluster = $c RETURN count(*)",
            dict(c=cluster),
            [(sum(c == cluster for c in view.partitioned.values()),)],
        )
        for column, default in view.columns.items():
            if column.startswith("c"):
                check(
                    f"MATCH (p:Person) WHERE p.{column} = $d RETURN count(*)",
                    dict(d=default),
                    [(sum(p[column] == default for p in persons.values()),)],
                )
