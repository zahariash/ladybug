"""Checks of the data, the catalog and a set of queries against the model."""

from __future__ import annotations

from collections import Counter
from contextlib import contextmanager

import hypothesis.strategies as st
from hypothesis.stateful import invariant, precondition, rule
from model import Model
from session import Session, diff, gated
from strategies import clusters, small_ids

USER_TABLES = {"Person", "Knows"}

gated(
    "in_lists", "in_list_nulls", "count_restricted", "txn_index_lookups", "pk_ranges", "lookup_join"
)


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
        if self.has_partitions():
            expected = Counter(self.model.partitioned)
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
        if self.has_partitions():
            expected.add("L")
        if self.options.loaded:
            expected.add("Doc")
        assert tables == expected, tables
        self.check_macros()
        indexes = self.rows("CALL show_indexes() RETURN table_name, index_name")
        person = {name for table, name in indexes if table == "Person"}
        assert person == ({self.model.pk_index, self.model.age_index} - {None}), person
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

    def check_index_paths(
        self, view: Model, conn: int, x: int, id: int, degrees: Counter, alone: bool
    ) -> None:
        """Queries that the IN, count, secondary-index and primary-key-range rewrites answer,
        each under a gate so that a build is only held to the paths it has fixed."""

        def check(query, params, expected):
            self.check(query, params, expected, False, conn, alone)

        skipped = self.options.skipped
        persons, knows = view.persons, view.knows
        ages = [p["age"] for p in persons.values() if p["age"] is not None]
        if "in_lists" not in skipped:
            listed = list(range(x, x + 19))
            # Before the list_contains fix a NULL element matched a NULL property.
            null = [] if "in_list_nulls" in skipped else [None]
            check(
                "MATCH (p:Person) WHERE p.age IN $ages RETURN count(*)",
                dict(ages=listed + null),
                [(sum(a in listed for a in ages),)],
            )
            literal = "[" + ", ".join(map(str, listed)) + (", NULL]" if null else "]")
            check(
                f"MATCH (p:Person) WHERE p.age IN {literal} RETURN count(*)",
                {},
                [(sum(a in listed for a in ages),)],
            )
            check(
                "MATCH (q:Person) WHERE q.age % 2 = 0 WITH collect(q.age) AS ages "
                "MATCH (p:Person) WHERE p.age IN ages RETURN count(*)",
                {},
                [(sum(a % 2 == 0 for a in ages),)],
            )
            check(
                "MATCH (p:Person) WHERE p.id = $a OR p.id = $b RETURN p.id",
                dict(a=id, b=id + 1),
                [(i,) for i in (id, id + 1) if i in persons],
            )
        if "count_restricted" not in skipped:
            check(
                "MATCH (a:Person)-[:Knows]->(:Person) WHERE a.id = $id "
                "RETURN count(*), count(DISTINCT a.id)",
                dict(id=id),
                [(degrees[id], 1 if degrees[id] else 0)],
            )
            check(
                "MATCH (a:Person)-[:Knows]->(:Person) WHERE a.age = $x RETURN count(*)",
                dict(x=x),
                [(sum(n for (src, _, _), n in knows.items() if persons.get(src, {}).get("age") == x),)],
            )
        # `alone` is a transaction's own view; its uncommitted rows reach index lookups only with
        # the secondary ART consistency fix.
        txn = self.txns.get(conn)
        if (alone or (txn and txn.kind == "write")) and "txn_index_lookups" in skipped:
            return
        check(
            "MATCH (p:Person) WHERE p.age = $x RETURN count(*)",
            dict(x=x),
            [(sum(a == x for a in ages),)],
        )
        check(
            "MATCH (p:Person) WHERE p.age IN $pair RETURN p.id",
            dict(pair=[x, x + 1]),
            [(i,) for i, p in persons.items() if p["age"] in (x, x + 1)],
        )
        if "pk_ranges" not in skipped:
            check(
                "MATCH (p:Person) WHERE p.id >= $lo AND p.id < $hi RETURN count(*)",
                dict(lo=id, hi=id + 10),
                [(sum(id <= i < id + 10 for i in persons),)],
            )
            check(
                "MATCH (p:Person) WHERE p.id > $lo RETURN count(*)",
                dict(lo=id),
                [(sum(i > id for i in persons),)],
            )

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

    def check_lookup_joins(self, view: Model, check, id: int) -> None:
        """Joins on a node ID whose other side is a bare scan of that node, the shape a lookup
        join fetches by offset instead of a scan and hash join: a few outer rows (the edges of
        one person) and the payload of the rows they reach, also columns added later."""
        persons, knows = view.persons, view.knows
        out = [(dst, n) for (src, dst, _), n in knows.items() if src == id]
        into = [(src, n) for (src, dst, _), n in knows.items() if dst == id]

        def payload(i):
            p = persons[i]
            return (i, p["name"], p["age"], p["score"])

        check(
            "MATCH (a:Person)-[:Knows]->(b:Person) WHERE a.id = $id "
            "MATCH (b) RETURN b.id, b.name, b.age, b.score",
            dict(id=id),
            [payload(dst) for dst, n in out for _ in range(n)],
        )
        check(
            "MATCH (a:Person)<-[:Knows]-(b:Person) WHERE a.id = $id "
            "MATCH (b) RETURN b.id, b.name, b.age, b.score",
            dict(id=id),
            [payload(src) for src, n in into for _ in range(n)],
        )
        for column in view.columns:
            check(
                f"MATCH (a:Person)-[:Knows]->(b:Person) WHERE a.id = $id MATCH (b) "
                f"RETURN b.id, b.{column}",
                dict(id=id),
                [(dst, persons[dst][column]) for dst, n in out for _ in range(n)],
            )
        # Two patterns sharing a node, as in LDBC's Q1.
        second = [
            (persons[b]["age"], c, persons[c]["age"])
            for b, n1 in out
            for (src, c, _), n2 in knows.items()
            if src == b
            for _ in range(n1 * n2)
        ]
        check(
            "MATCH (a:Person)-[:Knows]->(b:Person), (b)-[:Knows]->(c:Person) WHERE a.id = $id "
            "RETURN b.age, c.id, c.age",
            dict(id=id),
            second,
        )

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
        if self.has_partitions():
            check(
                "MATCH (n:L) WHERE n.cluster = $c RETURN count(*)",
                dict(c=cluster),
                [(sum(c == cluster for _, c in view.partitioned),)],
            )
        self.check_index_paths(view, conn, x, id, degrees, alone)
        if "lookup_join" not in self.options.skipped:
            self.check_lookup_joins(view, check, id)
        for column, default in view.columns.items():
            if column.startswith("c"):
                check(
                    f"MATCH (p:Person) WHERE p.{column} = $d RETURN count(*)",
                    dict(d=default),
                    [(sum(p[column] == default for p in persons.values()),)],
                )
