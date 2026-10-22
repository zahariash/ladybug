"""Checks of the data, the catalog and a set of queries against the model."""

from __future__ import annotations

import random
from collections import Counter, defaultdict
from contextlib import contextmanager

import hypothesis.strategies as st
from hypothesis.stateful import invariant, precondition, rule
from model import C_FP, Model, two_hops, walk_ends
from session import Session, diff, gated
from strategies import (
    CK2_SEED_SOURCE,
    CSR_BANDS,
    HUBS,
    VECTOR_CAPACITY,
    clusters,
    csr_id,
    csr_src_id,
    small_ids,
)

USER_TABLES = {"Person", "Knows"}

gated(
    "in_lists",
    "in_list_nulls",
    "count_restricted",
    "txn_index_lookups",
    "pk_ranges",
    "lookup_join",
    "count_keys",
    "rollback_prepared",
    "csr_local_offset_count",
    "csr_walk_counts",
    "repeated_aggregate",
    "hop_one",
    "hop_two",
    "hop_varlen",
    "hop_mixed",
    "hop_optional",
    "band_degrees",
    "big_aggregates",
    "big_rows",
    "big_order",
)


def band_sample(seed: int) -> list[int]:
    """Twelve C ids from both node groups' bands, in an order that is not the offsets'."""
    return random.Random(seed).sample(CSR_BANDS[0] + CSR_BANDS[-1], 12)


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
        if self.has_csr():
            actual = Counter(self.rows("MATCH (a:C)-[:CK]->(b:C) RETURN a.id, b.id"))
            assert actual == self.model.ck, diff(actual, self.model.ck)
            actual = Counter(self.rows("MATCH (a:C)-[k:CK2]->(b:C) RETURN a.id, b.id, k.w"))
            assert actual == self.model.ck2, diff(actual, self.model.ck2)
            rows = self.rows(f"MATCH (c:C) RETURN count(*), sum({C_FP})")
            big = self.model.big
            expected = [(big.count(), big.fingerprint())]
            assert rows == expected, (rows, expected)
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
        if self.has_csr():
            expected |= {"C", "CK", "CK2"}
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
            self.unoptimized = set(conns)
        yield
        if not enabled:
            self.unoptimized = set()
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

    def check_key_counts(
        self, view: Model, check, conn: int, x: int, id: int, degrees: Counter
    ) -> None:
        """count() and count(DISTINCT) over primary keys and id(), which the binder may count as
        the internal ID: a whole table, filtered, both ends of an edge, an OPTIONAL MATCH end
        that is null when unbound, a bound node, a renamed variable and a two-hop end. Edge
        counts also run with a single aggregate, which rel-degree and rel-count rewrites answer
        from storage."""
        persons, knows = view.persons, view.knows
        n = len(persons)
        ages = [p["age"] for p in persons.values() if p["age"] is not None]
        check(
            "MATCH (p:Person) "
            "RETURN count(p.id), count(DISTINCT p.id), count(id(p)), count(p.age)",
            {},
            [(n, n, n, len(ages))],
        )
        older = sum(a > x for a in ages)
        check(
            "MATCH (p:Person) WHERE p.age > $x RETURN count(p.id), count(DISTINCT p.id)",
            dict(x=x),
            [(older, older)],
        )
        check("MATCH (p:Person) WITH p AS q RETURN count(q.id)", {}, [(n,)])
        edges = [(src, dst) for (src, dst, _), k in knows.items() for _ in range(k)]
        txn = self.txns.get(conn)
        if not (txn and txn.kind == "read" and "snapshot_rel_count" in self.options.skipped):
            check(
                "MATCH (a:Person)-[:Knows]->(b:Person) "
                "RETURN count(b.id), count(DISTINCT b.id), count(DISTINCT a.id)",
                {},
                [(len(edges), len({d for _, d in edges}), len({s for s, _ in edges}))],
            )
            # One aggregate each, the shape the count-from-storage rewrites take.
            sources, targets = len({s for s, _ in edges}), len({d for _, d in edges})
            for query, expected in [
                ("MATCH (a:Person)-[:Knows]->(:Person) RETURN count(DISTINCT a.id)", sources),
                ("MATCH (a:Person)-[:Knows]->(:Person) RETURN count(DISTINCT id(a))", sources),
                ("MATCH (b:Person)<-[:Knows]-(:Person) RETURN count(DISTINCT b.id)", targets),
                ("MATCH (a:Person)-[:Knows]->(b:Person) RETURN count(b.id)", len(edges)),
                ("MATCH ()-[k:Knows]->() RETURN count(k)", len(edges)),
                ("MATCH ()-[k:Knows]->() RETURN sum(2)", 2 * len(edges) if edges else None),
            ]:
                check(query, {}, [(expected,)])
        query = (
            "MATCH (a:Person)-[:Knows]->(b:Person) "
            "RETURN count(b.id), count(DISTINCT b.id), count(id(b))"
        )
        names, rows = self.columns(query, conn)
        # An unaliased aggregate is named by its text, also when the key is counted as the ID.
        assert names == ["COUNT(b.id)", "COUNT(DISTINCT b.id)", "COUNT(b._ID)"], (query, names)
        assert rows == [(len(edges), len({d for _, d in edges}), len(edges))], (query, rows)
        query = "MATCH (p:Person) WITH p, count(p.id) AS c RETURN c, count(p.id)"
        names, _ = self.columns(query, conn)
        assert names == ["c", "COUNT(p.id)"], (query, names)
        if "repeated_aggregate" not in self.options.skipped:
            in_degrees = Counter(d for _, d in edges)
            fan_out = {s: len({d for t, d in edges if t == s}) for s, _ in edges}
            for query, expected in [
                (query, [(1, n)] if n else []),
                (
                    "MATCH (a:Person)-[:Knows]->(b:Person) WITH b, count(b.id) AS c "
                    "RETURN c, count(b.id)",
                    list(Counter(in_degrees.values()).items()),
                ),
                (
                    "MATCH (a:Person)-[:Knows]->(b:Person) WITH a, count(DISTINCT b.id) AS c "
                    "RETURN c, count(DISTINCT a.id)",
                    list(Counter(fan_out.values()).items()),
                ),
            ]:
                check(query, {}, expected)
        out = {d for s, d in edges if s == id}
        check(
            "MATCH (a:Person) OPTIONAL MATCH (a)-[:Knows]->(b:Person) "
            "RETURN a.id, count(b.id), count(DISTINCT b.id)",
            {},
            [(i, degrees[i], len({d for s, d in edges if s == i})) for i in persons],
        )
        bound = 1 if degrees[id] else 0
        check(
            "MATCH (a:Person)-[:Knows]->(b:Person) WHERE a.id = $id RETURN count(a.id), "
            "count(DISTINCT a.id), count(DISTINCT id(a)), count(DISTINCT b.id)",
            dict(id=id),
            [(degrees[id], bound, bound, len(out))],
        )
        if "rollback_prepared" not in self.options.skipped:
            reach = out | {d for s, d in edges if s in out}
            check(
                "MATCH (a:Person)-[:Knows*1..2]->(b:Person) WHERE a.id = $id "
                "RETURN count(DISTINCT b.id)",
                dict(id=id),
                [(len(reach),)],
            )
        if self.has_partitions():
            ids = {i for i, _ in view.partitioned}
            check(
                "MATCH (n:L) RETURN count(n.id), count(DISTINCT n.id)", {}, [(len(ids), len(ids))]
            )
        for i, values in view.tables.items():
            check(
                f"MATCH (t:T{i}) RETURN count(t.id), count(DISTINCT t.id)",
                {},
                [(len(values), len(values))],
            )

    def check_csr_counts(self, view: Model, check, conn: int, a: int, alone: bool) -> None:
        """Counts over CK, whose node table C is declared CSR, so that the optimizer answers them
        from rel degrees (one node's, top k, nodes with an edge), by bounded traversals (nodes
        reachable from one node, walks ending at each node) and from rel counts. Each also runs
        over the key, which the binder may count as the internal ID. The id is in the query:
        the offset rewrite takes a literal key."""
        edges = list(view.ck.elements())
        targets = defaultdict(list)
        for s, d in edges:
            targets[s].append(d)
        out = targets.get(a, [])
        reach = set(out) | {d for m in out for d in targets.get(m, [])}
        walks = Counter(d for _, d in edges) + Counter(
            d for _, m in edges for d in targets.get(m, [])
        )
        txn = self.txns.get(conn)
        writing = alone or (txn is not None and txn.kind == "write")
        local_gap = writing and "csr_local_offset_count" in self.options.skipped
        if not (txn and txn.kind == "read" and "snapshot_rel_count" in self.options.skipped):
            for counted in ("*", "b.id"):
                if not local_gap:
                    check(
                        f"MATCH (a:C)-[:CK]->(b:C) WHERE a.id = {a} RETURN count({counted})",
                        {},
                        [(len(out),)],
                    )
                self.check_top_degrees(counted, Counter(s for s, _ in edges), conn)
            check(
                "MATCH (a:C)-[:CK]->(:C) RETURN count(DISTINCT a.id)", {}, [(len(targets),)]
            )
            check("MATCH ()-[k:CK]->() RETURN count(*)", {}, [(len(edges),)])
        # Unoptimized, the walks start at every row of C: seconds per query.
        if conn in getattr(self, "unoptimized", ()) or "csr_walk_counts" in self.options.skipped:
            return
        for counted in ("b", "b.id"):
            check(
                f"MATCH (a:C)-[:CK*1..2]->(b:C) WHERE a.id = {a} RETURN count(DISTINCT {counted})",
                {},
                [(len(reach),)],
            )
            check(
                f"MATCH (a:C)-[:CK*1..2]->(b:C) RETURN b.id, count({counted})",
                {},
                list(walks.items()),
            )

    def check_csr_degree(self, src: int, table: str = "CK") -> None:
        """One node's CK or CK2 count right after a committed write to its edges, the moment a
        count from storage and the edges it has to subtract or add are most likely to disagree."""
        pairs = self.model.ck if table == "CK" else self.model.ck2_pairs()
        n = sum(k for (s, _), k in pairs.items() if s == src)
        for counted in ("*", "b.id"):
            query = f"MATCH (a:C)-[:{table}]->(b:C) WHERE a.id = {src} RETURN count({counted})"
            self.check(query, {}, [(n,)])

    def check_big(self, view: Model, check, id: int) -> None:
        """C's rows past its size boundaries: aggregates over every row, rows at shuffled band
        ids and at the end of the table, the top of an ORDER BY, and a hub's edges."""
        skipped, big = self.options.skipped, view.big
        if "big_aggregates" not in skipped:
            nulls = sum(v is None for i, v in big.updated.items() if i not in big.deleted)
            low, high = big.min_max()
            if all(abs(v) < 2**40 for v in big.updated_values()):
                query = "MATCH (c:C) RETURN count(*), count(c.v), sum(c.v), min(c.v), max(c.v)"
                check(query, {}, [(big.count(), big.count() - nulls, big.sum(), low, high)])
            else:  # the sum would overflow INT64
                query = "MATCH (c:C) RETURN count(*), count(c.v), min(c.v), max(c.v)"
                check(query, {}, [(big.count(), big.count() - nulls, low, high)])
        ids = list(dict.fromkeys(band_sample(id) + [big.rows - 1, big.rows - VECTOR_CAPACITY - 1]))
        if "big_rows" not in skipped:
            query = f"MATCH (c:C) WHERE c.id IN {ids} RETURN c.id, c.v"
            check(query, {}, [(i, big.value(i)) for i in ids if big.exists(i)])
            gone = next((i for i in ids if not big.exists(i)), ids[0])
            query = f"MATCH (c:C {{id: {gone}}}) RETURN c.id, c.v"
            check(query, {}, [(gone, big.value(gone))] if big.exists(gone) else [])
        if "big_order" not in skipped:
            k = (1, 5, VECTOR_CAPACITY + 1)[id % 3]
            query = (
                "MATCH (c:C) WHERE c.v IS NOT NULL RETURN c.id, c.v "
                f"ORDER BY c.v DESC, c.id LIMIT {k}"
            )
            check(query, {}, big.top(k), ordered=True)
        if "hub_edges" not in skipped:
            hub = HUBS[id % len(HUBS)]
            leaves = [d for (s, d), n in view.ck.items() if s == hub for _ in range(n)]
            query = f"MATCH (a:C)-[:CK]->(b:C) WHERE a.id = {hub} RETURN b.id"
            check(query, {}, [(d,) for d in leaves])
            if "lookup_join" not in skipped:
                query = f"MATCH (a:C)-[:CK]->(b:C) WHERE a.id = {hub} MATCH (b) RETURN b.id, b.v"
                check(query, {}, [(d, big.value(d)) for d in leaves])

    def check_hops(self, view: Model, check, conn: int, id: int, alone: bool) -> None:
        """Paths over CK2, whose sources all lie in C's first node group so that its rel table
        lacks the second, and over CK: a scan that reads a node past a rel table's last node
        group must still read the edges of the nodes it reads next. Each shape has its own gate.
        In a write transaction a two-hop batch across such a node misses uncommitted edges
        (txn_missing_group), so the multi-hop shapes run there only with that gate lifted."""
        skipped = self.options.skipped
        ck, ck2 = Counter(view.ck), view.ck2_pairs()
        a, b, ids = csr_src_id(id), csr_id(id + 1), band_sample(id)
        if "hop_one" not in skipped:
            query = f"MATCH (a:C)-[:CK2]->(b:C) WHERE a.id = {a} RETURN b.id"
            check(query, {}, [(d,) for s, d in ck2.elements() if s == a])
            query = f"MATCH (b:C)<-[:CK2]-(a:C) WHERE b.id = {b} RETURN a.id"
            check(query, {}, [(s,) for s, d in ck2.elements() if d == b])
        if "band_degrees" not in skipped:
            for table, pairs in (("CK", ck), ("CK2", ck2)):
                for pattern, end in ((f"-[:{table}]->", 0), (f"<-[:{table}]-", 1)):
                    degrees = Counter(edge[end] for edge in pairs.elements())
                    query = f"MATCH (a:C){pattern}(:C) WHERE a.id IN {ids} RETURN a.id, count(*)"
                    check(query, {}, [(i, degrees[i]) for i in ids if degrees[i]])
        if "hop_optional" not in skipped:
            degrees = Counter(s for s, _ in ck2.elements())
            query = (
                f"MATCH (a:C) WHERE a.id IN {ids} OPTIONAL MATCH (a)-[:CK2]->(b:C) "
                "RETURN a.id, count(b)"
            )
            check(query, {}, [(i, degrees[i]) for i in ids if view.big.exists(i)])
        if alone and "txn_missing_group" in skipped:
            return
        sources = sorted({a, CK2_SEED_SOURCE})
        if "hop_two" not in skipped:
            for src in sources:
                query = (
                    f"MATCH (a:C)-[:CK2]->(m:C)-[:CK2]->(b:C) WHERE a.id = {src} RETURN m.id, b.id"
                )
                check(query, {}, [(m, d) for _, m, d in two_hops(ck2, [src]).elements()])
            # An extend over another extend's output, in sources' offset order.
            for table, pairs in (("CK", ck), ("CK2", ck2)):
                query = (
                    f"MATCH (a:C)-[:{table}]->(m:C)-[:{table}]->(b:C) WHERE a.id < 40 "
                    "RETURN a.id, m.id, b.id"
                )
                sources = {s for s, _ in pairs if s < 40}
                check(query, {}, list(two_hops(pairs, sources).elements()))
        if "hop_mixed" not in skipped:
            both, c = ck + ck2, csr_id(id)
            query = f"MATCH (a:C)-[:CK|:CK2]->(b:C) WHERE a.id = {c} RETURN b.id"
            check(query, {}, [(d,) for s, d in both.elements() if s == c])
            query = (
                f"MATCH (a:C)-[:CK|:CK2]->(m:C)-[:CK|:CK2]->(b:C) WHERE a.id = {c} "
                "RETURN m.id, b.id"
            )
            check(query, {}, [(m, d) for _, m, d in two_hops(both, [c]).elements()])
        # Unoptimized, the walks start at every row of C: seconds per query.
        if "hop_varlen" in skipped or conn in getattr(self, "unoptimized", ()):
            return
        for src in sources:
            ends = walk_ends(ck2, src, 1, 3)
            if sum(ends.values()) > 50_000:
                continue
            query = f"MATCH (a:C)-[:CK2*1..3]->(b:C) WHERE a.id = {src} RETURN b.id"
            check(query, {}, [(d,) for d in ends.elements()])
            query = f"MATCH (a:C)-[:CK2*1..3]->(b:C) WHERE a.id = {src} RETURN count(DISTINCT b)"
            check(query, {}, [(len(ends),)])
        walks = sum(ck2.values()) + sum(two_hops(ck2, {s for s, _ in ck2}).values())
        check("MATCH (a:C)-[:CK2*1..2]->(b:C) RETURN count(*)", {}, [(walks,)])

    def check_top_degrees(self, counted: str, degrees: Counter, conn: int) -> None:
        """The three nodes with the most CK edges; ties may come back in any order."""
        query = (
            f"MATCH (a:C)-[:CK]->(b:C) RETURN a.id, count({counted}) AS d ORDER BY d DESC LIMIT 3"
        )
        rows = self.rows(query, None, conn)
        expected = sorted(degrees.values(), reverse=True)[:3]
        assert [d for _, d in rows] == expected, (query, rows, expected)
        assert all(degrees[i] == d for i, d in rows), (query, rows, dict(degrees))
        assert len({i for i, _ in rows}) == len(rows), (query, rows)

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
        if "count_keys" not in self.options.skipped:
            self.check_key_counts(view, check, conn, x, id, degrees)
        if self.has_csr():
            self.check_csr_counts(view, check, conn, csr_id(id), alone)
            self.check_hops(view, check, conn, id, alone)
            self.check_big(view, check, id)
        for column, default in view.columns.items():
            if column.startswith("c"):
                check(
                    f"MATCH (p:Person) WHERE p.{column} = $d RETURN count(*)",
                    dict(d=default),
                    [(sum(p[column] == default for p in persons.values()),)],
                )
