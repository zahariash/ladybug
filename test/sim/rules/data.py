"""Rules that write data: persons, Knows edges, bulk COPY, the partitioned table."""

from __future__ import annotations

import os
import tempfile

import hypothesis.strategies as st
from hypothesis.stateful import precondition, rule
from engine import EngineError
from session import DUPLICATE_KEY, Session, enabled
from strategies import clusters, csv_names, doubles, ints, names, small_ids

NO_PK_INDEX = "COPY into a non-empty primary-key node table without a hash index"


def csv_field(value) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return '"' + value.replace('"', '""') + '"'
    return str(value)


def copy_query(csv: str, rows) -> str:
    """The parallel CSV reader rejects quoted newlines."""
    newline = any(isinstance(name, str) and "\n" in name for name, _ in rows)
    return f'COPY Person FROM "{csv}"' + (" (parallel=false)" if newline else "")


class DataRules(Session):
    @enabled("persons")
    @rule(id=small_ids, name=names, age=ints, score=doubles)
    def insert_person(self, id, name, age, score):
        params = dict(id=id, name=name, age=age, score=score)
        query = "CREATE (:Person {id: $id, name: $name, age: $age, score: $score})"
        if id in self.model.persons:
            self.fails(query, params, DUPLICATE_KEY)
            return
        self.ok(query, params)
        self.model.add_persons({id: self.model.new_person(name=name, age=age, score=score)})

    @enabled("persons")
    @rule(id=small_ids, field=st.sampled_from(["name", "age", "score"]), data=st.data())
    def update_person(self, id, field, data):
        value = data.draw({"name": names, "age": ints, "score": doubles}[field])
        self.ok(f"MATCH (p:Person {{id: $id}}) SET p.{field} = $v", dict(id=id, v=value))
        if id in self.model.persons:
            self.model.persons[id][field] = value

    @enabled("persons")
    @rule(id=small_ids)
    def delete_person(self, id):
        self.ok("MATCH (p:Person {id: $id}) DETACH DELETE p", dict(id=id))
        self.model.forget(id)

    @enabled("persons")
    @rule(src=small_ids, dst=small_ids, since=ints)
    def insert_knows(self, src, dst, since):
        self.ok(
            "MATCH (a:Person {id: $a}), (b:Person {id: $b}) CREATE (a)-[:Knows {since: $s}]->(b)",
            dict(a=src, b=dst, s=since),
        )
        if src in self.model.persons and dst in self.model.persons:
            self.model.knows[(src, dst, since)] += 1

    @enabled("persons")
    @rule(src=small_ids, dst=small_ids)
    def delete_knows(self, src, dst):
        self.ok(
            "MATCH (a:Person {id: $a})-[k:Knows]->(b:Person {id: $b}) DELETE k",
            dict(a=src, b=dst),
        )
        for key in [k for k in self.model.knows if k[:2] == (src, dst)]:
            del self.model.knows[key]

    @enabled("knows")
    @rule(src=small_ids, dst=small_ids, since=ints)
    def update_knows(self, src, dst, since):
        self.ok(
            "MATCH (a:Person {id: $a})-[k:Knows]->(b:Person {id: $b}) SET k.since = $s",
            dict(a=src, b=dst, s=since),
        )
        self.model.set_since(src, dst, since)

    @enabled("knows")
    @precondition(lambda self: self.model.persons)
    @rule(data=st.data(), count=st.integers(1, 200), missing=st.booleans())
    def copy_knows(self, data, count, missing):
        """COPYs edges between existing persons; with an unknown endpoint the whole COPY
        fails and adds nothing."""
        ids = st.sampled_from(sorted(self.model.persons))
        edges = [
            (data.draw(ids), data.draw(ids), data.draw(st.one_of(st.none(), ints)))
            for _ in range(count)
        ]
        if missing:
            edges.insert(data.draw(st.integers(0, count)), (edges[0][0], -1, 0))
        fd, csv = tempfile.mkstemp(suffix=".csv", dir=self.files)
        w = "" if self.model.knows_w is None else f",{self.model.knows_w}"
        with os.fdopen(fd, "w") as f:
            f.writelines(f"{s},{d},{'' if v is None else v}{w}\n" for s, d, v in edges)
        if missing:
            self.fails(f'COPY Knows FROM "{csv}"', None, "Unable to find primary key value -1")
            return
        self.ok(f'COPY Knows FROM "{csv}"')
        self.model.knows.update(edges)

    def write_bulk_csv(self, rows: list[tuple[str | None, int]]) -> tuple[str, dict]:
        """Writes rows of (name, age) with fresh ids; returns the file and the new persons."""
        ids = self.model.take_bulk_ids(len(rows))
        csv = os.path.join(self.files, f"bulk{ids.start}.csv")
        added = {}
        with open(csv, "w") as f:
            for id, (name, age) in zip(ids, rows, strict=True):
                added[id] = self.model.new_person(name=name, age=age, score=age / 4)
                values = [csv_field(v) for v in added[id].values()]
                f.write(",".join([str(id), *values]) + "\n")
        return csv, added

    @enabled("bulk_copy")
    @rule(rows=st.lists(st.tuples(csv_names, st.integers(-1000, 1000)), min_size=1, max_size=3000))
    def bulk_copy(self, rows):
        csv, added = self.write_bulk_csv(rows)
        query = copy_query(csv, rows)
        if self.model.pk_index is None and self.model.persons:
            self.fails(query, None, NO_PK_INDEX)
            return
        if self.model.pk_index is None and self.model.persons_ever:
            # Deleted rows count as rows until they are compacted, so the COPY may be refused.
            try:
                self.ok(query)
            except EngineError as e:
                assert NO_PK_INDEX in str(e), e
                return
        else:
            self.ok(query)
        self.model.add_persons(added)

    @enabled("partitions")
    @rule(id=small_ids, cluster=clusters)
    def insert_partitioned(self, id, cluster):
        query = "CREATE (:L {id: $id, cluster: $c})"
        if id in self.model.partitioned:
            self.fails(query, dict(id=id, c=cluster), DUPLICATE_KEY)
            return
        self.ok(query, dict(id=id, c=cluster))
        self.model.partitioned[id] = cluster

    @enabled("partitions")
    @rule(id=small_ids)
    def delete_partitioned(self, id):
        self.ok("MATCH (n:L {id: $id}) DELETE n", dict(id=id))
        self.model.partitioned.pop(id, None)
