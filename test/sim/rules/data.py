"""Rules that write data: persons, Knows edges, bulk COPY, transactions, the partitioned table."""

from __future__ import annotations

import os

import hypothesis.strategies as st
from hypothesis.stateful import rule
from engine import EngineError
from session import DUPLICATE_KEY, Session, enabled
from strategies import clusters, csv_names, doubles, ints, names, small_ids

NO_PK_INDEX = "COPY into a non-empty primary-key node table without a hash index"


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
    @rule(src=small_ids, offset=st.integers(1, 40), since=ints)
    def insert_knows(self, src, offset, since):
        # No self-loops: they would make the 2-hop count depend on the path semantics.
        dst = (src + offset) % 41
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

    def write_bulk_csv(self, rows: list[tuple[str, int]]) -> tuple[str, dict]:
        """Writes rows of (name, age) with fresh ids; returns the file and the new persons."""
        ids = self.model.take_bulk_ids(len(rows))
        csv = os.path.join(self.dir, f"bulk{ids.start}.csv")
        added = {}
        with open(csv, "w") as f:
            for id, (name, age) in zip(ids, rows, strict=True):
                added[id] = self.model.new_person(name=name, age=age, score=age / 4)
                values = ["" if v is None else str(v) for v in added[id].values()]
                f.write(",".join([str(id), *values]) + "\n")
        return csv, added

    @enabled("bulk_copy")
    @rule(rows=st.lists(st.tuples(csv_names, st.integers(-1000, 1000)), min_size=1, max_size=3000))
    def bulk_copy(self, rows):
        csv, added = self.write_bulk_csv(rows)
        query = f'COPY Person FROM "{csv}"'
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

    @enabled("transaction")
    @rule(id=small_ids, age=ints, commit=st.booleans())
    def transaction(self, id, age, commit):
        self.ok("BEGIN TRANSACTION")
        self.ok("MATCH (p:Person {id: $id}) SET p.age = $v", dict(id=id, v=age))
        self.ok("MATCH (p:Person {id: $id}) DETACH DELETE p", dict(id=id + 1))
        self.ok("COMMIT" if commit else "ROLLBACK")
        if commit:
            if id in self.model.persons:
                self.model.persons[id]["age"] = age
            self.model.forget(id + 1)

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
