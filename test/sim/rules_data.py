"""Rules that write data: persons, Knows edges, bulk COPY, transactions, the partitioned table."""

from __future__ import annotations

import os

import hypothesis.strategies as st
from hypothesis import assume
from hypothesis.stateful import rule
from session import Session, enabled
from strategies import clusters, csv_names, doubles, ints, names, small_ids


class DataRules(Session):
    @rule(id=small_ids, name=names, age=ints, score=doubles)
    def insert_person(self, id, name, age, score):
        params = dict(id=id, name=name, age=age, score=score)
        query = "CREATE (:Person {id: $id, name: $name, age: $age, score: $score})"
        if id in self.model.persons:
            self.fails(query, params)
            return
        self.ok(query, params)
        self.model.persons[id] = self.model.new_person(name=name, age=age, score=score)

    @rule(id=small_ids, field=st.sampled_from(["name", "age", "score"]), data=st.data())
    def update_person(self, id, field, data):
        value = data.draw({"name": names, "age": ints, "score": doubles}[field])
        self.ok(f"MATCH (p:Person {{id: $id}}) SET p.{field} = $v", dict(id=id, v=value))
        if id in self.model.persons:
            self.model.persons[id][field] = value

    @rule(id=small_ids)
    def delete_person(self, id):
        self.ok("MATCH (p:Person {id: $id}) DETACH DELETE p", dict(id=id))
        self.model.forget(id)

    @rule(src=small_ids, dst=small_ids, since=ints)
    def insert_knows(self, src, dst, since):
        # Self-loops would make the 2-hop count depend on the path semantics.
        assume(src != dst)
        self.ok(
            "MATCH (a:Person {id: $a}), (b:Person {id: $b}) CREATE (a)-[:Knows {since: $s}]->(b)",
            dict(a=src, b=dst, s=since),
        )
        if src in self.model.persons and dst in self.model.persons:
            self.model.knows[(src, dst, since)] += 1

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

    @rule(rows=st.lists(st.tuples(csv_names, st.integers(-1000, 1000)), min_size=1, max_size=3000))
    def bulk_copy(self, rows):
        csv, added = self.write_bulk_csv(rows)
        if self.model.pk_index is None and self.model.persons:
            # COPY into a non-empty table needs the primary-key index.
            self.fails(f'COPY Person FROM "{csv}"')
            return
        self.ok(f'COPY Person FROM "{csv}"')
        self.model.persons.update(added)

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
            self.fails(query, dict(id=id, c=cluster))
            return
        self.ok(query, dict(id=id, c=cluster))
        self.model.partitioned[id] = cluster

    @enabled("partitions")
    @rule(id=small_ids)
    def delete_partitioned(self, id):
        self.ok("MATCH (n:L {id: $id}) DELETE n", dict(id=id))
        self.model.partitioned.pop(id, None)
