"""Rules for checkpoints, reopening, crashes and concurrent readers."""

from __future__ import annotations

import copy
import time

import hypothesis.strategies as st
from hypothesis import assume
from hypothesis.stateful import rule
from model import Model
from session import Session, diff, enabled
from strategies import BULK_ID_START, small_ids

RACE_OPS = ["insert", "delete", "set_age", "clear_age", "checkpoint"]


class DurabilityRules(Session):
    @rule()
    def checkpoint(self):
        self.ok("CHECKPOINT")

    @rule()
    def reopen(self):
        self.engine.close()
        self.reopen_engine()

    @rule()
    def crash(self):
        self.engine.kill()
        self.reopen_engine()

    @enabled("crash_during")
    @rule(
        op=st.sampled_from(["copy", "create", "delete", "checkpoint"]),
        size=st.integers(2000, 40000),
        delay=st.floats(0, 0.3),
    )
    def crash_during(self, op, size, delay):
        """Kills the engine while a large statement runs; recovery must see all of it or none."""
        assume(f"crash_{op}" not in self.options.skipped)
        after = copy.deepcopy(self.model)
        if op == "copy":
            assume(self.model.pk_index is not None or not self.model.persons)
            csv, added = self.write_bulk_csv([("bulk", i % 1000) for i in range(size)])
            after.persons.update(added)
            after.next_bulk_id = self.model.next_bulk_id
            query = f'COPY Person FROM "{csv}"'
        elif op == "create":
            ids = after.take_bulk_ids(size)
            self.model.next_bulk_id = after.next_bulk_id
            for id in ids:
                after.persons[id] = after.new_person(age=id % 1000)
            query = (
                f"UNWIND range({ids.start}, {ids.stop - 1}) AS i "
                "CREATE (:Person {id: i, age: i % 1000})"
            )
        elif op == "delete":
            for id in [id for id in after.persons if id >= BULK_ID_START]:
                after.forget(id)
            query = f"MATCH (p:Person) WHERE p.id >= {BULK_ID_START} DETACH DELETE p"
        else:
            query = "CHECKPOINT"
        self.engine.start(query)
        time.sleep(delay)
        self.engine.kill()
        self.reopen_engine()
        actual = self.person_rows()
        if actual == after.person_rows():
            self.model = after
        else:
            before = self.model.person_rows()
            assert actual == before, (
                f"after a crash during {op}, the data is neither the state before nor after it: "
                + diff(actual, before)
            )

    @enabled("race")
    @rule(
        ops=st.lists(
            st.tuples(st.sampled_from(RACE_OPS), small_ids, st.integers(-100, 100)),
            min_size=1,
            max_size=30,
        ),
        num_readers=st.integers(2, 4),
    )
    def race(self, ops, num_readers):
        """Writes race readers; every read transaction must see one committed state, in order."""
        if "race_checkpoint" in self.options.skipped:
            assume(self.config["checkpoint_threshold"] != 0)
        model = copy.deepcopy(self.model)
        writes, states = self.race_writes(model, ops)
        observations, errors = self.engine.race(writes, num_readers)
        assert not errors, errors
        for reader in observations:
            position = 0
            for seen in reader:
                matches = [i for i in range(position, len(states)) if states[i] == seen]
                assert matches, (
                    f"a reader saw {seen}, which is not a committed state at or after "
                    f"step {position}: {states}"
                )
                position = matches[0]
        self.model = model

    def race_writes(self, model: Model, ops: list) -> tuple[list, list]:
        """The statements for `ops`, applied to `model`, and the state after each of them."""
        states, writes = [model.state()], []
        for op, id, value in ops:
            if op == "checkpoint" and "race_checkpoint" in self.options.skipped:
                continue
            if op == "insert":
                if id in model.persons:
                    continue
                writes.append(("CREATE (:Person {id: $id, age: $v})", dict(id=id, v=value)))
                model.persons[id] = model.new_person(age=value)
            elif op == "delete":
                writes.append(("MATCH (p:Person {id: $id}) DETACH DELETE p", dict(id=id)))
                model.forget(id)
            elif op in ("set_age", "clear_age"):
                age = value if op == "set_age" else None
                writes.append(("MATCH (p:Person {id: $id}) SET p.age = $v", dict(id=id, v=age)))
                if id in model.persons:
                    model.persons[id]["age"] = age
            else:
                writes.append(("CHECKPOINT", {}))
            states.append(model.state())
        return writes, states
