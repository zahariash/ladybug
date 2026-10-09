"""Rules for checkpoints, reopening, crashes and concurrent readers."""

from __future__ import annotations

import copy
import time

import hypothesis.strategies as st
from hypothesis.stateful import precondition, rule
from model import Model
from session import Session, diff, enabled, gated
from strategies import BULK_ID_START, small_ids

CRASH_OPS = ["copy", "create", "delete", "checkpoint"]
RACE_OPS = ["insert", "delete", "set_age", "clear_age", "checkpoint"]
gated(*(f"crash_{op}" for op in CRASH_OPS), "race_checkpoint")

# The committed state a racing reader sees (see engine.read_state and race_state). Races write
# ages between -100 and 100; other ages are left out of the sum so that it cannot overflow.
STATE_QUERIES = [
    "MATCH (p:Person) RETURN count(*), sum(p.id), count(p.age), "
    "sum(CASE WHEN p.age >= -100 AND p.age <= 100 THEN p.age ELSE 0 END)",
    "MATCH (:Person)-[k:Knows]->(:Person) RETURN count(*)",
]


def race_state(model: Model) -> tuple:
    """What engine.read_state returns for STATE_QUERIES."""
    ages = [p["age"] for p in model.persons.values() if p["age"] is not None]
    small = sum(a for a in ages if -100 <= a <= 100)
    persons = (len(model.persons), sum(model.persons) if model.persons else None, len(ages))
    return (persons + ((small if model.persons else None),),), ((sum(model.knows.values()),),)


class DurabilityRules(Session):
    @enabled("checkpoint")
    @rule()
    def checkpoint(self):
        self.ok("CHECKPOINT")

    @enabled("reopen")
    @rule()
    def reopen(self):
        self.engine.close()
        self.reopen_engine()

    @enabled("crash")
    @rule()
    def crash(self):
        self.engine.kill()
        self.reopen_engine()

    @enabled("crash_during")
    @rule(data=st.data(), size=st.integers(2000, 40000), delay=st.floats(0.01, 0.3))
    def crash_during(self, data, size, delay):
        """Kills the engine while a large statement runs; recovery must see all of it or none."""
        ops = [op for op in CRASH_OPS if f"crash_{op}" not in self.options.skipped]
        if self.model.pk_index is None and self.model.persons_ever:
            ops = [op for op in ops if op != "copy"]
        if not ops:
            return
        op = data.draw(st.sampled_from(ops))
        after = copy.deepcopy(self.model)
        if op == "copy":
            csv, added = self.write_bulk_csv([("bulk", i % 1000) for i in range(size)])
            after.add_persons(added)
            after.next_bulk_id = self.model.next_bulk_id
            query = f'COPY Person FROM "{csv}"'
        elif op == "create":
            ids = after.take_bulk_ids(size)
            self.model.next_bulk_id = after.next_bulk_id
            after.add_persons({id: after.new_person(age=id % 1000) for id in ids})
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
            return
        before = self.model.person_rows()
        assert actual == before, (
            f"after a crash during {op}, the data is neither the state before nor after it: "
            + diff(actual, before)
        )
        # An interrupted statement may still have reserved rows, as for an interrupted COPY.
        self.model.persons_ever = after.persons_ever

    @enabled("race")
    @precondition(
        # Automatic checkpoints run whenever the threshold is not the default; see #1160.
        lambda self: (
            "race_checkpoint" not in self.options.skipped
            or self.config["checkpoint_threshold"] == -1
        )
    )
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
            # Empties the WAL, so that the race's writes cannot trigger an automatic checkpoint.
            self.ok("CHECKPOINT")
        # Races write only small ids, so only those rows are copied; bulk rows are shared.
        persons = {id: dict(p) for id, p in self.model.persons.items() if id < BULK_ID_START}
        model = Model(
            columns=self.model.columns,
            persons={**self.model.persons, **persons},
            knows=copy.copy(self.model.knows),
        )
        writes, states = self.race_writes(model, ops)
        self.check_race(writes, STATE_QUERIES, states, num_readers)
        self.model.persons, self.model.knows = model.persons, model.knows
        self.model.persons_ever = self.model.persons_ever or bool(model.persons)

    def race_writes(self, model: Model, ops: list) -> tuple[list, list]:
        """The statements for `ops`, applied to `model`, and the state after each of them."""
        states, writes = [race_state(model)], []
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
            states.append(race_state(model))
        return writes, states
