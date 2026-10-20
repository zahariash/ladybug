"""Rules for checkpoints, reopening, crashes and concurrent readers."""

from __future__ import annotations

import copy
import shutil

import hypothesis.strategies as st
from hypothesis.stateful import precondition, rule
from model import Model
from session import Session, diff, enabled, gated
from strategies import BULK_ID_START, CSR_ROWS, small_ids

CRASH_OPS = ["copy", "create", "delete", "checkpoint"]
RACE_OPS = ["insert", "delete", "set_age", "clear_age", "knows", "unknows", "checkpoint"]
# File syscalls a crash point can kill at, with how many calls a statement makes at most; the
# engine writes from two threads, each counted separately.
SYSCALLS = {
    "pwrite64": 600,
    "fdatasync": 12,
    "fsync": 12,
    "ftruncate": 6,
    "rename": 2,
    "unlink": 10,
}
gated(*(f"crash_{op}" for op in CRASH_OPS), "race_checkpoint", "race_edges")

# The committed state a racing reader sees (see engine.read_state and race_state). Races write
# ages between -100 and 100; other ages are left out of the sum so that it cannot overflow.
STATE_QUERIES = [
    "MATCH (p:Person) RETURN count(*), sum(p.id), count(p.age), "
    "sum(CASE WHEN p.age >= -100 AND p.age <= 100 THEN p.age ELSE 0 END)",
    "MATCH (:Person)-[k:Knows]->(:Person) RETURN count(*)",
    # Answered from rel degrees, which read the CSR index that committing edges change.
    "MATCH (a:Person)-[:Knows]->(:Person) RETURN count(DISTINCT a.id)",
]
# What a reader racing race_bulk sees of CK. The filtered count is a scan, planned from CK's
# degree statistics.
CK_STATE_QUERIES = [
    "MATCH ()-[k:CK]->() RETURN count(*)",
    "MATCH (a:C)-[:CK]->(:C) RETURN count(DISTINCT a.id)",
    "MATCH (a:C)-[k:CK]->(b:C) WHERE b.id >= 0 RETURN count(*)",
]


def race_state(model: Model) -> tuple:
    """What engine.read_state returns for STATE_QUERIES."""
    ages = [p["age"] for p in model.persons.values() if p["age"] is not None]
    small = sum(a for a in ages if -100 <= a <= 100)
    persons = (len(model.persons), sum(model.persons) if model.persons else None, len(ages))
    sources = len({src for src, _, _ in model.knows})
    return (
        (persons + ((small if model.persons else None),),),
        ((sum(model.knows.values()),),),
        ((sources,),),
    )


def ck_state(model: Model) -> tuple:
    """What engine.read_state returns for CK_STATE_QUERIES."""
    edges = sum(model.ck.values())
    return ((edges,),), ((len({src for src, _ in model.ck}),),), ((edges,),)


def bulk_edges(start: int, count: int, step: int) -> list[tuple[int, int]]:
    """The CK edges bulk_ck_query adds, from up to count distinct sources spread over C."""
    rows = range(start, start + count)
    return [((i * 67 + step) % CSR_ROWS, (i * step) % CSR_ROWS) for i in rows]


def bulk_ck_writes(kind: str, start: int, count: int, step: int) -> list[tuple[str, dict]]:
    """One COPY of bulk_edges(start, count, step), or one CREATE per edge: a CREATE that looks up
    many edges' endpoints from an UNWIND fills the buffer pool on a table as large as C."""
    if kind == "copy":
        src, dst = f"(i * 67 + {step}) % {CSR_ROWS}", f"(i * {step}) % {CSR_ROWS}"
        rows = f"UNWIND range({start}, {start + count - 1}) AS i RETURN {src}, {dst}"
        return [(f"COPY CK FROM ({rows})", {})]
    query = "MATCH (a:C {id: $a}), (b:C {id: $b}) CREATE (a)-[:CK]->(b)"
    return [(query, dict(a=src, b=dst)) for src, dst in bulk_edges(start, count, step)]


class DurabilityRules(Session):
    @enabled("checkpoint")
    @precondition(lambda self: self.checkpoint_keeps_snapshots())
    @rule()
    def checkpoint(self):
        self.ok("CHECKPOINT")
        self.model.checkpointed()

    @enabled("reopen", writes=False)
    @rule()
    def reopen(self):
        self.reopen_engine()

    @enabled("crash", writes=False)
    @rule()
    def crash(self):
        self.crash_engine()

    def crash_ops(self) -> list[str]:
        ops = [op for op in CRASH_OPS if f"crash_{op}" not in self.options.skipped]
        if self.model.pk_index is None and self.model.persons_ever:
            ops = [op for op in ops if op != "copy"]
        return ops

    def plan_crash(self, op: str, size: int) -> tuple[str, Model]:
        """The statement for a crash op and the model if it completes."""
        after = copy.deepcopy(self.model)
        if op == "copy":
            csv, added = self.write_bulk_csv([("bulk", i % 1000) for i in range(size)])
            after.add_persons(added)
            after.next_bulk_id = self.model.next_bulk_id
            return f'COPY Person FROM "{csv}"', after
        if op == "create":
            ids = after.take_bulk_ids(size)
            self.model.next_bulk_id = after.next_bulk_id
            after.add_persons({id: after.new_person(age=id % 1000) for id in ids})
            query = (
                f"UNWIND range({ids.start}, {ids.stop - 1}) AS i "
                "CREATE (:Person {id: i, age: i % 1000})"
            )
            return query, after
        if op == "delete":
            for id in [id for id in after.persons if id >= BULK_ID_START]:
                after.forget(id)
            return f"MATCH (p:Person) WHERE p.id >= {BULK_ID_START} DETACH DELETE p", after
        return "CHECKPOINT", after

    def check_crash(self, op: str, after: Model) -> None:
        """After a crash, the data is all of the statement or none of it."""
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

    @enabled("crash_during")
    @rule(data=st.data(), size=st.integers(2000, 40000), delay=st.floats(0.01, 0.3))
    def crash_during(self, data, size, delay):
        """Kills the engine while a large statement runs; recovery must see all of it or none."""
        if not (ops := self.crash_ops()):
            return
        op = data.draw(st.sampled_from(ops))
        query, after = self.plan_crash(op, size)
        self.crash_during_statement(query, delay)
        self.check_crash(op, after)

    @enabled("crash_at_syscall")
    @precondition(lambda self: shutil.which("strace") is not None)
    @rule(data=st.data(), size=st.integers(500, 5000), syscall=st.sampled_from(sorted(SYSCALLS)))
    def crash_at_syscall(self, data, size, syscall):
        """Kills the engine at the n-th call of a file syscall in a large statement, a crash
        point that does not depend on timing and shrinks with n."""
        if not (ops := self.crash_ops()):
            return
        op = data.draw(st.sampled_from(ops))
        n = data.draw(st.integers(1, SYSCALLS[syscall]))
        query, after = self.plan_crash(op, size)
        self.crash_at_syscall_point(query, syscall, n)
        self.check_crash(op, after)

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
        if "race_checkpoint" in self.options.skipped and self.checkpoint_keeps_snapshots():
            # Empties the WAL, so that the race's writes cannot trigger an automatic checkpoint.
            self.ok("CHECKPOINT")
        model = self.model.fork()
        writes, states = self.race_writes(model, ops)
        self.check_race(writes, STATE_QUERIES, states, num_readers)
        model.persons_ever = model.persons_ever or bool(model.persons)
        self.model = model

    def race_writes(self, model: Model, ops: list) -> tuple[list, list]:
        """The statements for `ops`, applied to `model`, and the state after each of them."""
        states, writes = [race_state(model)], []
        for op, id, value in ops:
            if op == "checkpoint" and (
                "race_checkpoint" in self.options.skipped or not self.checkpoint_keeps_snapshots()
            ):
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
                model.update(id, "age", age)
            elif op in ("knows", "unknows"):
                # Between existing persons, the second one picked by `value`.
                if id not in model.persons or "race_edges" in self.options.skipped:
                    continue
                dst = sorted(model.persons)[value % len(model.persons)]
                if op == "knows":
                    query = (
                        "MATCH (a:Person {id: $a}), (b:Person {id: $b}) "
                        "CREATE (a)-[:Knows {since: $s}]->(b)"
                    )
                    writes.append((query, dict(a=id, b=dst, s=value)))
                    model.knows[(id, dst, value)] += 1
                else:
                    query = "MATCH (a:Person {id: $a})-[k:Knows]->(b:Person {id: $b}) DELETE k"
                    writes.append((query, dict(a=id, b=dst)))
                    for key in [k for k in model.knows if k[:2] == (id, dst)]:
                        del model.knows[key]
            else:
                writes.append(("CHECKPOINT", {}))
            states.append(race_state(model))
        return writes, states

    @enabled("race_bulk")
    @precondition(
        lambda self: (
            self.has_csr()
            and "race_edges" not in self.options.skipped
            and (
                "race_checkpoint" not in self.options.skipped
                or self.config["checkpoint_threshold"] == -1
            )
        )
    )
    @rule(
        writes=st.lists(
            st.one_of(
                st.tuples(st.just("copy"), st.integers(0, 3000), st.integers(1, 3000)),
                st.tuples(st.just("create"), st.integers(0, 3000), st.integers(1, 100)),
            ),
            min_size=1,
            max_size=4,
        ),
        step=st.integers(1, 7),
        num_readers=st.integers(2, 4),
    )
    def race_bulk(self, writes, step, num_readers):
        """Many CK edges race readers of CK: COPYs of thousands, each giving many nodes their
        first in-memory edges or adding to earlier ones, and one-edge CREATEs, each a commit.
        Readers walk the CSR index those commits change; a COPY also forces a checkpoint."""
        checkpoints = (
            "race_checkpoint" not in self.options.skipped and self.checkpoint_keeps_snapshots()
        )
        if "race_checkpoint" in self.options.skipped and self.checkpoint_keeps_snapshots():
            self.ok("CHECKPOINT")
        model = self.model.fork()
        statements, states = [], [ck_state(model)]
        for kind, start, count in writes:
            if kind == "copy" and not checkpoints:
                kind, count = "create", min(count, 100)
            for query, params in bulk_ck_writes(kind, start, count, step):
                statements.append((query, params))
                if kind == "create":
                    model.ck[(params["a"], params["b"])] += 1
                    states.append(ck_state(model))
            if kind == "copy":
                model.ck.update(bulk_edges(start, count, step))
                states.append(ck_state(model))
        self.check_race(statements, CK_STATE_QUERIES, states, num_readers)
        self.model = model
