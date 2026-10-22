"""Rules for checkpoints, reopening, crashes and concurrent readers."""

from __future__ import annotations

import copy
import shutil

import hypothesis.strategies as st
from hypothesis.stateful import precondition, rule
from model import CK2_FP, CK_FP, C_FP, KNOWS_FP, PERSON_FP, Model, knows_fp, person_fp
from session import ONE_THREAD, RECOVERY_FAILURES, Session, copy_c, diff, enabled, gated
from strategies import BULK_ID_START, C_ROWS_CAP, CSR_ROWS, HUBS, small_ids

CRASH_OPS = ["copy", "create", "delete", "checkpoint", "copy_c", "delete_hub"]
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
gated(*(f"crash_{op}" for op in CRASH_OPS), "race_checkpoint", "race_edges", "art_copy")
gated(*RECOVERY_FAILURES)

# The committed state a racing reader sees (see engine.read_state and race_state): counts that
# fast paths answer, and row fingerprints that scans answer.
STATE_QUERIES = [
    f"MATCH (p:Person) RETURN count(*), sum(p.id), count(p.age), sum({PERSON_FP})",
    "MATCH (:Person)-[k:Knows]->(:Person) RETURN count(*)",
    # Answered from rel degrees, which read the CSR index that committing edges change.
    "MATCH (a:Person)-[:Knows]->(:Person) RETURN count(DISTINCT a.id)",
    f"MATCH (a:Person)-[k:Knows]->(b:Person) RETURN count(*), sum({KNOWS_FP})",
]
# What a reader racing race_bulk sees of CK. The filtered count is a scan, planned from CK's
# degree statistics.
CK_STATE_QUERIES = [
    "MATCH ()-[k:CK]->() RETURN count(*)",
    "MATCH (a:C)-[:CK]->(:C) RETURN count(DISTINCT a.id)",
    "MATCH (a:C)-[k:CK]->(b:C) WHERE b.id >= 0 RETURN count(*)",
    # CK2 is not written in the race; its sources are all in C's first node group.
    "MATCH ()-[k:CK2]->() RETURN count(*)",
    f"MATCH (a:C)-[:CK]->(b:C) RETURN count(*), sum({CK_FP})",
    f"MATCH (a:C)-[k:CK2]->(b:C) RETURN count(*), sum({CK2_FP})",
]


def race_state(model: Model) -> tuple:
    """What engine.read_state returns for STATE_QUERIES."""
    people = model.persons
    ages = sum(p["age"] is not None for p in people.values())
    fingerprint = sum(person_fp(id, p) for id, p in people.items()) if people else None
    persons = (len(people), sum(people) if people else None, ages, fingerprint)
    sources = len({src for src, _, _ in model.knows})
    edges = sum(model.knows.values())
    knows = sum(n * knows_fp(*key) for key, n in model.knows.items()) if edges else None
    return (persons,), ((edges,),), ((sources,),), ((edges, knows),)


def ck_state(model: Model) -> tuple:
    """What engine.read_state returns for CK_STATE_QUERIES."""
    edges, edges2 = sum(model.ck.values()), sum(model.ck2.values())
    sources = len({src for src, _ in model.ck})
    fp = sum(n * (s * 200003 + d) for (s, d), n in model.ck.items()) if edges else None
    fp2 = sum(n * (s * 200003 + d + w * 7) for (s, d, w), n in model.ck2.items())
    fp2 = fp2 if edges2 else None
    return ((edges,),), ((sources,),), ((edges,),), ((edges2,),), ((edges, fp),), ((edges2, fp2),)


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
        if (self.model.pk_index is None and self.model.persons_ever) or not self.copy_person_safe():
            ops = [op for op in ops if op != "copy"]
        if not self.has_csr() or self.model.big.rows >= C_ROWS_CAP:
            ops = [op for op in ops if op != "copy_c"]
        if not self.has_csr() or not any(self.model.big.exists(hub) for hub in HUBS):
            ops = [op for op in ops if op != "delete_hub"]
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
        if op == "copy_c":
            # The crash reopens the engine, which restores the thread count.
            self.ok(ONE_THREAD)
            after.big.rows += size
            return copy_c(self.model.big.rows, size), after
        if op == "delete_hub":
            hub = next(hub for hub in HUBS if after.big.exists(hub))
            after.forget_c(hub)
            return f"MATCH (c:C {{id: {hub}}}) DETACH DELETE c", after
        return "CHECKPOINT", after

    def crash_state(self, model: Model | None = None) -> tuple:
        """Person's rows, and C's rows (count, fingerprint, strings) and CK's edge count: in the
        engine, or in `model`."""
        if model is not None:
            persons, big = model.person_rows(), model.big
            texts = big.text_count_size()[0]
            state = (big.count(), big.fingerprint(), texts, sum(model.ck.values()))
            return persons, state if self.has_csr() else ()
        state = ()
        if self.has_csr():
            query = f"MATCH (c:C) RETURN count(*), sum({C_FP}), count(c.s)"
            [(rows, fingerprint, texts)] = self.rows(query)
            [(edges,)] = self.rows("MATCH ()-[k:CK]->() RETURN count(*)")
            state = (rows, fingerprint, texts, edges)
        return self.person_rows(), state

    def check_crash(self, op: str, after: Model) -> None:
        """After a crash, the data is all of the statement or none of it."""
        if self.started_over:
            return
        actual = self.crash_state()
        if actual == self.crash_state(after):
            self.model = after
            return
        before = self.crash_state(self.model)
        assert actual == before, (
            f"after a crash during {op}, the data is neither the state before nor after it: "
            f"{diff(actual[0], before[0])}; C rows and CK edges {actual[1]}, before {before[1]}, "
            f"after {self.crash_state(after)[1]}"
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
        if "snapshot_update" in self.options.skipped and self.model.updated_since_checkpoint:
            # A checkpoint in the race would move these updates to other rows under the readers'
            # snapshots (snapshot_update); no reader is open yet.
            self.ok("CHECKPOINT")
            self.model.checkpointed()
        if "snapshot_update" in self.options.skipped and (
            self.model.updated_since_checkpoint
            or self.config["checkpoint_threshold"] != -1
            or any(op == "checkpoint" for op, _, _ in ops)
        ):
            # A checkpoint while the readers hold snapshots moves updates made since the last
            # checkpoint to other rows (snapshot_update).
            ops = [o for o in ops if o[0] not in ("set_age", "clear_age")]
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
            # The writes' endpoints spread over all of C; a deleted one has no key to COPY to.
            and not self.model.big.deleted
            # COPY checkpoints while the readers hold snapshots (snapshot_update).
            and (
                "snapshot_update" not in self.options.skipped
                or not self.model.updated_since_checkpoint
            )
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
