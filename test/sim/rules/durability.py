"""Rules for checkpoints, reopening, crashes and concurrent readers."""

from __future__ import annotations

import copy
import os
import shutil
from collections import Counter

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


gated("snapshot_rel_scan")
# A CK edge per statement, so that each is a commit of its own.
WRITER_EDGE = "MATCH (a:C {id: $a}), (b:C {id: $b}) CREATE (a)-[:CK]->(b)"


def writer_edges(model: Model, writers: int, edges: int, start: int) -> list[tuple[int, list[int]]]:
    """For each writer a source node of its own and the distinct targets of its edges, in order;
    none is an edge the model holds already."""
    plans = []
    for w in range(writers):
        src = (start + 1009 * w) % CSR_ROWS
        targets = {(start + 3 * i + 1 + w) % CSR_ROWS for i in range(edges)}
        targets = sorted(d for d in targets if model.big.exists(d) and (src, d) not in model.ck)
        if model.big.exists(src) and targets:
            plans.append((src, targets))
    return plans


def writer_statements(plans: list) -> list[list]:
    return [[(WRITER_EDGE, dict(a=src, b=d)) for d in targets] for src, targets in plans]


def edge_fingerprint(src: int, dst: int) -> int:
    """An edge's share of the last CK_STATE_QUERIES fingerprint."""
    return src * 200003 + dst


def writer_states(base: tuple[int, int], plans: list) -> set[tuple[int, int]]:
    """The (edges, fingerprint) states concurrent writers can pass through: each writer's commits
    are in order, so a state holds a prefix of each writer's edges."""
    states = {base}
    for src, targets in plans:
        reached = set()
        for edges, fingerprint in states:
            reached.add((edges, fingerprint))
            for dst in targets:
                edges += 1
                fingerprint += edge_fingerprint(src, dst)
                reached.add((edges, fingerprint))
        states = reached
    return states


def check_writer_reads(
    observations: list, base: tuple, plans: list, final: tuple, one_snapshot: bool = True
) -> None:
    """Each read saw one state of writer_states and CK2 unchanged, no reader went back, and the
    last read, after all commits, saw the final state. `one_snapshot`: all queries of a read-only
    transaction saw the same state; otherwise, with automatic checkpoints on while snapshot_rel_scan
    is gated, each is checked on its own."""
    start = (base[0][0][0], base[4][0][1] or 0)
    states = writer_states(start, plans)
    for reader in observations:
        last = -1
        for seen in reader:
            edges = seen[4][0][0]
            fingerprint = seen[4][0][1] or 0
            consistent = not one_snapshot or edges == seen[0][0][0] == seen[2][0][0]
            valid = seen[3] == base[3] and seen[5] == base[5] and (edges, fingerprint) in states
            assert consistent and valid, (
                f"a reader saw {seen}, which no prefix of the writers' commits gives"
            )
            assert edges >= last, f"a reader went back from {last} edges to {seen}"
            last = edges
        assert reader[-1] == final, (reader[-1], final)


def read_acks(lines: list[str]) -> tuple[Counter, list[str]]:
    """How many commits returned for each writer, and the errors they hit, from what
    engine.start_writers wrote; a kill can leave the last line unfinished."""
    acked, errors = Counter(), []
    for line in lines:
        writer, _, error = line.partition(" ")
        if not line.endswith("\n"):
            continue
        if error:
            errors.append(line.strip())
        else:
            acked[int(writer)] += 1
    return acked, errors


def check_survivors(writer: int, targets: list[int], kept: Counter, acked: int) -> int:
    """How many of a writer's commits survived a crash: at least all that returned, and a prefix."""
    count = sum(kept.values())
    assert count >= acked, f"writer {writer}: {acked} commits returned, but only {count} survived"
    assert kept == Counter(targets[:count]), (
        f"writer {writer}: the {count} edges that survived are not its first ones: {kept}"
    )
    return count


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
        """Person's rows; C's rows (count, fingerprint, strings) and CK's edge count; the named
        graph's rows and Blob's: in the engine, or in `model`."""
        if model is not None:
            persons, big = model.person_rows(), model.big
            texts = big.text_count_size()[0]
            state = (big.count(), big.fingerprint(), texts, sum(model.ck.values()))
            blobs = Counter(model.blobs.items())
            return persons, state if self.has_csr() else (), model.graph_rows(), blobs
        state = ()
        if self.has_csr():
            query = f"MATCH (c:C) RETURN count(*), sum({C_FP}), count(c.s)"
            [(rows, fingerprint, texts)] = self.rows(query)
            [(edges,)] = self.rows("MATCH ()-[k:CK]->() RETURN count(*)")
            state = (rows, fingerprint, texts, edges)
        graph = self.graph_rows() if self.model.graph is not None else Counter()
        blobs = self.blob_rows() if self.has_blobs() else Counter()
        return self.person_rows(), state, graph, blobs

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
            f"after {self.crash_state(after)[1]}; graph {diff(actual[2], before[2])}; "
            f"blobs {diff(actual[3], before[3])}"
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

    @enabled("race_writers")
    @precondition(
        lambda self: self.multi_writes and self.has_csr() and self.checkpoint_keeps_snapshots()
    )
    @rule(
        writers=st.integers(2, 3),
        edges=st.integers(1, 20),
        start=st.integers(0, CSR_ROWS - 1),
        num_readers=st.integers(1, 3),
    )
    def race_writers(self, writers, edges, start, num_readers):
        """Writers commit CK edges concurrently while readers read: every read must be one state
        that some prefix of each writer's commits gives, and all must be there at the end."""
        plans = writer_edges(self.model, writers, edges, start)
        if len(plans) < 2:
            return
        threshold = self.config["checkpoint_threshold"]
        if "snapshot_rel_scan" in self.options.skipped and threshold == -1:
            # Empties the WAL, so that the writes cannot reach the default threshold mid-race.
            self.ok("CHECKPOINT")
            self.model.checkpointed()
        writes = writer_statements(plans)
        base = ck_state(self.model)
        op = dict(op="race_writers", writers=writes, queries=CK_STATE_QUERIES, readers=num_readers)
        observations, errors = self.traced(
            op, lambda: self.engine.race_writers(writes, CK_STATE_QUERIES, num_readers)
        )
        op["errors"] = errors
        assert not errors, errors
        for src, targets in plans:
            for dst in targets:
                self.model.ck[(src, dst)] += 1
        one_snapshot = (
            "snapshot_rel_scan" not in self.options.skipped
            or self.config["checkpoint_threshold"] == -1
        )
        check_writer_reads(observations, base, plans, ck_state(self.model), one_snapshot)

    @enabled("crash_writers")
    @precondition(
        lambda self: self.multi_writes and self.has_csr() and self.checkpoint_keeps_snapshots()
    )
    @rule(
        writers=st.integers(2, 4),
        edges=st.integers(5, 60),
        start=st.integers(0, CSR_ROWS - 1),
        delay=st.floats(0.0, 0.1),
    )
    def crash_writers(self, writers, edges, start, delay):
        """Kills the engine while writers commit concurrently. Recovery must keep every commit
        that returned, and of each writer's commits a prefix: nothing after a lost one."""
        plans = writer_edges(self.model, writers, edges, start)
        if not plans:
            return
        acks = os.path.join(self.files, f"acks-{len(self.trace)}")
        self.crash_during_writers(writer_statements(plans), acks, delay)
        if self.started_over:
            return
        lines = []
        if os.path.exists(acks):
            with open(acks) as f:
                lines = f.readlines()
        acked, errors = read_acks(lines)
        assert not errors, errors
        for w, (src, targets) in enumerate(plans):
            rows = self.rows("MATCH (a:C {id: $a})-[:CK]->(b:C) RETURN b.id", dict(a=src))
            before = Counter({d: n for (s, d), n in self.model.ck.items() if s == src})
            count = check_survivors(w, targets, Counter(row[0] for row in rows) - before, acked[w])
            # How many survive depends on timing, and rules draw from the model: removing them
            # keeps the workload replayable.
            if count:
                query = "MATCH (a:C {id: $a})-[k:CK]->(b:C) WHERE b.id IN $ids DELETE k"
                self.ok(query, dict(a=src, ids=targets[:count]))
        self.check_recovered(self.model)

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
