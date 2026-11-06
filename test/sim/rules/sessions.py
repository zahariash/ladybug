"""Rules that interleave transactions on several connections, one step per rule.

Connections 2-4 hold the sessions. A read-only transaction sees the committed state as of its
BEGIN for its whole life, a write transaction sees that plus its own writes, and a connection
without a transaction sees the committed state; nobody sees another transaction's uncommitted
writes. In single-writer mode only one write transaction can be open: a second BEGIN or an
auto-commit write elsewhere fails. In multi-writer mode they overlap: writesets.py says when a write
conflicts with another transaction's, and a commit replays the transaction's ops on the committed
state, since others may have committed since its snapshot. Every step is a rule of its own, so a
failure shrinks to the few steps that matter.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

import hypothesis.strategies as st
from engine import EngineError
from hypothesis.stateful import precondition, rule
from model import Model
from rules.data import copy_query
from rules.transactions import NO_TRANSACTION, apply, op, replay, seen_by, statement
from session import DUPLICATE_KEY, RULE_RUNS, Session, enabled, gated
from strategies import clusters, near_boundaries, small_ids
from writesets import (
    GATED_HIDDEN,
    GATED_INSERT,
    GATES,
    PERSON,
    Others,
    Writes,
    conflict,
    edge_count,
    record,
)

SESSIONS = (2, 3, 4)
READ_ONLY = "Can not execute a write query inside a read-only transaction"
ONE_WRITER = "Only one write transaction at a time"
POOL_FULL = "Buffer manager exception"
# The commit succeeded; the automatic checkpoint after it did not.
COMMITTED_BUT = "Transaction committed successfully, but the post-commit checkpoint failed"
# What CHECKPOINT and COPY report after waiting for an open write transaction to leave.
CHECKPOINT_BLOCKED = "Timeout waiting for active write transactions"
COPY_BLOCKED = "Timeout waiting for other write transactions"
# A checkpoint, or a commit past the checkpoint threshold, waits 5 s for open writers. Workloads may
# pay for this a few times; beyond that, overlapping writers run with automatic checkpoints off.
SLOW_WAITS = 2
# Beyond any WAL here; auto_checkpoint=false would also turn off the checkpoint that makes a COPY
# durable.
NO_AUTOMATIC_CHECKPOINT = 2**40
gated(*GATES)
# Few rows and entries, so that overlapping writers keep hitting each other's. The ops are listed,
# not composed, to keep the rule's draw small: Hypothesis drops a workload that draws too much.
HOT_PERSONS = range(3)
HOT_EDGE = (3, 67)
HOT_TABLE = 0
NEW_PERSONS = [("insert", id, age) for id in (6, 7) for age in (-5, 5)]
SET_AGES = [("set_age", id, age) for id in HOT_PERSONS for age in (-5, 5)]
DELETES = [("delete", id, None) for id in HOT_PERSONS]
update_ops = [*SET_AGES, *DELETES, *[("duplicate", id, None) for id in HOT_PERSONS]]
key_ops = [
    *NEW_PERSONS,
    *[("delete", id, None) for id in (6, 7)],
    *[("set_age", id, 5) for id in (6, 7)],
    *[("knows", a, b) for a, b in ((6, 7), (7, 6))],
]
detach_ops = [*DELETES, *[("knows", a, b) for a, b in ((0, 1), (1, 2), (2, 0))]]
edge_ops = [(kind, *HOT_EDGE) for kind in ("ck", "unck", "ck2", "unck2")]
catalog_ops = [
    *SET_AGES,
    *NEW_PERSONS,
    *[("add_column", column, value) for column in ("c0", "c1") for value in (-5, 5)],
    *[("drop_column", column, None) for column in ("c0", "c1")],
    *[("create_table", table, None) for table in (HOT_TABLE, 1)],
    *[("insert_t", table, 1) for table in (HOT_TABLE, 1)],
    *[("drop_table", table, None) for table in (HOT_TABLE, 1)],
    *[("create_macro", macro, 1) for macro in (0, 1)],
    *[("drop_macro", macro, None) for macro in (0, 1)],
]
serial_ops = [("insert_t", HOT_TABLE, value) for value in (1, 2)]
AUTO_COMMIT = 3


def overlap_round_on(ops: list):
    """Writes on one kind of entity. Each names its writer, AUTO_COMMIT for an auto-commit beside
    the transactions; the first transaction ends before the write at `first_ends`, so that the
    others keep writing after its commit."""
    write = st.tuples(st.integers(0, AUTO_COMMIT), st.sampled_from(ops))
    return st.fixed_dictionaries(
        {
            "writes": st.lists(write, min_size=3, max_size=10),
            "first_ends": st.integers(1, 6),
            "first_commits": st.sampled_from([True, True, True, False]),
            "end": st.sampled_from(["commit", "commit", "commit", "rollback", "crash", "reopen"]),
        }
    )


overlap_round = st.one_of(
    *map(
        overlap_round_on,
        (update_ops, key_ops, key_ops, detach_ops, edge_ops, catalog_ops, serial_ops),
    )
)
gated(
    "snapshot_checkpoint",
    "snapshot_rel_count",
    "snapshot_prepared",
    "snapshot_update",
    "snapshot_insert_delete",
)


@dataclass
class Txn:
    kind: str  # "read" or "write"
    view: Model  # what the transaction sees
    updated: bool = False  # a write transaction that updated an existing person
    start: int = 0  # the number of the last commit its snapshot holds
    ops: list = field(default_factory=list)  # (kind, a, b, seen_by) of its writes, in order
    writes: Writes = field(default_factory=Writes)


class SessionRules(Session):
    def free_sessions(self) -> list[int]:
        return [conn for conn in SESSIONS if conn not in self.txns]

    def readers(self) -> list[int]:
        return [conn for conn, txn in self.txns.items() if txn.kind == "read"]

    def session_writers(self) -> list[int]:
        return [conn for conn, txn in self.txns.items() if txn.kind == "write"]

    def overlap_allowed(self) -> bool:
        """Whether another commit may overlap a write transaction: past the checkpoint threshold
        it would wait 5 s for that transaction to leave."""
        return self.config["checkpoint_threshold"] == -1 or self.slow_waits > 0

    def may_begin_write(self) -> bool:
        if self.session_writer() is None:
            return True
        return self.multi_writes and self.overlap_allowed()

    def lookups_safe(self, txn: Txn) -> bool:
        """Whether to run the read battery in `txn`: its key lookups may hit a key inserted since
        the snapshot, which finds its own new row (hidden_key_lookup), and the lookups are not told
        apart from the rest of it."""
        if GATED_HIDDEN not in self.options.skipped or PERSON not in txn.writes.staged:
            return True
        return not any(w.inserted for n, w in self.commit_log if n > txn.start)

    def overlapping(self, conn: int | None = None) -> Others:
        """What the transactions overlapping connection `conn`'s write wrote; without `conn`, an
        auto-commit write, which overlaps only the open ones."""
        txn = self.txns.get(conn)
        return Others(
            open=[t.writes for c, t in self.txns.items() if t.kind == "write" and c != conn],
            committed=[w for n, w in self.commit_log if txn and n > txn.start],
        )

    def overlapping_step(self, view: Model, own: Writes, others: Others, write) -> tuple | None:
        """The statement for `write` and the error it must fail with; None if it does not run."""
        kind, a, b = write
        # What a rolled-back transaction leaves behind is in storage, whatever `view` predates.
        view.rolled_back_add |= self.model.rolled_back_add
        view.rolled_back_macro |= self.model.rolled_back_macro
        view.serial_rewound |= self.model.serial_rewound
        if not self.runs(view, kind, a):
            return None
        step = statement(view, kind, a, b)
        if step is None:
            return None
        query, params, error = step
        clash = conflict(kind, a, b, view, own, others)
        if clash:
            RULE_RUNS[f"clash: {clash}"] += 1
        if clash in GATES:
            if clash in self.options.skipped:
                return None
            # With the gate lifted: the second writer of a new key must be refused, and otherwise
            # the model says what should have happened.
            clash = DUPLICATE_KEY if clash == GATED_INSERT else None
        # A first insert into a changed table conflicts before it finds the key in the index.
        return query, params, (clash or error) if kind == "duplicate" else (error or clash)

    def note_commit(self, writes: Writes) -> None:
        """A commit: the open write transactions overlap it, and it waits for them if the
        checkpoint threshold is on."""
        self.commit_number += 1
        if self.session_writer() is None:
            self.commit_log = []
            return
        self.commit_log.append((self.commit_number, writes))
        if self.config["checkpoint_threshold"] != -1:
            self.slow_waits -= 1

    def commit_txn(self, txn: Txn) -> None:
        """The model after `txn` committed."""
        if not self.multi_writes:
            self.model = txn.view
            return
        for kind, a, b, seen in txn.ops:
            replay(self.model, kind, a, b, seen)
        # The engine numbers macros in creation order and WAL replay in commit order, so a DROP
        # MACRO after overlapping creates can drop another macro on recovery (rollback_macro).
        # Whether the other writer also creates one is not tracked; any overlap counts.
        overlapped = self.session_writer() is not None or self.commit_log
        created = any(kind == "create_macro" for kind, *_ in txn.ops)
        self.model.rolled_back_macro |= bool(overlapped and created)
        self.note_commit(txn.writes)

    def snapshots_safe(self) -> bool:
        """A checkpoint breaks open snapshots (#1184), and with automatic checkpoints any commit
        may run one."""
        return (
            "snapshot_checkpoint" not in self.options.skipped
            or self.config["checkpoint_threshold"] == -1
        )

    def seen_by_all_readers(self) -> set:
        """The persons every open read-only transaction sees."""
        views = [txn.view.persons for txn in self.txns.values() if txn.kind == "read"]
        return set.intersection(*map(set, views)) if views else set(self.model.persons)

    def no_pending_update(self) -> bool:
        """A checkpoint while a snapshot is open moves updates made since the last checkpoint to
        other rows (snapshot_update), also ones committed before the snapshot began or by a write
        transaction still open."""
        pending = self.model.updated_since_checkpoint or any(t.updated for t in self.txns.values())
        return "snapshot_update" not in self.options.skipped or not pending

    @enabled("sessions", writes=False)
    @precondition(
        lambda self: self.free_sessions() and self.snapshots_safe() and self.no_pending_update()
    )
    @rule(data=st.data())
    def begin_read(self, data):
        conn = data.draw(st.sampled_from(self.free_sessions()))
        self.ok("BEGIN TRANSACTION READ ONLY", conn=conn)
        self.txns[conn] = Txn("read", self.model.fork())

    @enabled("sessions", writes=False)
    @precondition(
        lambda self: self.free_sessions() and self.column_gap_safe() and self.may_begin_write()
    )
    @rule(data=st.data())
    def begin_write(self, data):
        conn = data.draw(st.sampled_from(self.free_sessions()))
        self.ok("BEGIN TRANSACTION", conn=conn)
        self.txns[conn] = Txn("write", self.model.fork(), start=self.commit_number)

    @enabled("sessions", writes=False)
    @precondition(lambda self: self.session_writer() is not None)
    @rule(data=st.data(), write=op)
    def session_write(self, data, write):
        """A write in an open write transaction; one that fails rolls the transaction back."""
        self.write_in_txn(data.draw(st.sampled_from(self.session_writers())), write)

    def write_in_txn(self, conn: int, write) -> None:
        txn = self.txns[conn]
        kind, a, b = write
        step = self.overlapping_step(txn.view, txn.writes, self.overlapping(conn), write)
        if step is None:
            return
        query, params, error = step
        if error is None:
            self.ok(query, params, conn)
            txn.updated |= kind == "set_age" and a in self.model.persons
            record(txn.writes, kind, a, b, txn.view)
            txn.ops.append((kind, a, b, seen_by(txn.view, kind, a, b)))
            apply(txn.view, kind, a, b)
            return
        self.fails(query, params, error, conn)
        del self.txns[conn]
        self.rolled_back(txn)

    # The two rules below write the id into the query: the driver caches a parameterized
    # statement even when preparing it failed, and would keep returning that error.

    @enabled("sessions", writes=False)
    @precondition(lambda self: self.readers())
    @rule(data=st.data(), id=small_ids)
    def write_in_read_only(self, data, id):
        """Fails without ending the transaction."""
        conn = data.draw(st.sampled_from(self.readers()))
        self.fails(f"CREATE (:Person {{id: {id}}})", None, READ_ONLY, conn)

    @enabled("writers", writes=False)
    @precondition(
        lambda self: (
            self.multi_writes and self.session_writer() is not None and self.overlap_allowed()
        )
    )
    @rule(data=st.data(), write=op)
    def commit_beside_writer(self, data, write):
        """An auto-commit write beside open write transactions, which they must not see."""
        self.auto_commit(data.draw(st.sampled_from([0, *self.free_sessions()])), write)

    def auto_commit(self, conn: int, write) -> None:
        kind, a, b = write
        own = Writes()
        step = self.overlapping_step(self.model, own, self.overlapping(), write)
        if step is None:
            return
        query, params, error = step
        if error is not None:
            self.fails(query, params, error, conn)
            return
        self.ok(query, params, conn)
        record(own, kind, a, b, self.model)
        replay(self.model, kind, a, b, seen_by(self.model, kind, a, b))
        self.note_commit(own)

    def check_txn_sees(self, conn: int) -> None:
        """What the connection's transaction sees of the tables overlapping writers write to."""
        view = self.txns[conn].view
        persons = self.person_rows(view, conn)
        assert persons == view.person_rows(), (conn, persons, view.person_rows())
        query = "MATCH (a:Person)-[k:Knows]->(b:Person) RETURN a.id, b.id, k.since"
        knows = Counter(self.rows(query, None, conn))
        assert knows == view.knows, (conn, knows, view.knows)
        if self.has_csr():
            query = "MATCH (a:C)-[k:CK]->(b:C) WHERE a.id < 100 RETURN a.id, b.id"
            edges = Counter(self.rows(query, None, conn))
            expected = Counter({key: n for key, n in view.ck.items() if key[0] < 100})
            assert edges == expected, (conn, edges, expected)
        for table, values in view.tables.items():
            rows = self.rows(f"MATCH (t:T{table}) RETURN t.id, t.v", None, conn)
            actual = Counter(v for _, v in rows)
            assert actual == Counter(values), (conn, table, actual, values)

    @enabled("writers", writes=False)
    @precondition(lambda self: self.multi_writes and self.column_gap_safe())
    @rule(writers=st.integers(2, 3), rounds=st.lists(overlap_round, min_size=1, max_size=3))
    def overlapping_writers(self, writers, rounds):
        """Write transactions that overlap on a few rows and catalog entries, with auto-commit
        writes beside them: each statement and commit must succeed or conflict as writesets.py
        says, and each transaction must see only its snapshot and its own writes."""
        for overlap in rounds:
            self.stop_slow_waits()
            self.overlap_round(writers, **overlap)

    def stop_slow_waits(self) -> None:
        """No commit of this process checkpoints once the workload has paid for its waits; a reopen
        is back at the default threshold, as for a workload that drew -1."""
        if not self.overlap_allowed():
            self.ok(f"CALL checkpoint_threshold={NO_AUTOMATIC_CHECKPOINT}")
            self.config = {**self.config, "checkpoint_threshold": -1}

    def overlap_round(self, writers, writes, first_ends, first_commits, end) -> None:
        for conn in sorted(self.txns)[: max(0, writers - len(self.free_sessions()))]:
            self.end_txn(conn, True)
        self.seed_overlap()
        conns = self.free_sessions()[:writers]
        for conn in conns:
            self.ok("BEGIN TRANSACTION", conn=conn)
            self.txns[conn] = Txn("write", self.model.fork(), start=self.commit_number)
        for position, (writer, write) in enumerate(writes):
            if position == first_ends and conns[0] in self.txns:
                self.end_txn(conns[0], first_commits)
                self.read_in_open_txns(conns)
            live = [conn for conn in conns if conn in self.txns]
            if writer != AUTO_COMMIT and live:
                self.write_in_txn(live[writer % len(live)], write)
            elif writer == AUTO_COMMIT and self.overlap_allowed():
                self.auto_commit(0, write)
        self.read_in_open_txns(conns)
        for conn in conns:
            if conn in self.txns and end in ("commit", "rollback"):
                self.end_txn(conn, end == "commit")
        if end == "crash":
            self.crash_engine()
        elif end == "reopen":
            self.reopen_engine()

    def seed_overlap(self) -> None:
        """What the round's writes conflict over, unless a transaction outside it would clash."""
        for id in HOT_PERSONS:
            if id not in self.model.persons:
                self.auto_commit(0, ("insert", id, id))
        if HOT_TABLE not in self.model.tables:
            self.auto_commit(0, ("create_table", HOT_TABLE, None))
        src, dst = HOT_EDGE
        if self.has_csr() and not self.model.ck[(src, dst)]:
            self.auto_commit(0, ("ck", src, dst))
        if self.has_csr() and not edge_count(self.model, "ck2", src, dst):
            self.auto_commit(0, ("ck2", src, dst))

    def read_in_open_txns(self, conns: list[int]) -> None:
        for conn in conns:
            if conn in self.txns:
                self.check_txn_sees(conn)

    def end_txn(self, conn: int, commit: bool) -> None:
        txn = self.txns.pop(conn)
        self.ok("COMMIT" if commit else "ROLLBACK", conn=conn)
        if txn.kind == "write" and commit:
            self.commit_txn(txn)
        elif txn.kind == "write":
            self.rolled_back(txn)

    @enabled("writers_checkpoint", writes=False)
    @precondition(
        lambda self: self.multi_writes and self.session_writer() is not None and self.slow_waits > 0
    )
    @rule()
    def checkpoint_beside_writer(self):
        """CHECKPOINT gives up waiting for an open write transaction, and changes nothing."""
        self.slow_waits -= 1
        self.fails("CHECKPOINT", None, CHECKPOINT_BLOCKED)

    @enabled("copy_rollback", writes=False)
    @precondition(
        lambda self: (
            self.multi_writes
            and (self.session_writer() is not None or self.free_sessions())
            and self.slow_waits > 0
            and self.copy_person_safe()
            and self.model.pk_index is not None
        )
    )
    @rule(ids=st.integers(1, 20))
    def copy_beside_writer(self, ids):
        """A COPY needs a checkpoint, so it waits for an open write transaction, gives up and
        rolls back; nothing of it stays, the next insert gets its own row, and the open
        transactions are not disturbed."""
        if any(PERSON in w.catalog for w in self.overlapping().all()):
            return  # the COPY would fail on the conflict instead
        if self.session_writer() is None:
            self.seed_overlap()
            conn = self.free_sessions()[0]
            self.ok("BEGIN TRANSACTION", conn=conn)
            self.txns[conn] = Txn("write", self.model.fork(), start=self.commit_number)
        rows = [("bulk", i) for i in range(ids)]
        csv, _ = self.write_bulk_csv(rows)
        self.slow_waits -= 1
        self.fails(copy_query(csv, rows), None, COPY_BLOCKED)
        self.model.persons_ever = True
        self.auto_commit(0, ("insert", self.model.take_bulk_ids(1).start, 1))

    @enabled("sessions", writes=False)
    @precondition(lambda self: not self.multi_writes and self.session_writer() is not None)
    @rule(data=st.data(), id=small_ids)
    def write_beside_writer(self, data, id):
        """An auto-commit write on another connection fails while a write transaction is open."""
        conn = data.draw(st.sampled_from([0, *self.free_sessions()]))
        self.fails(f"CREATE (:Person {{id: {id}}})", None, ONE_WRITER, conn)

    @enabled("second_writer", writes=False)
    @precondition(
        lambda self: (
            not self.multi_writes and self.session_writer() is not None and self.free_sessions()
        )
    )
    @rule(data=st.data(), then=st.sampled_from(["read", "write", "read_only", "commit"]))
    def second_writer(self, data, then):
        """A refused BEGIN leaves the connection without a transaction: its next statement runs
        on its own, a write fails beside the writer, a read-only BEGIN opens a snapshot and
        COMMIT finds nothing to commit."""
        conn = data.draw(st.sampled_from(self.free_sessions()))
        self.fails("BEGIN TRANSACTION", None, ONE_WRITER, conn)
        if then == "write":
            self.fails(f"CREATE (:Person {{id: {data.draw(small_ids)}}})", None, ONE_WRITER, conn)
        elif then == "read_only" and self.snapshots_safe() and self.no_pending_update():
            self.ok("BEGIN TRANSACTION READ ONLY", conn=conn)
            self.txns[conn] = Txn("read", self.model.fork())
        elif then == "commit":
            self.fails("COMMIT", None, NO_TRANSACTION, conn)
        persons = [(len(self.model.persons),)]
        self.check("MATCH (p:Person) RETURN count(*)", {}, persons, conn=conn)

    @enabled("sessions", writes=False)
    @rule(
        data=st.data(),
        x=st.integers(-200, 200),
        id=small_ids,
        limit=st.integers(1, 20),
        cluster=clusters,
        optimizer=st.booleans(),
    )
    def session_read(self, data, x, id, limit, cluster, optimizer):
        """The query battery on a session connection, against what it should see; mostly one
        that is in a transaction, where the views differ."""
        busy = sorted(self.txns)
        conn = data.draw(
            st.one_of(st.sampled_from(busy), st.sampled_from(busy), st.sampled_from(SESSIONS))
            if busy
            else st.sampled_from(SESSIONS)
        )
        view = self.txns[conn].view if conn in self.txns else self.model
        if conn in self.txns and not self.lookups_safe(self.txns[conn]):
            return
        with self.optimizer(optimizer, conn):
            self.check_view(view, conn, x, id, limit, cluster)

    @enabled("checkpoint")
    @precondition(lambda self: self.checkpoint_keeps_snapshots())
    @rule(data=st.data())
    def session_checkpoint(self, data):
        self.ok("CHECKPOINT", conn=data.draw(st.sampled_from([0, *self.free_sessions()])))
        self.model.checkpointed()

    @enabled("sessions", writes=False)
    @precondition(lambda self: self.txns)
    @rule(data=st.data(), commit=st.booleans())
    def end_session(self, data, commit):
        self.end_txn(data.draw(st.sampled_from(sorted(self.txns))), commit)

    @enabled("pool_commit", writes=False)
    @precondition(lambda self: self.session_writer() is not None)
    @rule(data=st.data())
    def commit_under_pressure(self, data):
        """COMMIT while another connection holds the buffer pool full: the transaction is either
        committed (also when only the checkpoint after it failed) or rolled back, and nothing
        else changes."""
        conn = self.session_writer()
        holder = data.draw(st.sampled_from([0, *self.free_sessions()]))
        txn = self.txns.pop(conn)
        self.hold_results(holder)
        try:
            self.ok("COMMIT", conn=conn)
            committed = True
        except EngineError as e:
            self.trace[-1]["expected"] = True
            committed = COMMITTED_BUT in str(e)
            assert committed or POOL_FULL in str(e), e
        self.release_results()
        if committed:
            self.commit_txn(txn)
        else:
            self.rolled_back(txn)

    @enabled("sessions")
    @precondition(lambda self: self.readers() and self.model.persons)
    @rule(data=st.data(), kind=st.sampled_from(["knows", "knows", "insert", "set_age", "delete"]))
    def commit_beside_snapshot(self, data, kind):
        """An auto-commit write while a snapshot is open, between existing persons, which the
        snapshot must not see."""
        existing = st.sampled_from(sorted(self.model.persons))
        target = near_boundaries(list(self.model.persons))
        hidden = sorted(set(self.model.persons) - self.seen_by_all_readers())
        if kind == "delete" and hidden:
            # A row a snapshot must not see, the case retention of deletes gets wrong.
            target = st.one_of(target, st.sampled_from(hidden))
        a = data.draw(small_ids if kind == "insert" else target)
        b = data.draw(existing if kind == "knows" else st.integers(-100, 100))
        if not self.snapshot_write_safe(kind, a):
            return
        step = statement(self.model, kind, a, b)
        if step is None:
            return
        query, params, _ = step
        self.ok(query, params)
        apply(self.model, kind, a, b)
