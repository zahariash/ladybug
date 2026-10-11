"""Rules that interleave transactions on several connections, one step per rule.

Connections 2-4 hold the sessions. A read-only transaction sees the committed state as of its
BEGIN for its whole life, a write transaction sees that plus its own writes, and a connection
without a transaction sees the committed state; nobody sees another transaction's uncommitted
writes. Only one write transaction can be open: a second BEGIN or an auto-commit write elsewhere
fails. Every step is a rule of its own, so a failure shrinks to the few steps that matter.
"""

from __future__ import annotations

from dataclasses import dataclass

import hypothesis.strategies as st
from engine import EngineError
from hypothesis.stateful import precondition, rule
from model import Model
from rules.transactions import apply, op, statement
from session import Session, enabled, gated
from strategies import clusters, small_ids

SESSIONS = (2, 3, 4)
READ_ONLY = "Can not execute a write query inside a read-only transaction"
ONE_WRITER = "Only one write transaction at a time"
POOL_FULL = "Buffer manager exception"
# The commit succeeded; the automatic checkpoint after it did not.
COMMITTED_BUT = "Transaction committed successfully, but the post-commit checkpoint failed"
gated("snapshot_checkpoint", "snapshot_rel_count", "snapshot_prepared")


@dataclass
class Txn:
    kind: str  # "read" or "write"
    view: Model  # what the transaction sees


class SessionRules(Session):
    def free_sessions(self) -> list[int]:
        return [conn for conn in SESSIONS if conn not in self.txns]

    def readers(self) -> list[int]:
        return [conn for conn, txn in self.txns.items() if txn.kind == "read"]

    def snapshots_safe(self) -> bool:
        """A checkpoint breaks open snapshots (#1184), and with automatic checkpoints any commit
        may run one."""
        return (
            "snapshot_checkpoint" not in self.options.skipped
            or self.config["checkpoint_threshold"] == -1
        )

    @enabled("sessions", writes=False)
    @precondition(lambda self: self.free_sessions() and self.snapshots_safe())
    @rule(data=st.data())
    def begin_read(self, data):
        conn = data.draw(st.sampled_from(self.free_sessions()))
        self.ok("BEGIN TRANSACTION READ ONLY", conn=conn)
        self.txns[conn] = Txn("read", self.model.fork())

    @enabled("sessions")
    @precondition(lambda self: self.free_sessions() and self.column_gap_safe())
    @rule(data=st.data())
    def begin_write(self, data):
        conn = data.draw(st.sampled_from(self.free_sessions()))
        self.ok("BEGIN TRANSACTION", conn=conn)
        self.txns[conn] = Txn("write", self.model.fork())

    @enabled("sessions", writes=False)
    @precondition(lambda self: self.session_writer() is not None)
    @rule(write=op)
    def session_write(self, write):
        """A write in the open write transaction; one that fails rolls the transaction back."""
        conn = self.session_writer()
        txn = self.txns[conn]
        kind, a, b = write
        if not self.runs(txn.view, kind, a):
            return
        step = statement(txn.view, kind, a, b)
        if step is None:
            return
        query, params, error = step
        if error is None:
            self.ok(query, params, conn)
            apply(txn.view, kind, a, b)
            return
        self.fails(query, params, error, conn)
        del self.txns[conn]
        self.model.discarded(txn.view)

    # The two rules below write the id into the query: the driver caches a parameterized
    # statement even when preparing it failed, and would keep returning that error.

    @enabled("sessions", writes=False)
    @precondition(lambda self: self.readers())
    @rule(data=st.data(), id=small_ids)
    def write_in_read_only(self, data, id):
        """Fails without ending the transaction."""
        conn = data.draw(st.sampled_from(self.readers()))
        self.fails(f"CREATE (:Person {{id: {id}}})", None, READ_ONLY, conn)

    @enabled("sessions", writes=False)
    @precondition(lambda self: self.session_writer() is not None)
    @rule(data=st.data(), id=small_ids)
    def write_beside_writer(self, data, id):
        """An auto-commit write on another connection fails while a write transaction is open."""
        conn = data.draw(st.sampled_from([0, *self.free_sessions()]))
        self.fails(f"CREATE (:Person {{id: {id}}})", None, ONE_WRITER, conn)

    @enabled("second_writer", writes=False)
    @precondition(lambda self: self.session_writer() is not None and self.free_sessions())
    @rule(data=st.data())
    def second_writer(self, data):
        conn = data.draw(st.sampled_from(self.free_sessions()))
        self.fails("BEGIN TRANSACTION", None, ONE_WRITER, conn)
        self.check("MATCH (p:Person) RETURN count(*)", {}, [(len(self.model.persons),)], conn=conn)

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
        conn = data.draw(st.sampled_from(sorted(self.txns)))
        txn = self.txns.pop(conn)
        self.ok("COMMIT" if commit else "ROLLBACK", conn=conn)
        if txn.kind == "write":
            self.model.discarded(txn.view)
            if commit:
                self.model = txn.view

    @enabled("pool_commit", writes=False)
    @precondition(lambda self: self.session_writer() is not None)
    @rule(data=st.data())
    def commit_under_pressure(self, data):
        """COMMIT while another connection holds the buffer pool full: the transaction is either
        committed (also when only the checkpoint after it failed) or rolled back, and nothing
        else changes."""
        conn = self.session_writer()
        txn = self.txns.pop(conn)
        self.hold_results(data.draw(st.sampled_from([0, *self.free_sessions()])))
        try:
            self.ok("COMMIT", conn=conn)
            committed = True
        except EngineError as e:
            committed = COMMITTED_BUT in str(e)
            assert committed or POOL_FULL in str(e), e
        self.release_results()
        self.model.discarded(txn.view)
        if committed:
            self.model = txn.view

    @enabled("sessions")
    @precondition(lambda self: self.readers() and self.model.persons)
    @rule(data=st.data(), kind=st.sampled_from(["knows", "knows", "insert", "set_age", "delete"]))
    def commit_beside_snapshot(self, data, kind):
        """An auto-commit write while a snapshot is open, between existing persons, which the
        snapshot must not see."""
        existing = st.sampled_from(sorted(self.model.persons))
        a = data.draw(small_ids if kind == "insert" else existing)
        b = data.draw(existing if kind == "knows" else st.integers(-100, 100))
        step = statement(self.model, kind, a, b)
        if step is None:
            return
        query, params, _ = step
        self.ok(query, params)
        apply(self.model, kind, a, b)
