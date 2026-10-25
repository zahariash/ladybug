"""Rules for failures around commits and checkpoints, and what recovery makes of them.

A CHECKPOINT taken while a read-only transaction is open keeps the WAL it applied (a pinned
checkpoint); pinned_checkpoint crashes before, during or instead of the checkpoint after one.
storage_limit fails writes past a file size, as a full disk does, or past a database size cap,
and commit_pressure_crash commits while the buffer pool is full. Each ends in a crash or a
reopen, after which every write that succeeded must be there and every one that failed must not,
whatever the engine had applied before failing. reopen_checksums opens a crashed database with
the other checksum setting, which must refuse a WAL written with the first rather than drop it.
graph_write writes GN, the named graph's table, which the data and crash checks read with the
rest.
"""

from __future__ import annotations

import os
import shutil
from collections.abc import Callable
from dataclasses import dataclass, field

import hypothesis.strategies as st
from engine import EngineError
from hypothesis.stateful import precondition, rule
from model import Model, blob_lengths, blob_query
from rules.durability import SYSCALLS
from rules.sessions import COMMITTED_BUT, POOL_FULL, Txn
from rules.transactions import apply, op, statement
from session import ONE_THREAD, Session, diff, enabled, gated
from strategies import small_ids

gated("limit_files", "limit_db_size", "limit_partitions", "limit_close")
# What a failed statement's error says about its transaction, besides that it failed.
FORCED_CHECKPOINT_FAILED = "but the checkpoint it requires failed"
APPLIED_ON_REOPEN = "applied when the database reopens"
POISONED = "refuses further writes"
DB_SIZE_CAP = "Maximum database size"
CHECKSUMS = "enableChecksums"
# What a write or checkpoint that a storage limit fails may say; anything else is a finding.
STORAGE_ERRORS = (
    "IO exception",
    "File too large",
    "Buffer manager exception",
    "Checkpoint rollback failed",
    POISONED,
    COMMITTED_BUT,
    FORCED_CHECKPOINT_FAILED,
    APPLIED_ON_REOPEN,
)
# max_db_size is a power of two of at least two page groups of 1024 pages.
MIN_DB_SIZE = 8 * 1024 * 1024
# Blob stops growing here, since the data checks read every row after every step.
BLOB_ROWS_CAP = 5000

GRAPH_OPS = ("gn_insert", "gn_set", "gn_delete")
graph_op = st.tuples(st.sampled_from(GRAPH_OPS), small_ids, st.integers(-100, 100))
# Auto-commit writes: the transaction ops without the column DDL, whose model flags describe a
# transaction's view, and the named graph's.
AUTO_KINDS = {"insert", "set_age", "delete", "knows", "duplicate", "insert_t"}
AUTO_KINDS |= {"ck", "unck", "ck2", "unck2", "create_table", "drop_table"}
write_op = st.one_of(op, graph_op)
# storage_limit's steps: writes, Blob rows of long strings that grow the data file at the next
# checkpoint, CHECKPOINT, and COPY, whose checkpoint is part of the statement.
limited_op = st.one_of(
    write_op,
    st.tuples(st.just("blob"), st.integers(1, 500), st.integers(10, 4000)),
    st.tuples(st.just("checkpoint"), st.none(), st.none()),
    st.tuples(st.just("copy"), st.integers(1, 3000), st.none()),
)


def graph_statement(model: Model, kind: str, a: int, b: int) -> str | None:
    """The statement for a graph op, with literals: the driver would keep a parameterized one
    prepared across USE GRAPH."""
    if kind == "gn_insert":
        return None if a in model.graph else f"CREATE (:GN {{id: {a}, v: {b}}})"
    if kind == "gn_set":
        return f"MATCH (n:GN {{id: {a}}}) SET n.v = {b}"
    return f"MATCH (n:GN {{id: {a}}}) DELETE n"


def graph_apply(model: Model, kind: str, a: int, b: int) -> None:
    if kind == "gn_insert":
        model.graph[a] = b
    elif kind == "gn_set" and a in model.graph:
        model.graph[a] = b
        model.updated_since_checkpoint = True
    elif kind == "gn_delete":
        model.graph.pop(a, None)


def write_outcome(error: str | None, copy: bool) -> str:
    """What happened to a write, from its error: "applied" (committed), "doubtful" (committed,
    but lost on a crash unless a later checkpoint succeeds), "on_reopen" (rolled back, but
    recovery applies it), "failed", or "auto_checkpoint": committed, and failed only because
    the automatic checkpoint after its commit failed, which must not fail the statement. A COPY
    needs its checkpoint, and the old message for it says the data is durable."""
    if error is None:
        return "applied"
    if COMMITTED_BUT in error:
        return "applied" if copy else "auto_checkpoint"
    if FORCED_CHECKPOINT_FAILED in error:
        return "doubtful"
    if APPLIED_ON_REOPEN in error:
        return "on_reopen"
    return "failed"


def capped_size(size: int) -> int:
    """The smallest max_db_size the engine takes that holds `size` bytes."""
    return max(MIN_DB_SIZE, 1 << (size - 1).bit_length())


@dataclass
class Write:
    run: Callable[[], None]  # runs the write on connection 0, outside a transaction
    change: Callable[[Model], None]  # applies it to a model
    error: str | None = None  # the error it must fail with when nothing else fails
    ids: set = field(default_factory=set)  # the persons a COPY adds


@dataclass
class Limited:
    """What storage_limit learned so far besides the model."""

    on_reopen: Model | None = None  # the data after recovery applies a rolled-back transaction
    doubt: set = field(default_factory=set)  # persons of a COPY no checkpoint has made durable
    poisoned: bool = False  # writes fail until a restart


class RecoveryRules(Session):
    def attempt(self, call: Callable[[], None]) -> str | None:
        """Runs `call`; returns the error of the statement that failed, marked as expected, or
        None."""
        start = len(self.trace)
        try:
            call()
        except EngineError as e:
            for step in self.trace[start:]:
                if step.get("outcome", "ok") != "ok":
                    step["expected"] = True
            return str(e)
        return None

    def graph_write_safe(self, kind: str, id: int) -> bool:
        """An update of an existing GN row meets snapshot_update like a Person update does."""
        readers = any(txn.kind == "read" for txn in self.txns.values())
        updates = kind == "gn_set" and id in self.model.graph
        return not (updates and readers and "snapshot_update" in self.options.skipped)

    def plan_write(self, kind: str, a, b) -> Write | None:
        """An op as a Write, or None when it does not run now."""
        if kind in GRAPH_OPS:
            if self.model.graph is None or not self.graph_write_safe(kind, a):
                return None
            query = graph_statement(self.model, kind, a, b)
            if query is None:
                return None
            return Write(
                lambda: self.in_graph(lambda: self.ok(query)),
                lambda m: graph_apply(m, kind, a, b),
            )
        if kind == "blob":
            if len(self.model.blobs) >= BLOB_ROWS_CAP:
                return None
            ids = self.model.take_blob_ids(a)
            lengths = blob_lengths(ids, b)
            return Write(lambda: self.ok(blob_query(ids, b)), lambda m: m.blobs.update(lengths))
        if kind == "copy":
            unsafe_pk = self.model.pk_index is None and self.model.persons_ever
            if unsafe_pk or not self.copy_person_safe():
                return None
            csv, added = self.write_bulk_csv([("bulk", i % 1000) for i in range(a)])
            return Write(
                lambda: self.ok(f'COPY Person FROM "{csv}"'),
                lambda m: m.add_persons(added),
                ids=set(added),
            )
        if kind not in AUTO_KINDS or kind not in self.allowed_ops():
            return None
        if not self.snapshot_write_safe(kind, a):
            return None
        step = statement(self.model, kind, a, b)
        if step is None:
            return None
        query, params, error = step
        return Write(lambda: self.ok(query, params), lambda m: apply(m, kind, a, b), error)

    def write(self, kind: str, a, b) -> None:
        """Runs an op that must succeed, or fail with its expected error."""
        if (write := self.plan_write(kind, a, b)) is None:
            return
        if write.error is None:
            write.run()
            write.change(self.model)
            return
        failure = self.attempt(write.run)
        assert failure is not None, f"expected an error with {write.error!r}"
        assert write.error in failure, f"failed with {failure!r}, expected {write.error!r}"

    def open_reader(self, data) -> int | None:
        """Opens a read-only transaction on a free session connection, if one may open now."""
        if not (self.free_sessions() and self.snapshots_safe() and self.no_pending_update()):
            return None
        conn = data.draw(st.sampled_from(self.free_sessions()))
        self.ok("BEGIN TRANSACTION READ ONLY", conn=conn)
        self.txns[conn] = Txn("read", self.model.fork())
        return conn

    def check_recovered(self, *models: Model) -> None:
        """After a restart the data is that of one of `models` (by default the model), which
        then becomes the model."""
        if getattr(self, "started_over", False):
            return
        models = models or (self.model,)
        actual = self.crash_state()
        for model in models:
            if actual == self.crash_state(model):
                self.model = model
                return
        expected = self.crash_state(models[0])
        raise AssertionError(
            f"after recovery, persons {diff(actual[0], expected[0])}; C rows and CK edges "
            f"{actual[1]}, expected {expected[1]}; graph {diff(actual[2], expected[2])}; "
            f"blobs {diff(actual[3], expected[3])}"
        )

    @enabled("graphs")
    @precondition(lambda self: self.model.graph is not None)
    @rule(
        ops=st.lists(graph_op, min_size=1, max_size=8),
        end=st.sampled_from([None, "commit", "rollback"]),
    )
    def graph_write(self, ops, end):
        """Writes GN in the named graph, a commit per write or in one transaction; the named
        graph's commits go to the main WAL."""
        if end is None:
            for kind, a, b in ops:
                self.write(kind, a, b)
            return
        pending = self.model.fork()

        def transaction():
            self.ok("BEGIN TRANSACTION")
            for kind, a, b in ops:
                if not self.graph_write_safe(kind, a):
                    continue
                if (query := graph_statement(pending, kind, a, b)) is not None:
                    self.ok(query)
                    graph_apply(pending, kind, a, b)
            self.ok(end.upper())

        self.in_graph(transaction)
        if end == "commit":
            self.model = pending

    @enabled("pinned_checkpoint")
    @precondition(
        lambda self: (
            self.free_sessions()
            and self.snapshots_safe()
            and self.no_pending_update()
            and "snapshot_checkpoint" not in self.options.skipped
        )
    )
    @rule(
        before=st.lists(write_op, max_size=4),
        after=st.lists(write_op, max_size=6),
        close_reader=st.booleans(),
        end=st.sampled_from(["crash", "kill_at", "kill_at", "reopen", "full_close"]),
        data=st.data(),
    )
    def pinned_checkpoint(self, before, after, close_reader, end, data):
        """Writes around a CHECKPOINT that a read-only transaction pins, which keeps the WAL it
        applied, so that the commits after it follow its record in the same WAL. Then a crash, a
        kill during the next checkpoint (at a file syscall), a reopen, or a close whose
        checkpoint fails on a full disk: every commit must survive."""
        conn = self.open_reader(data)
        for kind, a, b in before:
            self.write(kind, a, b)
        self.ok("CHECKPOINT")
        self.model.checkpointed()
        if close_reader:
            self.ok("COMMIT", conn=conn)
            del self.txns[conn]
        for kind, a, b in after:
            self.write(kind, a, b)
        if end == "kill_at" and shutil.which("strace"):
            # A small checkpoint makes few calls; later ones would kill it after it finished.
            syscall = data.draw(st.sampled_from(sorted(SYSCALLS)))
            n = data.draw(st.integers(1, min(8, SYSCALLS[syscall])))
            # One thread makes the n-th call the same call in every run.
            self.ok(ONE_THREAD)
            self.kill_at("CHECKPOINT", syscall, n)
        elif end == "reopen":
            self.reopen_engine()
        elif end == "full_close" and self.files_limit_safe(closes=True):
            # Writes past the WAL's size fail, so the checkpoint that closing runs fails too.
            self.limit_files(self.wal_size())
            self.reopen_engine(clean=False)
        else:
            self.crash_engine()
        self.check_recovered()

    @enabled("storage_limit")
    @precondition(lambda self: self.has_blobs())
    @rule(
        cap=st.sampled_from(["files", "wal", "db_size"]),
        slack=st.sampled_from([0, 4096, 1 << 16, 1 << 20, 1 << 23]),
        ops=st.lists(limited_op, min_size=1, max_size=10),
        pin=st.booleans(),
        end=st.sampled_from(["lift", "crash", "close"]),
        data=st.data(),
    )
    def storage_limit(self, cap, slack, ops, pin, end, data):
        """Writes, checkpoints and COPY against a full disk (every file limited to the data
        file's size plus `slack`), a file limit just above the WAL's size, which also fails the
        checkpoint writes past it, or a database size cap just above the data file, optionally
        with a reader that pins the checkpoints. Then the limit is lifted and a CHECKPOINT must
        succeed, or a crash, or a close whose checkpoint may fail. A write that succeeded must
        survive and one that failed must not; the automatic checkpoint after a commit must not
        fail it. A checkpoint reuses free pages before it grows the file, so a full disk fails
        only the writes that need more than the free space."""
        self.started_over = False
        size = os.path.getsize(self.path) + slack
        if cap == "db_size":
            if "limit_db_size" in self.options.skipped or not self.reopen_capped(size):
                return
        elif not self.files_limit_safe(closes=False):
            return
        elif cap == "wal":
            self.limit_files(self.wal_size() + slack)
        else:
            self.limit_files(size)
        if pin:
            self.open_reader(data)
        state = Limited()
        for kind, a, b in ops:
            if kind == "checkpoint":
                self.limited_checkpoint(state)
            elif (write := self.plan_write(kind, a, b)) is not None:
                self.limited_write(write, kind == "copy", state)
        if state.poisoned or state.on_reopen is not None:
            end = "crash" if end == "lift" else end
        if end == "close" and cap != "db_size" and not self.files_limit_safe(closes=True):
            end = "crash"
        if end == "lift":
            if cap == "files":
                self.limit_files(None)
            else:
                self.reopen_engine(clean=False)
            if self.checkpoint_keeps_snapshots():
                self.ok("CHECKPOINT")
                self.model.checkpointed()
                state.doubt.clear()
        elif end == "crash":
            self.crash_engine()
        else:
            self.reopen_engine(clean=False)
        recovered = [state.on_reopen or self.model]
        if state.doubt:
            recovered.append(recovered[0].fork())
            for id in state.doubt:
                recovered[1].forget(id)
        self.check_recovered(*recovered)

    def files_limit_safe(self, closes: bool) -> bool:
        """Whether a file limit stays clear of the known full-disk crashes: a checkpoint with a
        partitioned table (limit_partitions) and a close after a refused write (limit_close)."""
        skipped = self.options.skipped
        if "limit_files" in skipped or ("limit_partitions" in skipped and self.has_partitions()):
            return False
        return not (closes and "limit_close" in skipped)

    def wal_size(self) -> int:
        wal = f"{self.path}.wal"
        return os.path.getsize(wal) if os.path.exists(wal) else 0

    def reopen_capped(self, size: int) -> bool:
        """Reopens the database under the smallest size cap that holds `size` bytes; false when
        it does not open under it, after opening it again without."""
        try:
            self.reopen_engine({"max_db_size": capped_size(size)})
        except EngineError as e:
            self.trace[-1]["expected"] = True
            assert DB_SIZE_CAP in str(e), e
            self.open_again()
            return False
        return True

    def limited_checkpoint(self, state: Limited) -> None:
        if not self.checkpoint_keeps_snapshots():
            return
        failure = self.attempt(lambda: self.ok("CHECKPOINT"))
        if failure is None:
            self.model.checkpointed()
            state.doubt.clear()
            return
        assert any(error in failure for error in STORAGE_ERRORS), failure
        if POISONED in failure:
            self.after_poison(state)

    def limited_write(self, write: Write, copy: bool, state: Limited) -> None:
        failure = self.attempt(write.run)
        outcome = write_outcome(failure, copy)
        assert outcome != "auto_checkpoint", (
            f"a write failed for the automatic checkpoint after its commit: {failure}"
        )
        if outcome != "failed":
            assert write.error is None, f"expected an error with {write.error!r}"
        elif not (write.error and write.error in failure):
            assert any(error in failure for error in STORAGE_ERRORS), failure
        if copy:
            self.model.persons_ever = True  # a COPY reserves rows even when it fails
        if outcome in ("applied", "doubtful"):
            write.change(self.model)
            if state.on_reopen is not None:
                write.change(state.on_reopen)
        if outcome == "doubtful":
            state.doubt |= write.ids
        elif outcome == "on_reopen":
            state.on_reopen = state.on_reopen or self.model.fork()
            write.change(state.on_reopen)
            self.after_poison(state)
        elif outcome == "failed" and POISONED in failure:
            self.after_poison(state)

    def after_poison(self, state: Limited) -> None:
        """Once writes are refused, a refused BEGIN must leave the connection usable."""
        if state.poisoned:
            return
        state.poisoned = True
        if "second_writer" not in self.options.skipped:
            self.fails("BEGIN TRANSACTION", None, POISONED)
            self.ok("RETURN 1")

    @enabled("pool_commit_crash")
    @precondition(lambda self: len(self.free_sessions()) >= 1)
    @rule(
        ops=st.lists(op, min_size=1, max_size=6),
        end=st.sampled_from(["crash", "reopen"]),
        data=st.data(),
    )
    def commit_pressure_crash(self, ops, end, data):
        """COMMIT while another connection holds the buffer pool full, then a crash or a reopen:
        recovery must agree with what COMMIT reported, also when applying the committed
        transaction failed after its commit record was written."""
        conn = data.draw(st.sampled_from(self.free_sessions()))
        holder = data.draw(st.sampled_from([0, *(c for c in self.free_sessions() if c != conn)]))
        pending = self.model.fork()
        self.ok("BEGIN TRANSACTION", conn=conn)
        for kind, a, b in ops:
            if not self.runs(pending, kind, a):
                continue
            step = statement(pending, kind, a, b)
            if step is None or step[2] is not None:
                continue
            query, params, _ = step
            self.ok(query, params, conn)
            apply(pending, kind, a, b)
        self.hold_results(holder)
        failure = self.attempt(lambda: self.ok("COMMIT", conn=conn))
        self.release_results()
        outcome = write_outcome(failure, copy=False)
        assert outcome != "auto_checkpoint", (
            f"COMMIT failed for the automatic checkpoint after it: {failure}"
        )
        assert outcome in ("applied", "on_reopen") or POOL_FULL in failure, failure
        self.model.discarded(pending)
        recovered = pending if outcome in ("applied", "on_reopen") else self.model
        if outcome == "applied":
            self.model = pending
        if end == "crash":
            self.crash_engine()
        else:
            self.reopen_engine(clean=False)
        self.check_recovered(recovered)

    @enabled("checksums")
    @rule()
    def reopen_checksums(self):
        """A crash, then an open with the other checksum setting, which must refuse a WAL
        written with the workload's rather than drop it; then an open with the right one."""
        other = not self.config["enable_checksums"]
        try:
            self.crash_engine({"enable_checksums": other})
        except EngineError as e:
            self.trace[-1]["expected"] = True
            assert CHECKSUMS in str(e), e
            self.open_again()
        else:
            # The worker writes its own setting into the next WAL; closing retires it.
            self.check_recovered()
            self.reopen_engine()
        self.check_recovered()
