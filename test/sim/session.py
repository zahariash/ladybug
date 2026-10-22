"""The state every rule works on, and the helpers rules use to run statements and checks.

Every interaction with the engine goes through Session, which records it with its outcome in
a trace: the statements, crashes, crash points and races of one workload. replay.py reruns a
trace without the model and minimizes it.
"""

from __future__ import annotations

import json
import os
import sys
import time
from collections import Counter
from dataclasses import dataclass, field

from engine import Engine, EngineError
from hypothesis.stateful import precondition
from model import Model, canonical
from strategies import CSR_ROWS
from triage import probe_of, signature

# Every name rules are gated on, so --sim-skip can reject names that match nothing. Rules add
# theirs when their module is imported.
RULE_NAMES: set[str] = set()
EXTENSIONS = ("fts", "vector")

DUPLICATE_KEY = "duplicated primary key"
# C holds ids 0..rows-1 at the offsets of their ids, which SORTED BY CSR declares. Every open
# declares it again, since the declaration does not survive a reopen; only a new database's first
# open fills C, by COPY (straight into a node group on disk) or by CREATE (in-memory chunks).
C_SORTED = "ALTER TABLE C SET SORTED BY (id ASC) CSR"
# A COPY on several threads stores rows out of input order; the declaration is not checked.
ONE_THREAD = "CALL threads=1"


def copy_c(start: int, count: int) -> str:
    return f"COPY C FROM (UNWIND range({start}, {start + count - 1}) AS i RETURN i, i % 1000)"


def csr_setup(fill: str | None, threads: int) -> list[str]:
    fills = {
        "copy": [ONE_THREAD, copy_c(0, CSR_ROWS), f"CALL threads={threads}"],
        "create": [f"UNWIND range(0, {CSR_ROWS - 1}) AS i CREATE (:C {{id: i, v: i % 1000}})"],
    }
    return [
        "CREATE NODE TABLE IF NOT EXISTS C(id INT64 PRIMARY KEY, v INT64)",
        *(fills[fill] if fill else []),
        "CREATE REL TABLE IF NOT EXISTS CK(FROM C TO C)",
        "CREATE REL TABLE IF NOT EXISTS CK2(FROM C TO C, w INT64)",
        C_SORTED,
    ]


@dataclass(frozen=True)
class Options:
    skipped: frozenset = field(default_factory=frozenset)  # rule names and variants to skip
    extensions: tuple = ()  # statements that load extensions after every open
    keep: bool = False  # keep each database directory after the run
    debug: bool = False  # run workers under gdb, so that crashes report native stacks
    traces: str = ""  # where the current workload's trace and files are written
    scratch: str = ""  # where the databases are created

    @property
    def loaded(self) -> set[str]:
        """The extensions the load statements name."""
        return {name for name in EXTENSIONS if any(name in q.lower() for q in self.extensions)}


def gated(*names: str) -> None:
    """Registers names that rules check against Options.skipped without a decorator."""
    RULE_NAMES.update(names)


def enabled(name: str, writes: bool = True):
    """Runs the rule unless --sim-skip, --sim-focus or known.py turn `name` off. A rule that
    writes waits until no session holds a write transaction, since only one can be open."""
    gated(name)
    return precondition(
        lambda self: (
            self.engine is not None
            and name not in self.options.skipped
            and not (writes and self.session_writer() is not None)
        )
    )


def with_extension(name: str, extension: str | None = None):
    """Like enabled, and only when `extension` (or any extension, for None) is loaded."""
    gated(name)
    return precondition(
        lambda self: (
            self.engine is not None
            and bool(self.options.loaded if extension is None else extension in self.options.loaded)
            and name not in self.options.skipped
            and self.session_writer() is None
        )
    )


def diff(actual: Counter, expected: Counter) -> str:
    missing = list((expected - actual).items())[:5]
    unexpected = list((actual - expected).items())[:5]
    return f"missing={missing} unexpected={unexpected}"


RULE_MARKERS = ("hypothesis_stateful_rule", "hypothesis_stateful_invariant")
# Every failure this process recorded, for the terminal summary: kind, slug, directory, and for a
# wrong result the probe and its wrong rows.
FAILURES: list[tuple] = []
# How often each rule or invariant ran a statement, and how many workloads ended on a known
# recovery failure, for the terminal summary.
RULE_RUNS: Counter = Counter()
RECOVERY_SKIPS: Counter = Counter()
# Recovery failures after a kill that are known and not filed; while their gate is on, a workload
# that hits one is discarded instead of failing, so that crash ops keep running.
RECOVERY_FAILURES = {
    "recovery_duplicate_key": "Found duplicated primary key value",
    "recovery_wal_checksum": "Checksum verification failed",
    "recovery_catalog_exists": "already exists in catalog",
}


def failure_kind(error: BaseException) -> str:
    message = str(error)
    if isinstance(error, EngineError):
        return "error"
    if message.startswith("engine hung"):
        return "hung"
    if message.startswith("engine process died"):
        return "died"
    return "wrong result"


def outcome_of(error: BaseException) -> str:
    if isinstance(error, EngineError):
        return f"error: {error}"
    message = str(error)
    if message.startswith("engine hung"):
        return f"hung: {message}"
    if message.startswith("engine process died"):
        return f"died: {message}"
    return "ok"


class Session:
    options = Options()
    engine: Engine | None = None
    model: Model
    path: str  # the database
    dir: str  # the database's directory
    files: str  # where rules write the files they COPY, kept with the trace
    config: dict
    trace: list
    kept: dict  # names of the prepared statements the current worker keeps
    # Open transactions on the session connections (rules/sessions.py), by connection.
    txns: dict

    def session_writer(self) -> int | None:
        """The session connection holding the write transaction, if any."""
        return next((conn for conn, txn in self.txns.items() if txn.kind == "write"), None)

    def checkpoint_keeps_snapshots(self) -> bool:
        """Whether a CHECKPOINT now leaves the sessions' snapshots alone: a checkpoint breaks
        open read-only transactions (#1184), also the one COPY and CREATE INDEX force."""
        return "snapshot_checkpoint" not in self.options.skipped or not self.txns

    def snapshot_write_safe(self, kind: str, id) -> bool:
        """Whether a committed update or delete of Person `id` stays clear of two bugs that an
        open read-only snapshot hits at the next checkpoint: the updated value lands on another
        row (snapshot_update), and a row inserted after the snapshot began and then deleted
        becomes visible to it (snapshot_insert_delete)."""
        readers = [txn.view for txn in self.txns.values() if txn.kind == "read"]
        if not readers or id not in self.model.persons:
            return True
        skipped = self.options.skipped
        if kind in ("set_age", "update"):
            return "snapshot_update" not in skipped
        if kind == "delete":
            hidden = any(id not in view.persons for view in readers)
            return not (hidden and "snapshot_insert_delete" in skipped)
        return True

    def has_partitions(self) -> bool:
        """Whether the partitioned table L exists: only while its rules run, so that a gated
        partition bug cannot end workloads that never touch it."""
        return "partitions" not in self.options.skipped

    def copy_person_safe(self) -> bool:
        """Whether a COPY into Person keeps index lookups right: rows COPY loads never reach an
        ART index on a Person column (art_copy)."""
        return self.model.age_index is None or "art_copy" not in self.options.skipped

    def has_csr(self) -> bool:
        """Whether C and CK exist: only while the csr rules run."""
        return "csr" not in self.options.skipped

    def loads(self) -> list[str]:
        """The statements each open runs before the rules: extension loads and the C tables,
        filled on the database's first open."""
        if not self.has_csr():
            return list(self.options.extensions)
        fill = getattr(self, "c_fill", None) if self.model.big.rows == 0 else None
        return list(self.options.extensions) + csr_setup(fill, self.config["max_num_threads"])

    def reopen_loads(self) -> list[str]:
        """What loads() returns on every open after a database's first."""
        threads = self.config["max_num_threads"]
        return list(self.options.extensions) + (csr_setup(None, threads) if self.has_csr() else [])

    # Engine lifecycle

    def start_trace(self) -> None:
        self.trace = []
        op = dict(op="open", config=self.config, loads=self.loads())
        op["reopen_loads"] = self.reopen_loads()
        self.traced(op, self.open_engine)
        op["build"] = self.engine.build

    def current_rule(self) -> str | None:
        """The rule or invariant of this machine that is running, from the call stack."""
        frame = sys._getframe(1)
        while frame:
            method = getattr(type(self), frame.f_code.co_name, None)
            if frame.f_locals.get("self") is self and any(
                hasattr(method, marker) for marker in RULE_MARKERS
            ):
                return frame.f_code.co_name
            frame = frame.f_back
        return None

    def record_failure(self, error: BaseException) -> dict:
        """Appends the failure to the trace: its kind, its message (with native stacks for a
        crash or hang) and its signature (see triage.py)."""
        kind, message = failure_kind(error), str(error)
        step = next(
            (
                op
                for op in reversed(self.trace)
                if op.get("outcome", "ok") != "ok" and not op.get("expected")
            ),
            self.trace[-1],
        )
        record = dict(op="failure", kind=kind, message=message[:8000])
        record["signature"] = signature(kind, step, message)
        if kind == "wrong result" and (probe := probe_of(step, message)):
            record["probe"], record["rows"] = probe
        self.trace.append(record)
        return record

    def save_trace(self) -> None:
        with open(os.path.join(self.files, "trace.json"), "w") as f:
            json.dump(self.trace, f, indent=1, default=str)

    def traced(self, op: dict, call):
        """Records `op` with the rule it belongs to, runs `call` and records how it ended."""
        op["rule"] = self.current_rule()
        if op["rule"] != getattr(self, "last_rule", None):
            RULE_RUNS[op["rule"] or op["op"]] += 1
        self.last_rule = op["rule"]
        self.trace.append(op)
        start = time.monotonic()
        try:
            result = call()
        except (EngineError, AssertionError) as e:
            op["outcome"] = outcome_of(e)
            raise
        finally:
            op["ms"] = round((time.monotonic() - start) * 1000)
        op["outcome"] = "ok"
        return result

    def open_engine(self, inject: tuple[str, int] | None = None, crashed: bool = False) -> None:
        self.engine = None
        self.kept = {}
        # A new worker has new connections, without transactions.
        for txn in getattr(self, "txns", {}).values():
            self.model.discarded(txn.view)
        self.txns = {}
        # Closing checkpoints, so a clean reopen starts from a checkpoint; recovery after a crash
        # replays the WAL but leaves the column-ID gaps it found.
        if getattr(self, "model", None) is not None and not crashed:
            self.model.checkpointed()
        config = self.config
        if inject:
            # With one worker thread, the caller and the worker take turns, so the n-th call
            # of a syscall is the same call in every run.
            config = {**config, "max_num_threads": 1}
        try:
            self.engine = Engine(self.path, config, self.loads(), inject, self.options.debug)
        except EngineError as e:
            known = [gate for gate, text in RECOVERY_FAILURES.items() if text in str(e)]
            if crashed and known and known[0] in self.options.skipped:
                self.start_over(known[0])
                return
            raise

    def start_over(self, gate: str) -> None:
        """Goes on with a fresh database after a known recovery failure: discarding the workload
        would make Hypothesis see a timing-dependent outcome as flaky data generation."""
        RECOVERY_SKIPS[gate] += 1
        self.started_over = True
        self.new_database(fresh=True)

    def reopen_engine(self) -> None:
        self.traced(dict(op="reopen"), lambda: (self.engine.close(), self.open_engine()))

    def crash_engine(self) -> None:
        self.started_over = False
        self.traced(dict(op="crash"), lambda: (self.engine.kill(), self.open_engine(crashed=True)))

    def crash_during_statement(self, query: str, delay: float) -> None:
        """Sends a statement, kills the engine `delay` seconds later and opens it again."""

        self.started_over = False

        def call():
            self.engine.start(query)
            time.sleep(delay)
            self.engine.kill()
            self.open_engine(crashed=True)

        self.traced(dict(op="crash_during", query=query, delay=delay), call)

    def crash_at_syscall_point(self, query: str, syscall: str, n: int) -> None:
        """Reopens the engine under strace, which kills it at the n-th call of `syscall`, runs
        the statement, and opens the engine again normally."""

        self.started_over = False

        def call():
            self.engine.close()
            try:
                self.open_engine(inject=(syscall, n))
                self.engine.execute(query, {})
                self.engine.close()
            except AssertionError as e:
                if "died" not in str(e):
                    raise
            except EngineError:
                pass  # a statement killed during open fails, which is the crash being tested
            self.open_engine(crashed=True)

        self.traced(dict(op="crash_at", query=query, syscall=syscall, n=n), call)

    # Statements

    def run(self, query: str, params: dict | None = None, conn: int = 0) -> list:
        op = dict(op="execute", conn=conn, query=query, params=params or {})
        return self.traced(op, lambda: self.engine.execute(query, params or {}, conn))

    def ok(self, query: str, params: dict | None = None, conn: int = 0) -> None:
        self.run(query, params, conn)

    def fails(self, query: str, params: dict | None, error: str, conn: int = 0) -> None:
        """The statement must fail with a message containing `error`."""
        try:
            self.run(query, params, conn)
        except EngineError as e:
            self.trace[-1]["expected"] = True
            assert error in str(e), f"{query} {params} failed with {e!r}, expected {error!r}"
            return
        raise AssertionError(f"expected an error with {error!r}: {query} {params}")

    def hold_results(self, conn: int) -> int:
        return self.traced(dict(op="hold", conn=conn), lambda: self.engine.hold(conn))

    def release_results(self) -> None:
        self.traced(dict(op="release"), self.engine.release)

    def rows(self, query: str, params: dict | None = None, conn: int = 0) -> list[tuple]:
        return [canonical(row) for row in self.run(query, params, conn)]

    def columns(self, query: str, conn: int = 0) -> tuple[list[str], list[tuple]]:
        op = dict(op="columns", conn=conn, query=query, params={})
        names, rows = self.traced(op, lambda: self.engine.columns(query, {}, conn))
        return names, [canonical(row) for row in rows]

    def prepare_named(self, name: str, query: str, params: dict) -> None:
        op = dict(op="prepare", name=name, query=query, params=params)
        self.traced(op, lambda: self.engine.prepare(name, query, params))

    def run_named(self, name: str, params: dict) -> list[tuple]:
        op = dict(op="run", name=name, params=params)
        return [canonical(row) for row in self.traced(op, lambda: self.engine.run(name, params))]

    def check(
        self,
        query: str,
        params: dict,
        expected: list,
        ordered: bool = False,
        conn: int = 0,
        alone: bool = False,
    ) -> None:
        """Runs the query four ways and compares each result with `expected`: a statement
        prepared now, executed again so that it reuses its cached plan, the driver's own path
        (which caches prepared statements across steps when there are parameters), and on the
        single-threaded connection. Inside a transaction (`alone`, or a session connection)
        another connection would see another state, so the query runs on `conn` only."""
        expected = [canonical(row) for row in expected]
        runs = {"fresh plan": None, "cached plan": None}
        if conn not in self.txns or "snapshot_prepared" not in self.options.skipped:
            runs["driver"] = self.rows(query, params, conn)
        if conn == 0 and not alone:
            runs["single thread"] = self.rows(query, params, 1)
        op = dict(op="prepared", conn=conn, query=query, params=params)
        prepared = self.traced(op, lambda: self.engine.execute_prepared(query, params, conn))
        runs["fresh plan"], runs["cached plan"] = ([canonical(r) for r in rs] for rs in prepared)
        for how, actual in runs.items():
            same = actual == expected if ordered else Counter(actual) == Counter(expected)
            assert same, (query, params, how, actual, expected)

    def check_race(self, writes: list, queries: list[str], states: list, num_readers: int) -> None:
        """Runs `writes` against readers of `queries`; every read must be one of `states`, the
        committed state after each write, no reader may go back to an earlier one, and the last
        read, after all writes, must see the final state."""
        op = dict(op="race", writes=writes, queries=queries, readers=num_readers)
        observations, errors = self.traced(
            op, lambda: self.engine.race(writes, queries, num_readers)
        )
        op["errors"] = errors
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
            assert reader[-1] == states[-1], (reader[-1], states[-1])

    def person_rows(self, model: Model | None = None, conn: int = 0) -> Counter:
        """Person's rows, with the columns `model` (by default the committed one) has."""
        columns = "".join(f", p.{column}" for column in (model or self.model).columns)
        return Counter(self.rows(f"MATCH (p:Person) RETURN p.id{columns}", None, conn))
