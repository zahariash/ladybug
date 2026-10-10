"""The state every rule works on, and the helpers rules use to run statements and checks.

Every interaction with the engine goes through Session, which records it with its outcome in
a trace: the statements, crashes, crash points and races of one workload. replay.py reruns a
trace without the model and minimizes it.
"""

from __future__ import annotations

import json
import os
import time
from collections import Counter
from dataclasses import dataclass, field

from engine import Engine, EngineError
from hypothesis.stateful import precondition
from model import Model, canonical

# Every name rules are gated on, so --sim-skip can reject names that match nothing. Rules add
# theirs when their module is imported.
RULE_NAMES: set[str] = set()
EXTENSIONS = ("fts", "vector")

DUPLICATE_KEY = "duplicated primary key"


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


def enabled(name: str):
    """Runs the rule unless --sim-skip, --sim-focus or known.py turn `name` off."""
    gated(name)
    return precondition(lambda self: self.engine is not None and name not in self.options.skipped)


def with_extension(name: str, extension: str | None = None):
    """Like enabled, and only when `extension` (or any extension, for None) is loaded."""
    gated(name)
    return precondition(
        lambda self: (
            self.engine is not None
            and bool(self.options.loaded if extension is None else extension in self.options.loaded)
            and name not in self.options.skipped
        )
    )


def diff(actual: Counter, expected: Counter) -> str:
    missing = list((expected - actual).items())[:5]
    unexpected = list((actual - expected).items())[:5]
    return f"missing={missing} unexpected={unexpected}"


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

    # Engine lifecycle

    def start_trace(self) -> None:
        self.trace = []
        op = dict(op="open", config=self.config, loads=list(self.options.extensions))
        self.traced(op, self.open_engine)

    def save_trace(self) -> None:
        with open(os.path.join(self.files, "trace.json"), "w") as f:
            json.dump(self.trace, f, indent=1, default=str)

    def traced(self, op: dict, call):
        """Records `op`, runs `call` and records how it ended."""
        self.trace.append(op)
        try:
            result = call()
        except (EngineError, AssertionError) as e:
            op["outcome"] = outcome_of(e)
            raise
        op["outcome"] = "ok"
        return result

    def open_engine(self, inject: tuple[str, int] | None = None) -> None:
        self.engine = None
        self.kept = {}
        config = self.config
        if inject:
            # With one worker thread, the caller and the worker take turns, so the n-th call
            # of a syscall is the same call in every run.
            config = {**config, "max_num_threads": 1}
        self.engine = Engine(
            self.path, config, list(self.options.extensions), inject, self.options.debug
        )

    def reopen_engine(self) -> None:
        self.traced(dict(op="reopen"), lambda: (self.engine.close(), self.open_engine()))

    def crash_engine(self) -> None:
        self.traced(dict(op="crash"), lambda: (self.engine.kill(), self.open_engine()))

    def crash_during_statement(self, query: str, delay: float) -> None:
        """Sends a statement, kills the engine `delay` seconds later and opens it again."""

        def call():
            self.engine.start(query)
            time.sleep(delay)
            self.engine.kill()
            self.open_engine()

        self.traced(dict(op="crash_during", query=query, delay=delay), call)

    def crash_at_syscall_point(self, query: str, syscall: str, n: int) -> None:
        """Reopens the engine under strace, which kills it at the n-th call of `syscall`, runs
        the statement, and opens the engine again normally."""

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
            self.open_engine()

        self.traced(dict(op="crash_at", query=query, syscall=syscall, n=n), call)

    # Statements

    def run(self, query: str, params: dict | None = None, conn: int = 0) -> list:
        op = dict(op="execute", conn=conn, query=query, params=params or {})
        return self.traced(op, lambda: self.engine.execute(query, params or {}, conn))

    def ok(self, query: str, params: dict | None = None) -> None:
        self.run(query, params)

    def fails(self, query: str, params: dict | None, error: str) -> None:
        """The statement must fail with a message containing `error`."""
        try:
            self.run(query, params)
        except EngineError as e:
            self.trace[-1]["expected"] = True
            assert error in str(e), f"{query} {params} failed with {e!r}, expected {error!r}"
            return
        raise AssertionError(f"expected an error with {error!r}: {query} {params}")

    def rows(self, query: str, params: dict | None = None, conn: int = 0) -> list[tuple]:
        return [canonical(row) for row in self.run(query, params, conn)]

    def prepare_named(self, name: str, query: str, params: dict) -> None:
        op = dict(op="prepare", name=name, query=query, params=params)
        self.traced(op, lambda: self.engine.prepare(name, query, params))

    def run_named(self, name: str, params: dict) -> list[tuple]:
        op = dict(op="run", name=name, params=params)
        return [canonical(row) for row in self.traced(op, lambda: self.engine.run(name, params))]

    def check(self, query: str, params: dict, expected: list, ordered: bool = False) -> None:
        """Runs the query four ways and compares each result with `expected`: a statement
        prepared now, executed again so that it reuses its cached plan, the driver's own path
        (which caches prepared statements across steps when there are parameters), and on the
        single-threaded connection."""
        expected = [canonical(row) for row in expected]
        runs = {
            "fresh plan": None,
            "cached plan": None,
            "driver": self.rows(query, params, 0),
            "single thread": self.rows(query, params, 1),
        }
        op = dict(op="prepared", conn=0, query=query, params=params)
        prepared = self.traced(op, lambda: self.engine.execute_prepared(query, params))
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

    def person_rows(self) -> Counter:
        columns = "".join(f", p.{column}" for column in self.model.columns)
        return Counter(self.rows(f"MATCH (p:Person) RETURN p.id{columns}"))
