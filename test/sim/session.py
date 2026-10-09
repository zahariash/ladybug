"""The state every rule works on, and the helpers rules use to run statements and checks."""

from __future__ import annotations

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


class Session:
    options = Options()
    engine: Engine | None = None
    model: Model
    path: str
    dir: str
    config: dict

    def reopen_engine(self) -> None:
        self.engine = None
        self.engine = Engine(self.path, self.config, list(self.options.extensions))

    def run(self, query: str, params: dict | None = None, conn: int = 0) -> list:
        return self.engine.execute(query, params or {}, conn)

    def ok(self, query: str, params: dict | None = None) -> None:
        self.run(query, params)

    def fails(self, query: str, params: dict | None, error: str) -> None:
        """The statement must fail with a message containing `error`."""
        try:
            self.run(query, params)
        except EngineError as e:
            assert error in str(e), f"{query} {params} failed with {e!r}, expected {error!r}"
            return
        raise AssertionError(f"expected an error with {error!r}: {query} {params}")

    def rows(self, query: str, params: dict | None = None, conn: int = 0) -> list[tuple]:
        return [canonical(row) for row in self.run(query, params, conn)]

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
        prepared = self.engine.execute_prepared(query, params)
        runs["fresh plan"], runs["cached plan"] = ([canonical(r) for r in rs] for rs in prepared)
        for how, actual in runs.items():
            same = actual == expected if ordered else Counter(actual) == Counter(expected)
            assert same, (query, params, how, actual, expected)

    def check_race(self, writes: list, queries: list[str], states: list, num_readers: int) -> None:
        """Runs `writes` against readers of `queries`; every read must be one of `states`, the
        committed state after each write, no reader may go back to an earlier one, and the last
        read, after all writes, must see the final state."""
        observations, errors = self.engine.race(writes, queries, num_readers)
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
