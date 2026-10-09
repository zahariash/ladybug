"""The state every rule works on, and the helpers rules use to run statements and checks."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

from engine import Engine, EngineError
from hypothesis.stateful import precondition
from model import Model, canonical


@dataclass(frozen=True)
class Options:
    skipped: frozenset = field(default_factory=frozenset)  # rule names and variants to skip
    extensions: tuple = ()  # statements that load extensions after every open


def enabled(name: str):
    """Runs the rule unless --sim-skip or known.py turn it off."""
    return precondition(lambda self: self.engine is not None and name not in self.options.skipped)


def with_extensions(name: str):
    """Like enabled, and only when extensions are loaded."""
    return precondition(
        lambda self: (
            self.engine is not None
            and bool(self.options.extensions)
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
        self.engine = Engine(self.path, self.config, list(self.options.extensions))

    def run(self, query: str, params: dict | None = None, conn: int = 0) -> list:
        return self.engine.execute(query, params or {}, conn)

    def ok(self, query: str, params: dict | None = None) -> None:
        self.run(query, params)

    def fails(self, query: str, params: dict | None = None) -> None:
        try:
            self.run(query, params)
        except EngineError:
            return
        raise AssertionError(f"expected an error: {query} {params}")

    def rows(self, query: str, params: dict | None = None, conn: int = 0) -> list[tuple]:
        return [canonical(row) for row in self.run(query, params, conn)]

    def check(self, query: str, params: dict, expected: list, ordered: bool = False) -> None:
        """Runs the query twice on the main connection, so that a parameterized query reuses its
        cached plan, then on the single-threaded one; each result must match `expected`."""
        expected = [canonical(row) for row in expected]
        for conn in (0, 0, 1):
            actual = self.rows(query, params, conn)
            same = actual == expected if ordered else Counter(actual) == Counter(expected)
            assert same, (query, params, conn, actual, expected)

    def person_rows(self) -> Counter:
        columns = "".join(f", p.{column}" for column in self.model.columns)
        return Counter(self.rows(f"MATCH (p:Person) RETURN p.id{columns}"))
