"""What the database should contain after the steps so far."""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field, replace
from typing import NamedTuple

from strategies import BULK_ID_START


def canonical(row) -> tuple:
    """NaN never equals itself, so it compares as a string; lists become hashable tuples."""
    return tuple(_canonical(v) for v in row)


def _canonical(value):
    if isinstance(value, float) and math.isnan(value):
        return "NaN"
    if isinstance(value, list):
        return tuple(_canonical(v) for v in value)
    return value


class Doc(NamedTuple):
    words: list[str] | None
    emb: tuple[int, ...] | None

    @property
    def text(self) -> str | None:
        return " ".join(self.words) if self.words else None


@dataclass
class Model:
    # Person columns after id, in table order, with their defaults.
    columns: dict = field(default_factory=lambda: {"name": None, "age": None, "score": None})
    persons: dict = field(default_factory=dict)  # id -> {column: value}
    # Whether Person ever held rows; the engine counts deleted rows until they are compacted.
    persons_ever: bool = False
    knows: Counter = field(default_factory=Counter)  # (src, dst, since) -> count
    macros: dict = field(default_factory=dict)  # i -> k, for m{i}(x) = x + k
    tables: dict = field(default_factory=dict)  # i -> values of T{i}.v, in insertion order
    partitioned: dict = field(default_factory=dict)  # id -> cluster, in L
    pk_index: str | None = "_PK"  # the name of Person's primary-key index, if any
    docs: dict = field(default_factory=dict)  # id -> Doc
    doc_indexes: dict = field(default_factory=dict)  # "doc_fts"/"doc_vec" -> stemmer/metric
    next_bulk_id: int = BULK_ID_START

    def new_person(self, **values) -> dict:
        return {column: values.get(column, default) for column, default in self.columns.items()}

    def add_persons(self, persons: dict) -> None:
        self.persons.update(persons)
        self.persons_ever = self.persons_ever or bool(persons)

    def forget(self, id: int) -> None:
        """DETACH DELETE of a person."""
        self.persons.pop(id, None)
        for key in [k for k in self.knows if id in (k[0], k[1])]:
            del self.knows[key]

    def set_since(self, src: int, dst: int, since) -> None:
        """SET k.since on every Knows edge from src to dst."""
        count = sum(n for (s, d, _), n in self.knows.items() if (s, d) == (src, dst))
        for key in [k for k in self.knows if k[:2] == (src, dst)]:
            del self.knows[key]
        if count:
            self.knows[(src, dst, since)] += count

    def fork(self) -> Model:
        """A copy that transactions and races can change. They write only small ids, Knows,
        macros and T tables, so bulk rows and docs are shared, not copied."""
        persons = {id: dict(p) if id < BULK_ID_START else p for id, p in self.persons.items()}
        return replace(
            self,
            columns=dict(self.columns),
            persons=persons,
            knows=Counter(self.knows),
            macros=dict(self.macros),
            tables={i: list(values) for i, values in self.tables.items()},
        )

    def take_bulk_ids(self, count: int) -> range:
        ids = range(self.next_bulk_id, self.next_bulk_id + count)
        self.next_bulk_id += count
        return ids

    def person_rows(self) -> Counter:
        return Counter(canonical((id, *p.values())) for id, p in self.persons.items())
