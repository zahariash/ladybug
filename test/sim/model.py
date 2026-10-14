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
    knows_w: int | None = None  # the default of Knows.w, while the property exists
    macros: dict = field(default_factory=dict)  # i -> k, for m{i}(x) = x + k
    tables: dict = field(default_factory=dict)  # i -> values of T{i}.v, in insertion order
    # (id, cluster) rows of L; each cluster is its own partition, keys are unique per partition
    partitioned: set = field(default_factory=set)
    pk_index: str | None = "_PK"  # the name of Person's primary-key index, if any
    age_index: str | None = None  # the name of the ART index on Person.age, if any
    dropped_columns: set = field(default_factory=set)  # Person columns dropped at some point
    # Since the last checkpoint: a Person column was dropped, and then one was added. Column IDs
    # have a gap until a checkpoint compacts them.
    dropped_since_checkpoint: bool = False
    added_after_drop: bool = False
    # A transaction that added a column was rolled back since the last checkpoint.
    rolled_back_add: bool = False
    # In a transaction's view: it already ran a statement that prepared statements may reuse.
    wrote: bool = False
    # In a transaction's view: it added a Person column, which a Person insert prepared before it
    # (the driver caches prepared statements across steps) does not see.
    added_in_txn: bool = False

    def drop_column(self, column: str) -> None:
        del self.columns[column]
        self.persons = {
            id: {c: v for c, v in p.items() if c != column} for id, p in self.persons.items()
        }
        self.dropped_columns.add(column)
        self.dropped_since_checkpoint = True

    def add_column(self, column: str, default) -> None:
        self.columns[column] = default
        self.persons = {id: {**p, column: default} for id, p in self.persons.items()}
        self.added_after_drop |= self.dropped_since_checkpoint

    def checkpointed(self) -> None:
        self.dropped_since_checkpoint = self.added_after_drop = self.rolled_back_add = False

    def discarded(self, view: Model) -> None:
        """A transaction that saw `view` ended without committing: its inserts may still count
        as rows until compaction (see bulk_copy), and the columns it added leave data behind."""
        self.persons_ever |= view.persons_ever
        self.rolled_back_add |= bool(set(view.columns) - set(self.columns))

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
        """A copy that transactions and races can change, and that an open transaction keeps as
        its snapshot. Writes change only small ids, and DDL replaces a person's dict rather than
        changing it, so bulk rows and docs are shared, not copied."""
        persons = {id: dict(p) if id < BULK_ID_START else p for id, p in self.persons.items()}
        return replace(
            self,
            columns=dict(self.columns),
            persons=persons,
            knows=Counter(self.knows),
            macros=dict(self.macros),
            tables={i: list(values) for i, values in self.tables.items()},
            partitioned=set(self.partitioned),
            dropped_columns=set(self.dropped_columns),
            wrote=False,
        )

    def take_bulk_ids(self, count: int) -> range:
        ids = range(self.next_bulk_id, self.next_bulk_id + count)
        self.next_bulk_id += count
        return ids

    def person_rows(self) -> Counter:
        return Counter(canonical((id, *p.values())) for id, p in self.persons.items())
