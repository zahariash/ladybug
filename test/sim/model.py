"""What the database should contain after the steps so far."""

from __future__ import annotations

import heapq
import itertools
import math
from collections import Counter, defaultdict
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


# Row fingerprints: a reader sums one number per row, mixed from its columns, so that a wrong row
# changes the sum where counts match. Values are taken modulo FP_MOD first, so that no product
# leaves INT64; each Cypher expression below computes what its Python twin does.
FP_MOD = 1_000_000_007
PERSON_FP = (
    f"(p.id % {FP_MOD}) * 1009 + coalesce(p.age % {FP_MOD}, {FP_MOD + 1}) * 7919 "
    "+ coalesce(size(p.name), -1) * 104729 + CASE WHEN p.score IS NULL THEN 1 ELSE 0 END"
)
KNOWS_FP = f"(a.id % {FP_MOD}) * 1009 + (b.id % {FP_MOD}) * 7919 + coalesce(k.since % {FP_MOD}, -1)"
C_FP = f"c.id * 1009 + coalesce(c.v % {FP_MOD}, {FP_MOD + 1})"
CK_FP = "a.id * 200003 + b.id"
CK2_FP = "a.id * 200003 + b.id + k.w * 7"


def tmod(a: int, m: int = FP_MOD) -> int:
    """a % m as the engine computes it: with the sign of a."""
    r = abs(a) % m
    return -r if a < 0 else r


def person_fp(id: int, p: dict) -> int:
    age = tmod(p["age"]) if p["age"] is not None else FP_MOD + 1
    name = len(p["name"]) if p["name"] is not None else -1
    return tmod(id) * 1009 + age * 7919 + name * 104729 + (p["score"] is None)


def knows_fp(src: int, dst: int, since) -> int:
    return tmod(src) * 1009 + tmod(dst) * 7919 + (tmod(since) if since is not None else -1)


def c_fp(id: int, v) -> int:
    return id * 1009 + (tmod(v) if v is not None else FP_MOD + 1)


def filler(id: int) -> int:
    """C.v of a row that no update has changed."""
    return id % 1000


# C.s of a row that no update has changed: NULL, a string stored inline (8 bytes) or one past
# the 12 bytes strings keep inline (25 bytes). Ids are offset to 7 digits, so that strings of one
# length sort as their ids do. TEXT_QUERY computes it in Cypher for the id {i}.
SHORT, LONG = 8, 25


def filler_text(id: int) -> str | None:
    if id % 11 == 0:
        return None
    if id % 7 == 0:
        return f"long-{1_000_000 + id}-xxxxxxxxxxxx"
    return f"s{1_000_000 + id}"


TEXT_QUERY = (
    "CASE WHEN {i} % 11 = 0 THEN NULL "
    "WHEN {i} % 7 = 0 THEN concat('long-', CAST(1000000 + {i} AS STRING), '-xxxxxxxxxxxx') "
    "ELSE concat('s', CAST(1000000 + {i} AS STRING)) END"
)


def text_query(i: str) -> str:
    return TEXT_QUERY.format(i=i)


def cs_key(i: int) -> str:
    """CS's key for id i: 13 to 40 bytes."""
    return f"key-{1_000_000 + i}-x" + "x" * (i % 28)


def cs_key_query(i: str) -> str:
    xs = "x" * 27
    return f"concat('key-', CAST(1000000 + {i} AS STRING), '-x', substring('{xs}', 1, {i} % 28))"


@dataclass
class Big:
    """C: ids 0..rows-1 at the offsets of their ids, v = filler(id), less deleted ids and with
    updated values. Kept as changes against the formula, so that copies and aggregates cost the
    changes, not the rows."""

    rows: int = 0
    deleted: set = field(default_factory=set)
    updated: dict = field(default_factory=dict)  # id -> v, None for NULL
    texts: dict = field(default_factory=dict)  # id -> s, None for NULL
    columns: dict = field(default_factory=dict)  # added columns -> their defaults, in table order

    def copy(self) -> Big:
        return Big(
            self.rows, set(self.deleted), dict(self.updated), dict(self.texts), dict(self.columns)
        )

    def exists(self, id: int) -> bool:
        return 0 <= id < self.rows and id not in self.deleted

    def value(self, id: int):
        return self.updated[id] if id in self.updated else filler(id)

    def text(self, id: int) -> str | None:
        return self.texts[id] if id in self.texts else filler_text(id)

    def row(self, id: int) -> tuple:
        """id, v, s and the added columns."""
        return (id, self.value(id), self.text(id), *self.columns.values())

    def changed_rows(self) -> list[tuple]:
        """(id, v, s) of the rows whose v or s is not the formula's."""
        ids = sorted(id for id in self.updated.keys() | self.texts.keys() if self.exists(id))
        return [
            (id, self.value(id), self.text(id))
            for id in ids
            if (self.value(id), self.text(id)) != (filler(id), filler_text(id))
        ]

    def count(self) -> int:
        return self.rows - len(self.deleted)

    def changed(self) -> set:
        return self.deleted | self.updated.keys()

    def fingerprint(self) -> int | None:
        """The sum of C_FP over C's rows."""
        if not self.count():
            return None
        full, rest = divmod(self.rows, 1000)
        total = 1009 * self.rows * (self.rows - 1) // 2
        total += full * (999 * 1000 // 2) + rest * (rest - 1) // 2
        for id in self.changed():
            if id < self.rows:
                total -= c_fp(id, filler(id))
                if id not in self.deleted:
                    total += c_fp(id, self.updated[id])
        return total

    def updated_values(self) -> list:
        return [v for id, v in self.updated.items() if id not in self.deleted and v is not None]

    def sum(self) -> int:
        full, rest = divmod(self.rows, 1000)
        total = full * (999 * 1000 // 2) + rest * (rest - 1) // 2
        total -= sum(filler(id) for id in self.changed() if id < self.rows)
        return total + sum(self.updated_values())

    def _unchanged_filler(self, values) -> int | None:
        """The first of `values` that some unchanged row still holds."""
        changed = self.changed()
        for v in values:
            if any(id not in changed for id in range(v, self.rows, 1000)):
                return v
        return None

    def min_max(self) -> tuple:
        values = self.updated_values()
        low, high = self._unchanged_filler(range(1000)), self._unchanged_filler(range(999, -1, -1))
        values += [v for v in (low, high) if v is not None]
        return (min(values), max(values)) if values else (None, None)

    def text_count_size(self) -> tuple[int, int]:
        """count(c.s) and sum(size(c.s))."""
        nulls = -(-self.rows // 11)
        longs = -(-self.rows // 7) - -(-self.rows // 77)
        count, size = self.rows - nulls, (self.rows - nulls - longs) * SHORT + longs * LONG
        for id in self.deleted | self.texts.keys():
            if id < self.rows and (text := filler_text(id)) is not None:
                count, size = count - 1, size - len(text)
        for id, text in self.texts.items():
            if id not in self.deleted and text is not None:
                count, size = count + 1, size + len(text)
        return count, size

    def text_min_max(self) -> tuple:
        """min(c.s) and max(c.s): long strings sort before short ones, and each by id, so the
        formula's extremes are the first and last unchanged id of each kind."""
        changed = self.deleted | self.texts.keys()

        def first(ids, long):
            kind = (lambda id: id % 7 == 0) if long else (lambda id: id % 7 != 0)
            return next((id for id in ids if id % 11 and kind(id) and id not in changed), None)

        orders = (range(self.rows), range(self.rows - 1, -1, -1))
        ends = [first(order, long) for order in orders for long in (True, False)]
        values = [filler_text(id) for id in ends if id is not None]
        values += [t for id, t in self.texts.items() if id not in self.deleted and t is not None]
        return (min(values), max(values)) if values else (None, None)

    def top(self, k: int) -> list[tuple[int, int]]:
        """(id, v) of the k rows with a value, as ORDER BY v DESC, id LIMIT k returns them."""
        changed = self.changed()

        def unchanged():
            for v in range(999, -1, -1):
                for id in range(v, self.rows, 1000):
                    if id not in changed:
                        yield (-v, id)

        updated = sorted(
            (-v, id) for id, v in self.updated.items() if id not in self.deleted and v is not None
        )
        return [(id, -nv) for nv, id in itertools.islice(heapq.merge(updated, unchanged()), k)]


@dataclass
class Keys:
    """CS: the key cs_key(id) with n = id for ids 0..rows-1, less deleted ids, and the rows CREATE
    inserted, by id: ids at CS_FAR, and deleted ids again."""

    rows: int = 0
    deleted: set = field(default_factory=set)
    created: dict = field(default_factory=dict)  # id -> n

    def copy(self) -> Keys:
        return Keys(self.rows, set(self.deleted), dict(self.created))

    def n(self, id: int) -> int | None:
        """The row's n, or None if there is no row with the key."""
        if id in self.created:
            return self.created[id]
        return id if id < self.rows and id not in self.deleted else None

    def delete(self, id: int) -> None:
        self.created.pop(id, None)
        if id < self.rows:
            self.deleted.add(id)

    def count(self) -> int:
        return self.rows - len(self.deleted) + len(self.created)

    def total(self) -> int | None:
        """sum(c.n)."""
        if not self.count():
            return None
        return self.rows * (self.rows - 1) // 2 - sum(self.deleted) + sum(self.created.values())


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
    ck: Counter = field(default_factory=Counter)  # (src, dst) -> count of CK edges
    ck2: Counter = field(default_factory=Counter)  # (src, dst, w) -> count of CK2 edges
    big: Big = field(default_factory=Big)  # C's rows
    keys: Keys = field(default_factory=Keys)  # CS's rows
    pk_index: str | None = "_PK"  # the name of Person's primary-key index, if any
    age_index: str | None = None  # the name of the ART index on Person.age, if any
    dropped_columns: set = field(default_factory=set)  # Person columns dropped at some point
    # Since the last checkpoint: a Person column was dropped, and then one was added; or a column
    # other than the last was dropped. Column IDs have a gap until a checkpoint compacts them.
    dropped_since_checkpoint: bool = False
    added_after_drop: bool = False
    middle_dropped: bool = False
    # Since the last checkpoint, an existing person was updated.
    updated_since_checkpoint: bool = False
    # A transaction that added a column was rolled back since the last checkpoint.
    rolled_back_add: bool = False
    # In a transaction's view: it already ran a statement that prepared statements may reuse.
    wrote: bool = False
    # In a transaction's view: it added a Person column, which a Person insert prepared before it
    # (the driver caches prepared statements across steps) does not see.
    added_in_txn: bool = False

    def drop_column(self, column: str) -> None:
        self.middle_dropped |= column != list(self.columns)[-1]
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
        self.indexed |= self.keyed_tables()
        self.dropped_since_checkpoint = self.added_after_drop = self.middle_dropped = False
        self.rolled_back_add = self.updated_since_checkpoint = False

    def discarded(self, view: Model) -> None:
        """A transaction that saw `view` ended without committing: its inserts may still count
        as rows until compaction (see bulk_copy), and the columns it added leave data behind."""
        self.persons_ever |= view.persons_ever
        self.rolled_back_add |= bool(set(view.columns) - set(self.columns))

    docs: dict = field(default_factory=dict)  # id -> Doc
    doc_indexes: dict = field(default_factory=dict)  # "doc_fts"/"doc_vec" -> stemmer/metric
    next_bulk_id: int = BULK_ID_START
    # GN's rows in the named graph, id -> v, once the graph exists.
    graph: dict | None = None
    blobs: dict = field(default_factory=dict)  # id -> length of Blob.s
    next_blob_id: int = 0
    # The tables of keyed_tables() whose primary-key index held keys at a checkpoint known to
    # have succeeded.
    indexed: set = field(default_factory=set)

    def keyed_tables(self) -> set[str]:
        """The tables with a hash-index primary key that hold rows now."""
        held = {"Person": self.persons, "Blob": self.blobs, "GN": self.graph}
        return {table for table, rows in held.items() if rows}

    def first_index_pending(self) -> bool:
        """Whether a table's primary-key index has keys but none at any checkpoint yet."""
        return bool(self.keyed_tables() - self.indexed)

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
            ck=Counter(self.ck),
            ck2=Counter(self.ck2),
            big=self.big.copy(),
            keys=self.keys.copy(),
            dropped_columns=set(self.dropped_columns),
            wrote=False,
            graph=None if self.graph is None else dict(self.graph),
            blobs=dict(self.blobs),
            indexed=set(self.indexed),
        )

    def update(self, id: int, field: str, value) -> None:
        """Sets a field of an existing person. Replaces the person's dict, which forks may share."""
        if id in self.persons:
            self.persons[id] = {**self.persons[id], field: value}
            self.updated_since_checkpoint = True

    def forget_c(self, id: int) -> None:
        """DETACH DELETE of a C row: it and its CK and CK2 edges go."""
        self.big.deleted.add(id)
        self.big.updated.pop(id, None)
        self.big.texts.pop(id, None)
        for key in [k for k in self.ck if id in k]:
            del self.ck[key]
        for key in [k for k in self.ck2 if id in k[:2]]:
            del self.ck2[key]

    def ck2_pairs(self) -> Counter:
        """CK2's edges as (src, dst) -> count, whatever their w."""
        pairs = Counter()
        for (src, dst, _), n in self.ck2.items():
            pairs[(src, dst)] += n
        return pairs

    def take_bulk_ids(self, count: int) -> range:
        ids = range(self.next_bulk_id, self.next_bulk_id + count)
        self.next_bulk_id += count
        return ids

    def person_rows(self) -> Counter:
        return Counter(canonical((id, *p.values())) for id, p in self.persons.items())

    def graph_rows(self) -> Counter:
        return Counter((self.graph or {}).items())

    def take_blob_ids(self, count: int) -> range:
        ids = range(self.next_blob_id, self.next_blob_id + count)
        self.next_blob_id += count
        return ids


def blob_lengths(ids: range, units: int) -> dict:
    """The lengths of the strings blob_query writes."""
    return {i: units * (len(str(i)) + 1) for i in ids}


def blob_query(ids: range, units: int) -> str:
    """Blob rows of distinct long strings, which make the data file grow at the next checkpoint
    (identical strings would share a dictionary entry)."""
    rows = f"UNWIND range({ids.start}, {ids.stop - 1}) AS i"
    return f"{rows} CREATE (:Blob {{id: i, s: repeat(CAST(i AS STRING) + '-', {units})}})"


def walk_ends(pairs: Counter, src: int, lo: int, hi: int) -> Counter:
    """The end node of every walk of lo..hi edges from src over `pairs` ((src, dst) -> count)."""
    out = defaultdict(list)
    for (s, d), n in pairs.items():
        out[s] += [d] * n
    ends, frontier = Counter(), Counter({src: 1})
    for depth in range(1, hi + 1):
        step = Counter()
        for v, c in frontier.items():
            for d in out[v]:
                step[d] += c
        frontier = step
        if depth >= lo:
            ends += frontier
    return ends


def two_hops(pairs: Counter, sources) -> Counter:
    """(a, m, b) for every walk a->m->b over `pairs` with a in `sources`."""
    out = defaultdict(list)
    for (s, d), n in pairs.items():
        out[s] += [d] * n
    return Counter((a, m, b) for a in set(sources) for m in out[a] for b in out[m])
