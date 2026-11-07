"""What overlapping write transactions wrote, and whether the next write conflicts with it.

With multi-writer mode on, write transactions overlap and each works on its own snapshot. A write
fails with a write-write conflict, which rolls back its whole transaction, when it hits what another
transaction wrote and that transaction is still open or committed after this one began:

- updating a row another transaction updated;
- deleting a row or relationship another transaction deleted (DETACH DELETE counts each edge);
- changing a catalog entry another transaction changed;
- changing a table's columns, or dropping it, while another open transaction has inserts staged in
  it, and the first insert into a table whose columns another transaction changed or dropped.

Updating a row another transaction deleted is not a conflict, in either order, and the delete wins.
A row inserted later under the deleted id is another row: the update does not reach it, or conflict
with one made to it.

Two more overlaps should conflict but do not, so they are gated (see known.py): the same new primary
key committed twice, and a relationship added to a node another transaction deleted. Inserting a key
a transaction already inserted and deleted is gated too: beside another commit, recovery fails. So
is using a key inserted since the snapshot once the transaction has new rows: it finds its own row.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

WW_UPDATE = "Write-write conflict of updating the same row"
WW_DELETE = "Write-write conflict: deleting a row"
WW_CATALOG = "Write-write conflict on"
PERSON = "Person"

# What conflict() returns for a write that goes ahead.
OK = None
# A write the rule must not run while its gate is on.
GATED_INSERT = "concurrent_insert"
GATED_EDGE = "concurrent_detach"
GATED_REINSERT = "reinsert_beside_commit"
GATED_HIDDEN = "hidden_key_lookup"
GATES = (GATED_INSERT, GATED_EDGE, GATED_REINSERT, GATED_HIDDEN)


@dataclass
class Writes:
    """What one transaction wrote. Edges are (table, src, dst)."""

    updated: set = field(default_factory=set)  # (person, its generation in the writer's view)
    deleted: set = field(default_factory=set)  # persons
    inserted: set = field(default_factory=set)  # primary keys of new persons
    edges_deleted: set = field(default_factory=set)
    edges_added: set = field(default_factory=set)
    catalog: set = field(default_factory=set)  # entries it changed: "Person", "T1", "m1"
    staged: set = field(default_factory=set)  # tables with inserts it has staged
    own_edges: Counter = field(default_factory=Counter)  # CK and CK2 edges it added, uncommitted

    def touches_person(self, id: int) -> bool:
        """Whether it added an edge at person `id`, which that person's DETACH DELETE won't see."""
        return any(id in (src, dst) for table, src, dst in self.edges_added if table == "knows")


@dataclass
class Others:
    """What the transactions overlapping a write wrote: `open` ones may still commit or roll back,
    `committed` ones finished after the writer began."""

    open: list = field(default_factory=list)
    committed: list = field(default_factory=list)

    def all(self) -> list:
        return self.open + self.committed


def knows_edges(view, id: int) -> set:
    """The Knows edges of person `id` in `view`."""
    return {("knows", s, d) for s, d, _ in view.knows if id in (s, d)}


def edge_count(view, table: str, a: int, b: int) -> int:
    """How many CK or CK2 edges from `a` to `b` are in `view`."""
    if table == "ck":
        return view.ck[(a, b)]
    return sum(n for key, n in view.ck2.items() if key[:2] == (a, b))


def hidden_key(id: int, view, own: Writes, others: Others) -> bool:
    """Whether a lookup of `id` may find the transaction's own new row: the id was inserted since
    the snapshot, and it either does not see the id or sees a row that was deleted. The second case
    collides only if that row was checkpointed, which the model does not track, so it is gated
    whether or not it was."""
    if PERSON not in own.staged or not any(id in w.inserted for w in others.committed):
        return False
    return id not in view.persons or any(id in w.deleted for w in others.committed)


def conflict(kind: str, a, b, view, own: Writes, others: Others) -> str | None:
    """The error the op must fail with, a gate if it must not run, or None if it goes ahead. `view`
    is what the writing transaction sees, `own` what it wrote; an op that matches nothing in `view`
    writes nothing."""
    every = others.all()
    if kind in ("set_age", "delete") and hidden_key(a, view, own, others):
        return GATED_HIDDEN
    if kind == "knows" and (hidden_key(a, view, own, others) or hidden_key(b, view, own, others)):
        return GATED_HIDDEN
    if kind == "duplicate":
        return WW_CATALOG if any(PERSON in w.catalog for w in every) else OK
    if kind == "insert":
        if any(PERSON in w.catalog for w in every):
            return WW_CATALOG
        if a in own.inserted:
            return GATED_REINSERT
        return GATED_INSERT if any(a in w.inserted for w in every) else OK
    if kind == "set_age":
        row = (a, view.generation[a])
        return WW_UPDATE if a in view.persons and any(row in w.updated for w in every) else OK
    if kind == "delete":
        if a not in view.persons:
            return OK
        edges = knows_edges(view, a)
        if any(a in w.deleted or edges & w.edges_deleted for w in every):
            return WW_DELETE
        return GATED_EDGE if any(w.touches_person(a) for w in every) else OK
    if kind == "knows":
        exists = a in view.persons and b in view.persons
        return GATED_EDGE if exists and any(a in w.deleted or b in w.deleted for w in every) else OK
    if kind in ("unck", "unck2"):
        table = "ck" if kind == "unck" else "ck2"
        # Edges it added itself are its own; only those in its snapshot can conflict.
        if edge_count(view, table, a, b) <= own.own_edges[(table, a, b)]:
            return OK
        return WW_DELETE if any((table, a, b) in w.edges_deleted for w in every) else OK
    if kind == "insert_t":
        return WW_CATALOG if any(f"T{a}" in w.catalog for w in every) else OK
    if kind == "create_table" or kind == "create_macro" or kind == "drop_macro":
        entry = f"T{a}" if kind == "create_table" else f"m{a}"
        return WW_CATALOG if any(entry in w.catalog for w in every) else OK
    if kind == "drop_table":
        changed = any(f"T{a}" in w.catalog for w in every)
        return WW_CATALOG if changed or any(f"T{a}" in w.staged for w in others.open) else OK
    if kind in ("add_column", "drop_column"):
        changed = any(PERSON in w.catalog for w in every)
        return WW_CATALOG if changed or any(PERSON in w.staged for w in others.open) else OK
    return OK


def record(writes: Writes, kind: str, a, b, view) -> None:
    """Adds an op that went ahead to `writes`; `view` is the writer's view before the op."""
    if kind == "insert":
        writes.inserted.add(a)
        writes.staged.add(PERSON)
    elif kind == "set_age" and a in view.persons:
        writes.updated.add((a, view.generation[a]))
    elif kind == "delete" and a in view.persons:
        writes.deleted.add(a)
        writes.edges_deleted |= knows_edges(view, a)
    elif kind == "knows" and a in view.persons and b in view.persons:
        writes.edges_added.add(("knows", a, b))
    elif kind in ("unck", "unck2"):
        table = "ck" if kind == "unck" else "ck2"
        # Its DELETE removes its own edges too, and any later conflict is over the snapshot's.
        if edge_count(view, table, a, b) > writes.own_edges[(table, a, b)]:
            writes.edges_deleted.add((table, a, b))
        writes.own_edges.pop((table, a, b), None)
    elif kind in ("ck", "ck2") and view.big.exists(a) and view.big.exists(b):
        writes.own_edges[(kind, a, b)] += 1
    elif kind == "insert_t":
        writes.staged.add(f"T{a}")
    elif kind in ("create_table", "drop_table"):
        writes.catalog.add(f"T{a}")
    elif kind in ("create_macro", "drop_macro"):
        writes.catalog.add(f"m{a}")
    elif kind in ("add_column", "drop_column"):
        writes.catalog.add(PERSON)
