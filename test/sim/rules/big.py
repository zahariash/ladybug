"""Rules that write C at its size boundaries: appends across vector ends, updates and deletes at
the boundary ids, hubs whose CSR lists span several vectors, a dense CSR leaf region and column
DDL; and CS, whose string keys live past the inline bytes of a string."""

from __future__ import annotations

import hypothesis.strategies as st
from hypothesis.stateful import precondition, rule
from model import cs_key
from session import C_SORTED, DUPLICATE_KEY, ONE_THREAD, Session, copy_c, copy_cs, enabled
from strategies import (
    C_ROWS_CAP,
    CS_ROWS_CAP,
    HUB_DEGREES,
    HUBS,
    LEAF_START,
    c_texts,
    c_values,
    cs_ids,
    csr_ids,
    extra_columns,
)

APPENDS = [1, 2047, 2048, 2049]
# Pairs of consecutive band ids in the first node group: a node and the next one.
DENSE_SOURCES = [20, 1022, 2046]
CREATE_CK = "MATCH (a:C {id: $a}), (b:C {id: $b}) CREATE (a)-[:CK]->(b)"


class BigRules(Session):
    def c_write_safe(self, kind: str, id: int) -> bool:
        """snapshot_write_safe for C."""
        return self.big_write_safe(kind, lambda view: view.big.exists(id))

    def big_write_safe(self, kind: str, visible) -> bool:
        """snapshot_write_safe for C and CS: an update under an open snapshot hits
        snapshot_update, and deleting a row a snapshot does not see (`visible(view)` is false)
        hits snapshot_insert_delete."""
        readers = [txn.view for txn in self.txns.values() if txn.kind == "read"]
        if not readers:
            return True
        if kind == "update":
            return "snapshot_update" not in self.options.skipped
        hidden = any(not visible(view) for view in readers)
        return not (hidden and "snapshot_insert_delete" in self.options.skipped)

    def declare_sorted(self) -> None:
        """A write to C changes its epoch, which turns the CSR rewrites off; ids are still the
        offsets, so declare it again."""
        self.ok(C_SORTED)

    @enabled("big_appends")
    @precondition(
        lambda self: (
            self.has_csr()
            and self.model.big.rows < C_ROWS_CAP
            and self.checkpoint_keeps_snapshots()
        )
    )
    @rule(k=st.sampled_from(APPENDS))
    def append_c(self, k):
        self.ok(ONE_THREAD)
        self.ok(copy_c(self.model.big.rows, k))
        self.ok(f"CALL threads={self.config['max_num_threads']}")
        self.model.big.rows += k
        self.declare_sorted()

    @enabled("big_updates")
    @precondition(lambda self: self.has_csr())
    @rule(id=csr_ids, v=c_values)
    def update_c(self, id, v):
        if not self.c_write_safe("update", id):
            return
        self.ok("MATCH (c:C {id: $id}) SET c.v = $v", dict(id=id, v=v))
        if self.model.big.exists(id):
            self.model.big.updated[id] = v
            self.model.updated_since_checkpoint = True
        self.declare_sorted()

    @enabled("big_strings")
    @precondition(lambda self: self.has_csr())
    @rule(id=csr_ids, text=c_texts)
    def update_c_text(self, id, text):
        if not self.c_write_safe("update", id):
            return
        self.ok("MATCH (c:C {id: $id}) SET c.s = $s", dict(id=id, s=text))
        if self.model.big.exists(id):
            self.model.big.texts[id] = text
            self.model.updated_since_checkpoint = True
        self.declare_sorted()

    @enabled("big_deletes")
    @precondition(lambda self: self.has_csr())
    @rule(id=st.one_of(csr_ids, st.sampled_from(HUBS)))
    def delete_c(self, id):
        if not self.c_write_safe("delete", id):
            return
        self.ok("MATCH (c:C {id: $id}) DETACH DELETE c", dict(id=id))
        if self.model.big.exists(id):
            self.model.forget_c(id)
        self.declare_sorted()

    @enabled("hub_edges")
    @precondition(lambda self: self.has_csr() and self.checkpoint_keeps_snapshots())
    @rule(hub=st.sampled_from(HUBS), degree=st.sampled_from(HUB_DEGREES))
    def hub_edges(self, hub, degree):
        """Edges from a hub to `degree` leaves, by one COPY."""
        if not self.model.big.exists(hub):
            return
        leaves = range(LEAF_START, LEAF_START + degree)
        self.ok(f"COPY CK FROM (UNWIND range({leaves[0]}, {leaves[-1]}) AS i RETURN {hub}, i)")
        self.model.ck.update((hub, leaf) for leaf in leaves)
        self.check_csr_degree(hub)

    @enabled("dense_region")
    @precondition(lambda self: self.has_csr() and self.checkpoint_keeps_snapshots())
    @rule(src=st.sampled_from(DENSE_SOURCES))
    def dense_region(self, src):
        """1000 edges from one node by COPY, 100 from the next one by CREATE, then a checkpoint,
        which has to make room for them in one CSR leaf region."""
        if not (self.model.big.exists(src) and self.model.big.exists(src + 1)):
            return
        copied = range(LEAF_START, LEAF_START + 1000)
        self.ok(f"COPY CK FROM (UNWIND range({copied[0]}, {copied[-1]}) AS i RETURN {src}, i)")
        self.model.ck.update((src, leaf) for leaf in copied)
        for leaf in range(LEAF_START, LEAF_START + 100):
            self.ok(CREATE_CK, dict(a=src + 1, b=leaf))
            self.model.ck[(src + 1, leaf)] += 1
        self.ok("CHECKPOINT")
        self.model.checkpointed()
        self.check_csr_degree(src)
        self.check_csr_degree(src + 1)

    @enabled("big_ddl")
    @precondition(lambda self: self.has_csr())
    @rule(column=extra_columns, default=st.integers(-5, 5))
    def add_c_column(self, column, default):
        """A column of C's every row, also those that COPY appends later without it."""
        query = f"ALTER TABLE C ADD {column} INT64 DEFAULT {default}"
        if column in self.model.big.columns:
            self.fails(query, None, "already has property")
            return
        self.ok(query)
        self.model.big.columns[column] = default
        self.declare_sorted()

    @enabled("big_ddl")
    @precondition(lambda self: self.has_csr() and self.model.big.columns)
    @rule(data=st.data())
    def drop_c_column(self, data):
        column = data.draw(st.sampled_from(list(self.model.big.columns)))
        self.ok(f"ALTER TABLE C DROP {column}")
        del self.model.big.columns[column]
        self.declare_sorted()

    @enabled("big_ddl")
    @precondition(lambda self: self.has_csr() and self.model.big.columns)
    @rule(data=st.data(), new=extra_columns)
    def rename_c_column(self, data, new):
        old = data.draw(st.sampled_from(list(self.model.big.columns)))
        query = f"ALTER TABLE C RENAME {old} TO {new}"
        if new in self.model.big.columns:
            self.fails(query, None, "already has property")
            return
        self.ok(query)
        columns = self.model.big.columns
        self.model.big.columns = {new if c == old else c: d for c, d in columns.items()}
        self.declare_sorted()

    @enabled("string_keys")
    @precondition(lambda self: self.has_string_keys())
    @rule(id=cs_ids, n=st.integers(-1000, 1000))
    def insert_cs(self, id, n):
        """A key that exists must fail, a deleted one must go in again."""
        params = dict(k=cs_key(id), n=n)
        query = "CREATE (:CS {id: $k, n: $n})"
        if self.model.keys.n(id) is not None:
            self.fails(query, params, DUPLICATE_KEY)
            return
        self.ok(query, params)
        self.model.keys.created[id] = n

    @enabled("string_keys")
    @precondition(lambda self: self.has_string_keys())
    @rule(id=cs_ids)
    def delete_cs(self, id):
        if not self.big_write_safe("delete", lambda view: view.keys.n(id) is not None):
            return
        self.ok("MATCH (c:CS {id: $k}) DELETE c", dict(k=cs_key(id)))
        self.model.keys.delete(id)

    @enabled("string_keys")
    @precondition(
        lambda self: (
            self.has_string_keys()
            and self.model.keys.rows < CS_ROWS_CAP
            and self.checkpoint_keeps_snapshots()
        )
    )
    @rule(k=st.sampled_from([1, 2049, 5000]))
    def append_cs(self, k):
        """A COPY into CS's hash index, which grows it past the keys COPY built it from."""
        self.ok(copy_cs(self.model.keys.rows, k))
        self.model.keys.rows += k
