"""Rules that write C at its size boundaries: appends across vector ends, updates and deletes at
the boundary ids, hubs whose CSR lists span several vectors, and a dense CSR leaf region."""

from __future__ import annotations

import hypothesis.strategies as st
from hypothesis.stateful import precondition, rule
from session import C_SORTED, ONE_THREAD, Session, copy_c, enabled
from strategies import C_ROWS_CAP, HUB_DEGREES, HUBS, LEAF_START, c_values, csr_ids

APPENDS = [1, 2047, 2048, 2049]
# Pairs of consecutive band ids in the first node group: a node and the next one.
DENSE_SOURCES = [20, 1022, 2046]
CREATE_CK = "MATCH (a:C {id: $a}), (b:C {id: $b}) CREATE (a)-[:CK]->(b)"


class BigRules(Session):
    def c_write_safe(self, kind: str, id: int) -> bool:
        """snapshot_write_safe for C: an update under an open snapshot hits snapshot_update, and
        deleting a row the snapshot does not see hits snapshot_insert_delete."""
        readers = [txn.view for txn in self.txns.values() if txn.kind == "read"]
        if not readers:
            return True
        if kind == "update":
            return "snapshot_update" not in self.options.skipped
        hidden = any(not view.big.exists(id) for view in readers)
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
