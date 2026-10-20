"""Known product bugs the simulator would hit at once, and the rules that hit them.

These rules are off by default so that a run reaches new bugs; --sim-known turns them on. When a
bug is fixed, delete its entry and the rule runs again.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Known:
    rule: str  # a rule name, or a rule variant such as crash_copy or race_checkpoint
    reason: str


KNOWN_BUGS = [
    Known(
        "partitions",
        "a crash after a CHECKPOINT taken while a read-only transaction is open leaves a database "
        "with a partitioned table unopenable: the partition shadow file is missing (not filed)",
    ),
    Known(
        "txn_index_lookups",
        "#1318: a transaction's uncommitted rows are missing from secondary ART index lookups, and "
        "its updates and deletes reach the index at once and survive ROLLBACK",
    ),
    Known(
        "csr_walk_counts",
        "variable-length counts over CK (REACHABLE_COUNT, GROUPED_REACHABLE_COUNT) miss the edges "
        "of nodes scanned after a node past the rel table's last node group (not filed)",
    ),
    Known(
        "race_edges",
        "a query segfaults while another connection commits rels: commits edit the CSR index in "
        "place under readers (not filed; fix branch fix/csr-index-append-cow, 9b8c1116d)",
    ),
    Known("pk_index", "#1317: DROP INDEX on the primary-key index ignores the transaction"),
    Known("crash_create", "a kill mid-write can leave a WAL that fails its checksum on reopen"),
    Known(
        "rename_table",
        "renaming a table leaves its SERIAL sequence under the old name, so the name can't be "
        "reused (not filed)",
    ),
    Known("dotproduct", "dotproduct vector indexes return the least similar vectors first"),
    Known(
        "vector_recall",
        "#1023: inserts after CREATE_VECTOR_INDEX pick neighbours by cosine whatever the metric",
    ),
    Known(
        "snapshot_update",
        "a checkpoint while a read-only transaction is open puts updates made since the last "
        "checkpoint on other rows, durably (not filed)",
    ),
    Known(
        "snapshot_insert_delete",
        "a row inserted and deleted after a read-only transaction began becomes visible to it "
        "at the next checkpoint (not filed)",
    ),
    Known(
        "snapshot_prepared",
        "a prepared statement run inside an older snapshot is not rebound after DDL (not filed)",
    ),
    Known(
        "txn_column_gap",
        "after a column is dropped and another added (or a middle one dropped in the transaction), "
        "a transaction's new rows are misplaced until a checkpoint: wrong values after COMMIT, or "
        "a segfault reading them (not filed)",
    ),
    Known(
        "pool_commit",
        "after the buffer pool has been filled with results and released, rel scans fail with "
        "'buffer pool is full' until restart (not filed)",
    ),
    Known(
        "rollback_add",
        "a rolled-back ALTER TABLE ADD leaves its column data behind; the next ADD reads it until "
        "a checkpoint (not filed)",
    ),
    Known(
        "txn_prepared_ddl",
        "a statement prepared with parameters in a transaction is not rebound after ALTER TABLE "
        "ADD in the same transaction: the new column gets 0, not its default; after DROP and "
        "CREATE TABLE of the same name, an insert into the new table hits the old table's keys "
        "(not filed)",
    ),
    Known(
        "rollback_prepared",
        "#1220 (comment): a variable-length query the driver prepared inside a transaction that "
        "added columns and rolled back segfaults or fails with unordered_map::at afterwards",
    ),
    Known(
        "second_writer",
        "#1316: after a refused BEGIN TRANSACTION, the connection's next statement segfaults",
    ),
]


def known_rules() -> set[str]:
    return {known.rule for known in KNOWN_BUGS}
