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
        "#1304: a crash after a CHECKPOINT taken while a read-only transaction is open leaves a "
        "database with a partitioned table unopenable: the partition shadow file is missing",
    ),
    Known(
        "art_copy",
        "#1343: rows that COPY loads into a node table never reach an ART index on its columns, "
        "also after CHECKPOINT and reopen; CREATE does",
    ),
    Known(
        "txn_index_lookups",
        "#1318: a transaction's uncommitted rows are missing from secondary ART index lookups, and "
        "its updates and deletes reach the index at once and survive ROLLBACK",
    ),
    Known(
        "csr_walk_counts",
        "#1341: variable-length counts over CK (REACHABLE_COUNT, GROUPED_REACHABLE_COUNT) miss the "
        "edges of nodes scanned after a node past the rel table's last node group (fix: PR #1342)",
    ),
    Known(
        "hop_two",
        "#1341: an extend over another extend's output misses the edges of nodes it reads after "
        "a node past the rel table's last node group (fix: PR #1342)",
    ),
    Known(
        "hop_varlen",
        "#1341: variable-length matches miss the edges of nodes they expand after a node past the "
        "rel table's last node group (fix: PR #1342)",
    ),
    Known(
        "txn_missing_group",
        "in a write transaction, a two-hop match misses the uncommitted edges of a node batched "
        "with a node past the rel table's last node group, until COMMIT (not filed)",
    ),
    Known(
        "race_edges",
        "#1332: a query segfaults while another connection commits rels: commits edit the CSR "
        "index in place under readers (fix: PR #1333)",
    ),
    Known("pk_index", "#1317: DROP INDEX on the primary-key index ignores the transaction"),
    Known(
        "recovery_duplicate_key",
        "a kill during COPY (or other writes) can leave a database whose reopen fails with a "
        "duplicated primary key (not filed; ranking asked in #1341)",
    ),
    Known(
        "recovery_catalog_exists",
        "a kill during CHECKPOINT can leave a WAL whose replay recreates an existing table or "
        "macro: 'already exists in catalog' (not filed)",
    ),
    Known(
        "rename_table",
        "#1207: renaming a table leaves its SERIAL sequence under the old name, so the name can't "
        "be reused",
    ),
    Known("dotproduct", "#1210: dotproduct vector indexes return the least similar vectors first"),
    Known(
        "vector_recall",
        "#1023: inserts after CREATE_VECTOR_INDEX pick neighbours by cosine whatever the metric",
    ),
    Known(
        "snapshot_update",
        "#1303: a checkpoint while a read-only transaction is open puts updates made since the "
        "last checkpoint on other rows, durably",
    ),
    Known(
        "snapshot_insert_delete",
        "#1305: a row inserted and deleted after a read-only transaction began becomes visible to "
        "it at the next checkpoint",
    ),
    Known(
        "snapshot_prepared",
        "#1220: a prepared statement run inside an older snapshot is not rebound after DDL",
    ),
    Known(
        "txn_column_gap",
        "#1221: after a column is dropped and another added (or a middle one dropped in the "
        "transaction), a transaction's new rows are misplaced until a checkpoint: wrong values "
        "after COMMIT, or a segfault reading them",
    ),
    Known(
        "pool_commit",
        "after the buffer pool has been filled with results and released, rel scans fail with "
        "'buffer pool is full' until restart (not filed)",
    ),
    Known(
        "rollback_add",
        "#1221: a rolled-back ALTER TABLE ADD leaves its column data behind; the next ADD reads it "
        "until a checkpoint",
    ),
    Known(
        "txn_prepared_ddl",
        "#1220: a statement prepared with parameters in a transaction is not rebound after ALTER "
        "TABLE ADD in the same transaction: the new column gets 0, not its default; after DROP and "
        "CREATE TABLE of the same name, an insert into the new table hits the old table's keys "
        "(this case not filed)",
    ),
    Known(
        "rollback_prepared",
        "#1220 (comment): a variable-length query the driver prepared inside a transaction that "
        "added columns and rolled back segfaults or fails with unordered_map::at afterwards",
    ),
    Known(
        "second_writer",
        "#1316: after a refused BEGIN TRANSACTION, the connection's next statement segfaults "
        "(fix: PR #1375)",
    ),
    Known(
        "pinned_checkpoint",
        "#1304: after a CHECKPOINT taken while a read-only transaction is open, a crash during "
        "the next checkpoint, or a failed one, loses the commits since (fix: PR #1378)",
    ),
    Known(
        "graphs",
        "#1304: a crash after a CHECKPOINT taken while a read-only transaction is open leaves a "
        "database with a named graph unopenable: the graph shadow file is missing (fix: PR #1379)",
    ),
    Known(
        "limit_files",
        "#1199: when the automatic checkpoint after a commit fails, the committed statement fails "
        "too (fix: PR #1361)",
    ),
    Known(
        "limit_db_size",
        "#1386: after a write or checkpoint is refused at max_db_size, the next checkpoint, or the "
        "one closing the database runs, segfaults",
    ),
    Known(
        "failed_checkpoint_partitions",
        "with a partitioned table, a checkpoint that fails (a full disk, a full buffer pool) "
        "segfaults rolling back, in FileInfo::truncate (not filed)",
    ),
    Known(
        "wal_write_failure",
        "after a WAL write fails on a full disk, the next commit waits forever for durability "
        "and closing the database aborts on the WAL flush (not filed)",
    ),
    Known(
        "pool_commit_crash",
        "#1377: a COMMIT that fails on a full buffer pool after its WAL record is written is "
        "applied after a crash (fix: PR #1380)",
    ),
    Known(
        "checksums",
        "#1357: opening a database with the other enable_checksums setting drops its WAL "
        "(fix: PR #1358)",
    ),
]


def known_rules() -> set[str]:
    return {known.rule for known in KNOWN_BUGS}
