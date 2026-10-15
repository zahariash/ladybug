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
    Known("crash_copy", "#1159: SIGKILL during COPY's checkpoint can corrupt the database"),
    Known("crash_checkpoint", "#1159: SIGKILL during CHECKPOINT can corrupt the primary-key index"),
    Known(
        "race_checkpoint",
        "#1194: a relationship count(*) segfaults while another connection checkpoints",
    ),
    Known(
        "partitions",
        "a crash after a CHECKPOINT taken while a read-only transaction is open leaves a database "
        "with a partitioned table unopenable: the partition shadow file is missing (not filed)",
    ),
    Known(
        "txn_index_lookups",
        "a transaction's uncommitted rows are missing from secondary ART index lookups (not filed)",
    ),
    Known("pk_index", "DROP INDEX on the primary-key index ignores the transaction (not filed)"),
    Known("crash_create", "a kill mid-write can leave a WAL that fails its checksum on reopen"),
    Known(
        "rel_columns",
        "#1185: after DROP of a property, a crash and a CHECKPOINT, re-adding a property of the "
        "same name brings back the dropped values",
    ),
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
        "snapshot_checkpoint",
        "#1184: a CHECKPOINT on another connection, also the one COPY or CREATE INDEX forces, "
        "breaks open read-only transactions",
    ),
    Known(
        "snapshot_rel_count",
        "a relationship count(*) in a read-only transaction counts relationships another "
        "connection COPYs in after it began (not filed)",
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
        "readd_column",
        "re-adding a dropped Person column after crashes makes a later CHECKPOINT segfault "
        "(#1185 family, not filed)",
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
        "second_writer",
        "after a refused BEGIN TRANSACTION, the connection's next statement segfaults (not filed)",
    ),
]


def known_rules() -> set[str]:
    return {known.rule for known in KNOWN_BUGS}
