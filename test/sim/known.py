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
        "#1186: partitioned tables lose rows after a crash and accept duplicate primary keys",
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
        "#1184: a CHECKPOINT on another connection breaks open read-only transactions",
    ),
    Known(
        "snapshot_rel_count",
        "#1197: a relationship count(*) in a read-only transaction counts later commits (#1198)",
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
        "txn_drop_column",
        "dropping a middle column inside a write transaction mixes up the transaction's new rows: "
        "wrong values after COMMIT, or a segfault reading them (not filed)",
    ),
    Known(
        "pool_commit",
        "#1182: a COMMIT failing on a full buffer pool deletes checkpointed relationships (#1195)",
    ),
    Known(
        "second_writer",
        "after a refused BEGIN TRANSACTION, the connection's next statement segfaults (not filed)",
    ),
]


def known_rules() -> set[str]:
    return {known.rule for known in KNOWN_BUGS}
