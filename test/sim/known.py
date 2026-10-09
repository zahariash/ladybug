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
    Known("columns", "#1158: a prepared insert reused after ALTER TABLE ADD shifts the new column"),
    Known("crash_copy", "#1159: SIGKILL during COPY's checkpoint can corrupt the database"),
    Known("crash_checkpoint", "#1159: SIGKILL during CHECKPOINT can corrupt the primary-key index"),
    Known(
        "race_checkpoint",
        "#1160: reads fail or segfault during CHECKPOINT, also an automatic one",
    ),
    Known(
        "drop_macro", "DROP MACRO followed by a crash: WAL replay fails and the database won't open"
    ),
    Known(
        "partitions", "partitioned tables lose rows after a crash and accept duplicate primary keys"
    ),
    Known("pk_index", "a crash undoes DROP INDEX"),
    Known("crash_create", "a kill mid-write can leave a WAL that fails its checksum on reopen"),
    Known(
        "drop_doc_index",
        "a crash after DROP_FTS_INDEX or DROP_VECTOR_INDEX leaves Doc unwritable, or segfaulting",
    ),
    Known(
        "drop_table",
        "a prepared insert reused after DROP TABLE and CREATE TABLE of the same name fails or "
        "segfaults (prepared statements survive DDL, like #1158)",
    ),
    Known("dotproduct", "dotproduct vector indexes return the least similar vectors first"),
    Known(
        "vector_recall",
        "#1023: inserts after CREATE_VECTOR_INDEX pick neighbours by cosine whatever the metric",
    ),
]


def known_rules() -> set[str]:
    return {known.rule for known in KNOWN_BUGS}
