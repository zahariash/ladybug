"""The state machine: one database, the model of what it should contain, and every rule."""

from __future__ import annotations

import os
import shutil
import sys
import tempfile

import hypothesis.strategies as st
from checks import Checks
from hypothesis.stateful import RuleBasedStateMachine, initialize
from model import Model
from rules.catalog import CatalogRules
from rules.data import DataRules
from rules.durability import DurabilityRules
from rules.extensions import ExtensionRules
from rules.prepared import PreparedRules
from rules.sessions import SessionRules
from rules.transactions import TransactionRules
from search import SearchChecks
from session import FAILURES

SCHEMA = [
    "CREATE NODE TABLE Person(id INT64 PRIMARY KEY, name STRING, age INT64, score DOUBLE)",
    "CREATE REL TABLE Knows(FROM Person TO Person, since INT64)",
]
PARTITIONED_SCHEMA = [
    "CREATE NODE TABLE L(id INT64, cluster INT64, PRIMARY KEY(id)) PARTITION BY LIST (cluster)",
]
EXTENSION_SCHEMA = ["CREATE NODE TABLE Doc(id INT64 PRIMARY KEY, text STRING, emb FLOAT[3])"]


class LadybugSim(
    DataRules,
    TransactionRules,
    CatalogRules,
    PreparedRules,
    SessionRules,
    ExtensionRules,
    DurabilityRules,
    SearchChecks,
    Checks,
    RuleBasedStateMachine,
):
    dir = None

    @initialize(
        threads=st.integers(1, 8),
        compression=st.booleans(),
        checkpoint_threshold=st.sampled_from([-1, 0, 4096]),
    )
    def open_db(self, threads, compression, checkpoint_threshold):
        self.dir = tempfile.mkdtemp(prefix="db-", dir=self.options.scratch)
        self.path = os.path.join(self.dir, "db")
        # The files of the current workload live with its trace, so a failure can be replayed.
        self.files = os.path.join(self.options.traces or self.dir, f"current-{os.getpid()}")
        shutil.rmtree(self.files, ignore_errors=True)
        os.makedirs(self.files)
        self.config = dict(
            buffer_pool_size=256 * 1024 * 1024,
            max_num_threads=threads,
            compression=compression,
            checkpoint_threshold=checkpoint_threshold,
        )
        self.model = Model()
        self.start_trace()
        # SIM_ART_PK=1 gives Person an ART primary key instead of the default hash index, so that
        # primary-key ranges are index lookups.
        art_pk = bool(os.environ.get("SIM_ART_PK"))
        if art_pk:
            self.ok("CALL enable_default_hash_index=false")
        schema = SCHEMA + (PARTITIONED_SCHEMA if self.has_partitions() else [])
        for statement in schema + (EXTENSION_SCHEMA if self.options.loaded else []):
            self.ok(statement)
        if art_pk:
            self.ok("CREATE ART INDEX person_pk FOR (p:Person) ON (p.id)")
            self.model.pk_index = "person_pk"

    def teardown(self):
        # Hypothesis calls teardown in a finally block, so a failing step's error is in flight;
        # its own exceptions, which end an example early, are not failures.
        error = sys.exc_info()[1]
        if not isinstance(error, Exception) or type(error).__module__.startswith("hypothesis"):
            error = None
        if getattr(self, "trace", None) is not None:
            record = self.record_failure(error) if error is not None else None
            self.save_trace()
            if record:
                self.keep_failure(record)
        if self.engine:
            self.engine.kill()
        if self.dir and not self.options.keep:
            shutil.rmtree(self.dir, ignore_errors=True)

    def keep_failure(self, record: dict) -> None:
        """Copies the failing workload's trace and files to failed/<signature>/, which a smaller
        example of the same failure replaces while Hypothesis shrinks."""
        slug = record["signature"]["slug"]
        kept = os.path.join(self.options.traces or self.dir, "failed", slug)
        copy = f"{kept}.{os.getpid()}"
        shutil.rmtree(copy, ignore_errors=True)
        shutil.copytree(self.files, copy)
        shutil.rmtree(kept, ignore_errors=True)
        os.replace(copy, kept)
        failure = (record["kind"], slug, kept, record.get("probe"), record.get("rows"))
        if failure not in FAILURES:
            FAILURES.append(failure)
