"""The state machine: one database, the model of what it should contain, and every rule."""

from __future__ import annotations

import os
import shutil
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
from rules.transactions import TransactionRules
from search import SearchChecks

SCHEMA = [
    "CREATE NODE TABLE Person(id INT64 PRIMARY KEY, name STRING, age INT64, score DOUBLE)",
    "CREATE REL TABLE Knows(FROM Person TO Person, since INT64)",
    "CREATE NODE TABLE L(id INT64, cluster INT64, PRIMARY KEY(id)) PARTITION BY LIST (cluster)",
]
EXTENSION_SCHEMA = ["CREATE NODE TABLE Doc(id INT64 PRIMARY KEY, text STRING, emb FLOAT[3])"]


class LadybugSim(
    DataRules,
    TransactionRules,
    CatalogRules,
    PreparedRules,
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
        self.files = os.path.join(self.options.traces or self.dir, "current")
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
        for statement in SCHEMA + (EXTENSION_SCHEMA if self.options.loaded else []):
            self.ok(statement)

    def teardown(self):
        if getattr(self, "trace", None) is not None:
            self.save_trace()
        if self.engine:
            self.engine.kill()
        if self.dir and not self.options.keep:
            shutil.rmtree(self.dir, ignore_errors=True)
