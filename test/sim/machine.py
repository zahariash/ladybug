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
from search import SearchChecks

SCHEMA = [
    "CREATE NODE TABLE Person(id INT64 PRIMARY KEY, name STRING, age INT64, score DOUBLE)",
    "CREATE REL TABLE Knows(FROM Person TO Person, since INT64)",
    "CREATE NODE TABLE L(id INT64, cluster INT64, PRIMARY KEY(id)) PARTITION BY LIST (cluster)",
]
EXTENSION_SCHEMA = ["CREATE NODE TABLE Doc(id INT64 PRIMARY KEY, text STRING, emb FLOAT[3])"]


class LadybugSim(
    DataRules,
    CatalogRules,
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
        self.dir = tempfile.mkdtemp(prefix="lbug-sim-")
        self.path = os.path.join(self.dir, "db")
        self.config = dict(
            buffer_pool_size=256 * 1024 * 1024,
            max_num_threads=threads,
            compression=compression,
            checkpoint_threshold=checkpoint_threshold,
        )
        self.model = Model()
        self.reopen_engine()
        for statement in SCHEMA + (EXTENSION_SCHEMA if self.options.loaded else []):
            self.ok(statement)

    def teardown(self):
        if self.engine:
            self.engine.kill()
        if self.dir and not self.options.keep:
            shutil.rmtree(self.dir, ignore_errors=True)
