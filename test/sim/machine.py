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
from rules.big import BigRules
from rules.catalog import CatalogRules
from rules.data import DataRules
from rules.durability import DurabilityRules
from rules.extensions import ExtensionRules
from rules.prepared import PreparedRules
from rules.sessions import SessionRules
from rules.transactions import TransactionRules
from search import SearchChecks
from session import FAILURES
from strategies import CK2_SEED, CS_ROWS, CSR_ROWS

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
    BigRules,
    SearchChecks,
    Checks,
    RuleBasedStateMachine,
):
    dir = None

    @initialize(
        threads=st.integers(1, 8),
        compression=st.booleans(),
        checkpoint_threshold=st.sampled_from([-1, 0, 4096]),
        c_fill=st.sampled_from(["copy", "create"]),
    )
    def open_db(self, threads, compression, checkpoint_threshold, c_fill):
        # The files of the current workload live with its trace, so a failure can be replayed.
        self.files = os.path.join(self.options.traces or self.dir, f"current-{os.getpid()}")
        shutil.rmtree(self.files, ignore_errors=True)
        os.makedirs(self.files)
        self.config = dict(
            buffer_pool_size=256 * 1024 * 1024,
            max_num_threads=threads,
            compression=compression,
            checkpoint_threshold=checkpoint_threshold,
            # The engine's default: recovery drops a WAL tail a crash tore, which the crash checks
            # then verify; the Python driver's default refuses to open the database instead.
            throw_on_wal_replay_failure=False,
        )
        self.c_fill = c_fill
        self.new_database()

    def new_database(self, fresh: bool = False) -> None:
        """An empty database with the workload's settings and schema. A fresh one, after a known
        recovery failure, goes on in the same trace."""
        self.dir = tempfile.mkdtemp(prefix="db-", dir=self.options.scratch)
        self.path = os.path.join(self.dir, "db")
        self.model = Model()
        if fresh:
            self.traced(dict(op="fresh", loads=self.loads()), self.open_engine)
        else:
            self.start_trace()
        if self.has_csr():
            self.model.big.rows = CSR_ROWS
        if self.has_string_keys():
            self.model.keys.rows = CS_ROWS
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
        if self.has_csr():
            create = "MATCH (a:C {id: $a}), (b:C {id: $b}) CREATE (a)-[:CK2 {w: 0}]->(b)"
            for src, dst in CK2_SEED:
                self.ok(create, dict(a=src, b=dst))
                self.model.ck2[(src, dst, 0)] += 1

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
