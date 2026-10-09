"""Random workloads against an in-memory model of what the database should contain.

Hypothesis drives random sequences of steps on one database (see rules/):

- writes with edge values, transactions that commit or roll back, bulk COPY, and a partitioned
  table;
- DDL: macros, tables, ALTER TABLE ADD/DROP, Person's primary-key index, and with
  --sim-extensions full-text and vector indexes;
- checkpoints, clean reopens, crashes between statements, and crashes in the middle of a large
  statement, after which the data must be all of it or none;
- batches of writes racing readers, each of which must see committed states, in order.

The database runs in a worker process (engine.py), so a crash is a SIGKILL followed by WAL
replay, and an engine segfault or hang fails the test instead of the run. After every step the
data and the catalog are checked against the model (checks.py), and query rules compare their
results with it four ways (Session.check): a freshly prepared statement, the same statement
again on its cached plan, the driver's own path and a single-threaded connection. Failures
shrink to a minimal sequence of steps, except those that depend on timing.

Rules that hit the known bugs in known.py are off unless --sim-known is given. See
docs/testing.md for how to run it.
"""

from __future__ import annotations

from hypothesis.stateful import run_state_machine_as_test
from machine import LadybugSim


def test_simulation(sim_options, sim_settings) -> None:
    LadybugSim.options = sim_options
    run_state_machine_as_test(LadybugSim, settings=sim_settings)
