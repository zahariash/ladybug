# Testing Guide

## Unit Test Structure

```cpp
#include "test_helper/test_helper.h"
#include <gtest/gtest.h>

namespace lbug {
namespace testing {

class MyTest : public DBTest {
    void SetUp() override {
        BaseGraphTest::SetUp();
        // Test setup
    }
};

TEST_F(MyTest, TestCaseName) {
    // Test implementation
}

} // namespace testing
} // namespace lbug
```

## Test Categories

- `test/runner/` - End-to-end tests
- `test/storage/` - Storage layer tests
- `test/transaction/` - Transaction tests
- `test/api/` - API tests
- `test/c_api/` - C API tests
- `test/binder/` - Query binder tests
- `test/planner/` - Query planner tests
- `test/optimizer/` - Query optimizer tests
- `test/sim/` - Random workloads checked against a model (see below)

## Node.js API

Tests live in `tools/nodejs_api/test/` and use the Node.js built-in test runner (`node --test`). Run with `npm test` from `tools/nodejs_api/`.

For guidelines on writing and reviewing these tests, see [Node.js API — Testing Guide](../tools/nodejs_api/docs/nodejs_testing.md).

## Workload Simulation

`test/sim/` is a Hypothesis state machine that runs random workloads through the Python package
and checks the database against an in-memory model of what it should contain. Steps cover writes
with edge values, transactions (with DDL, ending in commit, rollback or a crash), bulk `COPY`
of nodes and edges, a partitioned table, DDL (macros, tables,
`ALTER TABLE`, the primary-key index, and with `--sim-extensions` full-text and vector indexes),
checkpoints, reopens, crashes between and in the middle of statements, and writes racing
concurrent readers. The database runs in a worker process, so a crash is a SIGKILL followed by
WAL replay, and a segfault or a hang fails the test. Query results are compared with the model
on a freshly prepared statement, on its cached plan, through the driver and on a single-threaded
connection. Failures shrink to a minimal sequence of steps, except those that depend on timing.

```bash
make python
PYTHONPATH=tools/python_api/build uv run --no-project --with hypothesis --with pytest \
    pytest test/sim --sim-examples 200 --sim-steps 50
```

Rules that hit known bugs are listed in `test/sim/known.py` and stay off unless `--sim-known` is
given; `--sim-skip <name>` turns off more, `--sim-focus extensions` keeps only the extension
rules, and `--sim-keep` keeps the database directories. They live in `/tmp/lbug-sim-run-<pid>`,
which a run removes at the end; the next run removes those of runs that were killed. With
`--sim-extensions "LOAD EXTENSION fts;LOAD EXTENSION vector"` (or `LOAD EXTENSION '<path>'` for
extensions built from `extension/`), the Doc table and the rules of each loaded extension are
added.
`SIM_TIMEOUT_SECONDS` (300 by default) bounds how long one statement may take; a hang reports the
native stack of every engine thread, and `--sim-gdb` starts workers under gdb so that a crash
reports its native stack too. With `strace` installed, crash points kill the engine at the n-th
call of a file syscall, which does not depend on timing; `--sim-enable <name>` turns a known-bug
rule back on, e.g. to verify a fix.

Every workload records its engine interactions in a trace, each step tagged with its rule.
Traces live in `.hypothesis/traces/` (`--sim-trace-dir`); a failing workload is kept there as
`failed/<signature>/trace.json`, with the failure as the trace's last entry, and the run ends by
printing each failure's directory and the command that starts its triage. Hypothesis saves
failing examples per set of enabled rules and replays them first when a run with the same
options starts again.

Session rules (`--sim-focus sessions`) interleave transactions on three more connections, one
step per rule: read-only snapshots, a write transaction, writes that must fail beside it, and
the query battery against what each connection should see.

### Triaging a failure

Crashes, hangs and errors where the original statement succeeded are replayed without the
model:

```bash
PYTHONPATH=tools/python_api/build uv run --no-project --with hypothesis \
    python test/sim/replay.py .hypothesis/traces/failed/<signature>/trace.json \
    --runs 5 --minimize --emit
```

`--runs` retries timing-dependent failures. `--minimize` delta-debugs the trace to the fewest
steps that still fail the same way, replaces steps with simpler ones (a crash with a clean
reopen, a timed or syscall crash with the statement and a crash, a prepared statement with a
plain one, another connection with the first), and drops the database settings the failure does
not need. `--emit` reruns it once under gdb and writes, next to the trace:

- `trace.min.cypher`: the statements in order, with the connection, crash and reopen steps
  marked and the failing statement flagged;
- `trace.min.py`: a standalone script, one process per worker lifetime, that exits 1 while the
  failure recurs; running it against another build (`PYTHONPATH=<build>`) tells whether that
  build fixes it;
- `finding.json`: the kind of failure, its signature (the failing operation, the statement with
  literals abstracted, the innermost engine frames or the error text), the settings it needs,
  the build and the commands run.

Two findings with the same signature are almost always one bug; the same frames or statement
shape after different steps usually mean one cause reached two ways. Before reporting, try the
script's variants (a node or a rel table, the last or a middle column, a crash or a clean
reopen) to state which conditions matter, check whether an open fix covers it by running the
script on that build, and add the failing rule or variant to `test/sim/known.py`, so that
later runs reach new bugs. Steps that call a parameterized statement a second time can meet
the Python driver's cache of prepared statements; rule the driver out before blaming the
engine.

Wrong results cannot be minimized by replay: dropping an earlier step changes the right answer.
The step list Hypothesis prints is already shrunk, and the trace maps each step to its
statements.

The simulator's own logic (model, search oracles, trace minimization, rule helpers) has unit
tests next to it that need no database: `pytest test/sim -k "not simulation"`.

The end-to-end runner also has a mode for the cached-plan path: with `E2E_REEXECUTE=1`, every
read-only query is prepared once and executed twice, and the second result is checked.

## Running Tests

See `AGENTS.md` for build and test commands.
