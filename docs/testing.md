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
rules, and `--sim-keep` keeps the database directories. With `--sim-extensions "LOAD EXTENSION
fts;LOAD EXTENSION vector"` (or `LOAD EXTENSION '<path>'` for extensions built from
`extension/`), the Doc table and the rules of each loaded extension are added.
`SIM_TIMEOUT_SECONDS` (300 by default) bounds how long one statement may take.

The end-to-end runner also has a mode for the cached-plan path: with `E2E_REEXECUTE=1`, every
read-only query is prepared once and executed twice, and the second result is checked.

## Running Tests

See `AGENTS.md` for build and test commands.
