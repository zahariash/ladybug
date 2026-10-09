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
- `test/storage_compat/` - Storage compatibility across versions (see below)

## Node.js API

Tests live in `tools/nodejs_api/test/` and use the Node.js built-in test runner (`node --test`). Run with `npm test` from `tools/nodejs_api/`.

For guidelines on writing and reviewing these tests, see [Node.js API — Testing Guide](../tools/nodejs_api/docs/nodejs_testing.md).

## Storage Compatibility

`test/storage_compat/` is a pytest suite that opens each database file with two builds and compares
what they read, every build through the Python package (`tools/python_api` on a build's
`liblbug.so`, or a release's wheel): this build reads, upgrades and replays the WAL of databases written by the base
commit and by past releases from PyPI, the base commit reads databases written by this build, and
when the storage version changes this build imports the base commit's export. CI runs it for every
change against the latest release (`storage-compat.yml`), and against every release
`test/storage_compat/releases.py` selects nightly and before anything is published.

```bash
make release
git submodule update --init dataset tools/python_api
uv run --no-project --with pytest pytest test/storage_compat \
    --lib build/release/src/liblbug.so --base-lib <base commit's liblbug.so> --releases latest
```

The releases are selected from `src/include/storage/storage_version_info.h` and PyPI: the last
patch of each of the 3 newest minor versions, and the newest release of every older storage version
this build reads; a release is picked up once its wheels are on PyPI. Known product bugs and the
fixture sections older releases cannot run are listed in `test/storage_compat/known.py`. A failing
run keeps the databases it wrote under pytest's temporary directory (`/tmp/pytest-of-$USER/` by
default); a passing run deletes them.

## Running Tests

See `AGENTS.md` for build and test commands.
