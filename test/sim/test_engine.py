"""Pieces of engine.py and the race state that work without a database."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys

import pytest

from engine import (
    CRASH_TRACE_SOURCE,
    GDB_WRAPPER,
    KEEP_MARKER,
    SCRATCH_PREFIX,
    STRACE_WRAPPER,
    backtrace_frames,
    crash_tracer,
    native_stacks,
    read_state,
    scratch_root,
    wrapper_executable,
)
from model import Model
from rules.durability import (
    CK_STATE_QUERIES,
    STATE_QUERIES,
    bulk_ck_writes,
    bulk_edges,
    ck_state,
    race_state,
)
from strategies import CSR_ROWS

GDB_OUTPUT = """\
[New LWP 12]
Thread 2 (Thread 0x7f (LWP 12) "python"):
#0  0x7f in __futex_abstimed_wait_common64 (private=0) at ./nptl/futex-internal.c:57
#5  0x7f in lbug::common::TaskScheduler::runWorkerThread() () from liblbug.so
Thread 1 (Thread 0x7e (LWP 11) "python"):
#0  0x7e in lbug::storage::Column::canCheckpointInPlace(lbug::storage::SegmentState const&) const
#1  0x7e in PyObject_Call ()
"""


class FakeResult:
    def __init__(self, rows: list) -> None:
        self.rows = rows

    def get_all(self) -> list:
        return self.rows


class FakeConnection:
    """Answers each state query with fixed rows, as the driver returns them."""

    def __init__(self, answers: dict) -> None:
        self.answers = answers

    def execute(self, query: str) -> FakeResult:
        return FakeResult(self.answers.get(query, []))


def test_native_stacks_keep_threads_and_engine_frames() -> None:
    assert native_stacks(GDB_OUTPUT).splitlines() == [
        "Thread 2",
        "#5 lbug::common::TaskScheduler::runWorkerThread",
        "Thread 1",
        "#0 lbug::storage::Column::canCheckpointInPlace",
    ]


def test_race_state_has_the_shape_read_state_returns() -> None:
    model = Model()
    model.add_persons(
        {1: model.new_person(age=5), 2: model.new_person(), 3: model.new_person(age=500)}
    )
    model.knows[(1, 2, 0)] += 2
    model.knows[(1, 3, 7)] += 1
    model.knows[(3, 1, 0)] += 1
    answers = {STATE_QUERIES[0]: [[3, 6, 2, 5]], STATE_QUERIES[1]: [[4]], STATE_QUERIES[2]: [[2]]}
    assert race_state(model) == read_state(FakeConnection(answers), STATE_QUERIES)


def test_race_state_of_an_empty_model() -> None:
    answers = {
        STATE_QUERIES[0]: [[0, None, 0, None]],
        STATE_QUERIES[1]: [[0]],
        STATE_QUERIES[2]: [[0]],
    }
    assert race_state(Model()) == read_state(FakeConnection(answers), STATE_QUERIES)


def test_ck_state_counts_edges_and_their_sources() -> None:
    model = Model()
    model.ck.update([(0, 5), (0, 5), (131072, 1)])
    model.ck2[(1, 131072, 0)] += 1
    answers = {CK_STATE_QUERIES[0]: [[3]], CK_STATE_QUERIES[1]: [[2]], CK_STATE_QUERIES[2]: [[3]]}
    answers[CK_STATE_QUERIES[3]] = [[1]]
    assert ck_state(model) == read_state(FakeConnection(answers), CK_STATE_QUERIES)


def test_bulk_edges_have_distinct_sources_and_match_the_query() -> None:
    edges = bulk_edges(1950, 100, 3)
    assert len({src for src, _ in edges}) == 100
    assert edges[0] == ((1950 * 67 + 3) % CSR_ROWS, 1950 * 3)
    creates = bulk_ck_writes("create", 1950, 100, 3)
    assert [(p["a"], p["b"]) for _, p in creates] == edges
    [(copy, params)] = bulk_ck_writes("copy", 0, 1, 1)
    rows = f"UNWIND range(0, 0) AS i RETURN (i * 67 + 1) % {CSR_ROWS}, (i * 1) % {CSR_ROWS}"
    assert copy == f"COPY CK FROM ({rows})"
    assert params == {}


BACKTRACE = """signal 11 address (nil)
/tmp/lbug-sim-crash-trace-1000-0123456789ab.so(+0x1472)[0x7e2fe2c3a472]
/lib/x86_64-linux-gnu/libc.so.6(+0x45330)[0x7e2fe2845330]
/x/_lbug.so(_ZNK4lbug7storage11ColumnChunk4scanEm+0x50)[0x1]
/x/_lbug.so(+0x123456)[0x2]
/x/_lbug.so(_ZNK4lbug7storage16ChunkedNodeGroup4scanEmm+0x59e)[0x3]
/lib/x86_64-linux-gnu/libstdc++.so.6(+0xf5c79)[0x7e2fdecf5c79]
"""


@pytest.mark.skipif(not shutil.which("c++filt"), reason="needs c++filt")
def test_backtrace_frames_keep_the_engine_frames_innermost_first() -> None:
    assert backtrace_frames(BACKTRACE).splitlines() == [
        "Thread that crashed: signal 11 address (nil)",
        "#0 lbug::storage::ColumnChunk::scan",
        "#1 lbug::storage::ChunkedNodeGroup::scan",
    ]


def test_the_crash_tracer_writes_the_backtrace_and_keeps_the_signal(tmp_path) -> None:
    library = crash_tracer()
    if library is None:
        pytest.skip("no C compiler")
    assert os.path.basename(CRASH_TRACE_SOURCE) == "crash_trace.c"
    crash = f"import ctypes; ctypes.CDLL({library!r}); ctypes.string_at(0)"
    env = dict(os.environ, SIM_CRASH_TRACE_DIR=str(tmp_path))
    result = subprocess.run([sys.executable, "-c", crash], env=env, capture_output=True)
    assert result.returncode == -11
    (trace,) = tmp_path.iterdir()
    assert trace.name.startswith("crash-")
    assert trace.read_text().startswith("signal 11 address")


def test_the_strace_wrapper_kills_at_the_requested_call() -> None:
    path = wrapper_executable("strace", STRACE_WRAPPER)
    assert os.access(path, os.X_OK)
    script = open(path).read()
    assert "signal=KILL:when=$SIM_STRACE_WHEN" in script and "exec strace" in script


def test_the_gdb_wrapper_starts_the_worker_and_writes_stacks_to_a_file() -> None:
    script = open(wrapper_executable("gdb", GDB_WRAPPER)).read()
    assert "-ex run" in script and '--args "$SIM_PYTHON"' in script
    assert script.rstrip().endswith('> "$SIM_GDB_OUT" 2>&1')


def test_scratch_roots_of_dead_processes_are_removed(tmp_path) -> None:
    finished = subprocess.run(
        [sys.executable, "-c", "import os; print(os.getpid())"], capture_output=True, text=True
    )
    dead = tmp_path / f"{SCRATCH_PREFIX}{finished.stdout.strip()}"
    kept = tmp_path / f"{SCRATCH_PREFIX}{int(finished.stdout) + 1}"
    other = tmp_path / f"{SCRATCH_PREFIX}notapid"
    for path in (dead, kept, other):
        (path / "db-1").mkdir(parents=True)
    (kept / KEEP_MARKER).touch()
    root = scratch_root(str(tmp_path))
    assert root == str(tmp_path / f"{SCRATCH_PREFIX}{os.getpid()}")
    assert not dead.exists()
    assert kept.exists() and other.exists()
