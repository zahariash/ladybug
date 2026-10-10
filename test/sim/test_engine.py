"""Pieces of engine.py and the race state that work without a database."""

from __future__ import annotations

import os
import subprocess
import sys

from engine import (
    KEEP_MARKER,
    SCRATCH_PREFIX,
    native_stacks,
    read_state,
    scratch_root,
    strace_executable,
)
from model import Model
from rules.durability import STATE_QUERIES, race_state

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
    conn = FakeConnection({STATE_QUERIES[0]: [[3, 6, 2, 5]], STATE_QUERIES[1]: [[2]]})
    assert race_state(model) == read_state(conn, STATE_QUERIES)


def test_race_state_of_an_empty_model() -> None:
    conn = FakeConnection({STATE_QUERIES[0]: [[0, None, 0, None]], STATE_QUERIES[1]: [[0]]})
    assert race_state(Model()) == read_state(conn, STATE_QUERIES)


def test_the_strace_wrapper_kills_at_the_requested_call() -> None:
    path = strace_executable()
    assert os.access(path, os.X_OK)
    script = open(path).read()
    assert "signal=KILL:when=$SIM_STRACE_WHEN" in script and "exec strace" in script


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
