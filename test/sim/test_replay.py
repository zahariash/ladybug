"""Trace analysis and minimization in replay.py, on canned traces and a fake failure."""

from __future__ import annotations

from replay import describe, minimize, original_failure

OPEN = {"op": "open", "config": {}, "loads": []}


def execute(query: str, outcome: str = "ok", expected: bool = False) -> dict:
    op = {"op": "execute", "conn": 0, "query": query, "params": {}, "outcome": outcome}
    return op | ({"expected": True} if expected else {})


def test_a_trace_that_failed_a_model_check_has_no_engine_failure() -> None:
    assert original_failure([OPEN, execute("RETURN 1")]) is None


def test_expected_errors_are_not_failures() -> None:
    trace = [OPEN, execute("CREATE (:P {id: 1})", "error: duplicated primary key", True)]
    assert original_failure(trace) is None


def test_the_first_engine_failure_is_found() -> None:
    trace = [
        OPEN,
        execute("CREATE (:P {id: 1})", "error: duplicated primary key", True),
        execute("MATCH (n) RETURN n", "error: unordered_map::at"),
        execute("CHECKPOINT", "died: engine process died with exit code -11"),
    ]
    assert original_failure(trace) == ("error", "unordered_map::at")
    assert original_failure([OPEN, trace[3]]) == ("died", "")


def test_race_errors_count_as_failures() -> None:
    race = {"op": "race", "writes": [], "queries": [], "readers": 2, "outcome": "ok"}
    assert original_failure([OPEN, race | {"errors": ["reader: unordered_map::at"]}]) == (
        "race",
        "",
    )


def test_minimize_keeps_only_the_steps_the_failure_needs() -> None:
    trace = [OPEN] + [execute(f"step {i}") for i in range(20)]

    def fails(candidate: list) -> bool:
        queries = {op.get("query") for op in candidate}
        return {"step 3", "step 7"} <= queries

    minimized = minimize(trace, fails)
    assert minimized == [OPEN, execute("step 3"), execute("step 7")]


def test_minimize_always_keeps_the_open_step() -> None:
    trace = [OPEN] + [execute(f"step {i}") for i in range(5)]
    minimized = minimize(trace, lambda candidate: True)
    assert minimized[0] == OPEN and len(minimized) == 2


def test_describe_names_crash_points_and_races() -> None:
    crash = {"op": "crash_at", "query": "CHECKPOINT", "syscall": "pwrite64", "n": 65}
    assert "pwrite64 #65 in CHECKPOINT" in describe(crash)
    race = {"op": "race", "writes": [1, 2, 3], "queries": [], "readers": 4}
    assert "3 writes, 4 readers" in describe(race)
