"""Trace analysis and minimization in replay.py, on canned traces and a fake failure."""

from __future__ import annotations

from collections import Counter

from replay import (
    agreement,
    describe,
    minimize,
    original_failure,
    relocate_files,
    same_failure,
)

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


def test_relocate_files_points_copies_at_the_kept_directory() -> None:
    copy = execute('COPY Person FROM "/runs/traces/current-4242/bulk1000.csv"')
    kept = "/runs/traces/failed/error-crash-abc"
    relocated = relocate_files([OPEN, copy, execute("RETURN '/x/current-1'")], kept)
    assert relocated[1]["query"] == f'COPY Person FROM "{kept}/bulk1000.csv"'
    assert relocated[2]["query"] == "RETURN '/x/current-1'"


def test_same_failure_compares_frames_then_error_text_then_operation() -> None:
    crash = "engine process died\nThread that crashed:\n#0 lbug::a\n#1 lbug::b\n#2 lbug::c"
    target = {
        "kind": "died",
        "frames": ["lbug::a", "lbug::b", "lbug::c"],
        "error": "",
        "operation": "MATCH",
    }
    assert same_failure(target, "died", execute("CHECKPOINT"), crash)
    assert not same_failure(
        target, "died", execute("MATCH (n) RETURN n"), crash.replace("::b", "::x")
    )
    assert not same_failure(target, "hung", execute("MATCH (n) RETURN n"), crash)
    error = {
        "kind": "error",
        "frames": [],
        "error": "Runtime exception: key ? missing",
        "operation": "MATCH",
    }
    assert same_failure(error, "error", execute("RETURN 2"), "Runtime exception: key 17 missing")
    assert not same_failure(error, "error", execute("RETURN 2"), "Runtime exception: other")
    bare = {"kind": "died", "frames": [], "error": "", "operation": "CHECKPOINT"}
    assert same_failure(bare, "died", execute("CHECKPOINT"), "engine process died")
    assert not same_failure(bare, "died", execute("RETURN 1"), "engine process died")


def test_agreement_groups_the_ways_by_their_rows() -> None:
    right, wrong = Counter({"[1]": 1, "[2]": 1}), Counter({"[1]": 1})
    ways = {"driver": wrong, "fresh plan": wrong, "cached plan": right, "no optimizer": right}
    assert agreement(ways) == [["cached plan", "no optimizer"], ["driver", "fresh plan"]]
    assert agreement({"driver": right, "fresh plan": Counter({"[2]": 1, "[1]": 1})}) == [
        ["driver", "fresh plan"]
    ]
