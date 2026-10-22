"""The steps shrinkray reduces, and the helpers around its run, without shrinkray or an engine."""

from __future__ import annotations

import json
import os

from shrinking import compact, distinctive, expand, latest_steps


def execute(query: str, outcome: str = "ok", expected: bool = False) -> dict:
    op = {"op": "execute", "conn": 0, "query": query, "params": {}, "outcome": outcome}
    return op | ({"expected": True} if expected else {})


def test_compact_keeps_only_what_a_replay_reads_and_expands_back() -> None:
    head = {
        "op": "open",
        "config": {"compression": False},
        "loads": ["CREATE NODE TABLE C(id INT64)"],
    }
    crash = {"op": "crash_at", "query": "CHECKPOINT", "syscall": "fsync", "n": 2, "rule": "x"}
    trace = [head, execute("RETURN 1", "error: boom", True), crash]
    data = compact(trace)
    assert data["steps"] == [
        {"op": "execute", "conn": 0, "query": "RETURN 1", "params": {}},
        {"op": "crash_at", "query": "CHECKPOINT", "syscall": "fsync", "n": 2},
    ]
    assert expand(data) == [{**head, "reopen_loads": head["loads"]}] + data["steps"]


def test_expand_rejects_what_a_reduction_broke() -> None:
    step = {"op": "execute", "conn": 0, "query": "RETURN 1", "params": {}}
    assert expand({"steps": [step]})[1:] == [step]
    for broken in (
        [],
        {"steps": [{**step, "op": "teleport"}]},
        {"steps": [{"op": "execute", "conn": 0, "query": "RETURN 1"}]},
        {"steps": [{**step, "conn": True}]},
        {"steps": [step], "loads": "CREATE"},
    ):
        assert expand(broken) is None, broken


def test_distinctive_signatures_need_more_than_the_operator_loop() -> None:
    loop = [
        "lbug::processor::PhysicalOperator::getNextTuple",
        "lbug::common::TaskScheduler::runWorkerThread",
    ]
    assert not distinctive({"frames": loop, "error": ""})
    assert distinctive({"frames": ["lbug::storage::ColumnChunk::scan", *loop], "error": ""})
    assert distinctive({"frames": [], "error": "Runtime exception: ?"})
    assert not distinctive(
        {"probe": "RETURN 1", "conn": 0, "agreement": [["driver"], ["no optimizer"]]}
    )


def test_latest_steps_falls_back_to_the_newest_recorded_reduction(tmp_path) -> None:
    (tmp_path / "steps.json").write_text('{"steps": [')
    for run, number, size in (("run-1", "0007", 9), ("run-2", "0002", 2), ("run-2", "0003", 3)):
        folder = tmp_path / ".shrinkray" / run / "reductions" / number
        os.makedirs(folder)
        (folder / "steps.json").write_text(json.dumps({"steps": [], "size": size}))
    assert latest_steps(str(tmp_path))["size"] == 3
