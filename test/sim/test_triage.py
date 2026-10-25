"""Triage helpers (triage.py) and replay.py's simplification passes, without an engine."""

from __future__ import annotations

import ast

import replay
import triage

STACKS = """Thread 3 "python3.12" received signal SIGSEGV, Segmentation fault.
#0 lbug::common::LogicalType::LogicalType
#1 lbug::storage::NodeGroupCollection::checkpoint
#2 lbug::storage::NodeTable::checkpoint
#3 lbug::storage::StorageManager::checkpoint
Thread 1
#0 lbug::common::TaskScheduler::runWorkerThread"""

TRACE = [
    dict(op="open", config=dict(compression=False, max_num_threads=7), loads=[]),
    dict(op="execute", conn=0, query="CREATE NODE TABLE T3(id INT64 PRIMARY KEY)", params={}),
    dict(op="execute", conn=2, query="CREATE (:T3 {id: $id, s: $s})", params=dict(id=1, s="a'b")),
    dict(op="crash"),
    dict(op="execute", conn=0, query="CREATE (:T3 {id: 1})", params={}, expected=True),
    dict(op="reopen"),
    dict(op="execute", conn=0, query="CHECKPOINT", params={}),
]


def test_top_frames_are_the_first_stack() -> None:
    assert triage.top_frames(STACKS) == [
        "lbug::common::LogicalType::LogicalType",
        "lbug::storage::NodeGroupCollection::checkpoint",
        "lbug::storage::NodeTable::checkpoint",
    ]
    assert triage.top_frames("Binder exception: no such table") == []


def test_shape_abstracts_literals_and_numbered_names() -> None:
    assert triage.shape("CREATE (:T3 {v: -42, s: 'x y'}) RETURN m2(1.5), $id") == (
        "CREATE (:T? {v: ?, s: ?}) RETURN m?(?), $id"
    )


def test_operation_names_the_change_an_alter_makes() -> None:
    assert triage.operation(dict(op="execute", query="ALTER TABLE P DROP c2")) == "ALTER DROP"
    assert triage.operation(dict(op="execute", query="checkpoint")) == "CHECKPOINT"
    assert triage.operation(dict(op="crash")) == "crash"


def test_signatures_ignore_literals_and_follow_frames() -> None:
    a = triage.signature("died", dict(op="execute", query="ALTER TABLE T1 ADD c2"), STACKS)
    b = triage.signature("died", dict(op="execute", query="ALTER TABLE T7 ADD c0"), STACKS)
    c = triage.signature("died", dict(op="execute", query="ALTER TABLE T1 ADD c2"), "#0 lbug::X")
    assert a["key"] == b["key"] != c["key"]
    assert a["slug"].startswith("died-alter-add-")
    error = triage.signature("error", dict(op="execute", query="Q"), "Table T4 does not exist")
    assert error["error"] == "Table T? does not exist" and error["frames"] == []


def test_cypher_keeps_parameters_and_marks_lifetimes() -> None:
    text = triage.cypher(TRACE, 6, "died: CHECKPOINT", TRACE[0]["config"])
    assert "// database: compression=False, max_num_threads=7" in text
    assert "// [connection 2]\nCREATE (:T3 {id: $id, s: $s});  // $id = 1, $s = 'a\\'b'" in text
    assert "// kill the process (SIGKILL), then open the database again" in text
    assert "CREATE (:T3 {id: 1});  // fails, as expected" in text
    assert text.rstrip().endswith("CHECKPOINT;\n// ^ died: CHECKPOINT")


def test_stages_split_at_crashes_and_reopens() -> None:
    stages = triage.stages(TRACE, 6)
    assert [stage["end"]["op"] for stage in stages] == ["crash", "reopen", "last"]
    assert [len(stage["steps"]) for stage in stages] == [2, 1, 1]


def test_script_compiles_and_embeds_the_stages() -> None:
    source = triage.script(TRACE, 6, "died", STACKS, TRACE[0]["config"], "trace.min.py")
    ast.parse(source)
    assert "STAGES = [" in source and "EXPECTED = {'kind': 'died'}" in source


def test_simplify_keeps_replacements_that_still_fail() -> None:
    trace = [TRACE[0], dict(op="crash"), dict(op="prepared", conn=2, query="Q", params={})]
    # Fails as long as the prepared statement stays on connection 2.
    simplified = replay.simplify(trace, lambda t: any(op.get("conn") == 2 for op in t))
    assert [op["op"] for op in simplified[1:]] == ["reopen", "execute"]
    assert simplified[2]["conn"] == 2


def test_reduce_config_keeps_only_needed_settings() -> None:
    trace, needed = replay.reduce_config(TRACE, lambda t: "compression" in t[0]["config"])
    assert needed == dict(compression=False) == trace[0]["config"]


def test_scripts_load_steps_with_nan_and_infinity() -> None:
    trace = [TRACE[0], dict(op="execute", conn=0, query="RETURN $x", params=dict(x=float("nan")))]
    trace[1]["params"]["y"] = float("-inf")
    source = triage.script(trace, 1, "died", STACKS, {}, "trace.min.py")
    namespace = {"__name__": "trace_min", "__file__": "trace.min.py"}
    exec(compile(source, "trace.min.py", "exec"), namespace)
    params = namespace["STAGES"][0]["steps"][0]["params"]
    assert params["x"] != params["x"] and params["y"] == float("-inf")
    assert triage.literal(float("nan")) == "CAST('nan' AS DOUBLE)"


def test_probes_come_from_either_assertion_format() -> None:
    check = (
        "('MATCH (p:Person) WHERE p.c1 = $d RETURN count(*)', {'d': -1}, 'driver', [(0,)], [(2,)])"
    )
    assert triage.probe_of({}, check) == (
        "MATCH (p:Person) WHERE p.c1 = -1 RETURN count(*)",
        [[0]],
    )
    diff = "missing=[((1, None, 1), 1)] unexpected=[((1, None, 0), 1)]"
    op = dict(op="execute", query="MATCH (p:Person) RETURN p.id, p.name, p.c2", params={})
    assert triage.probe_of(op, diff) == (
        "MATCH (p:Person) RETURN p.id, p.name, p.c2",
        [[1, None, 0]],
    )
    assert triage.probe_of(op, "something else") is None


def test_after_last_fresh_starts_the_trace_at_the_last_fresh_database() -> None:
    head = {"op": "open", "config": {"x": 1}, "loads": ["first"], "reopen_loads": ["again"]}
    fresh = {"op": "fresh", "loads": ["first'"]}
    step = {"op": "execute", "query": "RETURN 1"}
    trace = [head, step, fresh, step, dict(fresh, loads=["first''"]), step]
    assert triage.after_last_fresh(trace) == [dict(head, loads=["first''"]), step]
    assert triage.after_last_fresh([head, step]) == [head, step]


FAILURE_TRACE = [
    dict(op="open", config=dict(enable_checksums=True), loads=[]),
    dict(op="execute", conn=0, query="CREATE (:T3 {id: 1})", params={}),
    dict(op="crash", config=dict(enable_checksums=False), outcome="error: x", expected=True),
    dict(op="open_again"),
    dict(op="limit", size=4096),
    dict(op="execute", conn=0, query="CALL threads=1", params={}),
    dict(op="kill_at", query="CHECKPOINT", syscall="fsync", n=2),
    dict(op="execute", conn=0, query="MATCH (t:T3) RETURN t.id", params={}),
]


def test_stages_carry_the_settings_and_expected_failure_of_their_open() -> None:
    stages = triage.stages(FAILURE_TRACE, 7)
    assert [stage["end"]["op"] for stage in stages] == ["crash", "open_again", "kill_at", "last"]
    assert stages[1]["config"] == dict(enable_checksums=False) and stages[1]["open_fails"]
    assert stages[2]["config"] == {} and not stages[2]["open_fails"]
    assert [op["op"] for op in stages[2]["steps"]] == ["limit", "execute"]


def test_cypher_describes_limits_overrides_and_attached_kills() -> None:
    text = triage.cypher(FAILURE_TRACE, 7, "error: x", {})
    assert "then open the database again, with enable_checksums=False  // fails" in text
    assert "// writes past 4096 bytes of any file fail (RLIMIT_FSIZE)" in text
    assert "CHECKPOINT;  // killed at its fsync call 2 (strace attached" in text
    source = triage.script(FAILURE_TRACE, 7, "error", "x", {}, "trace.min.py")
    ast.parse(source)
    assert "def attach_strace" in source


def test_replay_simplifies_an_attached_kill_to_the_statement_and_a_crash() -> None:
    options = replay.simplifications(FAILURE_TRACE[6])
    assert [[op["op"] for op in option] for option in options] == [
        ["execute", "reopen"],
        ["execute", "crash"],
    ]
