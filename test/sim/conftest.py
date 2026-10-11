from __future__ import annotations

import hashlib
import os
import shutil
from collections.abc import Iterator

import machine  # noqa: F401  imports every rule module, so that RULE_NAMES is complete
import pytest
from engine import KEEP_MARKER, scratch_root
from hypothesis import HealthCheck, settings
from hypothesis.database import DirectoryBasedExampleDatabase
from known import KNOWN_BUGS, known_rules
from session import FAILURES, RULE_NAMES, Options

# Next to Hypothesis's own example database, which it already keeps out of git; not in the
# temporary directory, which wrappers that clean up after a run delete.
DEFAULT_TRACES = os.path.abspath(os.path.join(".hypothesis", "traces"))

# Rule groups each focus turns off.
FOCUS = {
    "extensions": {
        "persons",
        "bulk_copy",
        "transaction",
        "atomic_statement",
        "knows",
        "partitions",
        "macros",
        "tables",
        "columns",
        "rename_table",
        "rel_columns",
        "prepared_ddl",
        "pk_index",
        "race",
        "crash_during",
    },
    # Transactions on several connections over person and Knows writes and column DDL.
    "sessions": {
        "bulk_copy",
        "partitions",
        "macros",
        "tables",
        "rename_table",
        "rel_columns",
        "prepared_ddl",
        "pk_index",
        "race",
        "crash_during",
        "crash_at_syscall",
    },
}


def pytest_addoption(parser) -> None:
    group = parser.getgroup("sim", "workload simulation")
    group.addoption("--sim-examples", type=int, default=50, help="random workloads to run")
    group.addoption("--sim-steps", type=int, default=40, help="steps per workload")
    group.addoption(
        "--sim-skip",
        action="append",
        default=[],
        help="rule, rule variant (e.g. crash_copy) or vector metric to turn off; repeatable",
    )
    group.addoption(
        "--sim-known", action="store_true", help="also run the rules that hit known.py's bugs"
    )
    group.addoption(
        "--sim-enable",
        action="append",
        default=[],
        help="known.py rule to run anyway, e.g. to verify a fix; repeatable",
    )
    group.addoption(
        "--sim-focus",
        choices=sorted(FOCUS),
        help="turn off the rules outside one area; checkpoints, reopens and crashes stay on",
    )
    group.addoption(
        "--sim-extensions",
        default="",
        help='statements that load extensions, separated by ";", e.g. '
        '"LOAD EXTENSION fts;LOAD EXTENSION vector"; adds the Doc table and its index rules',
    )
    group.addoption(
        "--sim-keep", action="store_true", help="keep every database directory after the run"
    )
    group.addoption(
        "--sim-gdb",
        action="store_true",
        help="run workers under gdb, so that a crash reports its native stack (slower)",
    )
    group.addoption(
        "--sim-trace-dir",
        default=DEFAULT_TRACES,
        help="where traces go: each failing workload is kept as <dir>/failed/<signature>/, "
        "for replay.py",
    )


@pytest.fixture(scope="session")
def sim_options(pytestconfig) -> Iterator[Options]:
    skipped = set(pytestconfig.getoption("sim_skip"))
    enable = set(pytestconfig.getoption("sim_enable"))
    unknown = (skipped | enable) - RULE_NAMES
    if unknown:
        raise pytest.UsageError(f"--sim-skip/--sim-enable name no rule: {sorted(unknown)}")
    if pytestconfig.getoption("sim_focus"):
        skipped |= FOCUS[pytestconfig.getoption("sim_focus")]
    if not pytestconfig.getoption("sim_known"):
        skipped |= known_rules() - enable
    loads = [q for q in pytestconfig.getoption("sim_extensions").split(";") if q.strip()]
    os.makedirs(pytestconfig.getoption("sim_trace_dir"), exist_ok=True)
    scratch = scratch_root()
    if pytestconfig.getoption("sim_keep"):
        open(os.path.join(scratch, KEEP_MARKER), "w").close()
    yield Options(
        frozenset(skipped),
        tuple(loads),
        keep=pytestconfig.getoption("sim_keep"),
        debug=pytestconfig.getoption("sim_gdb"),
        traces=pytestconfig.getoption("sim_trace_dir"),
        scratch=scratch,
    )
    if not pytestconfig.getoption("sim_keep"):
        shutil.rmtree(scratch, ignore_errors=True)


@pytest.fixture(scope="session")
def sim_settings(pytestconfig, sim_options) -> settings:
    # Failing examples are saved and replayed first next time, but only for the same rules:
    # a run with another gate lifted must not start with this run's failures.
    rules = sorted(sim_options.skipped) + list(sim_options.extensions)
    rules.append(str(pytestconfig.getoption("sim_steps")))
    key = hashlib.sha1("\n".join(rules).encode()).hexdigest()[:12]
    return settings(
        max_examples=pytestconfig.getoption("sim_examples"),
        stateful_step_count=pytestconfig.getoption("sim_steps"),
        database=DirectoryBasedExampleDatabase(os.path.join(".hypothesis", "sim", key)),
        print_blob=True,
        deadline=None,
        # Steps are slow (a worker process, real I/O) and data is drawn inside rules.
        suppress_health_check=[HealthCheck.too_slow, HealthCheck.data_too_large],
    )


def pytest_report_header(config) -> list[str]:
    unknown = known_rules() - RULE_NAMES
    assert not unknown, f"known.py names no rule: {sorted(unknown)}"
    if config.getoption("sim_known"):
        return ["simulation: running the rules for known bugs too"]
    return [f"simulation: skipping {k.rule} ({k.reason})" for k in KNOWN_BUGS]


def pytest_terminal_summary(terminalreporter, exitstatus, config) -> None:
    """Each failure's kept trace and the command that starts its triage (docs/testing.md)."""
    for kind, slug, kept in FAILURES:
        trace = os.path.join(kept, "trace.json")
        if kind == "wrong result":
            command = "rerun it alone with the @reproduce_failure blob printed above"
        else:
            command = f"python test/sim/replay.py {trace} --runs 5 --minimize --emit"
        terminalreporter.write_line(f"{kind} {slug}: {trace}\n  triage: {command}")


def pytest_sessionfinish(session, exitstatus) -> None:
    """Failed workloads stay under failed/ for triage; the workload in progress does not."""
    traces = session.config.getoption("sim_trace_dir", None)
    if traces:
        shutil.rmtree(os.path.join(traces, f"current-{os.getpid()}"), ignore_errors=True)
