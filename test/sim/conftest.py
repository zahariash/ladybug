from __future__ import annotations

import glob
import hashlib
import importlib.util
import json
import os
import shutil
from collections.abc import Iterator

import machine  # noqa: F401  imports every rule module, so that RULE_NAMES is complete
import pytest
from engine import KEEP_MARKER, scratch_root
from hypothesis import HealthCheck, settings
from hypothesis.database import DirectoryBasedExampleDatabase
from known import KNOWN_BUGS, known_rules
from session import FAILURES, RECOVERY_SKIPS, RULE_NAMES, RULE_RUNS, Options

# Next to Hypothesis's own example database, which it already keeps out of git; not in the
# temporary directory, which wrappers that clean up after a run delete.
DEFAULT_TRACES = os.path.abspath(os.path.join(".hypothesis", "traces"))

# Rule groups each focus turns off.
FOCUS = {
    # C, CK and CK2 past their size boundaries, with sessions, checkpoints, crashes and races.
    "heavy": {
        "persons",
        "knows",
        "bulk_copy",
        "bulk_rows",
        "partitions",
        "macros",
        "drop_macro",
        "tables",
        "columns",
        "rename_table",
        "rel_columns",
        "prepared_ddl",
        "pk_index",
        "secondary_index",
        "atomic_statement",
        "in_lists",
        "count_keys",
        "pk_ranges",
        "repeated_aggregate",
    },
    "extensions": {
        "persons",
        "bulk_copy",
        "bulk_rows",
        "transaction",
        "atomic_statement",
        "knows",
        "partitions",
        "csr",
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
    # Writes racing readers, over persons, Knows and CK, with checkpoints, reopens and crashes.
    "races": {
        "atomic_statement",
        "bulk_copy",
        "columns",
        "count_keys",
        "count_restricted",
        "crash_at_syscall",
        "crash_during",
        "in_lists",
        "lookup_join",
        "macros",
        "partitions",
        "pk_index",
        "pk_ranges",
        "prepared_ddl",
        "rel_columns",
        "rename_table",
        "secondary_index",
        "sessions",
        "tables",
        "transaction",
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


def engine_build() -> str:
    """The engine library the workers load, by path, size and modification time; empty when it
    cannot be found without importing it."""
    if os.environ.get("LBUG_PYTHON_BACKEND") == "capi":
        libs = [os.environ.get("LBUG_C_API_LIB_PATH", "")]
    else:
        spec = importlib.util.find_spec("ladybug")
        if spec is None or spec.origin is None:
            return ""
        libs = sorted(glob.glob(os.path.join(os.path.dirname(spec.origin), "_lbug*.so")))
    stats = [(lib, os.stat(lib)) for lib in libs if os.path.exists(lib)]
    return "\n".join(f"{lib} {st.st_size} {st.st_mtime_ns}" for lib, st in stats)


@pytest.fixture(scope="session")
def sim_settings(pytestconfig, sim_options) -> settings:
    # Failing examples are saved and replayed first next time, but only for the same rules and
    # the same engine build: a run with another gate lifted, or on another build, must not
    # start with (and stop on) this run's failures.
    rules = sorted(sim_options.skipped) + list(sim_options.extensions)
    rules.append(str(pytestconfig.getoption("sim_steps")))
    rules.append(engine_build())
    key = hashlib.sha1("\n".join(rules).encode()).hexdigest()[:12]
    return settings(
        max_examples=pytestconfig.getoption("sim_examples"),
        stateful_step_count=pytestconfig.getoption("sim_steps"),
        database=DirectoryBasedExampleDatabase(os.path.join(".hypothesis", "sim", key)),
        print_blob=True,
        deadline=None,
        # Steps are slow (a worker process, real I/O) and data is drawn inside rules.
        suppress_health_check=[
            HealthCheck.too_slow,
            HealthCheck.data_too_large,
            HealthCheck.filter_too_much,
        ],
    )


def pytest_report_header(config) -> list[str]:
    unknown = known_rules() - RULE_NAMES
    assert not unknown, f"known.py names no rule: {sorted(unknown)}"
    if config.getoption("sim_known"):
        return ["simulation: running the rules for known bugs too"]
    return [f"simulation: skipping {k.rule} ({k.reason})" for k in KNOWN_BUGS]


def pytest_terminal_summary(terminalreporter, exitstatus, config) -> None:
    """Each failure's kept trace and the command that starts its triage (docs/testing.md)."""
    for kind, slug, kept, probe, rows in FAILURES:
        trace = os.path.join(kept, "trace.json")
        command = f"python test/sim/replay.py {trace} --runs 5 --minimize --emit"
        if kind == "wrong result":
            command = "rerun it alone with the @reproduce_failure blob printed above"
            if probe:
                command = (
                    f"python test/sim/replay.py {trace} --probe {json.dumps(probe)} "
                    f"--rows '{json.dumps(rows)}' --minimize --pin '<steps that set the answer>'"
                )
        terminalreporter.write_line(f"{kind} {slug}: {trace}\n  triage: {command}")
    if RULE_RUNS:
        runs = ", ".join(f"{rule} {n}" for rule, n in sorted(RULE_RUNS.items()))
        terminalreporter.write_line(f"rule runs: {runs}")
        skips = ", ".join(f"{gate} {n}" for gate, n in sorted(RECOVERY_SKIPS.items())) or "none"
        terminalreporter.write_line(f"workloads discarded on a known recovery failure: {skips}")
        path = os.path.join(config.getoption("sim_trace_dir"), "rule-runs.json")
        with open(path, "w") as f:
            json.dump(dict(rules=RULE_RUNS, recovery=RECOVERY_SKIPS), f, indent=1, sort_keys=True)


def pytest_sessionfinish(session, exitstatus) -> None:
    """Failed workloads stay under failed/ for triage; the workload in progress does not."""
    traces = session.config.getoption("sim_trace_dir", None)
    if traces:
        shutil.rmtree(os.path.join(traces, f"current-{os.getpid()}"), ignore_errors=True)
