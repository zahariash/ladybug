from __future__ import annotations

import machine  # noqa: F401  imports every rule module, so that RULE_NAMES is complete
import pytest
from hypothesis import HealthCheck, settings
from known import KNOWN_BUGS, known_rules
from session import RULE_NAMES, Options

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
        "pk_index",
        "race",
        "crash_during",
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


@pytest.fixture(scope="session")
def sim_options(pytestconfig) -> Options:
    skipped = set(pytestconfig.getoption("sim_skip"))
    unknown = skipped - RULE_NAMES
    if unknown:
        raise pytest.UsageError(f"--sim-skip names no rule: {sorted(unknown)}")
    if pytestconfig.getoption("sim_focus"):
        skipped |= FOCUS[pytestconfig.getoption("sim_focus")]
    if not pytestconfig.getoption("sim_known"):
        skipped |= known_rules()
    loads = [q for q in pytestconfig.getoption("sim_extensions").split(";") if q.strip()]
    return Options(frozenset(skipped), tuple(loads), pytestconfig.getoption("sim_keep"))


@pytest.fixture(scope="session")
def sim_settings(pytestconfig) -> settings:
    return settings(
        max_examples=pytestconfig.getoption("sim_examples"),
        stateful_step_count=pytestconfig.getoption("sim_steps"),
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
