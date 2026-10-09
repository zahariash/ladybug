from __future__ import annotations

import pytest
from hypothesis import HealthCheck, settings
from known import KNOWN_BUGS, known_rules
from session import Options


def pytest_addoption(parser) -> None:
    group = parser.getgroup("sim", "workload simulation")
    group.addoption("--sim-examples", type=int, default=50, help="random workloads to run")
    group.addoption("--sim-steps", type=int, default=40, help="steps per workload")
    group.addoption(
        "--sim-skip", action="append", default=[], help="rule or variant to turn off; repeatable"
    )
    group.addoption(
        "--sim-known", action="store_true", help="also run the rules that hit known.py's bugs"
    )
    group.addoption(
        "--sim-extensions",
        default="",
        help='statements that load extensions, separated by ";", e.g. '
        '"LOAD EXTENSION fts;LOAD EXTENSION vector"; adds the Doc table and its index rules',
    )


@pytest.fixture(scope="session")
def sim_options(pytestconfig) -> Options:
    skipped = set(pytestconfig.getoption("sim_skip"))
    if not pytestconfig.getoption("sim_known"):
        skipped |= known_rules()
    loads = [q for q in pytestconfig.getoption("sim_extensions").split(";") if q.strip()]
    return Options(frozenset(skipped), tuple(loads))


@pytest.fixture(scope="session")
def sim_settings(pytestconfig) -> settings:
    return settings(
        max_examples=pytestconfig.getoption("sim_examples"),
        stateful_step_count=pytestconfig.getoption("sim_steps"),
        deadline=None,
        suppress_health_check=list(HealthCheck),
    )


def pytest_report_header(config) -> list[str]:
    if config.getoption("sim_known"):
        return ["simulation: running the rules for known bugs too"]
    return [f"simulation: skipping {k.rule} ({k.reason})" for k in KNOWN_BUGS]
