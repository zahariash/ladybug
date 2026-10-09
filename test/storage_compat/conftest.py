from __future__ import annotations

import os
from pathlib import Path

import pytest
from databases import NAMES, REPO_ROOT, write_db
from releases import releases_to_check
from runners import Build

DEFAULT_LIB = REPO_ROOT / "build" / os.environ.get("LBUG_BUILD_TYPE", "release") / "src/liblbug.so"


def pytest_addoption(parser) -> None:
    parser.addoption("--lib", type=Path, default=DEFAULT_LIB, help="this build's liblbug.so")
    parser.addoption("--base-lib", type=Path, help="the base commit's liblbug.so")
    parser.addoption(
        "--release", action="append", default=[], help="ladybug version on PyPI; repeatable"
    )
    parser.addoption(
        "--releases",
        choices=["latest", "all"],
        help="add the newest release to check, or all of them (see releases.py; needs PyPI)",
    )


def releases(config) -> list[str]:
    """The releases to write databases with, computed once per session."""
    if not hasattr(config, "_compat_releases"):
        chosen = {"latest": lambda: releases_to_check()[-1:], "all": releases_to_check}
        selected = chosen[config.getoption("releases")]() if config.getoption("releases") else []
        config._compat_releases = config.getoption("release") + selected
    return config._compat_releases


def pytest_report_header(config) -> str:
    return f"storage compatibility releases: {' '.join(releases(config)) or 'none'}"


def pytest_generate_tests(metafunc) -> None:
    if "writer_name" in metafunc.fixturenames:
        names = ["base"] if metafunc.config.getoption("base_lib") else []
        metafunc.parametrize("writer_name", names + releases(metafunc.config))


@pytest.fixture(scope="session")
def build(pytestconfig) -> Build:
    return Build.local("this build", pytestconfig.getoption("lib").resolve())


@pytest.fixture(scope="session")
def base(pytestconfig) -> Build | None:
    path = pytestconfig.getoption("base_lib")
    return Build.local("the base commit", path.resolve()) if path else None


@pytest.fixture(scope="session")
def writers(pytestconfig, base) -> dict:
    return ({"base": base} if base else {}) | {v: Build.release(v) for v in releases(pytestconfig)}


@pytest.fixture(scope="session")
def version_changed(build, base) -> bool:
    if base is None:
        pytest.skip("no --base-lib")
    return build.storage_version() != base.storage_version()


class Corpora:
    """The databases each writer wrote, written on first use: name -> (path, failed statements)."""

    def __init__(self, work_dir: Path) -> None:
        self.work_dir, self.written, self.errors = work_dir, {}, {}

    def __call__(self, writer) -> dict[str, tuple[Path, dict]]:
        if writer.name in self.errors:
            raise self.errors[writer.name]
        if writer.name not in self.written:
            corpus = {}
            try:
                for name in NAMES:
                    db = self.work_dir / writer.name.replace(" ", "-") / name / "db.lbdb"
                    corpus[name] = db, write_db(writer, name, db)
            except Exception as e:
                self.errors[writer.name] = e
                raise
            self.written[writer.name] = corpus
        return self.written[writer.name]


@pytest.fixture(scope="session")
def corpora(tmp_path_factory) -> Corpora:
    return Corpora(tmp_path_factory.mktemp("storage-compat"))
