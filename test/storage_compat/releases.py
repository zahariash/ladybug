"""Which ladybug releases this build must read databases from.

The header src/include/storage/storage_version_info.h says which storage version each minor
version writes and which storage versions this build reads; PyPI says which releases exist and
can be installed. The selection is the last installable patch of each of the newest minor
versions, plus the newest release of every older storage version this build reads. A release
that is tagged but not on PyPI yet is not selected until it is, so the previous patch stays.
"""

from __future__ import annotations

import json
import re
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
VERSION_INFO = REPO_ROOT / "src/include/storage/storage_version_info.h"
CMAKE_LISTS = REPO_ROOT / "CMakeLists.txt"
NEWEST_MINORS = 3
# The wheel the suite installs (runners.RELEASE_PYTHON on Linux x86-64).
WHEEL = re.compile(r"-cp312-cp312-.*manylinux.*x86_64\.whl$")


def version_key(version: str) -> tuple[int, ...]:
    return tuple(int(part) for part in version.split("."))


def storage_versions(header: str) -> tuple[dict[str, set[int]], set[int]]:
    """Each minor version's storage versions, and the storage versions this build reads."""
    numbers = {name: int(n) for name, n in re.findall(r"(STORAGE_VERSION_\d+) = (\d+);", header)}
    minors: dict[str, set[int]] = {}
    for minor, name in re.findall(r'\{"(\d+\.\d+)\.\d+", (STORAGE_VERSION_\d+)\}', header):
        minors.setdefault(minor, set()).add(numbers[name])
    readable_body = header[header.index("canReadStorageVersion") :].split("}", 1)[0]
    readable = {numbers[name] for name in re.findall(r"STORAGE_VERSION_\d+", readable_body)}
    if "getStorageVersion()" in readable_body:
        readable.add(max(v for versions in minors.values() for v in versions))
    return minors, readable


def installable(pypi: dict) -> dict[str, str]:
    """The newest final release of each minor version that has the wheel the suite installs."""
    latest: dict[str, str] = {}
    for version, files in pypi["releases"].items():
        if not re.fullmatch(r"\d+\.\d+\.\d+", version):
            continue
        if not any(WHEEL.search(f["filename"]) and not f.get("yanked") for f in files):
            continue
        minor = version.rsplit(".", 1)[0]
        if minor not in latest or version_key(version) > version_key(latest[minor]):
            latest[minor] = version
    return latest


def select(minors: dict[str, set[int]], readable: set[int], available: dict[str, str]) -> list:
    """The releases to check, oldest first. Minor versions in the header newer than every
    installable release are under development, and older ones were never published to PyPI;
    both are ignored. A minor version in between that cannot be installed is an error."""
    supported = sorted((m for m in minors if minors[m] <= readable), key=version_key)
    on_pypi = [minor for minor in supported if minor in available]
    first, last = version_key(on_pypi[0]), version_key(on_pypi[-1])
    gaps = [m for m in supported if m not in available and first < version_key(m) < last]
    if gaps:
        raise ValueError(f"no installable release on PyPI for minor versions {gaps}")
    chosen = set(on_pypi[-NEWEST_MINORS:])
    # Oldest first, so each storage version keeps its newest minor version.
    chosen |= set({max(minors[minor]): minor for minor in on_pypi}.values())
    return [available[minor] for minor in sorted(chosen, key=version_key)]


def fetch_pypi() -> dict:
    with urllib.request.urlopen("https://pypi.org/pypi/ladybug/json", timeout=60) as response:
        return json.load(response)


def releases_to_check() -> list[str]:
    minors, readable = storage_versions(VERSION_INFO.read_text())
    return select(minors, readable, installable(fetch_pypi()))
