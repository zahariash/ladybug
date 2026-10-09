"""The release selection in releases.py, on canned data, and the header it relies on."""

from __future__ import annotations

import re

import pytest
from releases import CMAKE_LISTS, VERSION_INFO, installable, select, storage_versions

HEADER = """
    static constexpr storage_version_t STORAGE_VERSION_40 = 40;
    static constexpr storage_version_t STORAGE_VERSION_41 = 41;
    static constexpr storage_version_t STORAGE_VERSION_47 = 47;
    static constexpr storage_version_t STORAGE_VERSION_48 = 48;
        return {{"0.12.0", STORAGE_VERSION_40}, {"0.16.0", STORAGE_VERSION_40},
            {"0.17.0", STORAGE_VERSION_41}, {"0.19.0", STORAGE_VERSION_47},
            {"0.20.0", STORAGE_VERSION_47}, {"0.21.0", STORAGE_VERSION_47},
            {"0.22.0", STORAGE_VERSION_48}};
    static bool canReadStorageVersion(storage_version_t storageVersion) {
        return storageVersion == STORAGE_VERSION_40 || storageVersion == STORAGE_VERSION_41 ||
               storageVersion == STORAGE_VERSION_47 || storageVersion == getStorageVersion();
    }
"""
WHEEL = "ladybug-{}-cp312-cp312-manylinux_2_28_x86_64.whl"


def pypi(*versions: str, without_wheel=(), yanked=()) -> dict:
    releases = {}
    for version in versions:
        name = "ladybug-{}.tar.gz" if version in without_wheel else WHEEL
        releases[version] = [{"filename": name.format(version), "yanked": version in yanked}]
    return {"releases": releases}


def selected(header=HEADER, **pypi_args) -> list[str]:
    minors, readable = storage_versions(header)
    return select(minors, readable, installable(pypi(*pypi_args.pop("versions"), **pypi_args)))


VERSIONS = ["0.16.0", "0.16.1", "0.17.1", "0.19.0", "0.20.0", "0.20.4", "0.21.0", "0.21.2"]


def test_newest_minors_and_one_release_per_older_storage_version() -> None:
    # 0.12 was never on PyPI and 0.22 is under development; 0.19 shares 47 with newer minors.
    assert selected(versions=VERSIONS) == ["0.16.1", "0.17.1", "0.19.0", "0.20.4", "0.21.2"]


def test_a_new_patch_replaces_the_previous_one() -> None:
    assert selected(versions=[*VERSIONS, "0.21.3"])[-1] == "0.21.3"


def test_a_tagged_release_without_a_wheel_is_not_selected_yet() -> None:
    picked = selected(versions=[*VERSIONS, "0.21.3"], without_wheel={"0.21.3"})
    assert picked[-1] == "0.21.2"


def test_yanked_releases_are_skipped() -> None:
    assert selected(versions=[*VERSIONS, "0.21.3"], yanked={"0.21.3"})[-1] == "0.21.2"


def test_dropped_storage_version_drops_its_releases() -> None:
    header = HEADER.replace("storageVersion == STORAGE_VERSION_41 ||", "")
    assert "0.17.1" not in selected(header=header, versions=VERSIONS)


def test_unpublished_minor_between_published_ones_is_an_error() -> None:
    with pytest.raises(ValueError, match="0.17"):
        selected(versions=[v for v in VERSIONS if not v.startswith("0.17")])


def test_header_maps_this_version_and_each_minor_to_one_storage_version() -> None:
    header = VERSION_INFO.read_text()
    version = re.search(r"project\(Lbug VERSION ([\d.]+)", CMAKE_LISTS.read_text()).group(1)
    assert f'{{"{version}", STORAGE_VERSION_' in header, f"{version} has no storage version entry"
    minors, _ = storage_versions(header)
    assert {minor: v for minor, v in minors.items() if len(v) > 1} == {}
