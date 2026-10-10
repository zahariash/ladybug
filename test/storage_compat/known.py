"""Known product bugs the suite reports, and fixture sections older releases cannot run.

A known failure is expected exactly: if it stops failing, the test fails and asks for the entry
to be removed, so a fix is noticed and nothing else hides behind the entry.
"""

from __future__ import annotations

from dataclasses import dataclass


def version_tuple(version: str) -> tuple[int, ...]:
    return tuple(int(part) for part in version.split(".")[:3])


@dataclass(frozen=True)
class Known:
    reason: str
    tests: tuple[str, ...]
    cases: tuple[tuple[str, str], ...]  # (database, check)
    releases_before: str | None = None  # applies to databases written by releases before this
    releases: tuple[str, ...] = ()  # or by these releases

    def applies(self, test: str, writer_version: str | None, database: str) -> set[str]:
        """The checks expected to fail for this test, writer and database."""
        if test not in self.tests:
            return set()
        if self.releases_before or self.releases:
            if writer_version is None:
                return set()
            before = self.releases_before and version_tuple(writer_version) < version_tuple(
                self.releases_before
            )
            if not before and writer_version not in self.releases:
                return set()
        return {check for db, check in self.cases if db == database}


KNOWN_FAILURES = [
    Known(
        "0.21 changed the string hash; hash primary-key indexes written by older releases miss "
        "non-ASCII keys (follow-up to #882, #1092, #1108)",
        tests=("test_reads", "test_upgrade"),
        cases=(("tinysnb", "lookup movies middle"),),
        releases_before="0.21.0",
    ),
    Known(
        "the same, for the fixture's Tag table, which gets a hash primary-key index from "
        "releases without ART indexes",
        tests=("test_reads", "test_upgrade"),
        cases=(("fixture", "lookup Tag last"),),
        releases_before="0.17.0",
    ),
    Known(
        "ART primary-key indexes written by 0.17 cannot be read by 0.18 or later: lookups fail "
        "with 'Cannot read past the end of disk-backed ART storage' or crash",
        tests=("test_reads", "test_upgrade"),
        cases=(
            ("fixture", "lookup Tag first"),
            ("fixture", "lookup Tag middle"),
            ("fixture", "lookup Tag last"),
        ),
        releases=("0.17.1",),
    ),
    Known(
        "0.18.3 reads back some Person.name values it wrote as ''; this build reads what was written",
        tests=("test_reads", "test_upgrade"),
        cases=(("fixture", "node Person"),),
        releases=("0.18.3",),
    ),
    Known(
        "parquet export writes INT128 as DOUBLE, drops INTERVAL microseconds and misplaces "
        "nested STRUCT values after an empty STRUCT",
        tests=("test_imports_base_export",),
        cases=(
            ("tinysnb", "rel knows person->person"),
            ("tinysnb", "rel studyAt person->organisation"),
            ("tinysnb", "node person"),
            ("tinysnb", "node movies"),
            ("tinysnb-serial", "rel knows person->person"),
            ("tinysnb-serial", "node person"),
        ),
    ),
]

# The first minor release that runs each optional fixture section in full; older releases skip
# (part of) it, and nothing else may be skipped.
SECTIONS_SINCE = {"partitioned": "0.20.0", "csr": "0.20.0", "art_index": "0.17.0"}


def expected_skips(release: str) -> set[str]:
    return {
        section
        for section, since in SECTIONS_SINCE.items()
        if version_tuple(release)[:2] < version_tuple(since)[:2]
    }


def expected_failures(test: str, writer_version: str | None, database: str) -> dict[str, str]:
    """Known failing checks for this test, writer and database, with their reasons."""
    failures = {}
    for known in KNOWN_FAILURES:
        for check in known.applies(test, writer_version, database):
            failures[check] = known.reason
    return failures
