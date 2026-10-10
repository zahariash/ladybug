"""This build reads, upgrades and hands back databases written by other versions.

Every test opens one database file with two builds and compares what they read:

- this build reads databases written by the base commit or by a ladybug release;
- this build writes to a copy of such a database and checkpoints it, or writes and stops without
  checkpointing so that the next open replays the WAL, and reads what the writer reads after
  applying the same writes to its own copy;
- the base commit reads databases written by this build, unless this build changed the storage
  version (src/include/storage/storage_version_info.h);
- if it did, this build imports what the base commit exports.

Every build is read through the same Python package (this repository's tools/python_api on a
build's liblbug.so, or a release's wheel), with each value converted to its column's Arrow type, so
all of them return the same Python values.
Known product bugs are listed in known.py.

Run with: uv run --with pytest pytest test/storage_compat --lib <this build's liblbug.so>
--base-lib <the base commit's liblbug.so> --releases latest|all
"""

from __future__ import annotations

import pytest
from checks import compare, for_writer, plan, read
from databases import NAMES, copy_db, upgrade_statements, write_sections
from known import expected_failures, expected_skips

NO_CHECKPOINT = ["CALL force_checkpoint_on_close=false;", "CALL auto_checkpoint=false;"]


def test_fixture_runs_in_full_on_this_build(build, corpora) -> None:
    _, failed = corpora(build)["fixture"]
    assert not failed, f"fixture statements this build cannot run: {failed}"


def test_fixture_skips(writer_name, writers, corpora) -> None:
    """Releases skip exactly the fixture sections known.py says they lack."""
    writer = writers[writer_name]
    _, failed = corpora(writer)["fixture"]
    if writer.version is None:
        pytest.skip("the base commit may lack sections added by this change")
    assert set(failed) == expected_skips(writer.version)


@pytest.mark.parametrize("name", NAMES)
def test_reads(writer_name, name, writers, build, corpora) -> None:
    writer = writers[writer_name]
    db, _ = corpora(writer)[name]
    checks = for_writer(plan(build, db), writer)
    diffs = compare(read(writer, db, checks), read(build, db, checks), checks)
    assert_known("test_reads", writer.version, name, diffs)


@pytest.mark.parametrize("checkpoint", [True, False], ids=["checkpoint", "wal-replay"])
def test_upgrade(writer_name, checkpoint, writers, build, corpora) -> None:
    writer = writers[writer_name]
    db, _ = corpora(writer)["fixture"]
    label = "checkpoint" if checkpoint else "wal-replay"
    checks = for_writer(plan(build, db), writer)
    reference = copy_db(db, f"{label}-reference")
    write_sections(writer, reference, {"core": upgrade_statements()})
    expected = read(writer, reference, checks)

    upgraded = copy_db(db, label)
    if checkpoint:
        statements = upgrade_statements() + ["CHECKPOINT;"]
    else:
        statements = NO_CHECKPOINT + upgrade_statements()
    write_sections(build, upgraded, {"core": statements})
    if not checkpoint:
        assert upgraded.with_suffix(".lbdb.wal").exists(), "no WAL was left to replay"
    actual = read(build, upgraded, checks, read_only=checkpoint)
    diffs = compare(expected, actual, checks)
    assert_known("test_upgrade", writer.version, "fixture", diffs)


@pytest.mark.parametrize("name", NAMES)
def test_base_reads_this_build(name, build, base, corpora, version_changed) -> None:
    if version_changed:
        pytest.skip("this build changed the storage version")
    db, _ = corpora(build)[name]
    checks = plan(build, db)
    diffs = compare(read(build, db, checks), read(base, db, checks), checks)
    assert_known("test_base_reads_this_build", None, name, diffs)


@pytest.mark.parametrize("name", NAMES)
def test_imports_base_export(name, build, base, corpora, version_changed) -> None:
    if not version_changed:
        pytest.skip("this build kept the storage version")
    db, _ = corpora(base)[name]
    export_dir = db.parent.parent / f"{name}-export"
    export = [f"EXPORT DATABASE '{export_dir}';"]
    if error := base.execute_sections(db, {"export": export})["export"]["error"]:
        pytest.skip(f"the base commit cannot export {name}: {error}")
    imported = db.parent.parent / f"{name}-imported" / "db.lbdb"
    imported.parent.mkdir()
    write_sections(build, imported, {"core": [f"IMPORT DATABASE '{export_dir}';"]})
    checks = plan(build, imported)
    # Import restarts each sequence at its current value; currval() is still checked.
    del checks["sequences"]
    diffs = compare(read(base, db, checks), read(build, imported, checks), checks)
    assert_known("test_imports_base_export", None, name, diffs)


def assert_known(test: str, writer_version: str | None, database: str, diffs: dict) -> None:
    """Fails on any difference that is not a known failure, and on a known failure that no
    longer occurs; xfails when only known failures occur."""
    known = expected_failures(test, writer_version, database)
    unexpected = {check: diff for check, diff in diffs.items() if check not in known}
    assert not unexpected, "\n".join(f"{check}: {diff}" for check, diff in unexpected.items())
    fixed = sorted(set(known) - set(diffs))
    assert not fixed, f"known failures no longer occur, remove them from known.py: {fixed}"
    if known:
        pytest.xfail("; ".join(sorted(set(known.values()))))
