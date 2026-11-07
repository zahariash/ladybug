"""What Session keeps of a crash, and how reopen_image.py opens it again, without a database."""

from __future__ import annotations

import json
import os
import shutil

import pytest
import reopen_image
from engine import EngineError
from rules.sessions import Txn
from session import CRASH_IMAGE, IMAGE_OPEN, Options, Session


def write(path, text: str) -> None:
    with open(path, "w") as f:
        f.write(text)


def read(path) -> str:
    with open(path) as f:
        return f.read()


@pytest.fixture
def session(tmp_path) -> Session:
    session = Session()
    session.dir = str(tmp_path / "db-1")
    session.path = os.path.join(session.dir, "db")
    os.makedirs(session.dir)
    write(session.path, "pages")
    write(f"{session.path}.wal", "torn")
    session.loads = lambda: ["CALL threads=1"]
    return session


def test_an_image_keeps_the_files_as_the_crash_left_them(session) -> None:
    session.save_crash_image({"enable_checksums": True})
    write(f"{session.path}.wal", "")
    image = session.crash_image()
    assert read(os.path.join(image, "db.wal")) == "torn"
    assert read(os.path.join(image, "db")) == "pages"
    with open(os.path.join(image, IMAGE_OPEN)) as f:
        assert json.load(f) == {"config": {"enable_checksums": True}, "loads": ["CALL threads=1"]}


def test_an_open_after_a_crash_copies_the_files_before_recovery_changes_them(
    session, monkeypatch
) -> None:
    class Recovers:
        def __init__(self, path, *_) -> None:
            write(f"{path}.wal", "")

    monkeypatch.setattr("session.Engine", Recovers)
    session.config, session.trace = {"enable_checksums": False}, [{}]
    session.open_engine(crashed=True, overrides={"enable_checksums": True})
    assert read(f"{session.path}.wal") == ""
    assert "wal" in session.trace[-1]["wal"]
    image = session.crash_image()
    assert read(os.path.join(image, "db.wal")) == "torn"
    with open(os.path.join(image, IMAGE_OPEN)) as f:
        assert json.load(f)["config"] == {"enable_checksums": True}


def test_a_clean_open_copies_nothing(session, monkeypatch) -> None:
    monkeypatch.setattr("session.Engine", lambda *_: None)
    session.config = {}
    session.open_engine()
    assert not os.path.exists(session.crash_image())


def test_a_syscall_crash_stops_a_worker_whose_statement_failed(session, monkeypatch) -> None:
    class Worker:
        killed = False

        def execute(self, *_) -> None:
            raise EngineError("Binder exception")

        def close(self) -> None:
            pass

        def kill(self) -> None:
            self.killed = True

    worker = Worker()
    alive_at_recovery = []

    def open_engine(inject=None, crashed=False, overrides=None) -> None:
        if crashed:
            alive_at_recovery.append(not worker.killed)
        else:
            session.engine = worker

    session.engine, session.trace = worker, []
    session.open_engine = open_engine
    session.crash_at_syscall_point("RETURN 1", "pwrite64", 1)
    assert alive_at_recovery == [False]


def test_a_syscall_crash_during_open_has_no_worker_to_stop(session) -> None:
    class Closed:
        def close(self) -> None:
            pass

    def open_engine(inject=None, crashed=False, overrides=None) -> None:
        session.engine = None
        if inject:
            raise EngineError("killed during open")

    session.engine, session.trace = Closed(), []
    session.open_engine = open_engine
    session.crash_at_syscall_point("RETURN 1", "pwrite64", 1)
    assert session.trace[-1]["outcome"] == "ok"


def test_a_column_drop_waits_for_read_only_snapshots_only_while_gated(session) -> None:
    gated = Options(skipped=frozenset({"snapshot_column_drop"}))
    session.options = gated
    session.txns = {}
    assert session.column_drop_safe()
    session.txns = {3: Txn("write", None)}
    assert session.column_drop_safe()
    session.txns = {3: Txn("read", None)}
    assert not session.column_drop_safe()
    session.options = Options()
    assert session.column_drop_safe()


def test_a_later_crash_replaces_the_image(session) -> None:
    session.save_crash_image({})
    os.remove(f"{session.path}.wal")
    session.save_crash_image({})
    assert not os.path.exists(os.path.join(session.crash_image(), "db.wal"))


def test_without_room_for_an_image_the_workload_goes_on(session, monkeypatch, capsys) -> None:
    def no_room(*_, **__):
        raise OSError(28, "No space left on device")

    monkeypatch.setattr("shutil.copytree", no_room)
    session.save_crash_image({})
    assert not os.path.exists(session.crash_image())
    assert "no crash image" in capsys.readouterr().err


def test_a_kept_failure_gets_the_wal_files_and_the_image(session, tmp_path) -> None:
    write(f"{session.path}.wal.checkpoint", "frozen")
    session.save_crash_image({})
    write(f"{session.path}.wal", "after recovery")
    kept = tmp_path / "kept"
    kept.mkdir()
    session.keep_database_files(str(kept))
    assert read(kept / "db.wal") == "after recovery"
    assert read(kept / "db.wal.checkpoint") == "frozen"
    assert read(kept / CRASH_IMAGE / "db.wal") == "torn"


def test_an_image_that_cannot_be_kept_leaves_nothing_to_open(
    session, tmp_path, monkeypatch, capsys
) -> None:
    session.save_crash_image({})
    kept = tmp_path / "kept"
    kept.mkdir()
    real = shutil.copytree

    def fails_midway(source, destination, **kwargs):
        real(source, destination, **kwargs)
        raise OSError(28, "No space left on device")

    monkeypatch.setattr("shutil.copytree", fails_midway)
    session.keep_database_files(str(kept))
    assert sorted(os.listdir(kept)) == ["db.wal"]
    assert "not kept" in capsys.readouterr().err


def test_a_failure_without_a_crash_keeps_only_the_wal_files(session, tmp_path) -> None:
    kept = tmp_path / "kept"
    kept.mkdir()
    session.keep_database_files(str(kept))
    assert read(kept / "db.wal") == "torn"
    assert not (kept / CRASH_IMAGE).exists()


def image_of(session) -> str:
    session.save_crash_image({"max_num_threads": 1})
    return session.crash_image()


def test_every_open_starts_from_a_fresh_copy(session, tmp_path, monkeypatch) -> None:
    seen = []

    class Recovers:
        def __init__(self, path, config, loads, inject, debug) -> None:
            seen.append((read(f"{path}.wal"), config, loads))
            write(f"{path}.wal", "")

        def kill(self) -> None:
            pass

    monkeypatch.setattr(reopen_image, "Engine", Recovers)
    image = image_of(session)
    assert reopen_image.reopen(image, str(tmp_path), False) == "opened"
    assert reopen_image.reopen(image, str(tmp_path), False) == "opened"
    assert seen == [("torn", {"max_num_threads": 1}, ["CALL threads=1"])] * 2
    assert sorted(os.listdir(tmp_path)) == ["db-1", "db-1-image"]


@pytest.mark.parametrize(
    "error, kind",
    [
        (EngineError("Catalog exception: C already exists in catalog."), "error"),
        (AssertionError("engine process died with SIGSEGV on open\n#0 frame"), "died"),
    ],
)
def test_a_failed_open_is_described_the_same_way_each_time(
    session, tmp_path, monkeypatch, error, kind
) -> None:
    def fails(*_) -> None:
        raise error

    monkeypatch.setattr(reopen_image, "Engine", fails)
    image = image_of(session)
    first = reopen_image.reopen(image, str(tmp_path), False)
    assert first.startswith(f"{kind} ")
    assert str(error).splitlines()[0] in first
    assert reopen_image.reopen(image, str(tmp_path), False) == first


def test_one_failure_gets_one_text_whatever_the_copy_is_called() -> None:
    a = reopen_image.describe(
        EngineError("Cannot read /tmp/image-abc/copy/db.wal: gone"), "/tmp/image-abc"
    )
    b = reopen_image.describe(
        EngineError("Cannot read /tmp/image-xyz/copy/db.wal: gone"), "/tmp/image-xyz"
    )
    assert a.split(":")[0] == b.split(":")[0]


def test_crashes_that_share_a_first_line_are_told_apart() -> None:
    died = "engine process died with exit code -11 on open\n"
    a = reopen_image.describe(AssertionError(died + "signal 11 address (nil)\n#0 free"))
    b = reopen_image.describe(AssertionError(died + "signal 11 address (nil)\n#0 memcpy"))
    assert a != b


def test_addresses_do_not_split_one_crash() -> None:
    died = "engine process died with exit code -11 on open\nsignal 11 address "
    a = reopen_image.describe(AssertionError(died + "0x7f001234"))
    b = reopen_image.describe(AssertionError(died + "0x7f00abcd"))
    assert a == b
