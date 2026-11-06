"""What Session keeps of a crash, and how reopen_image.py opens it again, without a database."""

from __future__ import annotations

import json
import os

import pytest
import reopen_image
from engine import EngineError
from session import CRASH_IMAGE, IMAGE_OPEN, Session


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
    with open(os.path.join(image, IMAGE_OPEN)) as f:
        assert json.load(f) == {"config": {"enable_checksums": True}, "loads": ["CALL threads=1"]}


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
