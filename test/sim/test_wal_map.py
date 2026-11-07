"""Walking a WAL's frames, on files built here, without a database."""

from __future__ import annotations

import os
import pathlib
import re
import stat
import struct

import pytest
from wal_map import (
    ABORT,
    CHECKPOINT,
    COMMIT,
    RECORDS,
    TEAR_PLACES,
    Frame,
    boundary_inside,
    cut_point,
    describe_map,
    difference,
    fingerprint,
    last_commit_end,
    parse_wal,
    read_wal,
    save_wal_notes,
    state_difference,
    wal_state,
)

BEGIN, INSERT = 1, 30


def wal_bytes(*records: tuple, checksums: bool = False, database: bytes = b"\x01" * 16) -> bytes:
    """A WAL with one frame per (type, filler) record; the checksums are not real."""
    data = database + bytes([checksums]) + (b"\xee" * 8 if checksums else b"")
    for record_type, filler in records:
        payload = bytes([record_type]) + filler
        data += struct.pack("<Q", len(payload)) + payload + (b"\xee" * 8 if checksums else b"")
    return data


TXN = ((BEGIN, b""), (INSERT, b"row"), (COMMIT, b""))


@pytest.mark.parametrize("checksums, header, trailer", [(False, 17, 0), (True, 25, 8)])
def test_frames_follow_the_layout(checksums, header, trailer) -> None:
    wal = parse_wal(wal_bytes(*TXN, checksums=checksums))
    assert wal.header_end == header
    assert [frame.name for frame in wal.frames] == [
        "BEGIN_TRANSACTION_RECORD",
        "TABLE_INSERTION_RECORD",
        "COMMIT_RECORD",
    ]
    assert wal.frames[0].offset == header
    assert wal.frames[0].end == header + 8 + 1 + trailer
    assert wal.frames[1].end == wal.frames[0].end + 8 + 4 + trailer
    assert wal.frames[-1].end == wal.size
    assert wal.torn_bytes == 0


def test_a_torn_frame_ends_the_walk_and_says_where_it_would_end() -> None:
    data = wal_bytes(*TXN)
    wal = parse_wal(data[:-1])
    assert len(wal.frames) == 2
    assert wal.declared_end == len(data)
    assert wal.torn_bytes == 8


def test_a_cut_inside_a_length_field_leaves_no_declared_end() -> None:
    wal = parse_wal(wal_bytes(*TXN)[: 17 + 3])
    assert wal.frames == () and wal.declared_end is None and wal.torn_bytes == 3


@pytest.mark.parametrize("data", [b"", b"\x01" * 5, b"not a wal!", b"\x00" * 4096, b"\xff" * 4096])
def test_any_bytes_map_without_an_error(data) -> None:
    wal = parse_wal(data)
    assert wal.size == len(data)
    describe_map(wal)
    fingerprint(wal)


def test_zeroed_bytes_after_the_frames_end_the_walk() -> None:
    wal = parse_wal(wal_bytes(*TXN) + b"\x00" * 64)
    assert len(wal.frames) == 3 and wal.torn_bytes == 64


def test_a_huge_length_ends_the_walk() -> None:
    wal = parse_wal(wal_bytes(*TXN) + struct.pack("<Q", 1 << 60) + b"x")
    assert len(wal.frames) == 3 and wal.declared_end > wal.size


def test_the_records_match_the_engine() -> None:
    root = pathlib.Path(__file__).parents[2]
    header = root / "src/include/storage/wal/record/wal_record_base.h"
    declared = dict(re.findall(r"^\s+([A-Z_]+_RECORD) = (\d+)", header.read_text(), re.M))
    declared.pop("INVALID_RECORD")
    assert declared, "no records found"
    for name, value in declared.items():
        assert RECORDS.get(int(value)) == name


def test_a_missing_file_maps_like_an_empty_one(tmp_path) -> None:
    assert read_wal(str(tmp_path / "none")) == parse_wal(b"")


def test_last_commit_end_counts_only_whole_commits() -> None:
    wal = parse_wal(wal_bytes(*TXN, *TXN))
    first, second = [frame.end for frame in wal.frames if frame.type == COMMIT]
    assert last_commit_end(wal, 0) == 0
    assert last_commit_end(wal, first - 1) == 0
    assert last_commit_end(wal, first) == first
    assert last_commit_end(wal, second - 1) == first
    assert last_commit_end(wal, second) == second


@pytest.mark.parametrize(
    "offset, end, expected",
    [
        (4000, 4200, 4096),
        (4095, 4097, None),  # one byte past the boundary leaves nothing to cut into
        (4000, 9000, 4096),  # the first of two
        (100, 200, None),
        (4096, 4200, None),  # starts on a boundary
        (4096, 8200, 8192),
    ],
)
def test_boundary_inside(offset, end, expected) -> None:
    assert boundary_inside(Frame(offset, end, INSERT, 0)) == expected


def crossing_wal():
    records = [(BEGIN, b""), (INSERT, b"r" * 3000), (INSERT, b"s" * 3000), (COMMIT, b""), *TXN]
    return parse_wal(wal_bytes(*records, checksums=True))


@pytest.mark.parametrize("place", TEAR_PLACES)
@pytest.mark.parametrize("position", [0, 1, 7, 12345, 2**32])
def test_every_cut_lies_inside_the_wal(place, position) -> None:
    wal = crossing_wal()
    assert 0 <= cut_point(wal, place, position) <= wal.size


def test_each_place_means_what_it_says() -> None:
    wal = crossing_wal()
    commits = [frame.end for frame in wal.frames if frame.type == COMMIT]
    ends = {frame.end for frame in wal.frames}
    assert cut_point(wal, "header", 3) == 3
    assert {cut_point(wal, "commit_end", p) for p in range(4)} == set(commits)
    assert cut_point(wal, "frame_end", 2) in ends
    assert cut_point(wal, "first_byte", 0) == wal.frames[0].offset + 1
    past = cut_point(wal, "past_boundary", 5)
    crossing = next(f for f in wal.frames if f.offset < 4096 < f.end)
    assert 4096 < past < crossing.end


def test_a_wal_with_no_crossing_frame_still_gets_a_cut() -> None:
    wal = parse_wal(wal_bytes(*TXN))
    assert 0 < cut_point(wal, "past_boundary", 9) <= wal.size


def test_a_fingerprint_ignores_payload_bytes_of_one_length() -> None:
    assert fingerprint(parse_wal(wal_bytes((BEGIN, b""), (INSERT, b"ab"), (COMMIT, b"")))) == (
        fingerprint(parse_wal(wal_bytes((BEGIN, b""), (INSERT, b"cd"), (COMMIT, b""))))
    )


def test_a_fingerprint_sees_a_reordered_transaction_and_another_record_type() -> None:
    a = wal_bytes((BEGIN, b""), (INSERT, b"a"), (INSERT, b"bb"), (COMMIT, b""))
    b = wal_bytes((BEGIN, b""), (INSERT, b"bb"), (INSERT, b"a"), (COMMIT, b""))
    assert fingerprint(parse_wal(a)) != fingerprint(parse_wal(b))
    other = wal_bytes((BEGIN, b""), (32, b"a"), (INSERT, b"bb"), (COMMIT, b""))
    assert fingerprint(parse_wal(a)) != fingerprint(parse_wal(other))


def test_a_fingerprint_ignores_the_database_id_and_the_checksums() -> None:
    a = fingerprint(parse_wal(wal_bytes(*TXN, database=b"\x01" * 16)))
    b = fingerprint(parse_wal(wal_bytes(*TXN, database=b"\x02" * 16, checksums=True)))
    assert a == b
    assert [step.split(":")[0] for step in a["steps"]] == ["commit"]


def test_a_fingerprint_separates_transactions_checkpoints_and_aborts() -> None:
    records = (*TXN, (CHECKPOINT, b""), (BEGIN, b""), (ABORT, b""), (BEGIN, b""))
    print_ = fingerprint(parse_wal(wal_bytes(*records)))
    kinds = [step.split(":")[0] for step in print_["steps"]]
    assert kinds == ["commit", "checkpoint", "abort"]
    assert print_["open_frames"] == 1


def steps_of(*fillers: bytes) -> dict:
    records = [r for filler in fillers for r in ((BEGIN, b""), (INSERT, filler), (COMMIT, b""))]
    return fingerprint(parse_wal(wal_bytes(*records)))


def test_difference_tells_a_different_path_from_a_shorter_one() -> None:
    original = steps_of(b"a", b"b", b"c")
    assert difference(original, steps_of(b"a", b"b", b"c")) is None
    assert "other records at step 2" in difference(original, steps_of(b"a", b"xx", b"c"))
    assert "stopped earlier, after 1 of the original's 3" in difference(original, steps_of(b"a"))
    assert "went further, with 4 steps" in difference(original, steps_of(b"a", b"b", b"c", b"d"))
    checkpointed = fingerprint(parse_wal(wal_bytes(*TXN, (CHECKPOINT, b""))))
    aborted = fingerprint(parse_wal(wal_bytes(*TXN, (BEGIN, b""), (ABORT, b""))))
    assert "another record at step 2" in difference(checkpointed, aborted)


def test_difference_notes_a_different_tail() -> None:
    whole = wal_bytes(*TXN, (BEGIN, b""))
    note = difference(fingerprint(parse_wal(whole)), fingerprint(parse_wal(whole + b"xx")))
    assert "the same 1 steps" in note and "2 torn bytes" in note


def test_an_unfinished_header_counts_as_torn_bytes_whatever_the_database_id() -> None:
    torn = fingerprint(parse_wal(wal_bytes(*TXN)[:10]))
    assert torn["torn_bytes"] == 10
    assert difference(fingerprint(parse_wal(b"")), torn) is not None
    other = fingerprint(parse_wal(wal_bytes(*TXN, database=b"\x02" * 16)[:10]))
    assert difference(torn, other) is None


def test_torn_tails_compare_by_shape_not_bytes() -> None:
    whole = wal_bytes(*TXN, (INSERT, b"long enough to tear"))
    a = fingerprint(parse_wal(whole[:-3]))
    other = bytearray(whole[:-3])
    other[-1] ^= 0xFF
    assert difference(a, fingerprint(parse_wal(bytes(other)))) is None
    longer = wal_bytes(*TXN, (INSERT, b"long enough to tear, and more"))
    b = fingerprint(parse_wal(longer[: len(whole) - 3]))
    assert a["torn_bytes"] == b["torn_bytes"]
    note = difference(a, b)
    assert a["tail"] in note and b["tail"] in note
    updated = bytearray(whole[:-3])
    updated[wal_bytes(*TXN).__len__() + 8] = 32
    assert difference(a, fingerprint(parse_wal(bytes(updated)))) is not None


def test_state_difference_notes_a_file_only_one_side_has(tmp_path) -> None:
    same = dict(wal=steps_of(b"a"))
    assert state_difference(same, same) is None
    frozen = {**same, "wal.checkpoint": steps_of(b"b")}
    assert state_difference(frozen, same) == "db.wal.checkpoint exists only in the original"
    assert state_difference(same, frozen) == "db.wal.checkpoint exists only here"
    assert "db.wal: other records" in state_difference(same, dict(wal=steps_of(b"zz")))


def test_wal_state_reads_each_wal_file(tmp_path) -> None:
    database = tmp_path / "db"
    assert wal_state(str(database)) == {}
    (tmp_path / "db.wal").write_bytes(wal_bytes(*TXN))
    (tmp_path / "db.wal.checkpoint").write_bytes(wal_bytes(*TXN, *TXN))
    state = wal_state(str(database))
    assert len(state["wal"]["steps"]) == 1 and len(state["wal.checkpoint"]["steps"]) == 2


def test_describe_map_groups_runs_and_flags_a_torn_frame_crossing_the_buffer() -> None:
    rows = [(INSERT, b"r" * 500) for _ in range(9)]
    data = wal_bytes((BEGIN, b""), *rows, (COMMIT, b""))
    text = describe_map(parse_wal(data))
    assert "TABLE_INSERTION_RECORD x 9" in text and "1 commits" in text
    big = wal_bytes((BEGIN, b""), (INSERT, b"r" * 4000), (COMMIT, b""), (INSERT, b"s" * 2000))
    text = describe_map(parse_wal(big[:4200]))
    assert "crosses a 4096-byte read-buffer boundary" in text


def executable(path, text: str) -> str:
    path.write_text("#!/bin/sh\n" + text)
    path.chmod(path.stat().st_mode | stat.S_IEXEC)
    return str(path)


def test_notes_map_every_wal_file_and_nothing_else(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv("SIM_WAL_DUMP", raising=False)
    (tmp_path / "db.wal").write_bytes(wal_bytes(*TXN))
    (tmp_path / "db.wal.checkpoint").write_bytes(wal_bytes(*TXN))
    (tmp_path / "db.wal.other").write_bytes(b"x")
    save_wal_notes(str(tmp_path))
    text = (tmp_path / "wal-map.txt").read_text()
    assert "== db.wal\n" in text and "== db.wal.checkpoint\n" in text and "other" not in text
    assert not (tmp_path / "wal.txt").exists()


def test_without_a_wal_no_notes_are_written(tmp_path) -> None:
    save_wal_notes(str(tmp_path))
    assert os.listdir(tmp_path) == []


def test_wal_dump_output_and_a_crash_are_recorded(tmp_path, monkeypatch) -> None:
    kept = tmp_path / "kept"
    kept.mkdir()
    (kept / "db.wal").write_bytes(wal_bytes(*TXN))
    monkeypatch.setenv(
        "SIM_WAL_DUMP", executable(tmp_path / "dump", 'echo "read $1"\nkill -SEGV $$\n')
    )
    save_wal_notes(str(kept))
    text = (kept / "wal.txt").read_text()
    assert f"read --wal" in text and "exit code -11" in text


def test_a_missing_wal_dump_is_recorded(tmp_path, monkeypatch) -> None:
    (tmp_path / "db.wal").write_bytes(wal_bytes(*TXN))
    monkeypatch.setenv("SIM_WAL_DUMP", str(tmp_path / "missing"))
    save_wal_notes(str(tmp_path))
    assert "not run to the end" in (tmp_path / "wal.txt").read_text()
