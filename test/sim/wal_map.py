"""Reads a WAL's frames without the engine's reader, so that a torn or damaged file cannot crash it.

A WAL holds a 16-byte database ID, a checksum flag byte and, with the flag set, the header's u64
checksum. Then come frames: a u64 payload length, the payload, whose first byte is the record type,
and with the flag set a u64 checksum. Checksums are not verified, and a damaged length ends the
walk: there is no marker to find the next frame by. Only the layout is known here, so this also
reads the WALs of builds it was not built with.
"""

from __future__ import annotations

import glob
import hashlib
import itertools
import os
import struct
import subprocess
from dataclasses import dataclass

RECORDS = {
    1: "BEGIN_TRANSACTION_RECORD",
    2: "COMMIT_RECORD",
    3: "ABORT_RECORD",
    13: "COPY_TABLE_RECORD",
    14: "CREATE_CATALOG_ENTRY_RECORD",
    15: "CREATE_INDEX_RECORD",
    16: "DROP_CATALOG_ENTRY_RECORD",
    17: "ALTER_TABLE_ENTRY_RECORD",
    18: "UPDATE_SEQUENCE_RECORD",
    19: "UPDATE_SEQUENCE_NAMED_RECORD",
    30: "TABLE_INSERTION_RECORD",
    31: "NODE_DELETION_RECORD",
    32: "NODE_UPDATE_RECORD",
    33: "REL_DELETION_RECORD",
    34: "REL_DETACH_DELETE_RECORD",
    35: "REL_UPDATE_RECORD",
    100: "LOAD_EXTENSION_RECORD",
    254: "CHECKPOINT_RECORD",
}
COMMIT, ABORT, CHECKPOINT = 2, 3, 254
DATABASE_ID_SIZE = 16
U64_SIZE = 8
READ_BUFFER_SIZE = 4096  # BufferedFileReader's
WAL_SUFFIXES = ("wal", "wal.checkpoint")
TEAR_PLACES = ("header", "commit_end", "frame_end", "first_byte", "inside", "past_boundary")


@dataclass(frozen=True)
class Frame:
    offset: int
    end: int
    type: int
    length: int  # of the payload

    @property
    def name(self) -> str:
        return RECORDS.get(self.type, f"type {self.type}")


@dataclass(frozen=True)
class WalMap:
    size: int
    header_end: int
    frames: tuple[Frame, ...]
    # Where the frame after the last complete one says it ends, if its length is readable.
    declared_end: int | None = None
    torn_type: int | None = None  # the record type of that frame, if its first byte is there

    @property
    def complete_end(self) -> int:
        if self.frames:
            return self.frames[-1].end
        return self.header_end if self.size >= self.header_end else 0

    @property
    def torn_bytes(self) -> int:
        return self.size - self.complete_end

    def has(self, record_type: int) -> bool:
        return any(frame.type == record_type for frame in self.frames)


def digest_of(data: bytes) -> str:
    return hashlib.blake2b(data, digest_size=8).hexdigest()


def parse_wal(data: bytes) -> WalMap:
    checksums = len(data) > DATABASE_ID_SIZE and data[DATABASE_ID_SIZE] != 0
    trailer = U64_SIZE if checksums else 0
    header_end = DATABASE_ID_SIZE + 1 + trailer
    frames, offset, declared_end, torn_type = [], header_end, None, None
    while offset + U64_SIZE <= len(data):
        (length,) = struct.unpack_from("<Q", data, offset)
        end = offset + U64_SIZE + length + trailer
        if end > len(data) or length == 0:
            declared_end = end
            if offset + U64_SIZE < len(data):
                torn_type = data[offset + U64_SIZE]
            break
        frames.append(Frame(offset, end, data[offset + U64_SIZE], length))
        offset = end
    return WalMap(len(data), header_end, tuple(frames), declared_end, torn_type)


def read_wal(path: str) -> WalMap:
    """The map of the WAL at `path`, which may be missing: a missing file maps like an empty one."""
    if not os.path.exists(path):
        return parse_wal(b"")
    with open(path, "rb") as f:
        return parse_wal(f.read())


def last_commit_end(wal: WalMap, cut: int) -> int:
    """Where the last commit that lies wholly before `cut` ends, or 0 if none does."""
    return max((f.end for f in wal.frames if f.type == COMMIT and f.end <= cut), default=0)


def boundary_inside(frame: Frame) -> int | None:
    """The first read-buffer boundary with bytes of `frame` on both sides, if any."""
    boundary = (frame.offset // READ_BUFFER_SIZE + 1) * READ_BUFFER_SIZE
    return boundary if boundary < frame.end - 1 else None


def cut_point(wal: WalMap, place: str, position: int) -> int:
    """Where to cut a WAL with frames. `position` picks among the points of the kind `place` names."""
    if place == "header":
        return position % (wal.header_end + 1)
    if place == "commit_end":
        ends = [frame.end for frame in wal.frames if frame.type == COMMIT]
        if ends:
            return ends[position % len(ends)]
    frame = wal.frames[position % len(wal.frames)]
    if place == "past_boundary":
        crossing = [f for f in wal.frames if boundary_inside(f) is not None]
        if crossing:
            frame = max(crossing, key=lambda f: f.end - f.offset)
            boundary = boundary_inside(frame)
            return boundary + 1 + position % (frame.end - 1 - boundary)
    if place == "frame_end":
        return frame.end
    if place == "first_byte":
        return frame.offset + 1
    return frame.offset + 1 + position % (frame.end - frame.offset)


def shape_of(frames: list[Frame]) -> str:
    return digest_of(" ".join(f"{f.type}:{f.length}" for f in frames).encode())


def fingerprint(wal: WalMap) -> dict:
    """What a WAL holds, to compare with another run's: its transactions and checkpoints in order,
    each with a digest of its records' types and lengths, then the same of what follows the last of
    them, with the torn frame's type and declared end. No payload bytes are compared, since they
    differ between runs of the same statements: a NULL default holds uninitialized memory and a
    delete holds a physical offset. So two runs that wrote other values of the same lengths
    compare equal."""
    steps, frames = [], []
    for frame in wal.frames:
        frames.append(frame)
        if frame.type in (COMMIT, ABORT, CHECKPOINT):
            steps.append(f"{frame.name.removesuffix('_RECORD').lower()}:{shape_of(frames)}")
            frames = []
    torn = f"{wal.torn_type}:{wal.declared_end}"
    tail = digest_of(f"{shape_of(frames)} {torn}".encode())
    return dict(steps=steps, open_frames=len(frames), torn_bytes=wal.torn_bytes, tail=tail)


def wal_state(database: str) -> dict:
    """The fingerprint of each WAL file a database has."""
    paths = {suffix: f"{database}.{suffix}" for suffix in WAL_SUFFIXES}
    return {s: fingerprint(read_wal(path)) for s, path in paths.items() if os.path.exists(path)}


def difference(original: dict, replay: dict) -> str | None:
    """How a replay's fingerprint differs from the original run's, or None."""
    a, b = original["steps"], replay["steps"]
    common = sum(1 for _ in itertools.takewhile(lambda pair: pair[0] == pair[1], zip(a, b)))
    if common < min(len(a), len(b)):
        same_kind = a[common].split(":")[0] == b[common].split(":")[0]
        what = "other records" if same_kind else "another record"
        return f"{what} at step {common + 1}: {a[common]} in the original, {b[common]} here"
    if len(b) < len(a):
        return f"stopped earlier, after {len(b)} of the original's {len(a)} steps"
    if len(b) > len(a):
        return f"went further, with {len(b)} steps to the original's {len(a)}"
    if original != replay:
        return (
            f"the same {len(a)} steps, then another tail: {replay['open_frames']} frames and "
            f"{replay['torn_bytes']} torn bytes ({replay['tail']}) here, "
            f"{original['open_frames']} and {original['torn_bytes']} ({original['tail']}) "
            "in the original"
        )
    return None


def state_difference(original: dict, replay: dict) -> str | None:
    """How the WAL files after a crash in a replay differ from the original run's, or None."""
    notes = []
    for suffix in WAL_SUFFIXES:
        if suffix not in original and suffix not in replay:
            continue
        if suffix not in original or suffix not in replay:
            where = "here" if suffix in replay else "in the original"
            notes.append(f"db.{suffix} exists only {where}")
        elif note := difference(original[suffix], replay[suffix]):
            notes.append(f"db.{suffix}: {note}")
    return "; ".join(notes) or None


def describe_map(wal: WalMap) -> str:
    """A WAL's frames, one line per run of the same record type."""
    commits = sum(frame.type == COMMIT for frame in wal.frames)
    lines = [
        f"{wal.size} bytes, {len(wal.frames)} complete frames, {commits} commits, "
        f"last commit ends at {last_commit_end(wal, wal.size)}"
    ]
    for _, group in itertools.groupby(wal.frames, key=lambda frame: frame.type):
        run = list(group)
        count = f" x {len(run)}" if len(run) > 1 else ""
        lines.append(f"  [{run[0].offset}, {run[-1].end}) {run[0].name}{count}")
    if wal.torn_bytes:
        lines.append(
            f"  {wal.torn_bytes} bytes after the last complete frame, at {wal.complete_end}"
        )
    if wal.declared_end is not None:
        pages = wal.complete_end // READ_BUFFER_SIZE, (wal.size - 1) // READ_BUFFER_SIZE
        crosses = "crosses" if pages[0] != pages[1] else "stays inside"
        lines.append(
            f"  the next frame declares its end at {wal.declared_end}; the part in the file "
            f"{crosses} a {READ_BUFFER_SIZE}-byte read-buffer boundary"
        )
    return "\n".join(lines) + "\n"


def run_wal_dump(binary: str, wal: str, out) -> None:
    """Appends wal_dump's output and exit code to `out`. It uses the engine's reader, so it may
    crash on the file; that is recorded and changes nothing else."""
    out.write(f"== {binary} --wal {wal}\n")
    out.flush()
    try:
        code = subprocess.run(
            [binary, "--wal", wal], stdout=out, stderr=subprocess.STDOUT, timeout=60, check=False
        ).returncode
    except (OSError, subprocess.TimeoutExpired) as e:
        out.write(f"not run to the end: {e}\n")
    else:
        out.write(f"exit code {code}\n")


def save_wal_notes(directory: str) -> None:
    """Writes wal-map.txt for the WAL files in `directory` and, when SIM_WAL_DUMP names a wal_dump
    built from the engine under test, wal.txt with what it reads there."""
    wals = sorted(glob.glob(os.path.join(directory, "db.wal*")))
    wals = [wal for wal in wals if wal.endswith(tuple(f".{s}" for s in WAL_SUFFIXES))]
    if not wals:
        return
    with open(os.path.join(directory, "wal-map.txt"), "w") as f:
        for wal in wals:
            f.write(f"== {os.path.basename(wal)}\n{describe_map(read_wal(wal))}")
    if binary := os.environ.get("SIM_WAL_DUMP"):
        with open(os.path.join(directory, "wal.txt"), "w") as f:
            for wal in wals:
                run_wal_dump(binary, wal, f)
