"""Opens a crash image again, each time from a fresh copy.

A kept failure's crash image (<trace dir>/failed/<signature>/crash-image, see
Session.save_crash_image) holds the files the last recovery started from and how the simulator
opened them, so a failure that depends only on those files repeats without replaying the trace. One
that depends on thread scheduling during recovery can still pass some opens. The exit status is 0
if every open ended the same way.

    PYTHONPATH=tools/python_api/build uv run --no-project --with hypothesis \\
        python test/sim/reopen_image.py <crash-image> [--runs 5] [--gdb]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import sys
import tempfile
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from engine import Engine, EngineError, scratch_root  # noqa: E402
from session import IMAGE_OPEN, failure_kind  # noqa: E402


def describe(error: BaseException) -> str:
    """How an open failed. Its key covers the whole message, stacks included, apart from addresses
    and numbers, since two crashes can share a first line and have no engine frames."""
    message = str(error)
    shape = re.sub(r"0x[0-9a-fA-F]+|\b\d+\b", "?", message)
    key = hashlib.sha1(shape.encode()).hexdigest()[:10]
    return f"{failure_kind(error)} {key}: {message.splitlines()[0][:160]}"


def reopen(image: str, scratch: str, debug: bool) -> str:
    """Opens a fresh copy of `image`: "opened", or how it failed."""
    with open(os.path.join(image, IMAGE_OPEN)) as f:
        how = json.load(f)
    directory = tempfile.mkdtemp(prefix="image-", dir=scratch)
    copy = os.path.join(directory, "copy")
    shutil.copytree(image, copy)
    try:
        Engine(os.path.join(copy, "db"), how["config"], how["loads"], None, debug).kill()
        return "opened"
    except (EngineError, AssertionError) as e:
        return describe(e)
    finally:
        shutil.rmtree(directory, ignore_errors=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("image")
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--gdb", action="store_true", help="report native stacks of crashes")
    args = parser.parse_args()
    scratch = scratch_root()
    outcomes = Counter(reopen(args.image, scratch, args.gdb) for _ in range(args.runs))
    shutil.rmtree(scratch, ignore_errors=True)
    for outcome, count in outcomes.most_common():
        print(f"{count} of {args.runs}: {outcome}")
    sys.exit(0 if len(outcomes) == 1 else 1)


if __name__ == "__main__":
    main()
