"""Opens a crash image again, each time from a fresh copy.

A kept failure's crash image (<trace dir>/failed/<signature>/crash-image, see
Session.save_crash_image) holds the database's files as the last crash left them and how the
simulator opened them. Recovery starts from those files, whenever the crash happened, so a recovery
failure that replay.py hits once in twenty runs should fail here every time. The exit status is 0
if every open ended the same way.

    PYTHONPATH=tools/python_api/build uv run --no-project --with hypothesis \\
        python test/sim/reopen_image.py <crash-image> [--runs 5] [--gdb]
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from engine import Engine, EngineError, scratch_root  # noqa: E402
from session import IMAGE_OPEN, failure_kind  # noqa: E402
from triage import signature  # noqa: E402


def describe(error: BaseException) -> str:
    """How an open failed: the same failure gets the same text in every run."""
    kind, message = failure_kind(error), str(error)
    key = signature(kind, dict(op="open"), message)["key"]
    return f"{kind} {key}: {message.splitlines()[0][:160]}"


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
