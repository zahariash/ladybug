"""The shrinkray stage of replay.py --shrink (github.com/DRMacIver/shrinkray): the steps of a
trace in the form shrinkray reduces, and the run that reduces them while `replay.py --check`
says they still fail the same way."""

from __future__ import annotations

import glob
import json
import os
import shutil
import signal
import subprocess
import sys

REPLAY = os.path.join(os.path.dirname(os.path.abspath(__file__)), "replay.py")

# The fields of each step that a replay reads, by step kind; the rest record the original run.
STEP_FIELDS = {
    "execute": {"conn": int, "query": str, "params": dict},
    "prepared": {"conn": int, "query": str, "params": dict},
    "columns": {"conn": int, "query": str, "params": dict},
    "prepare": {"name": str, "query": str, "params": dict},
    "run": {"name": str, "params": dict},
    "reopen": {},
    "crash": {},
    "crash_during": {"query": str, "delay": (int, float)},
    "crash_at": {"query": str, "syscall": str, "n": int},
    "hold": {"conn": int},
    "release": {},
    "race": {"writes": list, "queries": list, "readers": int},
}
ALSO_INTERESTING = 101  # shrinkray's default: kept as a lead, not reduced towards


def compact(trace: list) -> dict:
    """What shrinkray reduces: the settings, the setup statements and each step's replay fields."""
    head = trace[0]
    steps = [{"op": op["op"], **{k: op[k] for k in STEP_FIELDS[op["op"]]}} for op in trace[1:]]
    loads = head.get("loads", [])
    return dict(
        config=head["config"],
        loads=loads,
        reopen_loads=head.get("reopen_loads", loads),
        steps=steps,
    )


def expand(data) -> list | None:
    """The trace for compact()'s form, or None if a reduction left it malformed."""
    if not isinstance(data, dict) or not isinstance(data.get("steps"), list):
        return None
    head = dict(op="open", config=data.get("config", {}), loads=data.get("loads", []))
    head["reopen_loads"] = data.get("reopen_loads", head["loads"])
    if not isinstance(head["config"], dict) or not all(
        isinstance(loads, list) and all(isinstance(q, str) for q in loads)
        for loads in (head["loads"], head["reopen_loads"])
    ):
        return None
    for op in data["steps"]:
        fields = STEP_FIELDS.get(op.get("op")) if isinstance(op, dict) else None
        if fields is None or not all(isinstance(op.get(k), t) for k, t in fields.items()):
            return None
        if isinstance(op.get("n"), bool) or isinstance(op.get("conn"), bool):
            return None
    return [head] + data["steps"]


INTERESTING = """#!/bin/sh
# shrinkray's interestingness test, written by replay.py --shrink: the steps in $1 must still show
# what target.json describes{fixed_note}.
"{python}" "{replay}" "$1" --check "{target}" || exit $?
"""
FIXED = """PYTHONPATH="{fixed}" "{python}" "{replay}" "$1" --check "{target}" && exit 1
exit 0
"""


# Frames a build without symbols reports for any crash in a query: the operator loop.
GENERIC_FRAMES = ("lbug::processor::PhysicalOperator::", "lbug::common::TaskScheduler::")


def distinctive(target: dict) -> bool:
    """Whether a signature tells one bug from another: an error's text, or engine frames other
    than the operator loop. A split of the ways of running a probe does not."""
    frames = target.get("frames") or []
    return bool(target.get("error")) or any(not f.startswith(GENERIC_FRAMES) for f in frames)


def latest_steps(work: str) -> dict | None:
    """The reduced steps: steps.json, or the newest reduction shrinkray recorded if stopping it
    cut a write short."""
    history = glob.glob(os.path.join(work, ".shrinkray", "*", "reductions", "*", "steps.json"))
    for path in [os.path.join(work, "steps.json"), *sorted(history, reverse=True)]:
        try:
            with open(path) as f:
                return json.load(f)
        except (OSError, ValueError):
            continue
    return None


def run_shrinkray(data: dict, target: dict, work: str, args) -> dict | None:
    """Reduces `data` (compact()'s form) with shrinkray in `work`, for up to --shrink-minutes,
    while `replay.py --check` accepts it against `target`; returns the smallest result. With
    --fixed it must also pass on that build: checked for every candidate when the signature alone
    could match another bug, else once at the end."""
    shrinkray = args.shrinkray or shutil.which("shrinkray")
    if not shrinkray:
        print("shrinkray is not installed (see docs/testing.md, Triaging a failure).")
        return None
    os.makedirs(work, exist_ok=True)
    paths = {
        name: os.path.join(work, name) for name in ("target.json", "steps.json", "interesting.sh")
    }
    with open(paths["target.json"], "w") as f:
        json.dump(target, f, indent=1)
    with open(paths["steps.json"], "w") as f:
        json.dump(data, f, indent=1, default=str)
    names = dict(python=sys.executable, replay=REPLAY, target=paths["target.json"])
    each = bool(args.fixed) and not distinctive(target)
    test = INTERESTING.format(fixed_note=", and not on the fixed build" if each else "", **names)
    if each:
        test += FIXED.format(fixed=args.fixed, **names)
    with open(paths["interesting.sh"], "w") as f:
        f.write(test)
    os.chmod(paths["interesting.sh"], 0o755)
    command = [
        shrinkray,
        "--no-llm",
        "--no-restart",
        "--ui",
        "basic",
        "--parallelism",
        str(args.parallelism),
        "--memory-limit",
        "0",
        "--timeout",
        str(args.shrink_timeout),
        "--input-type",
        "arg",
        "--formatter",
        "none",
        "--seed",
        "0",
    ]
    command += ["./interesting.sh", "steps.json"]
    print(f"Shrinking with shrinkray for up to {args.shrink_minutes} min in {work}...", flush=True)
    with open(os.path.join(work, "shrinkray.log"), "w") as log:
        process = subprocess.Popen(
            command, cwd=work, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
        )
        try:
            process.wait(timeout=args.shrink_minutes * 60)
        except subprocess.TimeoutExpired:
            # Like Ctrl-C: shrinkray then stops its tests, which run in process groups of their own.
            process.send_signal(signal.SIGINT)
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
    data = latest_steps(work)
    if data and args.fixed and not each:
        path = os.path.join(work, "steps.shrunk.json")
        with open(path, "w") as f:
            json.dump(data, f, indent=1, default=str)
        fixed = subprocess.run(
            [sys.executable, names["replay"], path, "--check", paths["target.json"]],
            env={**os.environ, "PYTHONPATH": args.fixed},
            capture_output=True,
        )
        if fixed.returncode == 0:
            print("Warning: the shrunk steps fail the same way on the fixed build too.")
    return data
