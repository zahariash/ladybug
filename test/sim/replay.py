"""Replays a simulator trace without the model, and minimizes it.

The simulator writes the engine interactions of each workload, with their outcomes, to
<trace dir>/current/trace.json (see Session). This script runs them again on a new database
and reports the first engine-level failure: a crash, a hang, a race error, or an error where
the original statement succeeded. Wrong results need the model and are not replayed. Timing
failures may need several runs; --runs N counts a failure in any of N.

With --minimize it delta-debugs the trace to the fewest steps that still fail the same way and
writes them next to the trace as trace.min.json. With --gdb a crash reports its native stack.

    PYTHONPATH=tools/python_api/build uv run --no-project --with hypothesis \\
        python test/sim/replay.py <trace.json> [--runs 5] [--minimize] [--gdb]
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from engine import Engine, EngineError, scratch_root  # noqa: E402


def original_failure(trace: list) -> tuple[str, str] | None:
    """The kind and detail of the first engine-level failure the original run recorded."""
    for op in trace:
        outcome = op.get("outcome", "ok")
        if op.get("errors"):
            return "race", ""
        if outcome.startswith(("died", "hung")):
            return outcome.split(":")[0], ""
        if outcome.startswith("error") and not op.get("expected"):
            return "error", outcome[7:87]
    return None


class Replayer:
    def __init__(self, trace: list, debug: bool) -> None:
        self.trace, self.debug = trace, debug
        self.dir = tempfile.mkdtemp(prefix="replay-", dir=scratch_root())
        self.path = os.path.join(self.dir, "db")
        self.config, self.loads = trace[0]["config"], trace[0]["loads"]
        self.engine = None

    def open(self, inject=None) -> None:
        self.engine = None
        config = {**self.config, "max_num_threads": 1} if inject else self.config
        self.engine = Engine(self.path, config, self.loads, inject, self.debug)

    def step(self, op: dict) -> None:
        kind = op["op"]
        if kind == "open":
            self.open()
        elif kind == "execute":
            self.engine.execute(op["query"], op["params"], op["conn"])
        elif kind == "prepared":
            self.engine.execute_prepared(op["query"], op["params"], op["conn"])
        elif kind == "prepare":
            self.engine.prepare(op["name"], op["query"], op["params"])
        elif kind == "run":
            self.engine.run(op["name"], op["params"])
        elif kind == "reopen":
            self.engine.close()
            self.open()
        elif kind == "crash":
            self.engine.kill()
            self.open()
        elif kind == "crash_during":
            self.engine.start(op["query"])
            time.sleep(op["delay"])
            self.engine.kill()
            self.open()
        elif kind == "crash_at":
            self.engine.close()
            try:
                self.open(inject=(op["syscall"], op["n"]))
                self.engine.execute(op["query"], {})
                self.engine.close()
            except AssertionError as e:
                if "died" not in str(e):
                    raise
            except EngineError:
                pass
            self.open()
        elif kind == "race":
            _, errors = self.engine.race(op["writes"], op["queries"], op["readers"])
            if errors:
                raise RaceError(errors)

    def run(self) -> tuple[str, str, int, str] | None:
        """Replays the trace; returns the first failure as (kind, detail, step, message)."""
        try:
            for i, op in enumerate(self.trace):
                try:
                    self.step(op)
                except EngineError as e:
                    if op.get("outcome", "ok") == "ok":
                        return "error", str(e)[:80], i, str(e)
                except RaceError as e:
                    return "race", "", i, str(e)
                except AssertionError as e:
                    kind = "hung" if str(e).startswith("engine hung") else "died"
                    return kind, "", i, str(e)
            return None
        finally:
            if self.engine:
                self.engine.kill()
            shutil.rmtree(self.dir, ignore_errors=True)


class RaceError(Exception):
    pass


def reproduces(trace: list, target: tuple[str, str] | None, runs: int, debug: bool):
    """The first failure in up to `runs` replays that matches `target` (any failure if None)."""
    for _ in range(runs):
        failure = Replayer(trace, debug).run()
        if failure and (target is None or failure[:2] == target):
            return failure
    return None


def minimize(trace: list, fails) -> list:
    """Delta debugging (ddmin) over the steps after the first open: the smallest trace for
    which `fails(trace)` still holds."""
    steps, n = trace[1:], 2
    while len(steps) >= 2:
        chunk = max(1, len(steps) // n)
        subsets = [steps[i : i + chunk] for i in range(0, len(steps), chunk)]
        for i in range(len(subsets)):
            complement = [op for j, part in enumerate(subsets) if j != i for op in part]
            if fails(trace[:1] + complement):
                steps, n = complement, max(n - 1, 2)
                print(f"  {len(steps)} steps still fail", flush=True)
                break
        else:
            if n >= len(steps):
                break
            n = min(n * 2, len(steps))
    return trace[:1] + steps


def describe(op: dict) -> str:
    detail = op.get("query") or op.get("name") or ""
    if op["op"] == "crash_at":
        detail = f"{op['syscall']} #{op['n']} in {detail}"
    elif op["op"] == "race":
        detail = f"{len(op['writes'])} writes, {op['readers']} readers"
    return f"{op['op']:12} {detail[:110]}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("trace")
    parser.add_argument("--runs", type=int, default=1, help="replays per check")
    parser.add_argument("--minimize", action="store_true")
    parser.add_argument("--gdb", action="store_true", help="report native stacks of crashes")
    args = parser.parse_args()
    with open(args.trace) as f:
        trace = json.load(f)
    target = original_failure(trace)
    if target is None:
        print("The original run failed a model check; replay reports engine failures only.")
    failure = reproduces(trace, target, args.runs, args.gdb)
    if not failure:
        print(f"No failure in {args.runs} replay(s).")
        return
    kind, _, step, message = failure
    print(f"{kind} at step {step}: {describe(trace[step])}\n{message}")
    if args.minimize:
        target = failure[:2]
        minimized = minimize(trace, lambda t: bool(reproduces(t, target, args.runs, False)))
        out = os.path.join(os.path.dirname(os.path.abspath(args.trace)), "trace.min.json")
        with open(out, "w") as f:
            json.dump(minimized, f, indent=1, default=str)
        print(f"\nMinimized to {len(minimized)} steps, written to {out}:")
        for op in minimized:
            print("  " + describe(op))


if __name__ == "__main__":
    main()
