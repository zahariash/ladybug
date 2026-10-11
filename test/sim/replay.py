"""Replays a simulator trace without the model, and minimizes it.

The simulator writes the engine interactions of each workload, with their outcomes, to
<trace dir>/current/trace.json (see Session). This script runs them again on a new database
and reports the first engine-level failure: a crash, a hang, a race error, or an error where
the original statement succeeded. Wrong results need the model and are not replayed. Timing
failures may need several runs; --runs N counts a failure in any of N.

With --minimize it delta-debugs the trace to the fewest steps that still fail the same way,
replaces steps with simpler ones (a crash with a clean reopen, a timed or syscall crash with the
statement and a crash, a prepared statement with a plain one, another connection with the first)
and drops the database settings the failure does not need, and writes trace.min.json next to the
trace. With --emit it then writes trace.min.cypher, trace.min.py and finding.json (see triage.py)
and runs trace.min.py once to confirm it fails. With --gdb a crash reports its native stack.

    PYTHONPATH=tools/python_api/build uv run --no-project --with hypothesis \\
        python test/sim/replay.py <trace.json> [--runs 5] [--minimize] [--gdb]
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import triage  # noqa: E402
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
        elif kind == "hold":
            self.engine.hold(op["conn"])
        elif kind == "release":
            self.engine.release()
        elif kind == "race":
            _, errors = self.engine.race(op["writes"], op["queries"], op["readers"])
            if errors:
                raise RaceError(errors)
        # "failure" records what the simulator saw; it is not an engine interaction.

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


def simplifications(op: dict) -> list[list[dict]]:
    """Simpler steps that could replace `op`, simplest first."""
    options = []
    if op["op"] == "crash":
        options.append([dict(op="reopen")])
    elif op["op"] in ("crash_during", "crash_at"):
        statement = dict(op="execute", conn=0, query=op["query"], params={})
        options += [[statement, dict(op="reopen")], [statement, dict(op="crash")]]
    elif op["op"] == "prepared":
        options.append([{**op, "op": "execute"}])
    if op.get("conn"):
        options.append([{**op, "conn": 0}])
    return options


def simplify(trace: list, fails) -> list:
    """Replaces steps with simpler ones while `fails(trace)` still holds."""
    i = 1
    while i < len(trace):
        for replacement in simplifications(trace[i]):
            candidate = trace[:i] + replacement + trace[i + 1 :]
            if fails(candidate):
                print(f"  step {i}: {trace[i]['op']} -> {[op['op'] for op in replacement]}")
                trace = candidate
                break
        i += 1
    return trace


def reduce_config(trace: list, fails) -> tuple[list, dict]:
    """Drops each database setting of the open step, so the engine's default applies, and the
    extension loads, while `fails(trace)` still holds. Returns the trace and the settings the
    failure needs."""
    head = trace[0]
    for key in list(head["config"]):
        config = {k: v for k, v in head["config"].items() if k != key}
        if fails([{**head, "config": config}] + trace[1:]):
            print(f"  not needed: {key}={head['config'][key]}")
            head = {**head, "config": config}
    if head.get("loads") and fails([{**head, "loads": []}] + trace[1:]):
        print("  not needed: the extension loads")
        head = {**head, "loads": []}
    return [head] + trace[1:], head["config"]


def emit(trace: list, failure: tuple, needed: dict | None, out: str) -> str:
    """Writes trace.min.cypher, trace.min.py and finding.json into `out`, runs the script once
    and returns a summary."""
    kind, _, step, message = failure
    config = trace[0]["config"]
    listing = triage.cypher(trace, step, f"{kind}: {message.splitlines()[0][:120]}", config)
    script = triage.script(trace, step, kind, message, config, "trace.min.py")
    paths = {"cypher": "trace.min.cypher", "py": "trace.min.py", "json": "finding.json"}
    paths = {key: os.path.join(out, name) for key, name in paths.items()}
    with open(paths["cypher"], "w") as f:
        f.write(listing)
    with open(paths["py"], "w") as f:
        f.write(script)
    check = subprocess.run(
        [sys.executable, paths["py"]], capture_output=True, text=True, timeout=1800
    )
    record = triage.finding(trace, step, kind, message, config, needed, [" ".join(sys.argv)])
    record["script_reproduces"] = check.returncode == 1
    record["script_output"] = check.stdout[-2000:]
    with open(paths["json"], "w") as f:
        json.dump(record, f, indent=1, default=str)
    verdicts = {1: "fails as recorded", 0: "does NOT reproduce it"}
    verdict = verdicts.get(check.returncode, "fails otherwise, see finding.json")
    return "\n".join(
        [
            f"Wrote {paths['cypher']}, {paths['py']} and {paths['json']}.",
            f"trace.min.py {verdict}; signature {record['signature']['slug']}",
            f"top frames: {record['signature']['frames'] or record['signature']['error']}",
        ]
    )


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
    parser.add_argument(
        "--emit", action="store_true", help="write trace.min.cypher, trace.min.py, finding.json"
    )
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
    target, needed = failure[:2], None
    out = os.path.dirname(os.path.abspath(args.trace))
    if args.minimize:
        fails = lambda t: bool(reproduces(t, target, args.runs, False))  # noqa: E731
        trace = minimize(trace, fails)
        print("Simplifying steps:")
        trace = simplify(trace, fails)
        print("Reducing the database settings:")
        trace, needed = reduce_config(trace, fails)
        with open(os.path.join(out, "trace.min.json"), "w") as f:
            json.dump(trace, f, indent=1, default=str)
        print(f"\nMinimized to {len(trace)} steps, written to {out}/trace.min.json:")
        for op in trace:
            print("  " + describe(op))
    if args.emit:
        # Once more under gdb for the native stack of a crash or hang.
        debug = kind in ("died", "hung")
        failure = reproduces(trace, target, args.runs, debug) or reproduces(
            trace, target, args.runs, False
        )
        if not failure:
            print("The trace no longer fails; nothing emitted.")
            return
        print(emit(trace, failure, needed, out))


if __name__ == "__main__":
    main()
