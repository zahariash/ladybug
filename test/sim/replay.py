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

--minimize also shrinks the statements: it drops the setup statements, table columns, column
defaults and list-parameter values the failure does not need, and halves range() bounds. With
--shrink it then hands the steps to shrinkray (github.com/DRMacIver/shrinkray) for up to
--shrink-minutes, which cuts anywhere in the statements, parameters, settings and setup as long as
the steps fail with the same signature (and pass on the --fixed build), and emits the result
into shrink/.

    PYTHONPATH=tools/python_api/build uv run --no-project --with hypothesis \\
        python test/sim/replay.py <trace.json> [--runs 5] [--minimize] [--gdb] [--shrink]
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import triage  # noqa: E402
from engine import Engine, EngineError, scratch_root  # noqa: E402
from reduction import ddmin, reduce_statements  # noqa: E402
from shrinking import ALSO_INTERESTING, compact, expand, run_shrinkray  # noqa: E402


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
        self.reopen_loads = trace[0].get("reopen_loads", self.loads)
        self.engine = None

    def open(self, inject=None, loads=None) -> None:
        self.engine = None
        config = {**self.config, "max_num_threads": 1} if inject else self.config
        loads = self.reopen_loads if loads is None else loads
        self.engine = Engine(self.path, config, loads, inject, self.debug)

    def step(self, op: dict) -> None:
        kind = op["op"]
        if kind == "open":
            self.open(loads=self.loads)
        elif kind == "execute":
            self.engine.execute(op["query"], op["params"], op["conn"])
        elif kind == "prepared":
            self.engine.execute_prepared(op["query"], op["params"], op["conn"])
        elif kind == "columns":
            self.engine.columns(op["query"], op["params"], op["conn"])
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
    """ddmin over the steps after the first open: the smallest trace for which `fails(trace)`
    still holds."""

    def steps_fail(steps: list) -> bool:
        if fails(trace[:1] + steps):
            print(f"  {len(steps)} steps still fail", flush=True)
            return True
        return False

    return trace[:1] + ddmin(trace[1:], steps_fail)


def after_replay(trace: list, then):
    """Replays the trace, skipping statement errors, and returns `then(engine)`; None if the
    engine crashes or hangs."""
    replayer = Replayer(trace, False)
    try:
        for op in trace:
            try:
                replayer.step(op)
            except (EngineError, RaceError):
                if replayer.engine is None:  # the database did not open
                    return None
        return then(replayer.engine)
    except AssertionError:
        return None
    finally:
        if replayer.engine:
            replayer.engine.kill()
        shutil.rmtree(replayer.dir, ignore_errors=True)


def probe_fails(trace: list, probe: str, wrong: list[tuple], conn: int) -> bool:
    """Replays the trace, ignoring statement errors, then runs `probe` on `conn`: true while it
    returns any of the `wrong` rows. A crash or hang counts as not failing this way."""

    def returns_wrong(engine: Engine) -> bool:
        try:
            rows = engine.execute(probe, {}, conn)
        except EngineError:
            return False
        return any(tuple(row) in wrong for row in rows)

    return bool(after_replay(trace, returns_wrong))


def probe_ways(engine: Engine, probe: str, conn: int) -> dict[str, Counter]:
    """The rows of the probe each way Session.check runs a query, and with the plan optimizer
    off; an error is a result of its own."""

    def outcomes(call) -> list[Counter]:
        try:
            return [Counter(json.dumps(row, default=str) for row in rows) for rows in call()]
        except EngineError as e:
            return [Counter([f"error: {triage.shape(str(e).splitlines()[0])}"])] * 2

    ways = {"driver": outcomes(lambda: [engine.execute(probe, {}, conn)])[0]}
    ways["fresh plan"], ways["cached plan"] = outcomes(
        lambda: engine.execute_prepared(probe, {}, conn)
    )
    if conn == 0:
        ways["single thread"] = outcomes(lambda: [engine.execute(probe, {}, 1)])[0]
    engine.execute("CALL enable_plan_optimizer=false", {}, conn)
    ways["no optimizer"] = outcomes(lambda: [engine.execute(probe, {}, conn)])[0]
    engine.execute("CALL enable_plan_optimizer=true", {}, conn)
    return ways


def agreement(ways: dict[str, Counter]) -> list[list[str]]:
    """The ways grouped by the rows they returned."""
    groups = defaultdict(list)
    for name, rows in ways.items():
        groups[frozenset(rows.items())].append(name)
    return sorted(sorted(names) for names in groups.values())


def probe_agreement(trace: list, probe: str, conn: int) -> list[list[str]] | None:
    """Replays the trace and groups the ways of running the probe by the rows they return."""
    return after_replay(trace, lambda engine: agreement(probe_ways(engine, probe, conn)))


def splits_ways(trace: list, probe: str, conn: int, split: list, runs: int) -> bool:
    """Whether the ways of running the probe split as `split` in any of `runs` replays."""
    return any(probe_agreement(trace, probe, conn) == split for _ in range(runs))


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
    if head.get("loads") and fails([{**head, "loads": [], "reopen_loads": []}] + trace[1:]):
        print("  not needed: the extension loads")
        head = {**head, "loads": [], "reopen_loads": []}
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


def minimize_wrong_result(trace: list, args) -> None:
    """--probe: wrong results have no engine failure to replay. With --rows the oracle is the probe
    returning the recorded wrong rows, and --pin keeps the steps that define the right answer;
    with --differential it is the ways of running the probe (see probe_ways) splitting into the
    same groups, which needs no right answer."""
    if args.differential:
        splits = [probe_agreement(trace, args.probe, args.conn) for _ in range(args.runs)]
        split = next((split for split in splits if split and len(split) > 1), None)
        if not split:
            print(f"The ways of running the probe agree on replay: {splits[0]}")
            return
        print(f"The ways of running the probe disagree: {split}")

        def fails(candidate: list) -> bool:
            return splits_ways(candidate, args.probe, args.conn, split, args.runs)

        failure = f"the ways disagree: {split}"
    else:
        wrong = [tuple(row) for row in json.loads(args.rows)]
        if not probe_fails(trace, args.probe, wrong, args.conn):
            print("The probe does not return those rows on replay.")
            return
        print("The probe returns the wrong rows on replay.")
        pins = [op for op in trace if any(re.search(p, op.get("query") or "") for p in args.pin)]

        def fails(candidate: list) -> bool:
            kept = all(op in candidate for op in pins)
            return kept and probe_fails(candidate, args.probe, wrong, args.conn)

        failure = f"returns {wrong}"
    out = os.path.dirname(os.path.abspath(args.trace))
    needed = trace[0]["config"]
    if args.minimize:
        trace = simplify(minimize(trace, fails), fails)
        trace, needed = reduce_config(trace, fails)
        trace = reduce_statements(trace, fails)
        with open(os.path.join(out, "trace.probe.min.json"), "w") as f:
            json.dump(trace, f, indent=1, default=str)
        listing = probe_listing(trace, args.probe, args.conn, failure, needed)
        with open(os.path.join(out, "trace.probe.min.cypher"), "w") as f:
            f.write(listing)
        print(f"\nMinimized to {len(trace)} steps (trace.probe.min.json):\n{listing}")
    if not args.shrink:
        return
    if not args.differential:
        print("--shrink needs --differential: a reduced case may return the right rows.")
        return
    target = dict(probe=args.probe, conn=args.conn, agreement=split)
    work = os.path.join(out, "shrink")
    data = run_shrinkray({**compact(trace), "probe": args.probe}, target, work, args)
    shrunk, probe = (expand(data), data.get("probe")) if data else (None, None)
    if not (
        shrunk
        and isinstance(probe, str)
        and splits_ways(shrunk, probe, args.conn, split, args.runs)
    ):
        print("The shrunk steps do not split the ways the same way on a replay; see shrinkray.log.")
        return
    shrunk = mark_errors(shrunk, len(shrunk))
    with open(os.path.join(out, "trace.probe.shrunk.json"), "w") as f:
        json.dump(dict(trace=shrunk, probe=probe, conn=args.conn), f, indent=1, default=str)
    listing = probe_listing(shrunk, probe, args.conn, failure, shrunk[0]["config"])
    with open(os.path.join(work, "trace.probe.cypher"), "w") as f:
        f.write(listing)
    print(f"\nShrunk to {len(shrunk)} steps (trace.probe.shrunk.json):\n{listing}")


def probe_listing(trace: list, probe: str, conn: int, failure: str, config: dict) -> str:
    step = dict(op="execute", conn=conn, query=probe, params={})
    return triage.cypher(trace + [step], len(trace), failure, config)


def same_failure(target: dict, kind: str, op: dict, message: str) -> bool:
    """Whether a failure has `target`'s signature: its innermost engine frames when it has them,
    else an error's abstracted text, else the operation that failed."""
    found = triage.signature(kind, op, message)
    if found["kind"] != target["kind"]:
        return False
    if target["frames"]:
        return found["frames"] == target["frames"]
    if target["error"]:
        return found["error"] == target["error"]
    return found["operation"] == target["operation"]


def check(trace: list, target: dict) -> tuple[int, int | None, str]:
    """Replays a reduced trace, skipping statement errors, which reductions often cause: 0 when it
    fails like `target`, ALSO_INTERESTING when it crashes or hangs otherwise, 1 when it does not;
    with the failing step and message."""
    replayer = Replayer(trace, False)
    try:
        for i, op in enumerate(trace):
            try:
                replayer.step(op)
            except EngineError as e:
                if same_failure(target, "error", op, str(e)):
                    return 0, i, str(e)
                if replayer.engine is None:  # the database did not open
                    return 1, None, ""
            except RaceError as e:
                return (0 if same_failure(target, "race", op, str(e)) else 1), i, str(e)
            except AssertionError as e:
                kind = "hung" if str(e).startswith("engine hung") else "died"
                status = 0 if same_failure(target, kind, op, str(e)) else ALSO_INTERESTING
                return status, i, str(e)
        return 1, None, ""
    finally:
        if replayer.engine:
            replayer.engine.kill()
        shutil.rmtree(replayer.dir, ignore_errors=True)


def check_probe(trace: list, probe, target: dict) -> int:
    """0 when the ways of running the probe split as in `target`, 1 otherwise."""
    if not isinstance(probe, str):
        return 1
    return 0 if probe_agreement(trace, probe, target["conn"]) == target["agreement"] else 1


def mark_errors(trace: list, failing: int) -> list:
    """Marks the steps before `failing` whose statements now fail as expected, so that the
    emitted script runs past them."""
    replayer = Replayer(trace[:failing], False)
    marked = list(trace)
    try:
        for i, op in enumerate(trace[:failing]):
            try:
                replayer.step(op)
            except (EngineError, RaceError) as e:
                marked[i] = {**op, "outcome": f"error: {str(e)[:200]}", "expected": True}
                if replayer.engine is None:
                    break
    finally:
        if replayer.engine:
            replayer.engine.kill()
        shutil.rmtree(replayer.dir, ignore_errors=True)
    return marked


def shrink(trace: list, failure: tuple, out: str, args) -> tuple[list, tuple] | None:
    """Reduces a failing trace with shrinkray in out/shrink/ and returns it with its failure."""
    kind, _, step, message = failure
    target = triage.signature(kind, trace[step], message)
    data = run_shrinkray(compact(trace), target, os.path.join(out, "shrink"), args)
    shrunk = expand(data) if data else None
    status, step, message = 1, None, ""
    for _ in range(args.runs if shrunk else 0):
        status, step, message = check(shrunk, target)
        if status == 0:
            break
    if status != 0:
        print("The shrunk steps do not fail the same way on a replay; see shrink/shrinkray.log.")
        return None
    shrunk = mark_errors(shrunk, step)
    if kind == "error":
        shrunk[step] = {**shrunk[step], "outcome": f"error: {message[:200]}"}
    with open(os.path.join(out, "trace.shrunk.json"), "w") as f:
        json.dump(shrunk, f, indent=1, default=str)
    return shrunk, (kind, "", step, message)


def describe(op: dict) -> str:
    detail = op.get("query") or op.get("name") or ""
    if op["op"] == "crash_at":
        detail = f"{op['syscall']} #{op['n']} in {detail}"
    elif op["op"] == "race":
        detail = f"{len(op['writes'])} writes, {op['readers']} readers"
    return f"{op['op']:12} {detail[:110]}"


def relocate_files(trace: list, trace_dir: str) -> list:
    """Points the steps' file paths at the copies kept next to the trace: the workload wrote its
    files to <traces>/current-<pid>, which is gone once the run ends (see keep_failure)."""
    text = re.sub(r'/[^"\s]*/current-\d+(?=/)', lambda _: trace_dir, json.dumps(trace))
    return json.loads(text)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("trace")
    parser.add_argument("--runs", type=int, default=1, help="replays per check")
    parser.add_argument("--minimize", action="store_true")
    parser.add_argument("--gdb", action="store_true", help="report native stacks of crashes")
    parser.add_argument(
        "--emit", action="store_true", help="write trace.min.cypher, trace.min.py, finding.json"
    )
    parser.add_argument("--probe", help="for a wrong result: the query that returns it")
    parser.add_argument("--rows", help="the wrong rows the probe returns, as a JSON list of lists")
    parser.add_argument("--conn", type=int, default=0, help="the connection to probe on")
    parser.add_argument(
        "--differential",
        action="store_true",
        help="for a wrong result: the ways of running the "
        "probe must disagree as on the original replay, instead of returning --rows",
    )
    parser.add_argument(
        "--pin", action="append", default=[], help="regex of steps --minimize must keep"
    )
    parser.add_argument(
        "--shrink", action="store_true", help="reduce further with shrinkray, into shrink/"
    )
    parser.add_argument("--fixed", help="--shrink: a build (PYTHONPATH) the steps must pass on")
    parser.add_argument("--parallelism", type=int, default=4, help="--shrink: tests at once")
    parser.add_argument(
        "--shrink-minutes", type=float, default=10, help="--shrink: stop shrinkray after this"
    )
    parser.add_argument(
        "--shrink-timeout", type=int, default=600, help="--shrink: seconds per test"
    )
    parser.add_argument("--shrinkray", help="--shrink: the shrinkray executable")
    parser.add_argument("--check", metavar="TARGET", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.check:  # shrinkray's interestingness test (see run_shrinkray): a steps file
        with open(args.check) as f:
            target = json.load(f)
        try:
            with open(args.trace) as f:
                data = json.load(f)
        except (OSError, ValueError):
            sys.exit(1)
        trace = expand(data)
        if trace is None:
            sys.exit(1)
        if "agreement" in target:
            sys.exit(check_probe(trace, data.get("probe"), target))
        sys.exit(check(trace, target)[0])
    with open(args.trace) as f:
        trace = triage.after_last_fresh([op for op in json.load(f) if op["op"] != "failure"])
    trace = relocate_files(trace, os.path.dirname(os.path.abspath(args.trace)))
    if args.probe:
        minimize_wrong_result(trace, args)
        return
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
        print("Reducing the statements:")
        trace = reduce_statements(trace, fails)
        with open(os.path.join(out, "trace.min.json"), "w") as f:
            json.dump(trace, f, indent=1, default=str)
        print(f"\nMinimized to {len(trace)} steps, written to {out}/trace.min.json:")
        for op in trace:
            print("  " + describe(op))
    if args.shrink:
        failure = reproduces(trace, target, args.runs, False)
        if not failure:
            print("The trace no longer fails; nothing to shrink.")
            return
        shrunk = shrink(trace, failure, out, args)
        if shrunk:
            trace, failure = shrunk
            print(f"\nShrunk to {len(trace)} steps, written to {out}/trace.shrunk.json:")
            for op in trace:
                print("  " + describe(op))
            print(emit(trace, failure, trace[0]["config"], os.path.join(out, "shrink")))
        return
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
