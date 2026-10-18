"""Turns a failing trace into a bug record for triage (see docs/testing.md, "Triaging a failure").

A finding has a signature for deduplication (the kind of failure, the operation that failed, the
statement with its literals abstracted and the innermost engine frames), the statements in order
as Cypher (trace.min.cypher), a standalone script that exits 1 while the failure recurs
(trace.min.py), and finding.json with all of it. Session writes a failure record into every
failing trace; replay.py --emit writes the rest after minimizing.
"""

from __future__ import annotations

import ast
import hashlib
import json
import math
import re

FRAME = re.compile(r"^#(\d+) (lbug::\S+)")
PAD = "UNWIND range(1, n) AS i RETURN 'pad-' + CAST(i AS STRING)"
# Steps that end the worker process; what follows runs in a new one.
LIFETIME_ENDS = ("crash", "reopen", "crash_during", "crash_at")


def top_frames(message: str, count: int = 3) -> list[str]:
    """The innermost engine frames of the first stack in `message` (see engine.native_stacks)."""
    frames = []
    for line in message.splitlines():
        frame = FRAME.match(line.strip())
        if not frame:
            continue
        if frame[1] == "0" and frames:
            break  # the next thread's stack
        frames.append(frame[2])
        if len(frames) == count:
            break
    return frames


def shape(query: str) -> str:
    """The statement with literals and numbered names abstracted: 'ab' -> ?, 42 -> ?, T3 -> T?."""
    query = re.sub(r"'(?:[^'\\]|\\.)*'", "?", query)
    query = re.sub(r"\b([A-Za-z_]+)\d+\b", r"\1?", query)
    query = re.sub(r"(?<![\w$?])-?\d+(?:\.\d+)?\b", "?", query)
    return re.sub(r"\s+", " ", query).strip()


def operation(op: dict) -> str:
    """The class of a step: its first keyword, and for ALTER the change it makes."""
    query = op.get("query")
    if op["op"] not in ("execute", "columns", "prepared", "prepare") or not query:
        return op["op"]
    words = query.split()
    if words[0].upper() == "ALTER" and len(words) > 3:
        return f"ALTER {words[3].upper()}"
    return words[0].upper()


def signature(kind: str, op: dict, message: str) -> dict:
    """What two findings with the same cause share. Stacks identify crashes and hangs; errors
    are identified by their text with numbers and quoted names removed."""
    frames = top_frames(message)
    fields = dict(
        kind=kind,
        operation=operation(op),
        statement=shape(op.get("query") or ""),
        frames=frames,
        error="" if frames else shape(message.splitlines()[0] if message else ""),
    )
    key = hashlib.sha1(json.dumps(fields, sort_keys=True).encode()).hexdigest()[:10]
    slug = re.sub(r"[^a-z0-9]+", "-", f"{kind}-{fields['operation']}".lower()).strip("-")
    return {**fields, "key": key, "slug": f"{slug}-{key}"}


def probe_of(op: dict, message: str) -> tuple[str, list] | None:
    """For a wrong result: the query that shows it and the wrong rows it returned, from the
    check's assertion (Session.check's tuple, or a data check's missing/unexpected diff)."""
    try:
        if message.startswith("("):
            query, params, _, actual, expected = ast.literal_eval(message)
            return inline(query, params), [list(r) for r in actual if r not in expected]
        unexpected = re.search(r"unexpected=(\[.*\])\s*$", message, re.S)
        if unexpected and op.get("query"):
            rows = [list(row) for row, _ in ast.literal_eval(unexpected[1])]
            return inline(op["query"], op.get("params") or {}), rows
    except (ValueError, SyntaxError):
        pass
    return None


def literal(value) -> str:
    """A Cypher literal for a parameter value."""
    if value is None:
        return "NULL"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float) and not math.isfinite(value):
        return f"CAST('{value}' AS DOUBLE)"
    if isinstance(value, str):
        return "'" + value.replace("\\", "\\\\").replace("'", "\\'") + "'"
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(literal(v) for v in value) + "]"
    if isinstance(value, dict):
        return "{" + ", ".join(f"{k}: {literal(v)}" for k, v in value.items()) + "}"
    return repr(value)


def inline(query: str, params: dict) -> str:
    """The statement with its parameters written in, longest names first."""
    for name in sorted(params, key=len, reverse=True):
        query = re.sub(rf"\${name}\b", lambda _, v=params[name]: literal(v), query)
    return query


def terminated(statement: str) -> str:
    """The statement with its semicolon before a trailing comment."""
    query, _, comment = statement.partition("  // ")
    return f"{query};  // {comment}" if comment else f"{query};"


def cypher(trace: list, failing: int, failure: str, config: dict) -> str:
    """The steps up to the failing one as Cypher, with comments for everything that is not a
    statement: connections, crashes, reopens and the step that fails."""
    settings = ", ".join(f"{k}={v}" for k, v in config.items()) or "defaults"
    lines = [f"// database: {settings}"]
    for query in trace[0].get("loads", []):
        lines.append(f"{query};")
    for i, op in enumerate(trace[1 : failing + 1], start=1):
        kind, conn = op["op"], op.get("conn", 0)
        where = f"// [connection {conn}]\n" if conn else ""
        # Parameters stay parameters: a driver reuses a parameterized statement's prepared plan,
        # which some failures need. Their values follow in a comment.
        params = op.get("params") or {}
        statement = op.get("query") or ""
        if params:
            values = ", ".join(f"${k} = {literal(v)}" for k, v in params.items())
            statement = f"{statement}  // {values}"
        if kind in ("execute", "columns"):
            text = f"{where}{terminated(statement)}"
        elif kind == "prepared":
            text = f"{where}// prepared, then executed twice:\n{terminated(statement)}"
        elif kind == "prepare":
            text = f"// prepare {op['name']}: {op['query']}"
        elif kind == "run":
            text = f"// execute {op['name']} with {op['params']}"
        elif kind == "crash":
            text = "// kill the process (SIGKILL), then open the database again"
        elif kind == "reopen":
            text = "// close the database and open it again"
        elif kind == "crash_during":
            text = f"{terminated(statement)}  // killed {op['delay']:.2f} s after it started"
        elif kind == "crash_at":
            text = f"{terminated(statement)}  // killed at {op['syscall']} call {op['n']}"
        elif kind == "race":
            text = f"// race: {len(op['writes'])} writes against {op['readers']} readers"
        elif kind == "hold":
            text = f"{where}// keep results of {PAD} unconsumed until the buffer pool is full"
        elif kind == "release":
            text = "// free those results"
        else:
            continue
        if op.get("expected"):
            text += "  // fails, as expected"
        if i == failing:
            text += f"\n// ^ {failure}"
        lines.append(text)
    return "\n".join(lines) + "\n"


def stages(trace: list, failing: int) -> list[dict]:
    """The steps up to the failing one, split into worker lifetimes. Each stage ends in "crash",
    "reopen", "crash_during", "crash_at" or "last"; the step that ends a stage is its "end"."""
    result, steps = [], []
    for op in trace[1 : failing + 1]:
        if op["op"] in LIFETIME_ENDS:
            result.append(dict(steps=steps, end=op))
            steps = []
        elif op["op"] != "failure":
            steps.append(op)
    result.append(dict(steps=steps, end=dict(op="last")))
    return result


SCRIPT = '''r"""Reproduces a simulator finding: {title}

Exits 1 while the failure recurs, 0 once it does not, so it also tells whether a build fixes
it, and 2 when it cannot tell:  PYTHONPATH=<python build> python {name}

{cypher}"""

import os
import subprocess
import sys
import tempfile
import threading
import time

nan, inf = float("nan"), float("inf")  # how repr() writes them in the steps below
CONFIG = {config!r}
LOADS = {loads!r}
STAGES = {stages!r}
EXPECTED = {expected!r}
TIMEOUT = 300
PAD = "UNWIND range(1, $n) AS i RETURN 'pad-' + CAST(i AS STRING)"


def run_stage(path, index):
    import ladybug as lb

    stage = STAGES[index]
    config = dict(CONFIG, max_num_threads=1) if stage["end"]["op"] == "crash_at" else CONFIG
    try:
        db = lb.Database(path, **config)
    except RuntimeError as e:
        print(f"stage {{index}} open: {{str(e).splitlines()[0]}}", flush=True)
        os._exit(3)
    conns, named, held = {{}}, {{}}, []

    def conn(i):
        if i not in conns:
            conns[i] = lb.Connection(db, num_threads=1) if i == 1 else lb.Connection(db)
        return conns[i]

    for query in LOADS:
        conn(0).execute(query)
    for n, op in enumerate(stage["steps"]):
        try:
            if op["op"] in ("execute", "columns"):
                conn(op["conn"]).execute(op["query"], op["params"]).get_all()
            elif op["op"] == "prepared":
                statement = conn(op["conn"]).prepare(op["query"], op["params"])
                for _ in range(2):
                    conn(op["conn"]).execute(statement, op["params"]).get_all()
            elif op["op"] == "prepare":
                named[op["name"]] = conn(0).prepare(op["query"], op["params"])
            elif op["op"] == "run":
                conn(0).execute(named[op["name"]], op["params"]).get_all()
            elif op["op"] == "hold":
                for rows in (200000, 20000, 2000, 200):
                    while len(held) < 2000:
                        try:
                            held.append(conn(op["conn"]).execute(PAD, {{"n": rows}}))
                        except RuntimeError:
                            break
            elif op["op"] == "release":
                held.clear()
            elif op["op"] == "race":
                print("a race step: replay the trace with replay.py instead", flush=True)
                os._exit(4)
        except RuntimeError as e:
            if not op.get("expected"):
                print(f"stage {{index}} step {{n}}: {{str(e).splitlines()[0]}}", flush=True)
                os._exit(3)
    end = stage["end"]
    if end["op"] == "crash":
        os._exit(0)
    if end["op"] == "crash_during":
        threading.Thread(target=lambda: conn(0).execute(end["query"]), daemon=True).start()
        time.sleep(end["delay"])
        os._exit(0)
    if end["op"] == "crash_at":  # strace kills the process at the recorded call
        conn(0).execute(end["query"])
        os._exit(0)
    conns.clear()
    db.close()


def outcome(path, index):
    """How a stage ended: "ok", "died" (a signal), "hung", or the error it printed."""
    command = [sys.executable, __file__, path, str(index)]
    end = STAGES[index]["end"]
    if end["op"] == "crash_at":
        syscall = end["syscall"]
        command = ["strace", "-f", "-qq", "-o", "/dev/null", "-e", f"trace={{syscall}}",
                   "-e", f"inject={{syscall}}:signal=KILL:when={{end['n']}}", *command]
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=TIMEOUT)
    except subprocess.TimeoutExpired:
        return "hung"
    sys.stdout.write(result.stdout)
    if end["op"] in ("crash_during", "crash_at") and result.returncode in (0, -9, 137):
        return "ok"
    if result.returncode < 0:
        return "died"
    if result.returncode == 3:
        return "error: " + result.stdout.strip().splitlines()[-1]
    if result.returncode != 0:
        return f"exit {{result.returncode}}: {{result.stderr.strip()[-300:]}}"
    return "ok"


def main():
    if len(sys.argv) == 3:
        run_stage(sys.argv[1], int(sys.argv[2]))
        return
    import ladybug as lb

    print("ladybug", getattr(lb, "__version__", "?"), "from", os.path.dirname(lb.__file__))
    path = os.path.join(tempfile.mkdtemp(), "db")
    for index in range(len(STAGES)):
        result = outcome(path, index)
        if result != "ok":
            break
    print("outcome:", result)
    if result.startswith(EXPECTED["kind"]) and EXPECTED.get("error", "") in result:
        print("the failure recurs")
        sys.exit(1)
    if result == "ok":
        print(f"expected {{EXPECTED}}: not reproduced")
        sys.exit(0)
    print(f"expected {{EXPECTED}}, but it failed otherwise: cannot tell")
    sys.exit(2)


if __name__ == "__main__":
    main()
'''


def script(trace: list, failing: int, kind: str, message: str, config: dict, name: str) -> str:
    """trace.min.py: the steps up to the failing one, one process per worker lifetime."""
    title = f"{kind} at {operation(trace[failing])}"
    expected = dict(kind=kind)
    if kind == "error":
        expected["error"] = message.splitlines()[0][:80]
    listing = cypher(trace, failing, f"{kind}: {message.splitlines()[0][:100]}", config)
    return SCRIPT.format(
        title=title,
        name=name,
        cypher=listing.replace('"""', "'''"),
        config=config,
        loads=trace[0].get("loads", []),
        stages=stages(trace, failing),
        expected=expected,
    )


def finding(
    trace: list, failing: int, kind: str, message: str, config: dict, needed: dict, commands
) -> dict:
    """finding.json: what the triage agent starts from."""
    op = trace[failing]
    return dict(
        signature=signature(kind, op, message),
        kind=kind,
        failing_step=failing,
        failing=dict(op=op["op"], query=op.get("query"), params=op.get("params")),
        message=message[:4000],
        config=config,
        needed_config=needed,
        build=trace[0].get("build"),
        steps=len(trace) - 1,
        rules=sorted({op["rule"] for op in trace if op.get("rule")}),
        commands=list(commands),
    )
