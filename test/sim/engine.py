"""The database under test, open in a worker process.

A crash is a SIGKILL of the worker followed by a new worker that opens the database again and
replays the WAL, and an engine segfault ends the worker instead of the test run. Statements and
their results travel over a pipe; errors come back as EngineError, and a worker that does not
answer within TIMEOUT_SECONDS is killed and reported as hung, with the native stack of every
thread. With debug=True the worker runs under gdb, so a crash reports its native stack too.

A worker can also run under strace, which kills it at the n-th call of one syscall: a crash
point that does not depend on timing.
"""

from __future__ import annotations

import ctypes
import multiprocessing
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import threading

TIMEOUT_SECONDS = int(os.environ.get("SIM_TIMEOUT_SECONDS", 300))
PR_SET_PTRACER, PR_SET_PTRACER_ANY = 0x59616D61, ctypes.c_ulong(-1)
GDB_SIGNALS = "SIGPIPE SIGUSR1 SIGUSR2 SIG32 SIG33 SIG34 SIG35"
STRACE_WRAPPER = """#!/bin/sh
exec strace -f -qq -o /dev/null -e trace="$SIM_STRACE_SYSCALL" \\
    -e inject="$SIM_STRACE_SYSCALL:signal=KILL:when=$SIM_STRACE_WHEN" "$SIM_PYTHON" "$@"
"""


class EngineError(Exception):
    """A statement failed in the engine."""


def read_state(conn, queries: list[str]) -> tuple:
    """The sorted rows of each query, read in one read-only transaction."""
    conn.execute("BEGIN TRANSACTION READ ONLY")
    state = tuple(tuple(sorted(map(tuple, conn.execute(q).get_all()))) for q in queries)
    conn.execute("COMMIT")
    return state


def race(db, connect, writer, writes: list, queries: list[str], num_readers: int) -> tuple:
    """Runs `writes` on `writer` while readers repeatedly read `queries`; returns every state
    each reader saw and the errors of the writer and the readers."""
    done = threading.Event()
    observations = [[] for _ in range(num_readers)]
    errors = []

    def reader(out: list) -> None:
        conn = connect(db)
        try:
            while not done.is_set():
                out.append(read_state(conn, queries))
            out.append(read_state(conn, queries))
        except Exception as e:
            errors.append(f"reader: {e}")
        finally:
            conn.close()

    threads = [threading.Thread(target=reader, args=(out,)) for out in observations]
    for thread in threads:
        thread.start()
    try:
        for query, params in writes:
            writer.execute(query, params)
    except Exception as e:
        errors.append(f"writer: {query}: {e}")
    finally:
        done.set()
        for thread in threads:
            thread.join()
    return observations, errors


def serve(conns: list, db, connect, named: dict, message: tuple):
    """Handles one request: ("race", writes, queries, readers); ("prepared", connection,
    query, params), which prepares the query and executes it twice; ("prepare", name,
    connection, query, params), which keeps the statement for ("run", name, params); or
    (connection, query, params)."""
    if message[0] == "race":
        _, writes, queries, num_readers = message
        return race(db, connect, conns[0], writes, queries, num_readers)
    if message[0] == "prepared":
        _, conn, query, params = message
        statement = conns[conn].prepare(query, params)
        if not statement.is_success():
            raise RuntimeError(statement.get_error_message())
        return [conns[conn].execute(statement, params).get_all() for _ in range(2)]
    if message[0] == "prepare":
        _, name, conn, query, params = message
        statement = conns[conn].prepare(query, params)
        if not statement.is_success():
            raise RuntimeError(statement.get_error_message())
        named[name] = (conns[conn], statement)
        return None
    if message[0] == "run":
        _, name, params = message
        conn, statement = named[name]
        return conn.execute(statement, params).get_all()
    conn, query, params = message
    return conns[conn].execute(query, params).get_all()


def worker_main(pipe, path: str, config: dict, loads: list[str]) -> None:
    """Opens the database with `config`, runs `loads` and serves requests until None."""
    # Lets the parent's gdb attach (ptrace_scope 1 only allows ancestors otherwise).
    ctypes.CDLL(None).prctl(PR_SET_PTRACER, PR_SET_PTRACER_ANY, 0, 0, 0)
    import ladybug as lb

    try:
        db = lb.Database(path, **config)
        conns = [lb.Connection(db), lb.Connection(db, num_threads=1)]
        for query in loads:
            conns[0].execute(query)
    except Exception as e:
        pipe.send(("error", f"open failed: {e}"))
        return
    pipe.send(("ok", os.getpid()))
    named = {}
    while (message := pipe.recv()) is not None:
        try:
            pipe.send(("ok", serve(conns, db, lb.Connection, named, message)))
        except Exception as e:
            pipe.send(("error", str(e)))
    named.clear()
    for conn in conns:
        conn.close()
    db.close()
    pipe.send(("ok", None))


def native_stacks(gdb_output: str, limit: int = 60) -> str:
    """The threads and engine frames of a gdb `thread apply all bt`, without addresses and
    argument lists."""
    lines = []
    for line in gdb_output.splitlines():
        if line.startswith("Thread "):
            lines.append(line.split(" (")[0])
        elif frame := re.match(r"(#\d+)\s+(?:0x[0-9a-f]+ in )?(lbug::[^(]+)", line):
            lines.append(f"{frame[1]} {frame[2].strip()}")
    return "\n".join(lines[:limit]) or gdb_output[-2000:]


def gdb_stacks(pid: int) -> str:
    """Attaches gdb to a live process and returns the native stack of every thread."""
    try:
        result = subprocess.run(
            ["gdb", "-q", "-batch", "-p", str(pid), "-ex", "thread apply all bt 25"],
            capture_output=True,
            text=True,
            timeout=120,
        )
    except (OSError, subprocess.TimeoutExpired) as e:
        return f"(no stacks: {e})"
    return native_stacks(result.stdout)


SCRATCH_PREFIX = "lbug-sim-run-"
KEEP_MARKER = "keep"


def pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        pass
    return True


def scratch_root(tmp: str | None = None) -> str:
    """This process's directory for databases. Roots left by processes that died without
    cleaning up, such as a run killed mid-workload, are removed first unless they were kept."""
    tmp = tmp or tempfile.gettempdir()
    for name in os.listdir(tmp):
        pid = name.removeprefix(SCRATCH_PREFIX)
        path = os.path.join(tmp, name)
        if (
            name.startswith(SCRATCH_PREFIX)
            and pid.isdigit()
            and not pid_alive(int(pid))
            and not os.path.exists(os.path.join(path, KEEP_MARKER))
        ):
            shutil.rmtree(path, ignore_errors=True)
    root = os.path.join(tmp, f"{SCRATCH_PREFIX}{os.getpid()}")
    os.makedirs(root, exist_ok=True)
    return root


def strace_executable() -> str:
    path = os.path.join(tempfile.gettempdir(), f"lbug-sim-strace-{os.getuid()}.sh")
    if not os.path.exists(path):
        with open(path, "w") as f:
            f.write(STRACE_WRAPPER)
        os.chmod(path, 0o755)
    return path


class Engine:
    """One open database in a worker process. Connection 0 uses the configured thread count,
    connection 1 a single thread. `inject` = (syscall, n) runs the worker under strace, which
    kills it at the n-th call of the syscall in any thread; `debug` runs it under gdb."""

    def __init__(
        self,
        path: str,
        config: dict,
        loads: list[str],
        inject: tuple[str, int] | None = None,
        debug: bool = False,
    ) -> None:
        context = multiprocessing.get_context("spawn")
        self.pipe, child = context.Pipe()
        self.process = context.Process(
            target=worker_main, args=(child, path, config, loads), daemon=True
        )
        self.pid, self.gdb, self.request = None, None, "open"
        if inject:
            env = dict(SIM_STRACE_SYSCALL=inject[0], SIM_STRACE_WHEN=str(inject[1]))
            os.environ.update(env, SIM_PYTHON=sys.executable)
            context.set_executable(strace_executable())
            try:
                self.process.start()
            finally:
                context.set_executable(sys.executable)
                for key in (*env, "SIM_PYTHON"):
                    os.environ.pop(key)
        else:
            self.process.start()
        # Without closing its copy of the worker's end, a dead worker never reads as EOF.
        child.close()
        try:
            self.pid = self.reply()
        except EngineError:
            self.process.join()
            self.pipe.close()
            raise
        if debug:
            self.gdb = subprocess.Popen(
                ["gdb", "-q", "-batch", "-p", str(self.pid)]
                + ["-ex", f"handle {GDB_SIGNALS} nostop noprint pass", "-ex", "continue"]
                + ["-ex", "thread apply all bt 25"],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )

    def send(self, message, request: str) -> None:
        self.request = request
        self.pipe.send(message)

    def execute(self, query: str, params: dict, conn: int = 0) -> list:
        self.send((conn, query, params), query)
        return self.reply()

    def execute_prepared(self, query: str, params: dict, conn: int = 0) -> list[list]:
        """The rows of two executions of one freshly prepared statement: the first plans the
        query, the second reuses the cached plan."""
        self.send(("prepared", conn, query, params), query)
        return self.reply()

    def prepare(self, name: str, query: str, params: dict, conn: int = 0) -> None:
        """Prepares a statement and keeps it in the worker under `name`."""
        self.send(("prepare", name, conn, query, params), f"prepare {query}")
        self.reply()

    def run(self, name: str, params: dict) -> list:
        """Executes the statement kept under `name`."""
        self.send(("run", name, params), f"execute prepared {name}")
        return self.reply()

    def start(self, query: str) -> None:
        """Sends a statement without waiting for it, to kill the worker while it runs."""
        self.send((0, query, {}), query)

    def race(self, writes: list, queries: list[str], num_readers: int) -> tuple[list, list]:
        self.send(("race", writes, queries, num_readers), f"race reading {queries}")
        return self.reply()

    def reply(self):
        if not self.pipe.poll(TIMEOUT_SECONDS):
            stacks = self.stacks_of_hang()
            self.kill()
            raise AssertionError(f"engine hung for {TIMEOUT_SECONDS} s on {self.request}\n{stacks}")
        try:
            status, value = self.pipe.recv()
        except EOFError:
            self.process.join()
            raise AssertionError(
                f"engine process died with exit code {self.process.exitcode} on {self.request}"
                + self.stacks_of_crash()
            ) from None
        if status == "error":
            raise EngineError(value)
        return value

    def stacks_of_hang(self) -> str:
        if self.pid is None:
            return ""
        if self.gdb:
            os.kill(self.pid, signal.SIGINT)  # gdb stops the worker and prints its stacks
            return self.stacks_of_crash()
        return gdb_stacks(self.pid)

    def stacks_of_crash(self) -> str:
        if not self.gdb:
            return ""
        try:
            output, _ = self.gdb.communicate(timeout=120)
        except subprocess.TimeoutExpired:
            self.gdb.kill()
            output, _ = self.gdb.communicate()
        self.gdb = None
        return "\n" + native_stacks(output)

    def close(self) -> None:
        self.send(None, "close")
        self.reply()
        self.process.join()
        self.pipe.close()
        self.stop_gdb()

    def kill(self) -> None:
        if self.pid is not None:
            try:
                os.kill(self.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        self.process.kill()
        self.process.join()
        self.pipe.close()
        self.stop_gdb()

    def stop_gdb(self) -> None:
        if self.gdb:
            self.gdb.kill()
            self.gdb.wait()
            self.gdb = None
