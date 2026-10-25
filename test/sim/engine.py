"""The database under test, open in a worker process.

A crash is a SIGKILL of the worker followed by a new worker that opens the database again and
replays the WAL, and an engine segfault ends the worker instead of the test run. Statements and
their results travel over a pipe; errors come back as EngineError, and a worker that does not
answer within TIMEOUT_SECONDS is killed and reported as hung, with the native stack of every
thread. With debug=True the worker runs under gdb, so a crash reports its native stack too.

A worker can also run under strace, which kills it at the n-th call of one syscall: a crash
point that does not depend on timing. strace either starts the worker or attaches to it while it
runs, which keeps what the worker holds in memory (open snapshots, a pinned WAL) up to the kill.
A worker can also limit the size of every file it writes (RLIMIT_FSIZE), so that writes past it
fail as on a full disk.
"""

from __future__ import annotations

import ctypes
import functools
import hashlib
import multiprocessing
import os
import re
import resource
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time

TIMEOUT_SECONDS = int(os.environ.get("SIM_TIMEOUT_SECONDS", 300))
PR_SET_PTRACER, PR_SET_PTRACER_ANY = 0x59616D61, ctypes.c_ulong(-1)
GDB_SIGNALS = "SIGPIPE SIGUSR1 SIGUSR2 SIG32 SIG33 SIG34 SIG35"
STRACE_WRAPPER = """#!/bin/sh
exec strace -f -qq -o /dev/null -e trace="$SIM_STRACE_SYSCALL" \\
    -e inject="$SIM_STRACE_SYSCALL:signal=KILL:when=$SIM_STRACE_WHEN" "$SIM_PYTHON" "$@"
"""
# The worker starts under gdb, so that even a crash in its first statement leaves a stack: the
# crashing thread's first, then every thread's.
GDB_WRAPPER = """#!/bin/sh
exec gdb -q -batch -ex "handle $SIM_GDB_SIGNALS nostop noprint pass" -ex run -ex "bt 25" \\
    -ex "thread apply all bt 25" --args "$SIM_PYTHON" "$@" > "$SIM_GDB_OUT" 2>&1
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


# Results of decreasing size that ("hold", ...) keeps unconsumed until the buffer pool is full.
PAD_ROWS = (200000, 20000, 2000, 200)
PAD_QUERY = "UNWIND range(1, $n) AS i RETURN 'pad-' + CAST(i AS STRING)"


def hold(conn, held: list) -> int:
    """Keeps results on `conn` unconsumed until the buffer pool cannot take another one."""
    for rows in PAD_ROWS:
        while len(held) < 2000:
            try:
                held.append(conn.execute(PAD_QUERY, {"n": rows}))
            except RuntimeError:
                break
    return len(held)


def limit_files(size: int | None) -> None:
    """Fails writes that would make any file larger than `size` bytes; None lifts the limit."""
    hard = resource.getrlimit(resource.RLIMIT_FSIZE)[1]
    resource.setrlimit(resource.RLIMIT_FSIZE, (hard if size is None else size, hard))


def serve(conns: list, db, connect, named: dict, held: list, message: tuple):
    """Handles one request: ("race", writes, queries, readers); ("prepared", connection,
    query, params), which prepares the query and executes it twice; ("columns", connection,
    query, params), the result's column names and rows; ("prepare", name,
    connection, query, params), which keeps the statement for ("run", name, params);
    ("hold", connection) and ("release",), which fill the buffer pool with unconsumed
    results and free them; ("limit", size), see limit_files; or (connection, query, params)."""
    if message[0] == "limit":
        return limit_files(message[1])
    if message[0] == "hold":
        return hold(conns[message[1]], held)
    if message[0] == "release":
        held.clear()
        return None
    if message[0] == "race":
        _, writes, queries, num_readers = message
        return race(db, connect, conns[0], writes, queries, num_readers)
    if message[0] == "prepared":
        _, conn, query, params = message
        statement = conns[conn].prepare(query, params)
        if not statement.is_success():
            raise RuntimeError(statement.get_error_message())
        return [conns[conn].execute(statement, params).get_all() for _ in range(2)]
    if message[0] == "columns":
        _, conn, query, params = message
        result = conns[conn].execute(query, params)
        return [result.get_column_names(), result.get_all()]
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


def worker_main(
    pipe, path: str, config: dict, loads: list[str], tracer: tuple[str, str] | None = None
) -> None:
    """Opens the database with `config`, runs `loads` and serves requests until None.
    `tracer` = (library, directory) loads crash_trace.c's library, which writes the backtrace of
    a crash to the directory."""
    # Lets the parent's gdb and strace attach (ptrace_scope 1 only allows ancestors otherwise).
    ctypes.CDLL(None).prctl(PR_SET_PTRACER, PR_SET_PTRACER_ANY, 0, 0, 0)
    # A write past the file-size limit then fails with EFBIG instead of killing the worker.
    signal.signal(signal.SIGXFSZ, signal.SIG_IGN)
    if tracer:
        os.environ["SIM_CRASH_TRACE_DIR"] = tracer[1]
        ctypes.CDLL(tracer[0])
    import ladybug as lb

    try:
        db = lb.Database(path, **config)
        # 0 is the default, 1 single-threaded, 2-4 hold the sessions' transactions.
        conns = [lb.Connection(db), lb.Connection(db, num_threads=1)]
        conns += [lb.Connection(db) for _ in range(3)]
        for query in loads:
            conns[0].execute(query)
    except Exception as e:
        pipe.send(("error", f"open failed: {e}"))
        return
    pipe.send(("ok", dict(pid=os.getpid(), version=lb.__version__, module=lb.__file__)))
    named, held = {}, []
    while (message := pipe.recv()) is not None:
        try:
            pipe.send(("ok", serve(conns, db, lb.Connection, named, held, message)))
        except Exception as e:
            pipe.send(("error", str(e)))
    named.clear()
    held.clear()
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


def demangle(names: list[str]) -> list[str]:
    try:
        result = subprocess.run(
            ["c++filt"], input="\n".join(names), capture_output=True, text=True, timeout=30
        )
    except (OSError, subprocess.SubprocessError):
        return names
    demangled = result.stdout.splitlines()
    return demangled if len(demangled) == len(names) else names


def backtrace_frames(trace: str, limit: int = 40) -> str:
    """The engine frames of a crash_trace.c backtrace, innermost first and numbered from #0 like
    native_stacks' frames, under a line naming the signal."""
    lines = trace.splitlines()
    mangled = [m[1] for line in lines[1:] if (m := re.search(r"\((_Z[^+)]+)", line))]
    names = [name.split("(")[0] for name in demangle(mangled) if name.startswith("lbug::")]
    head = f"Thread that crashed: {lines[0]}" if lines else "Thread that crashed"
    return "\n".join([head, *(f"#{i} {name}" for i, name in enumerate(names[:limit]))])


CRASH_TRACE_SOURCE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "crash_trace.c")


@functools.cache
def crash_tracer() -> str | None:
    """crash_trace.c built as a shared library, once per version of the source; None without a C
    compiler. Unlike gdb, it leaves the worker's timing alone, so races still crash under it."""
    with open(CRASH_TRACE_SOURCE, "rb") as f:
        digest = hashlib.sha1(f.read()).hexdigest()[:12]
    path = os.path.join(tempfile.gettempdir(), f"lbug-sim-crash-trace-{os.getuid()}-{digest}.so")
    if not os.path.exists(path):
        partial = f"{path}.{os.getpid()}"
        try:
            subprocess.run(
                ["cc", "-shared", "-fPIC", "-O2", "-o", partial, CRASH_TRACE_SOURCE],
                check=True,
                capture_output=True,
                timeout=120,
            )
        except (OSError, subprocess.SubprocessError):
            return None
        os.replace(partial, path)
    return path


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


def wrapper_executable(name: str, text: str) -> str:
    path = os.path.join(tempfile.gettempdir(), f"lbug-sim-{name}-{os.getuid()}.sh")
    if not os.path.exists(path) or open(path).read() != text:
        with open(path, "w") as f:
            f.write(text)
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
        # gdb stops the worker on the signal itself, and strace's kills are SIGKILL.
        self.crash_dir, tracer = None, None
        if not (inject or debug) and (library := crash_tracer()):
            self.crash_dir = os.path.join(tempfile.gettempdir(), f"lbug-sim-crashes-{os.getpid()}")
            os.makedirs(self.crash_dir, exist_ok=True)
            tracer = (library, self.crash_dir)
        self.process = context.Process(
            target=worker_main, args=(child, path, config, loads, tracer), daemon=True
        )
        self.pid, self.request, self.build = None, "open", {}
        self.gdb_output, self.strace = None, None
        if inject:
            env = dict(SIM_STRACE_SYSCALL=inject[0], SIM_STRACE_WHEN=str(inject[1]))
            executable = wrapper_executable("strace", STRACE_WRAPPER)
        elif debug:
            fd, self.gdb_output = tempfile.mkstemp(prefix="lbug-sim-gdb-", suffix=".txt")
            os.close(fd)
            env = dict(SIM_GDB_SIGNALS=GDB_SIGNALS, SIM_GDB_OUT=self.gdb_output)
            executable = wrapper_executable("gdb", GDB_WRAPPER)
        else:
            env, executable = {}, None
        if executable:
            os.environ.update(env, SIM_PYTHON=sys.executable)
            context.set_executable(executable)
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
            self.build = self.reply()
        except EngineError:
            self.process.join()
            self.pipe.close()
            raise
        self.pid = self.build.pop("pid")

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

    def columns(self, query: str, params: dict, conn: int = 0) -> list:
        """The column names and the rows of one execution."""
        self.send(("columns", conn, query, params), query)
        return self.reply()

    def prepare(self, name: str, query: str, params: dict, conn: int = 0) -> None:
        """Prepares a statement and keeps it in the worker under `name`."""
        self.send(("prepare", name, conn, query, params), f"prepare {query}")
        self.reply()

    def run(self, name: str, params: dict) -> list:
        """Executes the statement kept under `name`."""
        self.send(("run", name, params), f"execute prepared {name}")
        return self.reply()

    def hold(self, conn: int) -> int:
        """Fills the buffer pool with results kept unconsumed on `conn`; returns how many."""
        self.send(("hold", conn), "hold results")
        return self.reply()

    def release(self) -> None:
        self.send(("release",), "release results")
        self.reply()

    def limit(self, size: int | None) -> None:
        """Limits the size of every file the worker writes; None lifts the limit."""
        self.send(("limit", size), f"limit files to {size} bytes")
        self.reply()

    def kill_at(self, syscall: str, n: int) -> None:
        """Attaches strace to every thread of the running worker, to kill it at the n-th call of
        `syscall` from now on. stop_strace detaches it if the worker survives."""
        self.strace = subprocess.Popen(
            ["strace", "-f", "-qq", "-o", "/dev/null", "-p", str(self.pid)]
            + ["-e", f"trace={syscall}", "-e", f"inject={syscall}:signal=KILL:when={n}"],
            stdin=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        deadline = time.monotonic() + 30
        while not self.traced_by(self.strace.pid):
            if self.strace.poll() is not None or time.monotonic() > deadline:
                raise AssertionError(f"strace could not attach to the worker {self.pid}")
            time.sleep(0.01)

    def traced_by(self, tracer: int) -> bool:
        try:
            tasks = os.listdir(f"/proc/{self.pid}/task")
            for task in tasks:
                with open(f"/proc/{self.pid}/task/{task}/status") as f:
                    if f"TracerPid:\t{tracer}\n" not in f.read():
                        return False
        except OSError:
            return False
        return True

    def stop_strace(self) -> None:
        if self.strace:
            self.strace.terminate()  # strace detaches before it exits
            self.strace.wait()
            self.strace = None

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
            stacks = self.stacks_of_crash()
            # Under gdb the exit code is gdb's; the signal is in its output.
            signal_name = re.search(r"received signal (SIG\w+)", stacks)
            how = signal_name[1] if signal_name else f"exit code {self.process.exitcode}"
            raise AssertionError(
                f"engine process died with {how} on {self.request}" + stacks
            ) from None
        if status == "error":
            raise EngineError(value)
        return value

    def stacks_of_hang(self) -> str:
        if self.pid is None:
            return ""
        if self.gdb_output:
            os.kill(self.pid, signal.SIGINT)  # gdb stops the worker and prints its stacks
            return self.stacks_of_crash()
        return gdb_stacks(self.pid)

    def stacks_of_crash(self) -> str:
        """The stacks gdb wrote when the worker stopped (gdb exits once it has written them), or
        the crashing thread's backtrace from crash_trace.c."""
        if not self.gdb_output:
            trace = os.path.join(self.crash_dir or "", f"crash-{self.process.pid}.txt")
            if not self.crash_dir or not os.path.exists(trace):
                return ""
            with open(trace) as f:
                return "\n" + backtrace_frames(f.read())
        self.process.join(timeout=120)
        with open(self.gdb_output) as f:
            output = f.read()
        return "\n" + native_stacks(output)

    def close(self) -> None:
        self.stop_strace()
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
        self.stop_strace()
        self.stop_gdb()

    def stop_gdb(self) -> None:
        if self.gdb_output:
            os.remove(self.gdb_output)
            self.gdb_output = None
