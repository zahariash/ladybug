"""The database under test, open in a worker process.

A crash is a SIGKILL of the worker followed by a new worker that opens the database again and
replays the WAL, and an engine segfault ends the worker instead of the test run. Statements and
their results travel over a pipe; errors come back as EngineError, and a worker that does not
answer within TIMEOUT_SECONDS is killed and reported as hung.
"""

from __future__ import annotations

import multiprocessing
import os
import threading

TIMEOUT_SECONDS = int(os.environ.get("SIM_TIMEOUT_SECONDS", 300))


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


def serve(conns: list, db, connect, message: tuple):
    """Handles one request: ("race", writes, queries, readers), ("prepared", connection, query,
    params), which prepares the query once and executes it twice, or (connection, query,
    params)."""
    if message[0] == "race":
        _, writes, queries, num_readers = message
        return race(db, connect, conns[0], writes, queries, num_readers)
    if message[0] == "prepared":
        _, conn, query, params = message
        statement = conns[conn].prepare(query, params)
        return [conns[conn].execute(statement, params).get_all() for _ in range(2)]
    conn, query, params = message
    return conns[conn].execute(query, params).get_all()


def worker_main(pipe, path: str, config: dict, loads: list[str]) -> None:
    """Opens the database with `config`, runs `loads` and serves requests until None."""
    import ladybug as lb

    try:
        db = lb.Database(path, **config)
        conns = [lb.Connection(db), lb.Connection(db, num_threads=1)]
        for query in loads:
            conns[0].execute(query)
    except Exception as e:
        pipe.send(("error", f"open failed: {e}"))
        return
    pipe.send(("ok", None))
    while (message := pipe.recv()) is not None:
        try:
            pipe.send(("ok", serve(conns, db, lb.Connection, message)))
        except Exception as e:
            pipe.send(("error", str(e)))
    for conn in conns:
        conn.close()
    db.close()
    pipe.send(("ok", None))


class Engine:
    """One open database in a worker process. Connection 0 uses the configured thread count,
    connection 1 a single thread."""

    def __init__(self, path: str, config: dict, loads: list[str]) -> None:
        context = multiprocessing.get_context("spawn")
        self.pipe, child = context.Pipe()
        self.process = context.Process(
            target=worker_main, args=(child, path, config, loads), daemon=True
        )
        self.process.start()
        # Without closing its copy of the worker's end, a dead worker never reads as EOF.
        child.close()
        self.request = "open"
        try:
            self.reply()
        except EngineError:
            self.process.join()
            self.pipe.close()
            raise

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

    def start(self, query: str) -> None:
        """Sends a statement without waiting for it, to kill the worker while it runs."""
        self.send((0, query, {}), query)

    def race(self, writes: list, queries: list[str], num_readers: int) -> tuple[list, list]:
        self.send(("race", writes, queries, num_readers), f"race reading {queries}")
        return self.reply()

    def reply(self):
        if not self.pipe.poll(TIMEOUT_SECONDS):
            self.kill()
            raise AssertionError(f"engine hung for {TIMEOUT_SECONDS} s on {self.request}")
        try:
            status, value = self.pipe.recv()
        except EOFError:
            self.process.join()
            raise AssertionError(
                f"engine process died with exit code {self.process.exitcode} on {self.request}"
            ) from None
        if status == "error":
            raise EngineError(value)
        return value

    def close(self) -> None:
        self.send(None, "close")
        self.reply()
        self.process.join()
        self.pipe.close()

    def kill(self) -> None:
        self.process.kill()
        self.process.join()
        self.pipe.close()
