"""The database under test, open in a worker process.

A crash is a SIGKILL of the worker followed by a new worker that opens the database again and
replays the WAL, and an engine segfault ends the worker instead of the test run. Statements and
their results travel over a pipe; errors come back as EngineError.
"""

from __future__ import annotations

import multiprocessing
import threading

# The committed state a reader sees, read in one read-only transaction (see race).
STATE_QUERIES = [
    "MATCH (p:Person) RETURN count(*), sum(p.id), count(p.age)",
    "MATCH (:Person)-[k:Knows]->(:Person) RETURN count(*)",
]


class EngineError(Exception):
    """A statement failed in the engine."""


def read_state(conn) -> tuple:
    conn.execute("BEGIN TRANSACTION READ ONLY")
    state = tuple(tuple(conn.execute(q).get_all()[0]) for q in STATE_QUERIES)
    conn.execute("COMMIT")
    return state


def race(db, connect, writer, writes: list, num_readers: int) -> tuple[list, list]:
    """Runs `writes` on `writer` while readers repeatedly read the state; returns every state
    each reader saw and the errors of the writer and the readers."""
    done = threading.Event()
    observations = [[] for _ in range(num_readers)]
    errors = []

    def reader(out: list) -> None:
        conn = connect(db)
        try:
            while not done.is_set():
                out.append(read_state(conn))
            out.append(read_state(conn))
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


def worker_main(pipe, path: str, config: dict, loads: list[str]) -> None:
    """Opens the database with `config`, runs `loads` and serves requests until closed:
    (connection, query, params), ("race", writes, readers), or None to close."""
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
    while True:
        message = pipe.recv()
        if message is None:
            for conn in conns:
                conn.close()
            db.close()
            pipe.send(("ok", None))
            return
        if message[0] == "race":
            _, writes, num_readers = message
            pipe.send(("ok", race(db, lb.Connection, conns[0], writes, num_readers)))
            continue
        conn, query, params = message
        try:
            pipe.send(("ok", conns[conn].execute(query, params).get_all()))
        except Exception as e:
            pipe.send(("error", str(e)))


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
        self.reply()

    def execute(self, query: str, params: dict, conn: int = 0) -> list:
        self.pipe.send((conn, query, params))
        return self.reply()

    def start(self, query: str) -> None:
        """Sends a statement without waiting for it, to kill the worker while it runs."""
        self.pipe.send((0, query, {}))

    def race(self, writes: list, num_readers: int) -> tuple[list, list]:
        self.pipe.send(("race", writes, num_readers))
        return self.reply()

    def reply(self):
        try:
            status, value = self.pipe.recv()
        except EOFError:
            self.process.join()
            raise AssertionError(
                f"engine process died with exit code {self.process.exitcode}"
            ) from None
        if status == "error":
            raise EngineError(value)
        return value

    def close(self) -> None:
        self.pipe.send(None)
        self.reply()
        self.process.join()

    def kill(self) -> None:
        self.process.kill()
        self.process.join()
