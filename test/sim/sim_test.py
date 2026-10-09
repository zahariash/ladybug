"""Randomized workload simulation against an in-memory model.

Hypothesis drives random sequences of writes, transactions, DDL (macros, tables, columns, the
primary-key index, a partitioned table), checkpoints, bulk COPYs, clean reopens, crashes
between and in the middle of statements, and batches of writes racing concurrent readers.

The database runs in a worker process, so a crash is a SIGKILL followed by a reopen that
replays the WAL, and an engine segfault fails the test instead of killing it. After every
step, the data, the catalog and a set of queries are checked against the model and against a
single-threaded second connection. Failures shrink to a minimal step sequence and replay from
Hypothesis' example database.

    uv run --with hypothesis --with pytest --with ladybug pytest test/sim/sim_test.py
    SIM_EXAMPLES=500 SIM_STEPS=80 ... for longer runs
    SIM_SKIP_RULES=drop_macro,partitions ... to turn rules off
"""

import math
import multiprocessing
import os
import shutil
import tempfile
import threading
import time
from collections import Counter

import hypothesis.strategies as st
from hypothesis import HealthCheck, assume, settings
from hypothesis.stateful import RuleBasedStateMachine, initialize, invariant, precondition, rule

INT64_MIN, INT64_MAX = -(2**63), 2**63 - 1
SKIPPED_RULES = set(os.environ.get("SIM_SKIP_RULES", "").split(","))
SMALL_IDS = st.integers(0, 40)
BULK_ID_START = 1_000_000
MACROS = st.integers(0, 3)
TABLES = st.integers(0, 2)
EXTRA_COLUMNS = st.sampled_from(["c0", "c1"])
CLUSTERS = st.integers(1, 3)

ints = st.one_of(
    st.none(),
    st.integers(-100, 100),
    st.sampled_from([INT64_MIN, INT64_MIN + 1, INT64_MAX, 0, -1]),
    st.integers(INT64_MIN, INT64_MAX),
)
doubles = st.one_of(
    st.none(),
    st.floats(allow_nan=True, allow_infinity=True),
    st.sampled_from([0.0, -0.0, float("nan"), 1.5, -1e308]),
)
names = st.one_of(
    st.none(),
    st.text(st.characters(exclude_characters="\x00", exclude_categories=["Cs"]), max_size=40),
)
csv_names = st.text("abcdefghijklmnopqrstuvwxyz ", min_size=1, max_size=20)

STATE_QUERIES = [
    "MATCH (p:Person) RETURN count(*), sum(p.id), count(p.age)",
    "MATCH (:Person)-[k:Knows]->(:Person) RETURN count(*)",
]


def enabled(name):
    return precondition(lambda self: name not in SKIPPED_RULES and self.engine is not None)


class EngineError(Exception):
    pass


def read_state(conn):
    conn.execute("BEGIN TRANSACTION READ ONLY")
    state = tuple(tuple(conn.execute(q).get_all()[0]) for q in STATE_QUERIES)
    conn.execute("COMMIT")
    return state


def race(db, connect, writer, writes, num_readers):
    """Runs `writes` on `writer` while readers repeatedly read the state in a transaction."""
    done = threading.Event()
    observations = [[] for _ in range(num_readers)]
    errors = []

    def reader(out):
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
    for t in threads:
        t.start()
    try:
        for query, params in writes:
            writer.execute(query, params)
    except Exception as e:
        errors.append(f"writer: {query}: {e}")
    finally:
        done.set()
        for t in threads:
            t.join()
    return observations, errors


def worker_main(pipe, path, config):
    import ladybug as lb

    try:
        db = lb.Database(path, **config)
        conns = [lb.Connection(db), lb.Connection(db, num_threads=1)]
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
    """One open database in a worker process."""

    def __init__(self, path, config):
        context = multiprocessing.get_context("spawn")
        self.pipe, child = context.Pipe()
        self.process = context.Process(target=worker_main, args=(child, path, config), daemon=True)
        self.process.start()
        self.reply()

    def execute(self, query, params, conn):
        self.pipe.send((conn, query, params))
        return self.reply()

    def race(self, writes, num_readers):
        self.pipe.send(("race", writes, num_readers))
        return self.reply()

    def start(self, query):
        self.pipe.send((0, query, {}))

    def reply(self):
        try:
            status, value = self.pipe.recv()
        except EOFError:
            self.process.join()
            raise AssertionError(f"engine process died with exit code {self.process.exitcode}")
        if status == "error":
            raise EngineError(value)
        return value

    def close(self):
        self.pipe.send(None)
        self.reply()
        self.process.join()

    def kill(self):
        self.process.kill()
        self.process.join()


def canonical(row):
    return tuple("NaN" if isinstance(v, float) and math.isnan(v) else v for v in row)


def diff(actual, expected):
    return f"missing={list((expected - actual).items())[:5]} unexpected={list((actual - expected).items())[:5]}"


class LadybugSim(RuleBasedStateMachine):
    engine = None

    @initialize(
        threads=st.integers(1, 8),
        compression=st.booleans(),
        checkpoint_threshold=st.sampled_from([-1, 0, 4096]),
    )
    def open_db(self, threads, compression, checkpoint_threshold):
        self.dir = tempfile.mkdtemp(prefix="lbug-sim-")
        self.path = os.path.join(self.dir, "db")
        self.config = dict(
            buffer_pool_size=256 * 1024 * 1024,
            max_num_threads=threads,
            compression=compression,
            checkpoint_threshold=checkpoint_threshold,
        )
        self.columns = {"name": None, "age": None, "score": None}
        self.persons = {}
        self.knows = Counter()
        self.macros = {}
        self.tables = {}
        self.partitioned = {}
        self.pk_index = "_PK"
        self.next_bulk_id = BULK_ID_START
        self.engine = Engine(self.path, self.config)
        self.ok("CREATE NODE TABLE Person(id INT64 PRIMARY KEY, name STRING, age INT64, score DOUBLE)")
        self.ok("CREATE REL TABLE Knows(FROM Person TO Person, since INT64)")
        self.ok("CREATE NODE TABLE L(id INT64, cluster INT64, PRIMARY KEY(id)) PARTITION BY LIST (cluster)")

    def teardown(self):
        if self.engine:
            self.engine.kill()
            shutil.rmtree(self.dir, ignore_errors=True)

    def run(self, query, params=None, conn=0):
        return self.engine.execute(query, params or {}, conn)

    def ok(self, query, params=None):
        self.run(query, params)

    def fails(self, query, params=None):
        try:
            self.run(query, params)
        except EngineError:
            return
        raise AssertionError(f"expected an error: {query} {params}")

    def rows(self, query, params=None, conn=0):
        return [canonical(r) for r in self.run(query, params, conn)]

    def new_person(self, **values):
        return {c: values.get(c, default) for c, default in self.columns.items()}

    def forget(self, id):
        self.persons.pop(id, None)
        for key in [k for k in self.knows if id in (k[0], k[1])]:
            del self.knows[key]

    def reopen_engine(self):
        self.engine = Engine(self.path, self.config)

    # Data

    @rule(id=SMALL_IDS, name=names, age=ints, score=doubles)
    def insert_person(self, id, name, age, score):
        params = dict(id=id, name=name, age=age, score=score)
        query = "CREATE (:Person {id: $id, name: $name, age: $age, score: $score})"
        if id in self.persons:
            self.fails(query, params)
            return
        self.ok(query, params)
        self.persons[id] = self.new_person(name=name, age=age, score=score)

    @rule(id=SMALL_IDS, field=st.sampled_from(["name", "age", "score"]), data=st.data())
    def update_person(self, id, field, data):
        value = data.draw({"name": names, "age": ints, "score": doubles}[field])
        self.ok(f"MATCH (p:Person {{id: $id}}) SET p.{field} = $v", dict(id=id, v=value))
        if id in self.persons:
            self.persons[id][field] = value

    @rule(id=SMALL_IDS)
    def delete_person(self, id):
        self.ok("MATCH (p:Person {id: $id}) DETACH DELETE p", dict(id=id))
        self.forget(id)

    @rule(src=SMALL_IDS, dst=SMALL_IDS, since=ints)
    def insert_knows(self, src, dst, since):
        # Self-loops would make the 2-hop count depend on the path semantics.
        assume(src != dst)
        self.ok(
            "MATCH (a:Person {id: $a}), (b:Person {id: $b}) CREATE (a)-[:Knows {since: $s}]->(b)",
            dict(a=src, b=dst, s=since),
        )
        if src in self.persons and dst in self.persons:
            self.knows[(src, dst, since)] += 1

    @rule(src=SMALL_IDS, dst=SMALL_IDS)
    def delete_knows(self, src, dst):
        self.ok(
            "MATCH (a:Person {id: $a})-[k:Knows]->(b:Person {id: $b}) DELETE k",
            dict(a=src, b=dst),
        )
        for key in [k for k in self.knows if k[0] == src and k[1] == dst]:
            del self.knows[key]

    def write_bulk_csv(self, rows):
        """Writes `rows` of (name, age) with fresh ids; returns the path and the new persons."""
        csv = os.path.join(self.dir, f"bulk{self.next_bulk_id}.csv")
        added = {}
        with open(csv, "w") as f:
            for i, (name, age) in enumerate(rows):
                id = self.next_bulk_id + i
                added[id] = self.new_person(name=name, age=age, score=age / 4)
                f.write(",".join([str(id)] + ["" if v is None else str(v) for v in added[id].values()]) + "\n")
        self.next_bulk_id += len(rows)
        return csv, added

    @rule(rows=st.lists(st.tuples(csv_names, st.integers(-1000, 1000)), min_size=1, max_size=3000))
    def bulk_copy(self, rows):
        csv, added = self.write_bulk_csv(rows)
        if self.pk_index is None and self.persons:
            # COPY into a non-empty table needs the primary-key index.
            self.fails(f'COPY Person FROM "{csv}"')
            return
        self.ok(f'COPY Person FROM "{csv}"')
        self.persons.update(added)

    @rule(id=SMALL_IDS, age=ints, commit=st.booleans())
    def transaction(self, id, age, commit):
        self.ok("BEGIN TRANSACTION")
        self.ok("MATCH (p:Person {id: $id}) SET p.age = $v", dict(id=id, v=age))
        self.ok("MATCH (p:Person {id: $id}) DETACH DELETE p", dict(id=id + 1))
        self.ok("COMMIT" if commit else "ROLLBACK")
        if commit:
            if id in self.persons:
                self.persons[id]["age"] = age
            self.forget(id + 1)

    @enabled("partitions")
    @rule(id=SMALL_IDS, cluster=CLUSTERS)
    def insert_partitioned(self, id, cluster):
        query = "CREATE (:L {id: $id, cluster: $c})"
        if id in self.partitioned:
            self.fails(query, dict(id=id, c=cluster))
            return
        self.ok(query, dict(id=id, c=cluster))
        self.partitioned[id] = cluster

    @enabled("partitions")
    @rule(id=SMALL_IDS)
    def delete_partitioned(self, id):
        self.ok("MATCH (n:L {id: $id}) DELETE n", dict(id=id))
        self.partitioned.pop(id, None)

    # Catalog

    @enabled("macros")
    @rule(i=MACROS, k=st.integers(-5, 5))
    def create_macro(self, i, k):
        query = f"CREATE MACRO m{i}(x) AS x + {k}"
        if i in self.macros:
            self.fails(query)
            return
        self.ok(query)
        self.macros[i] = k

    @enabled("drop_macro")
    @rule(i=MACROS)
    def drop_macro(self, i):
        if i not in self.macros:
            self.fails(f"DROP MACRO m{i}")
            return
        self.ok(f"DROP MACRO m{i}")
        del self.macros[i]

    @enabled("tables")
    @rule(i=TABLES)
    def create_table(self, i):
        query = f"CREATE NODE TABLE T{i}(id SERIAL PRIMARY KEY, v INT64)"
        if i in self.tables:
            self.fails(query)
            return
        self.ok(query)
        self.tables[i] = []

    @enabled("tables")
    @rule(i=TABLES, values=st.lists(st.integers(-1000, 1000), min_size=1, max_size=5))
    def insert_into_table(self, i, values):
        assume(i in self.tables)
        self.ok(f"UNWIND $vs AS v CREATE (:T{i} {{v: v}})", dict(vs=values))
        self.tables[i] += values

    @enabled("tables")
    @rule(i=TABLES)
    def drop_table(self, i):
        if i not in self.tables:
            self.fails(f"DROP TABLE T{i}")
            return
        self.ok(f"DROP TABLE T{i}")
        del self.tables[i]

    @enabled("columns")
    @rule(column=EXTRA_COLUMNS, default=st.integers(-5, 5))
    def add_column(self, column, default):
        query = f"ALTER TABLE Person ADD {column} INT64 DEFAULT {default}"
        if column in self.columns:
            self.fails(query)
            return
        self.ok(query)
        self.columns[column] = default
        for person in self.persons.values():
            person[column] = default

    @enabled("columns")
    @rule(column=EXTRA_COLUMNS)
    def drop_column(self, column):
        query = f"ALTER TABLE Person DROP {column}"
        if column not in self.columns:
            self.fails(query)
            return
        self.ok(query)
        del self.columns[column]
        for person in self.persons.values():
            del person[column]

    @enabled("pk_index")
    @rule()
    def drop_pk_index(self):
        assume(self.pk_index is not None)
        self.ok(f"DROP INDEX Person.{self.pk_index}")
        self.pk_index = None

    @enabled("pk_index")
    @rule()
    def create_pk_index(self):
        assume(self.pk_index is None)
        self.ok("CREATE INDEX person_pk FOR (p:Person) ON (p.id)")
        self.pk_index = "person_pk"

    # Durability

    @rule()
    def checkpoint(self):
        self.ok("CHECKPOINT")

    @rule()
    def reopen(self):
        self.engine.close()
        self.reopen_engine()

    @rule()
    def crash(self):
        self.engine.kill()
        self.reopen_engine()

    @enabled("crash_during")
    @rule(
        op=st.sampled_from(["copy", "create", "delete", "checkpoint"]),
        size=st.integers(2000, 40000),
        delay=st.floats(0, 0.3),
    )
    def crash_during(self, op, size, delay):
        """Kills the engine while a large statement runs; recovery must see all of it or none."""
        before = {id: dict(p) for id, p in self.persons.items()}
        after = {id: dict(p) for id, p in self.persons.items()}
        knows_after = Counter(self.knows)
        if op == "copy":
            assume(self.pk_index is not None or not self.persons)
            csv, added = self.write_bulk_csv([("bulk", i % 1000) for i in range(size)])
            after.update(added)
            query = f'COPY Person FROM "{csv}"'
        elif op == "create":
            first = self.next_bulk_id
            self.next_bulk_id += size
            for id in range(first, first + size):
                after[id] = self.new_person(age=id % 1000)
            query = f"UNWIND range({first}, {first + size - 1}) AS i CREATE (:Person {{id: i, age: i % 1000}})"
        elif op == "delete":
            for id in [id for id in after if id >= BULK_ID_START]:
                del after[id]
            knows_after = Counter({k: n for k, n in self.knows.items()
                                   if k[0] < BULK_ID_START and k[1] < BULK_ID_START})
            query = f"MATCH (p:Person) WHERE p.id >= {BULK_ID_START} DETACH DELETE p"
        else:
            query = "CHECKPOINT"
        self.engine.start(query)
        time.sleep(delay)
        self.engine.kill()
        self.reopen_engine()
        actual = Counter(self.person_rows())
        if actual == Counter(self.model_rows(after)):
            self.persons, self.knows = after, knows_after
        else:
            assert actual == Counter(self.model_rows(before)), (
                f"after a crash during {op}, the data is neither the state before nor after it: "
                + diff(actual, Counter(self.model_rows(before))))

    # Concurrency

    @enabled("race")
    @rule(
        ops=st.lists(
            st.tuples(st.sampled_from(["insert", "delete", "set_age", "clear_age", "checkpoint"]),
                      SMALL_IDS, st.integers(-100, 100)),
            min_size=1, max_size=30),
        num_readers=st.integers(2, 4),
    )
    def race(self, ops, num_readers):
        """Writes race readers; every read transaction must see one committed state, in order."""
        persons = {id: dict(p) for id, p in self.persons.items()}
        knows = Counter(self.knows)
        states = [self.state_of(persons, knows)]
        writes = []
        for op, id, value in ops:
            if op == "insert":
                if id in persons:
                    continue
                writes.append(("CREATE (:Person {id: $id, age: $v})", dict(id=id, v=value)))
                persons[id] = self.new_person(age=value)
            elif op == "delete":
                writes.append(("MATCH (p:Person {id: $id}) DETACH DELETE p", dict(id=id)))
                persons.pop(id, None)
                knows = Counter({k: n for k, n in knows.items() if id not in (k[0], k[1])})
            elif op in ("set_age", "clear_age"):
                age = value if op == "set_age" else None
                writes.append(("MATCH (p:Person {id: $id}) SET p.age = $v", dict(id=id, v=age)))
                if id in persons:
                    persons[id]["age"] = age
            else:
                writes.append(("CHECKPOINT", {}))
            states.append(self.state_of(persons, knows))
        observations, errors = self.engine.race(writes, num_readers)
        assert not errors, errors
        for reader in observations:
            position = 0
            for seen in reader:
                matches = [i for i in range(position, len(states)) if states[i] == seen]
                assert matches, (f"a reader saw {seen}, which is not a committed state at or after "
                                 f"step {position}: {states}")
                position = matches[0]
        self.persons, self.knows = persons, knows

    def state_of(self, persons, knows):
        ages = [p["age"] for p in persons.values() if p["age"] is not None]
        return ((len(persons), sum(persons) if persons else None, len(ages)), (sum(knows.values()),))

    # Checks

    def person_rows(self):
        columns = "".join(f", p.{c}" for c in self.columns)
        return self.rows(f"MATCH (p:Person) RETURN p.id{columns}")

    def model_rows(self, persons):
        return [canonical((id, *p.values())) for id, p in persons.items()]

    @invariant()
    def data_matches_model(self):
        if not self.engine:
            return
        actual = Counter(self.person_rows())
        expected = Counter(self.model_rows(self.persons))
        assert actual == expected, diff(actual, expected)
        actual = Counter(self.rows("MATCH (a:Person)-[k:Knows]->(b:Person) RETURN a.id, b.id, k.since"))
        assert actual == self.knows, diff(actual, self.knows)
        actual = Counter(self.rows("MATCH (n:L) RETURN n.id, n.cluster"))
        assert actual == Counter(self.partitioned.items()), diff(actual, Counter(self.partitioned.items()))

    @invariant()
    def catalog_matches_model(self):
        if not self.engine:
            return
        tables = {name for (name,) in self.rows("CALL show_tables() RETURN name") if not name.startswith("L_p")}
        assert tables == {"Person", "Knows", "L"} | {f"T{i}" for i in self.tables}, tables
        for i, values in self.tables.items():
            self.check(f"MATCH (t:T{i}) RETURN count(*), sum(t.v)", {}, [(len(values), sum(values) if values else None)])
        macros = {name.lower() for (name,) in self.rows("CALL show_macros() RETURN name")}
        assert macros == {f"m{i}" for i in self.macros}, macros
        for i, k in self.macros.items():
            self.check(f"RETURN m{i}(10)", {}, [(10 + k,)])
        indexes = {name for table, name in self.rows("CALL show_indexes() RETURN table_name, index_name")
                   if table == "Person"}
        assert indexes == ({self.pk_index} if self.pk_index else set()), indexes

    @precondition(lambda self: self.engine is not None)
    @rule(x=st.integers(-200, 200), id=SMALL_IDS, limit=st.integers(1, 20), cluster=CLUSTERS)
    def queries_match_model(self, x, id, limit, cluster):
        ages = [p["age"] for p in self.persons.values() if p["age"] is not None]
        self.check("MATCH (p:Person) WHERE p.age > $x RETURN count(*)", dict(x=x),
                   [(sum(a > x for a in ages),)])
        self.check("MATCH (p:Person) WHERE p.age IS NOT NULL RETURN min(p.age), max(p.age), count(*)", {},
                   [(min(ages), max(ages), len(ages)) if ages else (None, None, 0)])
        person = self.persons.get(id)
        self.check("MATCH (p:Person {id: $id}) RETURN p.name, p.age", dict(id=id),
                   [(person["name"], person["age"])] if person else [])
        self.check("MATCH (p:Person) WHERE p.id IN [$id, $id + 1] RETURN p.id", dict(id=id),
                   [(i,) for i in (id, id + 1) if i in self.persons])
        top = sorted((p["age"], k) for k, p in self.persons.items() if p["age"] is not None)[:limit]
        self.check("MATCH (p:Person) WHERE p.age IS NOT NULL RETURN p.age, p.id ORDER BY p.age, p.id LIMIT $l",
                   dict(l=limit), top, ordered=True)
        out = Counter()
        for (src, _, _), n in self.knows.items():
            out[src] += n
        self.check("MATCH (a:Person)-[:Knows]->(:Person) RETURN a.id, count(*)", {}, list(out.items()))
        two_hop = sum(n1 * n2 for (_, b, _), n1 in self.knows.items()
                      for (c, _, _), n2 in self.knows.items() if b == c)
        self.check("MATCH (:Person)-[:Knows]->(:Person)-[:Knows]->(:Person) RETURN count(*)", {},
                   [(two_hop,)])
        self.check("MATCH (n:L) WHERE n.cluster = $c RETURN count(*)", dict(c=cluster),
                   [(sum(c == cluster for c in self.partitioned.values()),)])
        for column, default in self.columns.items():
            if column.startswith("c"):
                self.check(f"MATCH (p:Person) WHERE p.{column} = $d RETURN count(*)", dict(d=default),
                           [(sum(p[column] == default for p in self.persons.values()),)])

    def check(self, query, params, expected, ordered=False):
        expected = [canonical(r) for r in expected]
        # The second run on the same connection reuses the cached plan for parameterized queries.
        for conn in (0, 0, 1):
            actual = self.rows(query, params, conn)
            if ordered:
                assert actual == expected, (query, params, conn, actual, expected)
            else:
                assert Counter(actual) == Counter(expected), (query, params, conn, actual, expected)


LadybugSim.TestCase.settings = settings(
    max_examples=int(os.environ.get("SIM_EXAMPLES", 50)),
    stateful_step_count=int(os.environ.get("SIM_STEPS", 40)),
    deadline=None,
    suppress_health_check=list(HealthCheck),
)
TestLadybugSim = LadybugSim.TestCase
