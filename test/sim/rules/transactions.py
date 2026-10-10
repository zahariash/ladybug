"""Rules for explicit transactions and statement atomicity.

A transaction applies its writes to a fork of the model. It sees its own writes, COMMIT keeps
them, and ROLLBACK or a crash before COMMIT drops them. A statement that fails inside the
transaction rolls all of it back; the statements after it run on their own, and COMMIT then
fails because no transaction is active.
"""

from __future__ import annotations

from collections import Counter

import hypothesis.strategies as st
from engine import EngineError
from hypothesis.stateful import precondition, rule
from model import Model
from session import DUPLICATE_KEY, Session, diff, enabled, gated
from strategies import ints, macro_ids, small_ids, table_ids

ENDS = ["commit", "rollback", "crash"]
gated("txn_ddl", "txn_error", *(f"txn_{end}" for end in ENDS))
NO_TRANSACTION = "No active transaction"

op = st.one_of(
    st.tuples(st.just("insert"), small_ids, ints),
    st.tuples(st.just("set_age"), small_ids, ints),
    st.tuples(st.just("delete"), small_ids, st.none()),
    st.tuples(st.just("knows"), small_ids, small_ids),
    st.tuples(st.just("create_table"), table_ids, st.none()),
    st.tuples(st.just("insert_t"), table_ids, st.integers(-1000, 1000)),
    st.tuples(st.just("create_macro"), macro_ids, st.integers(-5, 5)),
    st.tuples(st.just("duplicate"), small_ids, st.none()),
)
DDL_OPS = {"create_table", "create_macro"}


def statement(model: Model, kind: str, a, b) -> tuple[str, dict, str | None] | None:
    """The statement for an op, and the error it must fail with, or None for a no-op."""
    if kind == "insert":
        if a in model.persons:
            return None
        return "CREATE (:Person {id: $id, age: $v})", dict(id=a, v=b), None
    if kind == "set_age":
        return "MATCH (p:Person {id: $id}) SET p.age = $v", dict(id=a, v=b), None
    if kind == "delete":
        return "MATCH (p:Person {id: $id}) DETACH DELETE p", dict(id=a), None
    if kind == "knows":
        query = "MATCH (x:Person {id: $a}), (y:Person {id: $b}) CREATE (x)-[:Knows {since: 0}]->(y)"
        return query, dict(a=a, b=b), None
    if kind == "create_table":
        if a in model.tables:
            return None
        return f"CREATE NODE TABLE T{a}(id SERIAL PRIMARY KEY, v INT64)", {}, None
    if kind == "insert_t":
        if a not in model.tables:
            return None
        return f"CREATE (:T{a} {{v: $v}})", dict(v=b), None
    if kind == "create_macro":
        if a in model.macros:
            return None
        return f"CREATE MACRO m{a}(x) AS x + {b}", {}, None
    if a not in model.persons:
        return None
    return "CREATE (:Person {id: $id})", dict(id=a), DUPLICATE_KEY


def apply(model: Model, kind: str, a, b) -> None:
    if kind == "insert":
        model.add_persons({a: model.new_person(age=b)})
    elif kind == "set_age" and a in model.persons:
        model.persons[a]["age"] = b
    elif kind == "delete":
        model.forget(a)
    elif kind == "knows" and a in model.persons and b in model.persons:
        model.knows[(a, b, 0)] += 1
    elif kind == "create_table":
        model.tables[a] = []
    elif kind == "insert_t":
        model.tables[a].append(b)
    elif kind == "create_macro":
        model.macros[a] = b


class TransactionRules(Session):
    def allowed_ops(self) -> list:
        skipped = self.options.skipped
        return [
            kind
            for kind in ("insert", "set_age", "delete", "knows", "insert_t", "duplicate")
            + tuple(DDL_OPS)
            if not (kind in DDL_OPS and "txn_ddl" in skipped)
            and not (kind == "duplicate" and "txn_error" in skipped)
        ]

    @enabled("transaction")
    @rule(ops=st.lists(op, min_size=1, max_size=12), data=st.data())
    def transaction(self, ops, data):
        """Runs ops in a transaction, checks that it sees its own writes, then ends it."""
        ends = [end for end in ENDS if f"txn_{end}" not in self.options.skipped]
        end = data.draw(st.sampled_from(ends or ["commit"]))
        allowed = self.allowed_ops()
        pending = self.model.fork()  # what the transaction sees
        active = True
        self.ok("BEGIN TRANSACTION")
        for kind, a, b in (o for o in ops if o[0] in allowed):
            target = pending if active else self.model
            step = statement(target, kind, a, b)
            if step is None:
                continue
            query, params, error = step
            if error is None:
                self.ok(query, params)
                apply(target, kind, a, b)
                continue
            self.fails(query, params, error)
            # The failed statement rolled the transaction back; later ones run on their own.
            self.model.persons_ever |= pending.persons_ever
            active = False
        if active:
            self.check_own_writes(pending)
        self.end(end, pending, active)

    def check_own_writes(self, pending: Model) -> None:
        actual = self.person_rows()
        expected = pending.person_rows()
        assert actual == expected, "inside the transaction: " + diff(actual, expected)
        query = "MATCH (a:Person)-[k:Knows]->(b:Person) RETURN a.id, b.id, k.since"
        actual = Counter(self.rows(query))
        assert actual == pending.knows, "inside the transaction: " + diff(actual, pending.knows)

    def end(self, end: str, pending: Model, active: bool) -> None:
        if not active:
            self.fails("COMMIT" if end == "commit" else "ROLLBACK", None, NO_TRANSACTION)
            return
        # Rolled-back inserts may still count as rows until compaction (see bulk_copy).
        self.model.persons_ever |= pending.persons_ever
        if end == "commit":
            self.ok("COMMIT")
            self.model = pending
        elif end == "rollback":
            self.ok("ROLLBACK")
        else:
            self.crash_engine()

    @enabled("atomic_statement")
    @precondition(lambda self: any(id < 41 for id in self.model.persons))
    @rule(new=st.lists(small_ids, min_size=1, max_size=5, unique=True), data=st.data())
    def failing_statement_is_atomic(self, new, data):
        """A multi-row CREATE that hits a duplicate key midway must leave nothing behind."""
        existing = data.draw(st.sampled_from(sorted(i for i in self.model.persons if i < 41)))
        ids = [i for i in new if i not in self.model.persons]
        ids.insert(data.draw(st.integers(0, len(ids))), existing)
        try:
            self.run("UNWIND $ids AS i CREATE (:Person {id: i})", dict(ids=ids))
        except EngineError as e:
            assert DUPLICATE_KEY in str(e), e
            self.model.persons_ever = True
            return
        raise AssertionError(f"inserting existing id {existing} among {ids} succeeded")
