"""Rules for prepared statements that outlive schema changes.

prepare_statement keeps a prepared statement in the worker; execute_prepared runs it later,
after whatever DDL the steps in between did. It must behave as if it had been prepared just
now: return what the model says, apply its write, or, when its table or column no longer
exists, fail cleanly. Reopening or crashing the engine drops the kept statements.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import hypothesis.strategies as st
from engine import EngineError
from hypothesis.stateful import precondition, rule
from model import Model
from session import DUPLICATE_KEY, Session, enabled
from strategies import small_ids

values = st.integers(-100, 100)


@dataclass(frozen=True)
class Template:
    query: str
    write: bool = False
    needs_t0: bool = False
    needs_c0: bool = False

    def binds(self, model: Model) -> bool:
        """Whether the statement refers only to tables and columns that exist."""
        return (not self.needs_t0 or 0 in model.tables) and (
            not self.needs_c0 or "c0" in model.columns
        )


TEMPLATES = {
    "person_by_id": Template("MATCH (p:Person) WHERE p.id = $id RETURN p.id, p.age"),
    "count_age": Template("MATCH (p:Person) WHERE p.age > $v RETURN count(*)"),
    "knows_from": Template("MATCH (a:Person {id: $id})-[k:Knows]->(b:Person) RETURN b.id"),
    "read_c0": Template("MATCH (p:Person) WHERE p.id = $id RETURN p.c0", needs_c0=True),
    "count_t0": Template("MATCH (t:T0) RETURN count(*)", needs_t0=True),
    "insert_person": Template("CREATE (:Person {id: $id, age: $v})", write=True),
    "set_age": Template("MATCH (p:Person {id: $id}) SET p.age = $v", write=True),
    "insert_t0": Template("CREATE (:T0 {v: $v})", write=True, needs_t0=True),
}


def params_for(name: str, id: int, v: int) -> dict:
    query = TEMPLATES[name].query
    return {key: value for key, value in (("id", id), ("v", v)) if f"${key}" in query}


def expected_rows(model: Model, name: str, id: int, v: int) -> list:
    persons = model.persons
    if name == "person_by_id":
        return [(id, persons[id]["age"])] if id in persons else []
    if name == "count_age":
        return [(sum(p["age"] is not None and p["age"] > v for p in persons.values()),)]
    if name == "knows_from":
        return [(dst,) for (src, dst, _), n in model.knows.items() if src == id for _ in range(n)]
    if name == "read_c0":
        return [(persons[id]["c0"],)] if id in persons else []
    return [(len(model.tables[0]),)]


class PreparedRules(Session):
    @enabled("prepared_ddl")
    @rule(name=st.sampled_from(sorted(TEMPLATES)), id=small_ids, v=values)
    def prepare_statement(self, name, id, v):
        template = TEMPLATES[name]
        params = params_for(name, id, v)
        if not template.binds(self.model):
            try:
                self.prepare_named(name, template.query, params)
            except EngineError:
                self.trace[-1]["expected"] = True
                return
            raise AssertionError(f"preparing {template.query} without its table/column worked")
        self.prepare_named(name, template.query, params)
        self.kept[name] = template

    @enabled("prepared_ddl")
    @precondition(lambda self: self.kept)
    @rule(data=st.data(), id=small_ids, v=values)
    def execute_prepared(self, data, id, v):
        """Runs a kept statement; it must act as if prepared now."""
        name = data.draw(st.sampled_from(sorted(self.kept)))
        template, params = TEMPLATES[name], params_for(name, id, v)
        if not template.binds(self.model):
            try:
                self.run_named(name, params)
            except EngineError:
                self.trace[-1]["expected"] = True
                return
            raise AssertionError(f"{template.query} ran although its table/column is gone")
        if name == "insert_person" and id in self.model.persons:
            try:
                self.run_named(name, params)
            except EngineError as e:
                self.trace[-1]["expected"] = True
                assert DUPLICATE_KEY in str(e), e
                return
            raise AssertionError(f"duplicate id {id} was accepted by a kept statement")
        actual = self.run_named(name, params)
        if not template.write:
            expected = expected_rows(self.model, name, id, v)
            assert Counter(actual) == Counter(expected), (name, params, actual, expected)
        elif name == "insert_person":
            self.model.add_persons({id: self.model.new_person(age=v)})
        elif name == "set_age" and id in self.model.persons:
            self.model.persons[id]["age"] = v
        elif name == "insert_t0":
            self.model.tables[0].append(v)
