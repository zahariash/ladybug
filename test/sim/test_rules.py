"""Rule helpers and the rule names that options and known.py refer to, without an engine."""

from __future__ import annotations

import ast
import pathlib
from types import SimpleNamespace

import conftest
import pytest
from known import known_rules
from model import Model
from rules.data import copy_query, csv_field
from rules.durability import DurabilityRules
from rules.prepared import TEMPLATES, expected_rows, params_for
from rules.transactions import apply, statement
from session import DUPLICATE_KEY, RULE_NAMES, Options


def test_known_bugs_and_focus_groups_name_registered_rules() -> None:
    assert known_rules() <= RULE_NAMES
    for names in conftest.FOCUS.values():
        assert names <= RULE_NAMES


def test_csv_fields_quote_strings_and_leave_null_empty() -> None:
    assert csv_field(None) == ""
    assert csv_field("") == '""'
    assert csv_field('say "hi", ok') == '"say ""hi"", ok"'
    assert csv_field(3) == "3"


def test_copy_uses_the_serial_reader_for_quoted_newlines() -> None:
    assert copy_query("f.csv", [("a", 1)]) == 'COPY Person FROM "f.csv"'
    assert copy_query("f.csv", [("a\nb", 1), (None, 2)]).endswith("(parallel=false)")


def small_model() -> Model:
    model = Model()
    model.add_persons({1: model.new_person(age=10), 2: model.new_person(age=-5)})
    model.knows[(1, 2, 0)] += 2
    return model


def test_race_edges_join_existing_persons_and_update_the_model() -> None:
    model = small_model()
    rules = SimpleNamespace(options=Options(), checkpoint_keeps_snapshots=lambda: True)
    ops = [("knows", 1, 3), ("knows", 9, 0), ("unknows", 1, 1)]
    writes, states = DurabilityRules.race_writes(rules, model, ops)
    assert [params for _, params in writes] == [dict(a=1, b=2, s=3), dict(a=1, b=2)]
    edges_and_sources = [state[1:3] for state in states]
    assert edges_and_sources == [(((2,),), ((1,),)), (((3,),), ((1,),)), (((0,),), ((0,),))]
    assert not model.knows


def test_transaction_statements_skip_no_ops_and_flag_duplicates() -> None:
    model = small_model()
    assert statement(model, "insert", 1, 0) is None
    assert statement(model, "insert_t", 0, 1) is None  # no table T0
    assert statement(model, "duplicate", 1, None)[2] == DUPLICATE_KEY
    assert statement(model, "duplicate", 9, None) is None
    assert statement(model, "insert", 9, 3)[2] is None


def test_transaction_ops_apply_to_the_model() -> None:
    model = small_model()
    apply(model, "insert", 9, 3)
    apply(model, "knows", 9, 1)
    apply(model, "knows", 9, 42)  # unknown endpoint: no edge
    apply(model, "create_table", 0, None)
    apply(model, "insert_t", 0, 7)
    apply(model, "delete", 2, None)
    assert model.persons[9]["age"] == 3
    assert model.knows == {(9, 1, 0): 1}
    assert model.tables == {0: [7]}


@pytest.mark.parametrize("name", sorted(TEMPLATES))
def test_prepared_templates_get_only_the_parameters_they_use(name) -> None:
    params = params_for(name, 1, 2)
    for key in params:
        assert f"${key}" in TEMPLATES[name].query
    assert all(f"${key}" not in TEMPLATES[name].query for key in {"id", "v"} - set(params))


def test_prepared_templates_bind_only_when_their_table_and_column_exist() -> None:
    model = small_model()
    assert TEMPLATES["person_by_id"].binds(model)
    assert not TEMPLATES["count_t0"].binds(model)
    assert not TEMPLATES["read_c0"].binds(model)
    model.tables[0] = []
    model.columns["c0"] = 1
    assert TEMPLATES["count_t0"].binds(model) and TEMPLATES["read_c0"].binds(model)


def test_prepared_reads_expect_what_the_model_holds() -> None:
    model = small_model()
    assert expected_rows(model, "person_by_id", 1, 0) == [(1, 10)]
    assert expected_rows(model, "person_by_id", 7, 0) == []
    assert expected_rows(model, "count_age", 0, 0) == [(1,)]
    assert expected_rows(model, "knows_from", 1, 0) == [(2,), (2,)]


def test_transaction_ddl_ops_change_the_catalog_and_flag_conflicts() -> None:
    model = small_model()
    assert statement(model, "add_column", "c1", 3) == (
        "ALTER TABLE Person ADD c1 INT64 DEFAULT 3",
        {},
        None,
    )
    apply(model, "add_column", "c1", 3)
    assert model.columns["c1"] == 3 and all(p["c1"] == 3 for p in model.persons.values())
    assert statement(model, "add_column", "c1", 0)[2] == "already has property"
    apply(model, "drop_column", "c1", None)
    assert "c1" not in model.columns and "c1" in model.dropped_columns
    assert statement(model, "drop_column", "c1", None)[2] == "does not have property"
    assert statement(model, "drop_table", 4, None) is None
    model.tables[4] = [1]
    assert statement(model, "drop_table", 4, None) == ("DROP TABLE T4", {}, None)


def test_session_rules_are_rules_of_the_machine() -> None:
    """A helper of the same name in an earlier mixin would shadow a rule."""
    from machine import LadybugSim

    for name in ("begin_read", "begin_write", "session_write", "session_read", "end_session"):
        assert hasattr(getattr(LadybugSim, name), "hypothesis_stateful_rule"), name


def test_rules_mark_the_errors_they_expect() -> None:
    """Triage and replay take any unmarked error in a trace for the failure."""
    for path in pathlib.Path(__file__).parent.glob("rules/*.py"):
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.ExceptHandler) and ast.unparse(node.type) == "EngineError":
                body = ast.unparse(node)
                assert "['expected'] = True" in body or "raise" in body, (
                    f"{path.name}:{node.lineno}"
                )
