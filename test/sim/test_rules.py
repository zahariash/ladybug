"""Rule helpers and the rule names that options and known.py refer to, without an engine."""

from __future__ import annotations

import conftest
import pytest
from known import known_rules
from model import Model
from rules.data import copy_query, csv_field
from rules.prepared import TEMPLATES, expected_rows, params_for
from rules.transactions import apply, statement
from session import DUPLICATE_KEY, RULE_NAMES


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
