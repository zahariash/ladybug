"""Rule helpers and the rule names that options and known.py refer to, without an engine."""

from __future__ import annotations

import ast
import pathlib
from collections import Counter
from types import SimpleNamespace

import conftest
import pytest
from known import known_rules
from model import Model
from rules.data import copy_query, csv_field
from rules.durability import (
    DurabilityRules,
    check_survivors,
    check_writer_reads,
    ck_state,
    read_acks,
    edge_fingerprint,
    writer_states,
)
from rules.prepared import TEMPLATES, expected_rows, params_for
from rules.sessions import SessionRules, Txn
from rules.transactions import apply, replay, seen_by, statement
from session import DUPLICATE_KEY, RULE_NAMES, Options, Session
from writesets import (
    GATED_EDGE,
    GATED_HIDDEN,
    GATED_INSERT,
    GATED_REINSERT,
    PERSON,
    WW_CATALOG,
    WW_DELETE,
    WW_UPDATE,
    Others,
    Writes,
    conflict,
    record,
)


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


def test_write_outcomes_follow_what_the_error_says() -> None:
    from rules.recovery import write_outcome

    assert write_outcome(None, copy=False) == "applied"
    old = "Transaction committed successfully, but the post-commit checkpoint failed. ..."
    assert write_outcome(old, copy=False) == "auto_checkpoint"
    assert write_outcome(old, copy=True) == "applied"
    forced = "Transaction committed successfully, but the checkpoint it requires failed. ..."
    assert write_outcome(forced, copy=True) == "doubtful"
    reopen = "... logging its abort failed, so it is applied when the database reopens; ..."
    assert write_outcome(reopen, copy=False) == "on_reopen"
    assert write_outcome("IO exception: File too large", copy=False) == "failed"


def test_size_caps_are_powers_of_two_of_at_least_8_mib() -> None:
    from rules.recovery import capped_size

    assert capped_size(1) == 8 << 20
    assert capped_size(8 << 20) == 8 << 20
    assert capped_size((8 << 20) + 1) == 16 << 20


def test_graph_ops_and_blobs_change_the_model_as_their_statements_do() -> None:
    from model import blob_lengths, blob_query
    from rules.recovery import graph_apply, graph_statement

    model = Model(graph={})
    assert graph_statement(model, "gn_insert", 3, -1) == "CREATE (:GN {id: 3, v: -1})"
    graph_apply(model, "gn_insert", 3, -1)
    assert graph_statement(model, "gn_insert", 3, 5) is None
    graph_apply(model, "gn_set", 3, 5)
    graph_apply(model, "gn_set", 4, 5)
    assert model.graph == {3: 5} and model.updated_since_checkpoint
    graph_apply(model, "gn_delete", 3, 0)
    assert model.graph == {}
    model.next_blob_id = 9
    ids = model.take_blob_ids(2)
    assert ids == range(9, 11) and model.next_blob_id == 11
    assert blob_lengths(ids, 3) == {9: 6, 10: 9}
    query = blob_query(ids, 3)
    assert "range(9, 10)" in query and "repeat(CAST(i AS STRING) + '-', 3)" in query


def overlapped(**writes) -> Others:
    """One open transaction that wrote `writes`."""
    return Others(open=[Writes(**writes)])


def test_writes_conflict_with_what_overlapping_transactions_wrote() -> None:
    view = small_model()
    own = Writes()
    assert conflict("set_age", 1, 5, view, own, overlapped(updated={(1, 0)})) == WW_UPDATE
    assert conflict("set_age", 2, 5, view, own, overlapped(updated={(1, 0)})) is None
    assert conflict("delete", 1, None, view, own, overlapped(deleted={1})) == WW_DELETE
    # Update and delete of one row do not conflict, in either order.
    assert conflict("set_age", 1, 5, view, own, overlapped(deleted={1})) is None
    assert conflict("delete", 1, None, view, own, overlapped(updated={(1, 0)})) is None
    # A row the transaction does not see matches nothing.
    assert conflict("set_age", 9, 5, view, own, overlapped(updated={9})) is None
    # DETACH DELETE deletes the edges it sees.
    deleted = overlapped(edges_deleted={("knows", 1, 2)})
    assert conflict("delete", 2, None, view, own, deleted) == WW_DELETE


def test_catalog_conflicts_are_per_entry_and_staged_inserts_block_column_changes() -> None:
    view, own = small_model(), Writes()
    assert conflict("create_table", 1, None, view, own, overlapped(catalog={"T1"})) == WW_CATALOG
    assert conflict("create_table", 2, None, view, own, overlapped(catalog={"T1"})) is None
    assert conflict("add_column", "c1", 1, view, own, overlapped(catalog={PERSON})) == WW_CATALOG
    assert conflict("insert", 7, 1, view, own, overlapped(catalog={PERSON})) == WW_CATALOG
    assert conflict("add_column", "c1", 1, view, own, overlapped(staged={PERSON})) == WW_CATALOG
    # Inserts that are already committed no longer block the change.
    committed = Others(committed=[Writes(staged={PERSON})])
    assert conflict("add_column", "c1", 1, view, own, committed) is None
    assert conflict("drop_table", 1, None, view, own, overlapped(staged={"T1"})) == WW_CATALOG


def test_overlaps_the_engine_gets_wrong_are_gated() -> None:
    view, own = small_model(), Writes()
    assert conflict("insert", 7, 1, view, own, overlapped(inserted={7})) == GATED_INSERT
    assert conflict("knows", 1, 2, view, own, overlapped(deleted={2})) == GATED_EDGE
    added = overlapped(edges_added={("knows", 2, 1)})
    assert conflict("delete", 1, None, view, own, added) == GATED_EDGE


def test_deleting_edges_conflicts_only_over_the_snapshots() -> None:
    view, own = small_model(), Writes()
    view.ck[(1, 2)] = 1
    deleted = overlapped(edges_deleted={("ck", 1, 2)})
    assert conflict("unck", 1, 2, view, own, deleted) == WW_DELETE
    own.own_edges[("ck", 1, 2)] = 1  # the one it sees is its own
    assert conflict("unck", 1, 2, view, own, deleted) is None


def test_commit_replay_removes_only_the_edges_the_transaction_saw() -> None:
    view, committed = small_model(), small_model()
    view.ck[(1, 2)] = 2
    committed.ck[(1, 2)] = 3  # another transaction committed one more since the snapshot
    replay(committed, "unck", 1, 2, seen_by(view, "unck", 1, 2))
    assert committed.ck[(1, 2)] == 1
    view.ck2[(1, 2, 0)] = 1
    committed.ck2[(1, 2, 0)] = 1
    committed.ck2[(1, 2, 5)] = 1
    replay(committed, "unck2", 1, 2, seen_by(view, "unck2", 1, 2))
    assert committed.ck2 == {(1, 2, 5): 1}


def test_writer_states_hold_a_prefix_of_each_writers_edges() -> None:
    plans = [(1, [10, 11]), (2, [20])]
    states = writer_states((0, 0), plans)
    fp = edge_fingerprint
    assert (1, fp(1, 10)) in states
    assert (2, fp(1, 10) + fp(2, 20)) in states
    assert (3, fp(1, 10) + fp(1, 11) + fp(2, 20)) in states
    assert (1, fp(1, 11)) not in states  # the second edge without the first
    assert len(states) == 6


def test_writer_reads_must_be_consistent_and_never_go_back() -> None:
    model = small_model()
    base = ck_state(model)
    plans = [(1, [10]), (2, [20])]
    first = small_model()
    first.ck[(1, 10)] += 1
    final = small_model()
    final.ck[(1, 10)] += 1
    final.ck[(2, 20)] += 1
    check_writer_reads([[base, ck_state(first), ck_state(final)]], base, plans, ck_state(final))
    with pytest.raises(AssertionError, match="went back"):
        reads = [ck_state(first), base, ck_state(final)]
        check_writer_reads([reads], base, plans, ck_state(final))
    torn = list(ck_state(first))
    torn[2] = ((0,),)  # one query of the transaction missed the edge the others saw
    with pytest.raises(AssertionError, match="no prefix"):
        check_writer_reads([[tuple(torn), ck_state(final)]], base, plans, ck_state(final))
    # With the gate on, only that query-to-query check is off.
    check_writer_reads([[tuple(torn), ck_state(final)]], base, plans, ck_state(final), False)
    with pytest.raises(AssertionError):
        check_writer_reads([[ck_state(first)]], base, plans, ck_state(final))


def test_a_crash_keeps_every_returned_commit_as_a_prefix_of_each_writer() -> None:
    targets = [10, 11, 12]
    assert check_survivors(0, targets, Counter(), 0) == 0
    assert check_survivors(0, targets, Counter({10: 1, 11: 1}), 1) == 2  # one that never returned
    with pytest.raises(AssertionError, match="only 1 survived"):
        check_survivors(0, targets, Counter({10: 1}), 2)
    with pytest.raises(AssertionError, match="not its first"):
        check_survivors(0, targets, Counter({11: 1}), 1)  # the first commit lost, the second kept


def test_acks_count_finished_lines_and_report_errors() -> None:
    acked, errors = read_acks(["0\n", "1\n", "0\n", "1 Runtime exception: boom\n", "0"])
    assert acked == {0: 2, 1: 1}
    assert errors == ["1 Runtime exception: boom"]


def test_an_update_stays_with_the_row_its_snapshot_saw() -> None:
    committed = small_model()
    apply(committed, "insert", 5, 1)
    updater = committed.fork()
    seen = seen_by(updater, "set_age", 5, 100)
    apply(committed, "delete", 5, None)
    apply(committed, "insert", 5, 7)  # another row under the same id
    replay(committed, "set_age", 5, 100, seen)
    assert committed.persons[5]["age"] == 7
    replay(committed, "set_age", 5, 50, seen_by(committed, "set_age", 5, 50))
    assert committed.persons[5]["age"] == 50


def test_updating_a_reinserted_row_does_not_conflict_with_an_update_of_the_old_one() -> None:
    view = small_model()
    apply(view, "insert", 5, 1)
    old = Writes()
    record(old, "set_age", 5, 100, view)
    assert conflict("set_age", 5, 0, view, Writes(), Others(open=[old])) == WW_UPDATE
    apply(view, "delete", 5, None)
    apply(view, "insert", 5, 7)
    assert conflict("set_age", 5, 0, view, Writes(), Others(open=[old])) is None


def rollback_session(model, *open_writes, commit_log=()):
    txns = {i: Txn("write", model.fork(), writes=w) for i, w in enumerate(open_writes)}
    return SimpleNamespace(model=model, txns=txns, commit_log=list(commit_log))


def test_a_rollback_rewinds_serial_only_beside_an_insert_into_the_same_table() -> None:
    model = small_model()
    txn = Txn("write", model.fork(), writes=Writes(staged={"T0"}))
    Session.rolled_back(rollback_session(model, Writes(staged={"T1"})), txn)
    assert not model.serial_rewound
    Session.rolled_back(rollback_session(model, Writes(staged={"Person"})), txn)
    assert not model.serial_rewound
    Session.rolled_back(rollback_session(model, Writes(staged={"T0"})), txn)
    assert model.serial_rewound
    model = small_model()
    earlier = rollback_session(model, commit_log=[(0, Writes(staged={"T0"}))])
    Session.rolled_back(earlier, txn)
    assert not model.serial_rewound  # committed before the transaction began, ids already moved on
    model = small_model()
    later = rollback_session(model, commit_log=[(1, Writes(staged={"T0"}))])
    Session.rolled_back(later, Txn("write", model.fork(), writes=Writes(staged={"T0"})))
    assert model.serial_rewound


def committing_session(model, overlapped: bool):
    return SimpleNamespace(
        model=model,
        multi_writes=True,
        session_writer=lambda: 2 if overlapped else None,
        commit_log=[],
        note_commit=lambda writes: None,
    )


def test_a_commit_sets_no_rollback_flag() -> None:
    model = small_model()
    view = model.fork()
    apply(view, "add_column", "c0", 1)
    txn = Txn("write", view, ops=[("add_column", "c0", 1, None)])
    SessionRules.commit_txn(committing_session(model, False), txn)
    assert "c0" in model.columns and not model.rolled_back_add and not model.rolled_back_macro


def test_a_macro_committed_beside_another_writer_keeps_later_drops_from_recovery() -> None:
    for overlapped in (False, True):
        model = small_model()
        view = model.fork()
        apply(view, "create_macro", 0, 1)
        txn = Txn("write", view, ops=[("create_macro", 0, 1, None)])
        SessionRules.commit_txn(committing_session(model, overlapped), txn)
        assert model.rolled_back_macro == overlapped


def test_deletes_and_edges_replay_only_what_the_snapshot_matched() -> None:
    committed = small_model()
    view = committed.fork()
    seen_delete = seen_by(view, "delete", 5, None)  # no row 5 in the snapshot: matches nothing
    seen_edge = seen_by(view, "knows", 5, 1)
    seen_update = seen_by(view, "set_age", 5, 100)
    apply(committed, "insert", 5, 7)
    replay(committed, "delete", 5, None, seen_delete)
    replay(committed, "knows", 5, 1, seen_edge)
    replay(committed, "set_age", 5, 100, seen_update)
    assert committed.persons[5]["age"] == 7
    assert not committed.knows[(5, 1, 0)]
    replay(committed, "delete", 5, None, seen_by(committed, "delete", 5, None))
    assert 5 not in committed.persons


def test_inserting_a_key_the_transaction_inserted_and_deleted_is_gated() -> None:
    view, own = small_model(), Writes()
    record(own, "insert", 5, 1, view)
    apply(view, "insert", 5, 1)
    record(own, "delete", 5, None, view)
    apply(view, "delete", 5, None)
    assert conflict("insert", 5, 1, view, own, Others()) == GATED_REINSERT
    assert conflict("insert", 6, 1, view, own, Others()) is None


def test_a_key_committed_since_the_snapshot_is_gated_once_the_transaction_has_new_rows() -> None:
    view, own = small_model(), Writes()
    others = Others(committed=[Writes(inserted={7})])
    assert conflict("set_age", 7, 1, view, own, others) is None
    record(own, "insert", 6, 1, view)
    for kind, a, b in (("set_age", 7, 1), ("delete", 7, None), ("knows", 7, 1), ("knows", 1, 7)):
        assert conflict(kind, a, b, view, own, others) == GATED_HIDDEN
    assert conflict("set_age", 1, 1, view, own, others) is None


def test_a_key_replaced_since_the_snapshot_is_gated_once_the_transaction_has_new_rows() -> None:
    view, own = small_model(), Writes()
    record(own, "insert", 6, 1, view)
    replaced = Others(committed=[Writes(deleted={1}), Writes(inserted={1})])
    assert conflict("set_age", 1, 1, view, own, replaced) == GATED_HIDDEN
    assert conflict("delete", 1, None, view, own, replaced) == GATED_HIDDEN
    assert conflict("set_age", 1, 1, view, Writes(), replaced) is None
    deleted_only = Others(committed=[Writes(deleted={1})])
    assert conflict("set_age", 1, 1, view, own, deleted_only) is None


def test_the_read_battery_waits_for_a_key_inserted_since_the_snapshot() -> None:
    session = SimpleNamespace(
        options=Options(skipped={GATED_HIDDEN}), commit_log=[(1, Writes(inserted={7}))]
    )
    new_rows = Txn("write", small_model(), writes=Writes(staged={PERSON}))
    assert not SessionRules.lookups_safe(session, new_rows)
    assert SessionRules.lookups_safe(session, Txn("write", small_model()))
    later = Txn("write", small_model(), start=1, writes=new_rows.writes)
    assert SessionRules.lookups_safe(session, later)


def test_a_snapshot_of_a_dropped_table_holds_back_checkpoints() -> None:
    model = small_model()
    snapshot = model.fork()
    snapshot.tables[0] = []
    session = SimpleNamespace(
        options=Options(skipped={"snapshot_dropped_table"}),
        model=model,
        txns={2: Txn("read", snapshot)},
        config={"checkpoint_threshold": 0},
    )
    assert not Session.checkpoint_keeps_snapshots(session)
    assert not Session.drop_table_safe(session, 0)
    assert Session.drop_table_safe(session, 1)
    session.config = {"checkpoint_threshold": -1}
    assert Session.drop_table_safe(session, 0)
    model.tables[0] = []
    assert Session.checkpoint_keeps_snapshots(session)
