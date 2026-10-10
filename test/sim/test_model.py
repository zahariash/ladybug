"""The model's bookkeeping, without an engine."""

from __future__ import annotations

import math
from collections import Counter

from model import Doc, Model, canonical
from strategies import BULK_ID_START


def model_with(persons: dict, knows: dict | None = None) -> Model:
    model = Model()
    model.add_persons({id: model.new_person(age=age) for id, age in persons.items()})
    model.knows = Counter(knows or {})
    return model


def test_canonical_makes_nan_comparable_and_lists_hashable() -> None:
    assert canonical((1, math.nan, [1.0, [2.0]])) == (1, "NaN", (1.0, (2.0,)))
    assert canonical((math.nan,)) == canonical((float("nan"),))


def test_fork_copies_what_transactions_change_and_shares_bulk_rows() -> None:
    model = model_with({1: 10, BULK_ID_START: 20}, {(1, 1, 0): 1})
    model.tables[0] = [5]
    model.macros[0] = 1
    fork = model.fork()
    fork.persons[1]["age"] = 99
    fork.knows[(1, 1, 0)] += 1
    fork.tables[0].append(6)
    fork.macros[1] = 2
    fork.columns["c0"] = 0
    assert model.persons[1]["age"] == 10
    assert model.knows == Counter({(1, 1, 0): 1})
    assert model.tables == {0: [5]}
    assert model.macros == {0: 1}
    assert "c0" not in model.columns
    assert fork.persons[BULK_ID_START] is model.persons[BULK_ID_START]


def test_forget_removes_a_person_and_every_edge_touching_it() -> None:
    model = model_with({1: 1, 2: 2, 3: 3}, {(1, 2, 0): 1, (2, 3, 0): 2, (2, 2, 5): 1, (1, 3, 0): 1})
    model.forget(2)
    assert set(model.persons) == {1, 3}
    assert model.knows == Counter({(1, 3, 0): 1})


def test_set_since_moves_every_edge_between_two_persons_to_the_new_value() -> None:
    model = model_with({1: 1, 2: 2}, {(1, 2, 5): 2, (1, 2, None): 1, (2, 1, 5): 1})
    model.set_since(1, 2, 7)
    assert model.knows == Counter({(1, 2, 7): 3, (2, 1, 5): 1})
    model.set_since(2, 2, 9)
    assert model.knows == Counter({(1, 2, 7): 3, (2, 1, 5): 1})


def test_persons_ever_stays_set_after_deletes() -> None:
    model = Model()
    model.add_persons({})
    assert not model.persons_ever
    model.add_persons({1: model.new_person()})
    model.forget(1)
    assert model.persons_ever and not model.persons


def test_bulk_ids_do_not_overlap() -> None:
    model = Model()
    first, second = model.take_bulk_ids(3), model.take_bulk_ids(2)
    assert list(first) == [BULK_ID_START, BULK_ID_START + 1, BULK_ID_START + 2]
    assert second.start == first.stop


def test_new_persons_get_column_defaults() -> None:
    model = Model()
    model.columns["c0"] = 4
    assert model.new_person(age=1) == {"name": None, "age": 1, "score": None, "c0": 4}


def test_doc_text_is_null_without_words() -> None:
    assert Doc(None, None).text is None
    assert Doc([], None).text is None
    assert Doc(["apple", "grape"], None).text == "apple grape"
