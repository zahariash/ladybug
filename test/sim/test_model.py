"""The model's bookkeeping, without an engine."""

from __future__ import annotations

import math
import random
from collections import Counter

from model import Big, Doc, Model, canonical, filler, two_hops, walk_ends
from strategies import BULK_ID_START, CSR_BANDS, CSR_ROWS, CSR_SECOND_GROUP


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


def test_a_discarded_transaction_flags_columns_it_added_until_a_checkpoint() -> None:
    model = Model()
    view = model.fork()
    view.add_column("c1", 3)
    model.discarded(view)
    assert model.rolled_back_add and "c1" not in model.columns
    model.checkpointed()
    assert not model.rolled_back_add


def test_dropping_then_adding_a_column_marks_the_gap_until_a_checkpoint() -> None:
    model = Model()
    model.add_column("c1", 1)
    assert not model.added_after_drop
    model.drop_column("c1")
    model.add_column("c2", 2)
    assert model.dropped_since_checkpoint and model.added_after_drop
    model.checkpointed()
    assert not (model.dropped_since_checkpoint or model.added_after_drop)


def test_dropping_a_middle_column_marks_the_gap_until_a_checkpoint() -> None:
    model = Model()
    for column in ("c0", "c1", "c2"):
        model.add_column(column, 0)
    model.drop_column("c2")
    assert not model.middle_dropped
    model.drop_column("c0")
    assert model.middle_dropped
    model.checkpointed()
    assert not model.middle_dropped


def enumerated_walk_ends(edges: list, src: int, length: int) -> list:
    """The end node of every walk of `length` edges from src, by enumerating the walks."""
    walks = [[src]]
    for _ in range(length):
        walks = [walk + [d] for walk in walks for s, d in edges if s == walk[-1]]
    return [walk[-1] for walk in walks]


def test_walk_ends_counts_every_walk() -> None:
    edges = [(1, 2), (2, 3), (1, 3), (3, 3), (2, 3), (1, 2)]
    for lo, hi in [(1, 1), (1, 2), (2, 3), (1, 3)]:
        expected = Counter(e for n in range(lo, hi + 1) for e in enumerated_walk_ends(edges, 1, n))
        assert walk_ends(Counter(edges), 1, lo, hi) == expected


def test_two_hops_counts_each_pair_of_edges() -> None:
    edges = [(1, 2), (2, 3), (2, 3), (1, 2), (4, 1)]
    expected = Counter((a, m, b) for a, m in edges for x, b in edges if x == m and a in (1, 4))
    assert two_hops(Counter(edges), [1, 4]) == expected


def test_fork_copies_ck2_and_pairs_ignore_w() -> None:
    model = Model()
    model.ck2[(1, 2, 0)] += 1
    model.ck2[(1, 2, 5)] += 2
    fork = model.fork()
    fork.ck2[(3, 4, 0)] += 1
    assert (3, 4, 0) not in model.ck2
    assert model.ck2_pairs() == Counter({(1, 2): 3})


def test_csr_bands_stay_inside_c() -> None:
    first, second = CSR_BANDS
    assert first[:40] == list(range(40)) and CSR_SECOND_GROUP - 1 in first
    assert second[-1] == CSR_ROWS - 1 and CSR_SECOND_GROUP + 2049 in second
    assert CSR_SECOND_GROUP + 4095 not in second


def test_big_aggregates_and_top_match_the_rows() -> None:
    rng = random.Random(7)
    for rows in (0, 1, 999, 1000, 2500):
        big = Big(rows)
        ids = range(rows)
        for id in rng.sample(ids, min(rows, 40)):
            if rng.random() < 0.5:
                big.deleted.add(id)
            else:
                big.updated[id] = rng.choice([None, -1, 0, 999, 1000, rng.randint(-5, 2000)])
        table = {id: big.value(id) for id in ids if id not in big.deleted}
        values = [v for v in table.values() if v is not None]
        assert big.count() == len(table)
        assert big.sum() == sum(values)
        assert big.min_max() == ((min(values), max(values)) if values else (None, None))
        ranked = sorted(((id, v) for id, v in table.items() if v is not None), key=lambda r: (-r[1], r[0]))
        for k in (1, 5, 2049):
            assert big.top(k) == ranked[:k]


def test_forget_c_drops_the_row_and_its_edges() -> None:
    model = Model(big=Big(100))
    model.big.updated[5] = 1
    model.ck.update([(5, 6), (6, 5), (7, 8)])
    model.ck2[(5, 9, 0)] += 1
    model.forget_c(5)
    assert not model.big.exists(5) and 5 not in model.big.updated
    assert model.ck == Counter({(7, 8): 1}) and not model.ck2
    assert model.fork().big.deleted == {5} and filler(1005) == 5
