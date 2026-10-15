"""The search oracles in search.py, without an engine."""

from __future__ import annotations

import math

import pytest
from search import create_statement, drop_statement, rank, unstemmed_match


def test_rank_matches_each_metrics_distance() -> None:
    a, b = (1, 0, 0), (1, 1, 0)
    assert rank("l2sq", a, b) == 1
    assert rank("l2", a, (1, 3, 4)) == 5
    assert rank("cosine", a, b) == pytest.approx(1 - 1 / math.sqrt(2))
    assert rank("cosine", a, (2, 0, 0)) == pytest.approx(0)


def test_rank_puts_the_larger_dot_product_nearer() -> None:
    query = (1, 1, 1)
    assert rank("dotproduct", query, (5, 5, 5)) < rank("dotproduct", query, (-1, -1, -1))


@pytest.mark.parametrize(
    "query, words, found",
    [
        (["apple"], ["Apple", "grape"], True),
        (["APPLE"], ["apple"], True),
        (["apple"], ["Apples"], False),
        (["the"], ["the", "apple"], False),
        (["the", "grape"], ["the", "grape"], True),
        (["café"], ["CAFÉ"], True),
        (["apple"], None, False),
    ],
)
def test_unstemmed_match_lowercases_and_drops_stopwords(query, words, found) -> None:
    assert unstemmed_match(query, words) is found


def test_index_statements_carry_their_option() -> None:
    assert "stemmer := 'none'" in create_statement("doc_fts", "none")
    assert "metric := 'l2'" in create_statement("doc_vec", "l2")
    assert create_statement("doc_fts", "english", "fresh").startswith(
        "CALL CREATE_FTS_INDEX('Doc', 'fresh'"
    )
    assert drop_statement("doc_vec") == "CALL DROP_VECTOR_INDEX('Doc', 'doc_vec')"
