"""Reductions below the level of steps (reduction.py), on canned statements and fake oracles."""

from __future__ import annotations

import json
import re

from reduction import (
    ddmin,
    drop_column,
    reduce_statements,
    split_top,
    statement_variants,
    table_columns,
)


def test_ddmin_finds_the_items_a_failure_needs() -> None:
    assert ddmin(list(range(30)), lambda items: {4, 17} <= set(items)) == [4, 17]
    assert ddmin([1], lambda items: True) == [1]


def test_split_top_keeps_brackets_and_quotes_together() -> None:
    text = "a INT64, b STRUCT(x INT64, y STRING), c STRING DEFAULT 'p, q', d INT64[]"
    assert split_top(text) == [
        "a INT64",
        " b STRUCT(x INT64, y STRING)",
        " c STRING DEFAULT 'p, q'",
        " d INT64[]",
    ]


def test_table_columns_ends_at_the_matching_parenthesis() -> None:
    query = "CREATE NODE TABLE L(id INT64, c INT64, PRIMARY KEY(id)) PARTITION BY LIST (c)"
    table, before, items, after = table_columns(query)
    assert table == "L" and before == "CREATE NODE TABLE L(" and after == ") PARTITION BY LIST (c)"
    assert [item.strip() for item in items] == ["id INT64", "c INT64", "PRIMARY KEY(id)"]
    assert table_columns("MATCH (n) RETURN n") is None


def test_statement_variants_drop_defaults_and_halve_ranges() -> None:
    table = "CREATE NODE TABLE T(id INT64 PRIMARY KEY, a INT64 DEFAULT -2, s STRING DEFAULT 'x')"
    assert statement_variants(table) == [
        "CREATE NODE TABLE T(id INT64 PRIMARY KEY, a INT64, s STRING DEFAULT 'x')",
        "CREATE NODE TABLE T(id INT64 PRIMARY KEY, a INT64 DEFAULT -2, s STRING)",
    ]
    assert statement_variants("UNWIND range(0, 100) AS i RETURN i") == [
        "UNWIND range(0, 50) AS i RETURN i"
    ]
    assert statement_variants("UNWIND range(0, 1) AS i RETURN i") == []


def test_drop_column_removes_it_wherever_the_table_is_defined_copied_or_set() -> None:
    order = ["id", "v", "s"]
    create = "CREATE NODE TABLE IF NOT EXISTS C(id INT64 PRIMARY KEY, v INT64, s STRING)"
    assert drop_column(create, "C", "v", order) == (
        "CREATE NODE TABLE IF NOT EXISTS C(id INT64 PRIMARY KEY, s STRING)"
    )
    copy = (
        "COPY C(id, v, s) FROM "
        "(UNWIND range(0, 9) AS i RETURN i, i % 1000, CASE WHEN i > 1 THEN 'a' END)"
    )
    assert drop_column(copy, "C", "s", order) == (
        "COPY C(id, v) FROM (UNWIND range(0, 9) AS i RETURN i, i % 1000)"
    )
    unlisted = "COPY C FROM (UNWIND range(0, 9) AS i RETURN i, i % 1000, 'x')"
    assert drop_column(unlisted, "C", "v", order) == (
        "COPY C FROM (UNWIND range(0, 9) AS i RETURN i, 'x')"
    )
    fill = "UNWIND range(0, 9) AS i CREATE (:C {id: i, v: i % 1000})"
    assert drop_column(fill, "C", "v", order) == "UNWIND range(0, 9) AS i CREATE (:C {id: i})"
    edge = "MATCH (a:C {id: $a}), (b:C {id: $b}) CREATE (a)-[:CK2 {w: 0}]->(b)"
    assert drop_column(edge, "CK2", "w", ["w"]) == (
        "MATCH (a:C {id: $a}), (b:C {id: $b}) CREATE (a)-[:CK2]->(b)"
    )
    other = "COPY CK FROM (UNWIND range(0, 9) AS i RETURN i, i)"
    assert drop_column(other, "C", "v", order) == other


def test_reduce_statements_keeps_only_what_the_failure_needs() -> None:
    head = {
        "op": "open",
        "config": {},
        "loads": [
            "CREATE NODE TABLE C(id INT64 PRIMARY KEY, v INT64)",
            "CREATE NODE TABLE CS(id STRING PRIMARY KEY)",
            "COPY C FROM (UNWIND range(0, 1000) AS i RETURN i, i)",
        ],
    }
    head["reopen_loads"] = head["loads"][:2]
    step = {
        "op": "execute",
        "conn": 0,
        "query": "MATCH (c:C) RETURN c.v",
        "params": {"ids": [5, 6, 7, 8]},
    }

    def fails(trace: list) -> bool:
        text = json.dumps(trace)
        # needs table C with v, at least 300 rows and the id 7
        bounds = [int(n) for n in re.findall(r"range\(0, (\d+)\)", text)]
        return (
            "C(id INT64 PRIMARY KEY, v INT64)" in text
            and bounds
            and bounds[0] >= 300
            and 7 in trace[1]["params"]["ids"]
        )

    reduced = reduce_statements([head, step], fails)
    assert reduced[0]["loads"] == [
        "CREATE NODE TABLE C(id INT64 PRIMARY KEY, v INT64)",
        "COPY C FROM (UNWIND range(0, 500) AS i RETURN i, i)",
    ]
    assert reduced[0]["reopen_loads"] == ["CREATE NODE TABLE C(id INT64 PRIMARY KEY, v INT64)"]
    assert reduced[1]["params"] == {"ids": [7]}
