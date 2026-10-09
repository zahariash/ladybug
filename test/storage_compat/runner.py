"""Runs statements or checks on a database with a ladybug Python package.

Run by runners.Build under uv: ``python runner.py <execute|read> <db>``, with a JSON spec on
stdin. "execute" runs every statement of each named section and reports the first error and the
indices of the statements that failed, running the rest. "read" runs each query and prints its
rows, or "<error> <message>" for a query that failed. Rows are converted to the Arrow types of the
query's columns, so that every package returns the same Python values whatever its own conversions.
Release packages read rows one by one. Local builds, whose C-API backend is slow at that, read
through the build's Arrow export and fall back to rows one by one when Arrow cannot convert a
result.
"""

import json
import sys

import ladybug as lb
import pyarrow as pa


def execute(conn, sections: dict[str, list[str]]) -> dict:
    results = {}
    for name, statements in sections.items():
        results[name] = {"error": None, "failed": []}
        for i, statement in enumerate(statements):
            try:
                conn.execute(statement)
            except Exception as e:
                results[name]["error"] = results[name]["error"] or str(e)
                results[name]["failed"].append(i)
    return results


def rows(conn, query: str, arrow_first: bool) -> list[list]:
    if arrow_first:
        try:
            return _pylist(conn.execute(query).get_as_arrow())
        except pa.ArrowException:
            pass
    return _pylist(_row_by_row(conn, query))


def _row_by_row(conn, query: str) -> pa.Table:
    schema = conn.execute(f"{query} LIMIT 0").get_as_arrow().schema
    result = conn.execute(query)
    found = []
    while result.has_next():
        found.append(result.get_next())
    columns = list(zip(*found)) if found else [[] for _ in schema]
    return pa.table([pa.array(c, type=f.type) for c, f in zip(columns, schema)], schema=schema)


def _pylist(table: pa.Table) -> list[list]:
    return [list(row) for row in zip(*(column.to_pylist() for column in table.columns))]


def read(conn, queries: dict[str, str], arrow_first: bool) -> dict:
    results = {}
    for name, query in queries.items():
        try:
            results[name] = rows(conn, query, arrow_first)
        except Exception as e:
            results[name] = f"<error> {e}"
    return results


def main() -> None:
    mode, path = sys.argv[1:]
    spec = json.load(sys.stdin)
    db = lb.Database(path, read_only=spec["read_only"])
    conn = lb.Connection(db)
    if mode == "execute":
        results = execute(conn, spec["sections"])
    else:
        results = read(conn, spec["queries"], spec["arrow_first"])
    conn.close()
    db.close()
    # Decimal, UUID, dates, intervals and bytes become their str() form on every side alike.
    print(json.dumps(results, default=str))


if __name__ == "__main__":
    main()
