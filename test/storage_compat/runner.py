"""Runs statements or checks on a database with a ladybug Python package.

Run by runners.Build under uv: ``python runner.py <execute|read> <db>``, with a JSON spec on
stdin. "execute" runs every statement of each named section and reports the first error and the
indices of the statements that failed, running the rest. "read" runs each query and prints its
rows, or "<error> <message>" for a query that failed. Rows are read one by one and converted to the
Arrow types of the query's columns, so that every package returns the same Python values whatever
its own conversions; the build's Arrow export supplies only those types.
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


def rows(conn, query: str) -> list[list]:
    schema = conn.execute(f"{query} LIMIT 0").get_as_arrow().schema
    result = conn.execute(query)
    found = []
    while result.has_next():
        found.append(result.get_next())
    if not found:
        return []
    columns = [pa.array(values, type=f.type).to_pylist() for values, f in zip(zip(*found), schema)]
    return [list(row) for row in zip(*columns)]


def read(conn, queries: dict[str, str]) -> dict:
    results = {}
    for name, query in queries.items():
        try:
            results[name] = rows(conn, query)
        except Exception as e:
            results[name] = f"<error> {e}"
    return results


def main() -> None:
    mode, path = sys.argv[1:]
    spec = json.load(sys.stdin)
    db = lb.Database(path, read_only=spec["read_only"])
    conn = lb.Connection(db)
    results = execute(conn, spec["sections"]) if mode == "execute" else read(conn, spec["queries"])
    conn.close()
    db.close()
    # Decimal, UUID, dates, intervals and bytes become their str() form on every side alike.
    print(json.dumps(results, default=str))


if __name__ == "__main__":
    main()
