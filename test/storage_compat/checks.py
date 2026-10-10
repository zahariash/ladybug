"""Checks generated from a database's catalog, and how two builds' results are compared.

Every node and rel table is read in full (rels keyed by their endpoints' primary keys), the
first, middle and last primary keys are looked up through the index, every sequence's currval()
is read, and the catalog's tables, comments, sequences, macros and indexes are listed.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from runners import Build, CompatError, is_error

KEY_TYPES = {"INT8", "INT16", "INT32", "INT64", "UINT8", "UINT16", "UINT32", "UINT64", "SERIAL"}
KEY_TYPES |= {"STRING"}  # primary keys looked up through the index


@dataclass
class Check:
    query: str
    may_fail: bool = False  # currval() fails for a sequence nextval() was never called on
    # Catalog listings whose columns and contents change between versions are compared only
    # between this build and the base commit.
    local_only: bool = False


def for_writer(checks: dict[str, Check], writer: Build) -> dict[str, Check]:
    return {name: c for name, c in checks.items() if not (c.local_only and writer.version)}


def value(var: str, name: str, type_: str) -> str:
    """How a property is read. INTERVAL goes through its text form: Arrow converts it to a
    duration with a months-to-days factor that changed between releases. UNION, which pyarrow
    cannot build from Python values, is read as its member's name and its text form."""
    expr = f"{var}.`{name}`"
    kind = type_.upper()
    if kind == "INTERVAL":
        return f"CAST({expr} AS STRING)"
    if kind.startswith("UNION(") and kind.endswith(")"):
        return f"union_tag({expr}), CAST({expr} AS STRING)"
    return expr


def read(reader: Build, db: Path, checks: dict[str, Check], read_only: bool = True) -> dict:
    return reader.read(db, {name: check.query for name, check in checks.items()}, read_only)


def rows_of(build: Build, db: Path, queries: dict[str, str]) -> dict[str, list[list]]:
    results = build.read(db, queries)
    if errors := {name: rows for name, rows in results.items() if is_error(rows)}:
        raise CompatError(f"{build.name} could not plan the checks for {db}: {errors}")
    return results


def plan(build: Build, db: Path) -> dict[str, Check]:
    """Checks for every table, sequence and catalog object in the database."""
    catalog = rows_of(
        build,
        db,
        {
            "tables": "CALL show_tables() RETURN name, type",
            "sequences": "CALL show_sequences() RETURN name",
        },
    )
    tables = dict(catalog["tables"])
    info = {f"columns {t}": f"CALL table_info('{t}') RETURN name, type" for t in tables}
    info |= {
        f"pk {t}": f"CALL table_info('{t}') WHERE `primary key` RETURN name, type"
        for t, k in tables.items()
        if k == "NODE"
    }
    info |= {
        f"pairs {t}": f"CALL show_connection('{t}') RETURN `source table name`, "
        "`destination table name`, `source table primary key`, `destination table primary key`"
        for t, kind in tables.items()
        if kind == "REL"
    }
    info = rows_of(build, db, info)
    primary_keys = {t: info[f"pk {t}"][0] for t, kind in tables.items() if kind == "NODE"}
    keys = rows_of(
        build,
        db,
        {
            t: f"MATCH (n:`{t}`) RETURN n.`{pk}`"
            for t, (pk, pk_type) in primary_keys.items()
            if pk_type.upper() in KEY_TYPES
        },
    )

    checks = {}
    for table, (pk, _) in primary_keys.items():
        columns = info[f"columns {table}"]
        properties = [(pk, dict(columns)[pk])] + [(n, t) for n, t in columns if n != pk]
        query = f"MATCH (n:`{table}`) RETURN " + ", ".join(value("n", n, t) for n, t in properties)
        checks[f"node {table}"] = Check(query)
        checks |= lookups(table, pk, sorted(row[0] for row in keys.get(table, [])))
    for table, kind in tables.items():
        if kind != "REL":
            continue
        properties = info[f"columns {table}"]
        for src, dst, src_pk, dst_pk in info[f"pairs {table}"]:
            values = [f"a.`{src_pk}`", f"b.`{dst_pk}`"] + [value("r", n, t) for n, t in properties]
            query = f"MATCH (a:`{src}`)-[r:`{table}`]->(b:`{dst}`) RETURN " + ", ".join(values)
            checks[f"rel {table} {src}->{dst}"] = Check(query)
    for (sequence,) in catalog["sequences"]:
        checks[f"sequence {sequence}"] = Check(f"RETURN currval('{sequence}')", may_fail=True)
    checks["tables"] = Check("CALL show_tables() RETURN name, type")
    checks["comments"] = Check("CALL show_tables() RETURN name, comment", local_only=True)
    checks["sequences"] = Check(
        "CALL show_sequences() RETURN name, `start value`, increment, `min value`, `max value`, cycle",
        local_only=True,
    )
    checks["macros"] = Check("CALL show_macros() RETURN name, definition", local_only=True)
    checks["indexes"] = Check(
        "CALL show_indexes() RETURN table_name, index_name, index_type, property_names",
        local_only=True,
    )
    return checks


def lookups(table: str, pk: str, keys: list) -> dict[str, Check]:
    """Lookups of the first, middle and last primary keys, which go through the index."""
    if not keys:
        return {}
    picks = {"first": keys[0], "middle": keys[len(keys) // 2], "last": keys[-1]}
    checks = {}
    for position, key in picks.items():
        literal = _string_literal(key) if isinstance(key, str) else str(key)
        query = f"MATCH (n:`{table}`) WHERE n.`{pk}` = {literal} RETURN count(*)"
        checks[f"lookup {table} {position}"] = Check(query)
    return checks


def _string_literal(value: str) -> str:
    escaped = value.replace("\\", "\\\\").replace("'", "\\'").replace("\n", "\\n")
    return f"'{escaped}'"


def compare(expected: dict, actual: dict, checks: dict[str, Check]) -> dict[str, str]:
    """The checks whose results differ, by name, each with a sample of the rows that differ."""
    diffs = {}
    for name, check in checks.items():
        want, got = expected.get(name), actual.get(name)
        if check.may_fail and is_error(want) and is_error(got):
            continue
        if is_error(want) or is_error(got) or None in (want, got):
            diffs[name] = f"expected {_size(want)}, got {_size(got)}"
            continue
        want_rows = Counter(json.dumps(row, sort_keys=True) for row in want)
        got_rows = Counter(json.dumps(row, sort_keys=True) for row in got)
        if want_rows != got_rows:
            missing = list((want_rows - got_rows).elements())[:3]
            extra = list((got_rows - want_rows).elements())[:3]
            diffs[name] = (
                f"{len(want)} rows expected, {len(got)} read; missing {missing}, unexpected {extra}"
            )
    return diffs


def _size(value) -> str:
    return str(value) if value is None or is_error(value) else f"{len(value)} rows"
