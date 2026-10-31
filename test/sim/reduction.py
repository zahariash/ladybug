"""Reductions of a failing trace below the level of its steps, for replay.py --minimize: the setup
statements, columns, defaults, list parameters and range() bounds the failure does not need.
Each takes `fails(trace)` as the oracle and keeps a change only while it still holds."""

from __future__ import annotations

import re

DEFAULT = re.compile(
    r"\s+DEFAULT\s+(?:-?\d+(?:\.\d+)?\b|'(?:[^'\\]|\\.)*'|NULL\b|TRUE\b|FALSE\b)", re.I
)
RANGE = re.compile(r"\brange\(\s*(-?\d+)\s*,\s*(-?\d+)\s*\)", re.I)
TABLE = re.compile(
    r"\s*CREATE\s+(?:NODE|REL)\s+TABLE\s+(?:IF\s+NOT\s+EXISTS\s+)?`?(\w+)`?\s*\(", re.I
)
# Items of a table's definition that are not a column: the rel table's ends, keys, multiplicity.
NOT_A_COLUMN = re.compile(r"\s*(?:FROM\b|PRIMARY\s+KEY\b|(?:ONE|MANY)_(?:ONE|MANY)\b)", re.I)


def ddmin(items: list, fails) -> list:
    """Delta debugging: the smallest sublist of `items` for which `fails(sublist)` still holds."""
    n = 2
    while len(items) >= 2:
        chunk = max(1, len(items) // n)
        subsets = [items[i : i + chunk] for i in range(0, len(items), chunk)]
        for i in range(len(subsets)):
            complement = [item for j, part in enumerate(subsets) if j != i for item in part]
            if fails(complement):
                items, n = complement, max(n - 1, 2)
                break
        else:
            if n >= len(items):
                break
            n = min(n * 2, len(items))
    return items


def closing(text: str, start: int) -> int | None:
    """The index of the bracket that closes the one at `start`, quotes skipped."""
    depth, quote = 0, None
    for i in range(start, len(text)):
        ch = text[i]
        if quote:
            quote = None if ch == quote and text[i - 1] != "\\" else quote
        elif ch in "'\"`":
            quote = ch
        elif ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1
            if depth == 0:
                return i
    return None


def split_top(text: str) -> list[str]:
    """`text` split at the commas outside brackets and quotes."""
    parts, depth, quote, start = [], 0, None, 0
    for i, ch in enumerate(text):
        if quote:
            quote = None if ch == quote and text[i - 1] != "\\" else quote
        elif ch in "'\"`":
            quote = ch
        elif ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1
        elif ch == "," and depth == 0:
            parts.append(text[start:i])
            start = i + 1
    return parts + [text[start:]]


def name_of(item: str) -> str:
    """The name of a column definition or of a property map entry."""
    return re.split(r"[\s:]", item.strip(), maxsplit=1)[0].strip("`")


def table_columns(query: str) -> tuple[str, str, list[str], str] | None:
    """A CREATE TABLE statement as its table, the text before its definition list, the list's
    items and the text from its closing parenthesis on."""
    head = TABLE.match(query)
    end = closing(query, head.end() - 1) if head else None
    if end is None:
        return None
    return head[1], query[: head.end()], split_top(query[head.end() : end]), query[end:]


def is_column(item: str) -> bool:
    return not NOT_A_COLUMN.match(item) and "PRIMARY KEY" not in item.upper()


def without(items: list[str], index: int) -> str:
    return ",".join(items[:index] + items[index + 1 :]).strip()


def drop_column(query: str, table: str, column: str, order: list[str]) -> str:
    """`query` without `column` of `table` (whose columns are `order`): its definition, its place
    in a COPY's column list or, without a list, its position, with the matching item of the
    COPY subquery's RETURN, and its entry in property maps of the table's nodes or rels."""
    if (definition := table_columns(query)) and definition[0] == table:
        _, before, items, after = definition
        names = [name_of(item) for item in items]
        if column in names:
            return before + without(items, names.index(column)) + after
    copy = re.match(rf"\s*COPY\s+`?{table}`?\s*(\(|FROM\b)", query, re.I)
    if copy:
        query = drop_copied_column(query, copy, column, order)
    for entry in reversed(list(re.finditer(rf":\s*`?{table}`?(\s*)\{{", query))):
        end = closing(query, entry.end() - 1)
        items = split_top(query[entry.end() : end]) if end else []
        names = [name_of(item) for item in items]
        if column in names:
            rest = without(items, names.index(column))
            if rest:
                query = query[: entry.end()] + rest + query[end:]
            else:  # the map's last entry: the map goes
                query = query[: entry.start(1)] + query[end + 1 :]
    return query


def drop_copied_column(query: str, copy: re.Match, column: str, order: list[str]) -> str:
    """A COPY into the table without `column`: its place in the COPY's column list (or, without
    a list, in `order`) and the matching item of the subquery's RETURN."""
    names, list_end = order, None
    if copy[1] == "(":
        list_end = closing(query, copy.end() - 1)
        names = [name_of(item) for item in split_top(query[copy.end() : list_end])]
    if column not in names:
        return query
    index = names.index(column)
    start = list_end + 1 if list_end else copy.start(1)
    source = re.compile(r"\s*FROM\s*\(", re.I).match(query, start)
    if source and (end := closing(query, source.end() - 1)):
        inner = query[source.end() : end]
        tops = [
            m for m in re.finditer(r"\bRETURN\b", inner, re.I) if depth_at(inner, m.start()) == 0
        ]
        items = split_top(inner[tops[-1].end() :]) if tops else []
        if len(items) == len(names):
            returned = source.end() + tops[-1].end()
            query = query[:returned] + " " + without(items, index) + query[end:]
    if list_end:  # before the subquery, so its offsets still hold
        items = split_top(query[copy.end() : list_end])
        query = query[: copy.end()] + without(items, index) + query[list_end:]
    return query


def depth_at(text: str, position: int) -> int:
    """How many brackets are open at `position`, quotes skipped."""
    depth, quote = 0, None
    for i, ch in enumerate(text[:position]):
        if quote:
            quote = None if ch == quote and text[i - 1] != "\\" else quote
        elif ch in "'\"`":
            quote = ch
        elif ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1
    return depth


def statement_variants(query: str) -> list[str]:
    """Simpler versions of a statement, one change each: a column without its DEFAULT, a range()
    with half as many values."""
    variants = [query[: m.start()] + query[m.end() :] for m in DEFAULT.finditer(query)]
    for m in RANGE.finditer(query):
        low, high = int(m[1]), int(m[2])
        if high - low >= 2:
            variants.append(query[: m.start(2)] + str(low + (high - low) // 2) + query[m.end(2) :])
    return variants


def statements(trace: list) -> list[str]:
    """Every statement the trace runs: the setup statements and the steps' queries."""
    head = trace[0]
    queries = head.get("loads", []) + head.get("reopen_loads", [])
    return queries + [op["query"] for op in trace[1:] if isinstance(op.get("query"), str)]


def map_statements(trace: list, change) -> list:
    """The trace with `change(query)` applied to every statement it runs."""
    head = trace[0]
    head = {**head, "loads": [change(q) for q in head.get("loads", [])]}
    if "reopen_loads" in trace[0]:
        head["reopen_loads"] = [change(q) for q in trace[0]["reopen_loads"]]
    steps = [
        {**op, "query": change(op["query"])} if isinstance(op.get("query"), str) else op
        for op in trace[1:]
    ]
    return [head] + steps


def with_statement(trace: list, where: tuple, new: str) -> list:
    """The trace with one statement replaced: a setup statement (("load", text)) wherever the
    open step runs it, or a step's (("step", index))."""
    if where[0] == "load":
        return map_statements(trace[:1], lambda q: new if q == where[1] else q) + trace[1:]
    i = where[1]
    return trace[:i] + [{**trace[i], "query": new}] + trace[i + 1 :]


def tables(trace: list) -> dict[str, tuple[list[str], list[str]]]:
    """The tables the trace creates: the columns that could go, and every column in order."""
    found = {}
    for query in statements(trace):
        if definition := table_columns(query):
            items = definition[2]
            droppable = [name_of(item) for item in items if is_column(item)]
            found[definition[0]] = droppable, [
                name_of(i) for i in items if not NOT_A_COLUMN.match(i)
            ]
    return found


def reduce_setup(trace: list, fails) -> list:
    """The setup statements the failure needs, by ddmin."""
    head, loads = trace[0], trace[0].get("loads", [])
    if len(loads) < 2:
        return trace

    def with_loads(kept: list) -> list:
        reopen = [q for q in head.get("reopen_loads", loads) if q in kept or q not in loads]
        return [{**head, "loads": kept, "reopen_loads": reopen}] + trace[1:]

    kept = ddmin(loads, lambda subset: fails(with_loads(subset)))
    if kept != loads:
        print(f"  setup: {len(kept)} of {len(loads)} statements needed", flush=True)
    return with_loads(kept)


def reduce_columns(trace: list, fails) -> list:
    """Drops each table column everywhere it is defined, copied or set, while the failure holds."""
    progress = True
    while progress:
        progress = False
        for table, (droppable, order) in tables(trace).items():
            for column in droppable:
                candidate = map_statements(
                    trace, lambda q, c=column: drop_column(q, table, c, order)
                )
                if candidate != trace and fails(candidate):
                    print(f"  not needed: column {table}.{column}", flush=True)
                    trace, progress = candidate, True
                    break
            if progress:
                break
    return trace


def reduce_statements(trace: list, fails) -> list:
    """Shrinks what the steps run while `fails(trace)` still holds: the setup statements and
    table columns the failure does not need, column defaults and halves of range() bounds in
    every statement, and elements of list parameters."""
    trace = reduce_columns(reduce_setup(trace, fails), fails)
    progress = True
    while progress:
        progress = False
        places = [("load", q) for q in trace[0]["loads"]]
        places += [
            ("step", i) for i, op in enumerate(trace) if i and isinstance(op.get("query"), str)
        ]
        for where in places:
            query = where[1] if where[0] == "load" else trace[where[1]]["query"]
            for variant in statement_variants(query):
                candidate = with_statement(trace, where, variant)
                if fails(candidate):
                    print(f"  {query[:70]}\n    -> {variant[:70]}", flush=True)
                    trace, progress = candidate, True
                    break
    for i in range(1, len(trace)):
        for name, value in (trace[i].get("params") or {}).items():
            if not isinstance(value, list) or len(value) < 2:
                continue

            def with_list(subset: list, i=i, name=name) -> list:
                params = {**trace[i]["params"], name: subset}
                return trace[:i] + [{**trace[i], "params": params}] + trace[i + 1 :]

            kept = ddmin(value, lambda subset: fails(with_list(subset)))
            if kept != value:
                print(f"  ${name}: {len(kept)} of {len(value)} values needed", flush=True)
                trace = with_list(kept)
    return trace
