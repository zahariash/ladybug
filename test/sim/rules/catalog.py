"""Rules that change the catalog: macros, tables, columns and Person's primary-key index."""

from __future__ import annotations

import hypothesis.strategies as st
from hypothesis.stateful import precondition, rule
from session import Session, enabled
from strategies import extra_columns, macro_ids, table_ids


class CatalogRules(Session):
    @enabled("macros")
    @rule(i=macro_ids, k=st.integers(-5, 5))
    def create_macro(self, i, k):
        query = f"CREATE MACRO m{i}(x) AS x + {k}"
        if i in self.model.macros:
            self.fails(query, None, "already exists")
            return
        self.ok(query)
        self.model.macros[i] = k

    @enabled("drop_macro")
    @rule(i=macro_ids)
    def drop_macro(self, i):
        if i not in self.model.macros:
            self.fails(f"DROP MACRO m{i}", None, "does not exist")
            return
        self.ok(f"DROP MACRO m{i}")
        del self.model.macros[i]

    @enabled("tables")
    @rule(i=table_ids)
    def create_table(self, i):
        query = f"CREATE NODE TABLE T{i}(id SERIAL PRIMARY KEY, v INT64)"
        if i in self.model.tables:
            self.fails(query, None, "already exists")
            return
        self.ok(query)
        self.model.tables[i] = []

    @enabled("tables")
    @precondition(lambda self: self.model.tables)
    @rule(data=st.data(), values=st.lists(st.integers(-1000, 1000), min_size=1, max_size=5))
    def insert_into_table(self, data, values):
        i = data.draw(st.sampled_from(sorted(self.model.tables)))
        self.ok(f"UNWIND $vs AS v CREATE (:T{i} {{v: v}})", dict(vs=values))
        self.model.tables[i] += values

    @enabled("drop_table")
    @rule(i=table_ids)
    def drop_table(self, i):
        if i not in self.model.tables:
            self.fails(f"DROP TABLE T{i}", None, "does not exist")
            return
        self.ok(f"DROP TABLE T{i}")
        del self.model.tables[i]

    @enabled("columns")
    @rule(column=extra_columns, default=st.integers(-5, 5))
    def add_column(self, column, default):
        query = f"ALTER TABLE Person ADD {column} INT64 DEFAULT {default}"
        if column in self.model.columns:
            self.fails(query, None, "already has property")
            return
        self.ok(query)
        self.model.columns[column] = default
        for person in self.model.persons.values():
            person[column] = default

    @enabled("columns")
    @rule(column=extra_columns)
    def drop_column(self, column):
        query = f"ALTER TABLE Person DROP {column}"
        if column not in self.model.columns:
            self.fails(query, None, "does not have property")
            return
        self.ok(query)
        del self.model.columns[column]
        for person in self.model.persons.values():
            del person[column]

    @enabled("pk_index")
    @precondition(lambda self: self.model.pk_index is not None)
    @rule()
    def drop_pk_index(self):
        self.ok(f"DROP INDEX Person.{self.model.pk_index}")
        self.model.pk_index = None

    @enabled("pk_index")
    @precondition(lambda self: self.model.pk_index is None)
    @rule()
    def create_pk_index(self):
        self.ok("CREATE INDEX person_pk FOR (p:Person) ON (p.id)")
        self.model.pk_index = "person_pk"
