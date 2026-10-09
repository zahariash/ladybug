"""Values the rules draw: small key ranges so that rules hit the same rows, and edge values."""

from __future__ import annotations

import hypothesis.strategies as st

INT64_MIN, INT64_MAX = -(2**63), 2**63 - 1
# Ids at or above this come from bulk loads, never from the small-id rules.
BULK_ID_START = 1_000_000

small_ids = st.integers(0, 40)
macro_ids = st.integers(0, 3)
table_ids = st.integers(0, 2)
extra_columns = st.sampled_from(["c0", "c1"])
clusters = st.integers(1, 3)

ints = st.one_of(
    st.none(),
    st.integers(-100, 100),
    st.sampled_from([INT64_MIN, INT64_MIN + 1, INT64_MAX, 0, -1]),
    st.integers(INT64_MIN, INT64_MAX),
)
doubles = st.one_of(
    st.none(),
    st.floats(allow_nan=True, allow_infinity=True),
    st.sampled_from([0.0, -0.0, float("nan"), 1.5, -1e308]),
)
names = st.one_of(
    st.none(),
    st.text(st.characters(exclude_characters="\x00", exclude_categories=["Cs"]), max_size=40),
)
csv_names = st.text("abcdefghijklmnopqrstuvwxyz ", min_size=1, max_size=20)

# Plain words, case and Unicode variants, and stopwords, which full-text queries never match.
WORDS = ["apple", "grape", "melon", "lemon", "mango", "peach"]
HARD_WORDS = ["Apples", "GRAPE", "café", "Straße", "日本語", "naïve"]
STOPWORDS = ["the", "and", "of"]
ALL_WORDS = WORDS + HARD_WORDS + STOPWORDS
doc_words = st.one_of(st.none(), st.lists(st.sampled_from(ALL_WORDS), min_size=1, max_size=5))
embeddings = st.one_of(st.none(), st.tuples(*[st.integers(-3, 3)] * 3).filter(any))
