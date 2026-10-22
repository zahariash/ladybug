"""Values the rules draw: small key ranges so that rules hit the same rows, and edge values."""

from __future__ import annotations

import hypothesis.strategies as st

INT64_MIN, INT64_MAX = -(2**63), 2**63 - 1
# Ids at or above this come from bulk loads, never from the small-id rules.
BULK_ID_START = 1_000_000

small_ids = st.integers(0, 40)
VECTOR_CAPACITY = 2048


def near_boundaries(ids: list) -> st.SearchStrategy:
    """One of `ids`, half of the time one at the rows on either side of a half or full vector
    boundary (1023/1024, 2047/2048, ...), where per-vector bookkeeping changes and few writes
    reach. Row positions follow `ids`' order, the insertion order, so deletes shift them."""
    half = VECTOR_CAPACITY // 2
    edges = [id for i, id in enumerate(ids) if i > 2 and i % half in (half - 2, half - 1, 0, 1)]
    anywhere = st.sampled_from(ids)
    return st.one_of(anywhere, st.sampled_from(edges)) if edges else anywhere
macro_ids = st.integers(0, 3)
table_ids = st.integers(0, 2)
EXTRA_COLUMNS = ["c0", "c1", "c2"]
extra_columns = st.sampled_from(EXTRA_COLUMNS)
clusters = st.integers(1, 3)
# C, the node table declared CSR, holds ids 0..CSR_ROWS-1 at the offsets of their ids: a full node
# group (2^17 rows, the default build) and a second one that ends past a vector. Edges join the
# first 40 ids of each group and ids at the size boundaries in each group (CSR leaf regions of
# 1024 lists, vectors of 2048 rows, 32768-row INT64 segments, the group's last rows).
CSR_SECOND_GROUP = 2**17
CSR_ROWS = CSR_SECOND_GROUP + VECTOR_CAPACITY + 40
BOUNDARY_OFFSETS = [1022, 1023, 1024, 1025, 2046, 2047, 2048, 2049, 4095, 4096]
BOUNDARY_OFFSETS += [32766, 32767, 32768, 32769, 65534, 65535, 65536, 65537]


def csr_bands(rows: int = CSR_ROWS) -> list[list[int]]:
    """The C ids the rules draw, by node group: the first 40 and the boundary ids."""
    bands = []
    for start in range(0, rows, CSR_SECOND_GROUP):
        size = min(CSR_SECOND_GROUP, rows - start)
        offsets = set(range(40)) | set(BOUNDARY_OFFSETS) | set(range(size - 4, size))
        bands.append([start + i for i in sorted(offsets) if i < size])
    return bands


CSR_BANDS = csr_bands()
CSR_BOUNDARY_IDS = [id for band in CSR_BANDS for id in band if id % CSR_SECOND_GROUP >= 40]
csr_ids = st.one_of(
    st.integers(0, 39),
    st.integers(CSR_SECOND_GROUP, CSR_SECOND_GROUP + 39),
    st.sampled_from(CSR_BOUNDARY_IDS),
)
# CK2's sources lie in the first node group only, so that its rel table lacks the second.
csr_src_ids = st.one_of(
    st.integers(0, 39), st.sampled_from([id for id in CSR_BOUNDARY_IDS if id < CSR_SECOND_GROUP])
)

# CK2's edges in every new database: node 1 reaches 17, 18 and a node of the second group, with two
# more levels below 17 and 18, and 11 reaches 12 and 13. Variable-length matches from 1 and
# two-hop matches over 1 and 11 then read first-group nodes after the second-group one.
CK2_SEED = [(1, 17), (1, CSR_SECOND_GROUP + 8), (1, 18), (17, 5), (18, 6), (5, 7), (6, 8)]
CK2_SEED += [(11, 12), (12, 13)]
CK2_SEED_SOURCE = 1
# Hubs, one per node group, take CK edges to leaves, ids that no band holds, so that their CSR
# lists span several vectors without adding two-hop walks. A degree around 1024 also moves the
# planner's lookup-join cut-off.
HUBS = [7, CSR_SECOND_GROUP + 7]
HUB_DEGREES = [1023, 1024, 1025, 2047, 2048, 2049, 4097]
LEAF_START = 60_000
# Appends to C stop here, so that C stays near two node groups.
C_ROWS_CAP = CSR_ROWS + 50_000
# C.v values an update writes: around the formula's range, and ones whose width no longer fits a
# segment compressed for 0..999.
c_values = st.one_of(
    st.integers(-5, 1005), st.sampled_from([INT64_MAX, INT64_MIN, -1, 999, 1000]), st.none()
)


def csr_id(i: int) -> int:
    """A C id for a small id: in the first node group for even ones, the second for odd ones."""
    return i % 40 if i % 2 == 0 else CSR_SECOND_GROUP + i % 40


def csr_src_id(i: int) -> int:
    """A first-group C id for a small id, a boundary id for every third."""
    first = [id for id in CSR_BANDS[0] if id >= 40]
    return first[i % len(first)] if i % 3 == 2 else i % 40

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
# Names for CSV files: commas, quotes, newlines and Unicode must survive quoting; "" is an
# empty string and an unquoted empty field is NULL.
csv_names = st.one_of(st.none(), st.text('abcxyz ,"\néß日', max_size=12))

# Plain words, case and Unicode variants, and stopwords, which full-text queries never match.
WORDS = ["apple", "grape", "melon", "lemon", "mango", "peach"]
HARD_WORDS = ["Apples", "GRAPE", "café", "Straße", "日本語", "naïve"]
STOPWORDS = ["the", "and", "of"]
ALL_WORDS = WORDS + HARD_WORDS + STOPWORDS
doc_words = st.one_of(st.none(), st.lists(st.sampled_from(ALL_WORDS), min_size=1, max_size=5))
embeddings = st.one_of(st.none(), st.tuples(*[st.integers(-3, 3)] * 3).filter(any))
