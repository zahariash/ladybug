"""The databases each build writes: the fixture in fixture/ and a few test datasets."""

from __future__ import annotations

import re
import shutil
from pathlib import Path

from runners import CompatError

FIXTURE_DIR = Path(__file__).with_name("fixture")
REPO_ROOT = Path(__file__).resolve().parents[2]
# Fixture sections in the order they run. A writer that cannot run a statement skips it, as old
# releases lack partitioning or CSR tables; "core" must run in full.
SECTIONS = ["core", "partitioned", "csr", "catalog", "art_index"]
DATASETS = ["tinysnb", "tinysnb-serial", "long-string-pk-tests", "rel-group"]
NAMES = ["fixture", *DATASETS]


def statements(path: Path) -> list[str]:
    return [line for line in path.read_text().splitlines() if line.strip()]


def upgrade_statements() -> list[str]:
    """Writes applied to a copy of the fixture to test upgrading it in place."""
    return statements(FIXTURE_DIR / "upgrade.cypher")


def dataset_statements(name: str) -> list[str]:
    """A dataset's schema and copy scripts, rewritten as TestHelper::executeScript does: single
    quotes become double quotes and data file paths are resolved."""
    result = []
    for script in ("schema.cypher", "copy.cypher"):
        path = REPO_ROOT / "dataset" / name / script
        if not path.exists():
            continue
        for line in statements(path):
            line = line.strip().replace("'", '"')
            line = re.sub(r'"([^"]*)"', lambda m: f'"{_data_path(m.group(1), path.parent)}"', line)
            result.append(line.rstrip(";") + ";")
    if not result:
        raise CompatError(f"dataset {name} has no scripts; is the dataset submodule checked out?")
    return result


def _data_path(file: str, dataset_dir: Path) -> str:
    if not re.search(r"\.(csv|parquet|npy|ttl|nq|json)", file.lower()) or Path(file).is_absolute():
        return file
    return str((dataset_dir if Path(file).parent == Path() else REPO_ROOT) / file)


def write_sections(writer, db: Path, sections: dict[str, list[str]]) -> dict[str, list[int]]:
    """Runs the sections; returns the indices of the failed statements of each section that had
    a failure. Raises if "core" failed."""
    results = writer.execute_sections(db, sections)
    for name, result in results.items():
        if result["error"] and name == "core":
            raise CompatError(f"{writer.name} could not write {db}: {result['error']}")
        if result["error"]:
            print(f"{writer.name} skipped part of fixture section {name}: {result['error']}")
    return {name: result["failed"] for name, result in results.items() if result["error"]}


def write_db(writer, name: str, db: Path) -> dict[str, list[int]]:
    """Writes the fixture or a dataset; returns the fixture statements that failed, by section."""
    db.parent.mkdir(parents=True)
    if name != "fixture":
        write_sections(writer, db, {"core": dataset_statements(name) + ["CHECKPOINT;"]})
        return {}
    sections = {section: statements(FIXTURE_DIR / f"{section}.cypher") for section in SECTIONS}
    failures = write_sections(writer, db, sections)
    write_sections(writer, db, {"checkpoint": ["CHECKPOINT;"]})
    return failures


def copy_db(db: Path, label: str) -> Path:
    """Copies the database's directory, which holds partition files next to it."""
    target_dir = db.parent.parent / f"{db.parent.name}-{label}"
    shutil.copytree(db.parent, target_dir)
    return target_dir / db.name
