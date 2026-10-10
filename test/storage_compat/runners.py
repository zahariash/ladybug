"""The builds under test, each run as a ladybug Python package through uv: a release from PyPI, or
this repository's tools/python_api on a build's liblbug.so through the C-API backend."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

PYTHON = "3.12"  # a Python version the releases publish wheels for
TIMEOUT_SECONDS = 900
ERROR = "<error>"  # a check that failed reads as ERROR followed by the message
RUNNER = Path(__file__).with_name("runner.py")
PYTHON_API = Path(__file__).resolve().parents[2] / "tools/python_api"


def is_error(result) -> bool:
    return isinstance(result, str) and result.startswith(ERROR)


class CompatError(Exception):
    pass


class Build:
    def __init__(self, name: str, package: str, lib: Path | None, version: str | None) -> None:
        self.name, self.package, self.lib, self.version = name, package, lib, version

    @classmethod
    def release(cls, version: str) -> Build:
        return cls(f"ladybug {version}", f"ladybug=={version}", None, version)

    @classmethod
    def local(cls, name: str, lib: Path) -> Build:
        return cls(name, str(PYTHON_API), lib, None)

    def _run(self, mode: str, db: Path, spec: dict) -> dict:
        args = ["uv", "run", "--quiet", "--no-project", "--python", PYTHON]
        args += ["--with", self.package, "--with", "pyarrow", "python", str(RUNNER), mode, str(db)]
        env = os.environ.copy()
        if self.lib:
            env |= {"LBUG_PYTHON_BACKEND": "capi", "LBUG_C_API_LIB_PATH": str(self.lib)}
        try:
            result = subprocess.run(
                args,
                input=json.dumps(spec),
                env=env,
                capture_output=True,
                text=True,
                timeout=TIMEOUT_SECONDS,
                check=False,
            )
        except subprocess.TimeoutExpired as e:
            raise CompatError(f"{self.name} timed out after {TIMEOUT_SECONDS} s on {db}") from e
        if result.returncode != 0:
            raise CompatError(
                f"{self.name} failed on {db} (exit {result.returncode}):\n{result.stderr}"
            )
        return json.loads(result.stdout)

    def execute_sections(self, db: Path, sections: dict[str, list[str]]) -> dict[str, dict]:
        """Runs each section read-write; reports its first error and its failed statements."""
        return self._run("execute", db, {"read_only": False, "sections": sections})

    def read(self, db: Path, queries: dict[str, str], read_only: bool = True) -> dict:
        """Rows of each query as lists of JSON values, or an error (see is_error)."""
        return self._run("read", db, {"read_only": read_only, "queries": queries})

    def storage_version(self) -> int:
        query = {"v": "CALL storage_version() RETURN version"}
        rows = self.read(Path(":memory:"), query, read_only=False)["v"]
        if is_error(rows):
            raise CompatError(f"{self.name} cannot report its storage version: {rows}")
        return rows[0][0]
