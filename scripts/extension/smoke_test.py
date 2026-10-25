"""Installs and loads published extensions with the Ladybug releases that read them.

Every check runs in its own process with an empty home directory, so a cached
extension cannot hide what the repository serves, and a crash at exit fails it.
Each line is tested with its newest patch release, or with every patch release
when listed in --all-patches. Releases come from PyPI; the line a release reads
comes from LBUG_EXTENSION_VERSION in its tag's CMakeLists.txt.
"""

import argparse
import functools
import http.server
import json
import os
import pathlib
import platform
import re
import subprocess
import sys
import tempfile
import threading
import urllib.request
from concurrent.futures import ThreadPoolExecutor

OFFICIAL_REPO = "https://extension.ladybugdb.com/"
CMAKE_URL = "https://raw.githubusercontent.com/LadybugDB/ladybug/v{}/CMakeLists.txt"
PYPI_URL = "https://pypi.org/pypi/ladybug/json"
KNOWN_FAILURES = pathlib.Path(__file__).with_name("smoke_test_known_failures.txt")
LINE_PATTERN = re.compile(r"v(\d+)\.(\d+)\.(\d+)")
CHECK_TIMEOUT_SECONDS = 300

CHILD = """
import sys
import ladybug

repo, extension = sys.argv[1:]
conn = ladybug.Connection(ladybug.Database(":memory:"))
conn.execute(f"INSTALL {extension} FROM '{repo}'")
conn.execute(f"LOAD EXTENSION {extension}")
print("LOADED")
"""


def get_platform():
    system = {"Linux": "linux", "Darwin": "osx", "Windows": "win"}[platform.system()]
    machine = platform.machine().lower()
    arch = "arm64" if machine in ("arm64", "aarch64") else "amd64"
    return f"{system}_{arch}"


def fetch(url):
    # The extension CDN rejects urllib's default User-Agent from CI runners.
    request = urllib.request.Request(url, headers={"User-Agent": "ladybug-extension-smoke-test"})
    with urllib.request.urlopen(request, timeout=60) as response:
        return response.read().decode()


def list_dirs(url):
    """Directory names in an nginx or http.server index page."""
    return sorted(set(re.findall(r'href="([^"/?.][^"/]*)/"', fetch(url))))


def version_key(version):
    return tuple(int(part) for part in version.split("."))


def released_versions():
    releases = json.loads(fetch(PYPI_URL))["releases"]
    return sorted((v for v, files in releases.items() if files and re.fullmatch(r"\d+\.\d+\.\d+", v)),
                  key=version_key)


@functools.cache
def extension_line(version):
    match = re.search(r'LBUG_EXTENSION_VERSION="([^"]+)"', fetch(CMAKE_URL.format(version)))
    return "v" + match.group(1)


def engines_for_line(line, versions, all_patches):
    major_minor = LINE_PATTERN.fullmatch(line).groups()[:2]
    candidates = [v for v in versions if tuple(v.split(".")[:2]) == major_minor]
    engines = [v for v in candidates if extension_line(v) == line]
    return engines if all_patches else engines[-1:]


def load_known_failures():
    """Maps (line, platform, extension) to a reason; line and platform may be "*"."""
    known = {}
    for raw in KNOWN_FAILURES.read_text().splitlines():
        entry, _, reason = raw.partition("#")
        if entry.strip():
            line, plat, extension = entry.split()
            known[(line, plat, extension)] = reason.strip()
    return known


def known_reason(known, line, plat, extension):
    for key in ((line, plat, extension), ("*", plat, extension), (line, "*", extension), ("*", "*", extension)):
        if key in known:
            return known[key]
    return None


def make_venv(work_dir, python, version):
    venv = work_dir / f"ladybug-{version}"
    subprocess.run(["uv", "venv", "-q", "--python", python, str(venv)], check=True)
    bin_dir = venv / ("Scripts" if platform.system() == "Windows" else "bin")
    interpreter = bin_dir / ("python.exe" if platform.system() == "Windows" else "python")
    subprocess.run(["uv", "pip", "install", "-q", "--python", str(interpreter), f"ladybug=={version}"],
                   check=True)
    return interpreter


def check(interpreter, repo, extension, work_dir):
    home = tempfile.mkdtemp(prefix="home-", dir=work_dir)
    env = dict(os.environ, HOME=home, USERPROFILE=home)
    try:
        result = subprocess.run([str(interpreter), "-I", "-c", CHILD, repo, extension], env=env,
                                capture_output=True, text=True, timeout=CHECK_TIMEOUT_SECONDS)
    except subprocess.TimeoutExpired:
        return f"timed out after {CHECK_TIMEOUT_SECONDS} s"
    if result.returncode == 0 and "LOADED" in result.stdout:
        return None
    lines = [line.strip() for line in (result.stdout + result.stderr).replace(home, "~").splitlines()]
    # dyld follows the missing library's name with long "Referenced from" and "Reason" lines.
    lines = [line for line in lines
             if line.strip("^~ ") and not line.startswith(("Referenced from:", "Reason:"))]
    return f"exit code {result.returncode}: " + " | ".join(lines[-2:])[-500:]


class QuietHandler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *args):
        pass


def serve(directory):
    handler = functools.partial(QuietHandler, directory=directory)
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return f"http://127.0.0.1:{server.server_port}/"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--repo", default=OFFICIAL_REPO, help="extension repository URL")
    source.add_argument("--serve", metavar="DIR", help="serve a local releases directory and test it")
    parser.add_argument("--line", nargs="*", default=[], help="lines to test, e.g. v0.21.0 (default: all)")
    parser.add_argument("--all-patches", nargs="*", default=[], help="lines to test with every patch release")
    parser.add_argument("--extension", nargs="*", default=[], help="extensions to test (default: all)")
    parser.add_argument("--python", default="3.12", help="Python version for the release venvs")
    parser.add_argument("--jobs", type=int, default=4, help="checks to run at once")
    args = parser.parse_args()

    repo = serve(args.serve) if args.serve else args.repo.rstrip("/") + "/"
    plat = get_platform()
    lines = args.line or [d for d in list_dirs(repo) if LINE_PATTERN.fullmatch(d)]
    versions = released_versions()
    known = load_known_failures()
    work_dir = pathlib.Path(tempfile.mkdtemp(prefix="extension-smoke-"))

    checks = []
    for line in sorted(lines, key=lambda l: version_key(l[1:])):
        engines = engines_for_line(line, versions, line in args.all_patches)
        if not engines:
            print(f"{line}: skipped, no release on PyPI reads it")
            continue
        extensions = args.extension or [d for d in list_dirs(f"{repo}{line}/{plat}/") if d != "common"]
        print(f"{line}: {', '.join(engines)} x {len(extensions)} extensions on {plat}")
        for engine in engines:
            interpreter = make_venv(work_dir, args.python, engine)
            checks += [(line, engine, extension, interpreter) for extension in extensions]

    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        errors = list(pool.map(lambda c: check(c[3], repo, c[2], work_dir), checks))

    unexpected = 0
    for (line, engine, extension, _), error in zip(checks, errors):
        reason = known_reason(known, line, plat, extension)
        if error is None:
            status = "ok" if reason is None else f"ok, listed in {KNOWN_FAILURES.name}"
        elif reason is not None:
            status = f"known failure ({reason}): {error}"
        else:
            status = f"FAILED: {error}"
            unexpected += 1
        print(f"{line:9} {engine:8} {extension:14} {status}")
    print(f"{len(checks)} checks, {unexpected} unexpected failures")
    return 1 if unexpected else 0


if __name__ == "__main__":
    sys.exit(main())
