"""Check that the built sdist and wheel contain the complete aitlas package.

Run from the repository root after ``uv build``:

    python tools/check_dist.py dist

The check fails if:

* the directory does not hold exactly one wheel and one sdist (for example,
  leftovers from an earlier build),
* any ``aitlas/**/*.py`` file in the source tree is missing from the wheel or
  the sdist, or either of them holds a module that the source tree does not
  have (for example, stale files from an old ``build/`` directory),
* the wheel contains anything other than ``aitlas/`` and its ``.dist-info``,
* after installing the wheel into a fresh virtual environment, ``aitlas`` is
  not imported from that environment, or one of its subpackages is not found.

Only the standard library is used, and the wheel is installed without its
dependencies, so the check is fast and does not need torch. Any Python 3.9+
can run it; the wheel's Python version requirement is ignored for the install.
"""

import argparse
import json
import subprocess
import sys
import tarfile
import tempfile
import venv
import zipfile
from pathlib import Path, PurePosixPath


PACKAGE = "aitlas"
REPO_ROOT = Path(__file__).resolve().parent.parent
MAX_LISTED = 10

# Runs inside the fresh environment. Importing only the top-level package and
# looking up (not importing) its direct subpackages avoids loading torch.
PROBE = """
import importlib.util, json, pathlib, sys
import aitlas
print(json.dumps({
    "location": str(pathlib.Path(aitlas.__file__).resolve().parent),
    "unresolved": [name for name in sys.argv[1:] if importlib.util.find_spec(name) is None],
}))
"""


def find_single(dist_dir, pattern):
    matches = sorted(dist_dir.glob(pattern))
    if len(matches) != 1:
        names = ", ".join(m.name for m in matches) or "none"
        sys.exit(
            f"Expected exactly one {pattern} in {dist_dir}, found {len(matches)} ({names}). "
            "Empty the directory and build again."
        )
    return matches[0]


def compare(label, expected, found):
    problems = []
    for what, names in (
        ("missing", expected - found),
        ("not in the source tree", found - expected),
    ):
        if names:
            listed = sorted(names)[:MAX_LISTED]
            more = (
                f"\n      ... and {len(names) - len(listed)} more"
                if len(names) > len(listed)
                else ""
            )
            problems.append(
                f"{label}: {len(names)} module(s) {what}:\n      " + "\n      ".join(listed) + more
            )
    return problems


def check_wheel(wheel, expected):
    with zipfile.ZipFile(wheel) as zf:
        names = zf.namelist()
    modules = {n for n in names if n.startswith(f"{PACKAGE}/") and n.endswith(".py")}
    print(f"wheel  {wheel.name}: {len(modules)} modules")

    problems = compare("wheel", expected, modules)
    top_level = {n.split("/", 1)[0] for n in names}
    for entry in sorted(top_level):
        if entry != PACKAGE and not entry.endswith(".dist-info"):
            problems.append(f"wheel: unexpected top-level entry {entry!r}")
    return problems


def check_sdist(sdist, expected):
    with tarfile.open(sdist) as tf:
        names = [m.name for m in tf.getmembers() if m.isfile()]
    # Every member sits below a single "<name>-<version>/" directory.
    modules = set()
    for name in names:
        parts = PurePosixPath(name).parts
        if len(parts) > 2 and parts[1] == PACKAGE and name.endswith(".py"):
            modules.add("/".join(parts[1:]))
    print(f"sdist  {sdist.name}: {len(modules)} modules")
    return compare("sdist", expected, modules)


def check_install(wheel, subpackages):
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp).resolve()
        env_dir = tmp / "venv"
        venv.EnvBuilder(with_pip=True).create(env_dir)
        python = env_dir / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
        subprocess.run(
            [
                python,
                "-m",
                "pip",
                "install",
                "--quiet",
                "--disable-pip-version-check",
                "--no-deps",
                "--no-index",
                "--ignore-requires-python",
                wheel,
            ],
            check=True,
        )
        # Isolated mode (-I) from an empty working directory, so nothing can be
        # imported from the source checkout.
        result = subprocess.run(
            [python, "-I", "-c", PROBE, *subpackages],
            cwd=tmp,
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            return [f"clean install: 'import {PACKAGE}' failed:\n{result.stderr}"]

        info = json.loads(result.stdout)
        location = Path(info["location"])
        print(f"install: {PACKAGE} imported from {location}")
        problems = []
        if not location.is_relative_to(env_dir):
            problems.append(
                f"clean install: {PACKAGE} was imported from outside the new environment"
            )
        if info["unresolved"]:
            problems.append(
                f"clean install: {len(info['unresolved'])} of {len(subpackages)} subpackages "
                f"not found: {', '.join(info['unresolved'])}"
            )
        return problems


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "dist_dir",
        nargs="?",
        type=Path,
        default=REPO_ROOT / "dist",
        help="directory with the built sdist and wheel (default: dist/)",
    )
    args = parser.parse_args()

    package_dir = REPO_ROOT / PACKAGE
    expected = {p.relative_to(REPO_ROOT).as_posix() for p in package_dir.rglob("*.py")}
    subpackages = sorted(f"{PACKAGE}.{p.parent.name}" for p in package_dir.glob("*/__init__.py"))
    wheel = find_single(args.dist_dir, "*.whl")
    sdist = find_single(args.dist_dir, "*.tar.gz")

    print(f"source {PACKAGE}/: {len(expected)} modules")
    problems = check_wheel(wheel, expected) + check_sdist(sdist, expected)
    problems += check_install(wheel, subpackages)

    if problems:
        print("\nFAILED")
        for problem in problems:
            print(f"  - {problem}")
        return 1
    print(
        f"\nOK: wheel and sdist contain all {len(expected)} modules; "
        f"clean install finds all {len(subpackages)} subpackages"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
