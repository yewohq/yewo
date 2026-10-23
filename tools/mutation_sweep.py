#!/usr/bin/env python3
"""Break the judging and the gate on purpose, and check something notices.

    python -m tools.mutation_sweep             every mutant, up to four at once
    python -m tools.mutation_sweep --jobs 1    one mutant at a time
    python -m tools.mutation_sweep --only crash

No third-party mutation engine is configured for this repository; a general
one would spend most of its time on code where a surviving mutant means
nothing. This focused sweep holds a written list instead: each entry is a
defect somebody could really introduce, in the logic that decides a verdict
or decides whether the gate passes.

A mutant is **caught** when the check that owns it fails. The judge's mutants
are owned by the eval corpus, so the gate has to fail. The gate's own mutants
are owned by its tests, so pytest has to fail.

Each worker receives an isolated copy of the files needed by the checks. No
mutant touches the working tree, and independent workers can run in parallel.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
import tempfile

ROOT = Path(__file__).resolve().parents[1]
MUTANTS = Path(__file__).resolve().with_name("mutants.json")

JUDGE = "judge"
"""Owned by the eval corpus: the gate has to fail."""

GATE = "gate"
"""Owned by the gate's own tests: pytest has to fail."""

GATE_TESTS = (
    "tests/tools/test_eval_gate.py",
    "tests/tools/test_eval_tiers.py",
    "tests/tools/test_eval_corpus.py",
    "tests/judge/test_auditor_answers.py",
    "tests/report/test_rejudge.py",
    "tests/report/test_manifest.py",
)

WORKSPACE_DIRECTORIES = ("eval", "src", "tests", "tools")
WORKSPACE_FILES = ("pyproject.toml", "yewo-schema.json")
DEFAULT_JOBS = min(4, os.cpu_count() or 1)


@dataclass(frozen=True)
class Mutant:
    """One defect, and where it goes."""

    name: str
    owner: str
    path: str
    find: str
    replace: str


def _python() -> str:
    """The interpreter running this, so a sweep uses the same environment."""
    return sys.executable


def _checked(mutant: Mutant, root: Path) -> subprocess.CompletedProcess:
    environment = os.environ.copy()
    python_paths = [str(root / "src"), str(root / "tests")]
    if existing := environment.get("PYTHONPATH"):
        python_paths.append(existing)
    environment["PYTHONPATH"] = os.pathsep.join(python_paths)
    if mutant.owner == JUDGE:
        return subprocess.run(
            [_python(), "-m", "tools.eval_gate"],
            cwd=root,
            capture_output=True,
            text=True,
            env=environment,
        )
    return subprocess.run(
        [_python(), "-m", "pytest", *GATE_TESTS, "-q"],
        cwd=root,
        capture_output=True,
        text=True,
        env=environment,
    )


def _applied(mutant: Mutant, root: Path) -> tuple[bool, str]:
    """Runs one mutant, and says whether the check caught it."""
    path = root / mutant.path
    source = path.read_text(encoding="utf-8")
    if mutant.find not in source:
        return False, "the code it edits has moved. Rewrite the mutant."
    try:
        path.write_text(source.replace(mutant.find, mutant.replace, 1), encoding="utf-8")
        done = _checked(mutant, root)
    finally:
        path.write_text(source, encoding="utf-8")
    if done.returncode != 0:
        return True, _told(done.stdout)
    return False, "nothing failed"


SUMMARY = re.compile(r"^\d+ failed")


def _told(output: str) -> str:
    """The line of a failure worth putting in a sweep's own report.

    Matched on the shape pytest ends with, not on the word. A docstring in
    the test that just failed can hold "failed" and be reported instead.
    """
    for line in output.splitlines():
        stripped = line.strip()
        if "wrong:" in stripped or "unanswered:" in stripped or SUMMARY.match(stripped):
            return stripped
    return "the check failed"


def _read() -> list[Mutant]:
    loaded = json.loads(MUTANTS.read_text(encoding="utf-8"))
    return [Mutant(**entry) for entry in loaded]


def _positive_count(value: str) -> int:
    count = int(value)
    if count < 1:
        raise argparse.ArgumentTypeError(f"jobs must be at least 1, not {count}")
    return count


def _copy_workspace(destination: Path) -> None:
    ignored = shutil.ignore_patterns("__pycache__", "*.pyc")
    for directory in WORKSPACE_DIRECTORIES:
        shutil.copytree(ROOT / directory, destination / directory, ignore=ignored)
    for file_name in WORKSPACE_FILES:
        shutil.copy2(ROOT / file_name, destination / file_name)


def _run_partition(
    indexed_mutants: list[tuple[int, Mutant]],
) -> list[tuple[int, Mutant, bool, str]]:
    with tempfile.TemporaryDirectory(prefix="yewo-mutation-") as directory:
        root = Path(directory)
        _copy_workspace(root)
        return [
            (index, mutant, *_applied(mutant, root))
            for index, mutant in indexed_mutants
        ]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", default=None, help="Run the mutants whose name holds this.")
    parser.add_argument(
        "--jobs",
        type=_positive_count,
        default=DEFAULT_JOBS,
        help=f"Run this many isolated workers (default: {DEFAULT_JOBS}).",
    )
    args = parser.parse_args(argv)

    mutants = [m for m in _read() if not args.only or args.only in m.name]
    worker_count = min(args.jobs, len(mutants)) if mutants else 1
    indexed = list(enumerate(mutants))
    partitions = [indexed[offset::worker_count] for offset in range(worker_count)]
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        nested_results = executor.map(_run_partition, partitions)
    results = sorted(
        (result for partition in nested_results for result in partition),
        key=lambda result: result[0],
    )

    caught = 0
    stale = 0
    for _index, mutant, was_caught, detail in results:
        if was_caught:
            caught += 1
            print(f"caught    [{mutant.owner}] {mutant.name}  ->  {detail}")
        elif "has moved" in detail:
            stale += 1
            print(f"STALE     [{mutant.owner}] {mutant.name}  ->  {detail}")
        else:
            print(f"SURVIVED  [{mutant.owner}] {mutant.name}  ->  {detail}")

    print(f"\n{caught} of {len(mutants)} mutants caught.")
    if stale:
        print(f"{stale} mutant(s) no longer match the code. A stale mutant tests nothing.")
    return 0 if caught == len(mutants) else 1


if __name__ == "__main__":
    raise SystemExit(main())
