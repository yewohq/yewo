#!/usr/bin/env python3
"""Enforce Yewo's type-checking ratchet with Pyright.

Pyright has no baseline of its own. Every file stays checked, and each recorded
diagnostic is matched by file, rule, and normalized message. A different error
cannot replace an old one merely because the count stayed the same.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tomllib

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
BASELINE_PATH = Path(__file__).resolve().with_name("type-baseline.toml")


@dataclass(frozen=True)
class Diagnostic:
    path: str
    rule: str
    message: str

    def label(self) -> str:
        return f"{self.path} [{self.rule}] {self.message}"


@dataclass(frozen=True)
class Violation:
    diagnostic: Diagnostic
    found: int
    allowed: int

    def render(self) -> str:
        if self.allowed:
            return (
                f"{self.diagnostic.label()}: found {self.found}, "
                f"above its baseline of {self.allowed}"
            )
        return f"{self.diagnostic.label()}: new type error"


def _load_baseline() -> Counter[Diagnostic]:
    with BASELINE_PATH.open("rb") as baseline_file:
        entries = tomllib.load(baseline_file).get("errors", [])
    return Counter(
        {
            Diagnostic(entry["path"], entry["rule"], entry["message"]): entry["count"]
            for entry in entries
        }
    )


def _run_pyright(paths: list[str]) -> dict:
    executable = shutil.which("pyright")
    if executable is None:
        raise SystemExit(
            "pyright is not on PATH. Install the dev group: uv sync --group dev"
        )
    completed = subprocess.run(
        [executable, "--outputjson", "--level", "error", *paths],
        capture_output=True,
        text=True,
        cwd=REPOSITORY_ROOT,
    )
    try:
        return json.loads(completed.stdout)
    except json.JSONDecodeError:
        raise SystemExit(
            f"pyright produced no report (exit {completed.returncode}):\n{completed.stderr}"
        ) from None


def _diagnostics(report: dict) -> Counter[Diagnostic]:
    diagnostics: Counter[Diagnostic] = Counter()
    for diagnostic in report.get("generalDiagnostics", []):
        if diagnostic.get("severity") != "error":
            continue
        path = Path(diagnostic["file"])
        try:
            relative = path.relative_to(REPOSITORY_ROOT)
        except ValueError:
            relative = path
        message = " ".join(str(diagnostic.get("message", "")).split())
        rule = str(diagnostic.get("rule") or "unknown")
        diagnostics[Diagnostic(relative.as_posix(), rule, message)] += 1
    return diagnostics


def _render_baseline(diagnostics: Counter[Diagnostic]) -> str:
    lines = [
        "# Recorded Pyright errors, identified without line numbers so ordinary",
        "# edits do not move them. A changed rule or message is a new error.",
    ]
    for diagnostic, count in sorted(
        diagnostics.items(),
        key=lambda item: (item[0].path, item[0].rule, item[0].message),
    ):
        lines.extend(
            [
                "",
                "[[errors]]",
                f"path = {json.dumps(diagnostic.path)}",
                f"rule = {json.dumps(diagnostic.rule)}",
                f"message = {json.dumps(diagnostic.message)}",
                f"count = {count}",
            ]
        )
    return "\n".join(lines) + "\n"


def _violations(
    found: Counter[Diagnostic], baseline: Counter[Diagnostic]
) -> list[Violation]:
    return [
        Violation(diagnostic, count, baseline.get(diagnostic, 0))
        for diagnostic, count in found.items()
        if count > baseline.get(diagnostic, 0)
    ]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "paths",
        nargs="*",
        default=["src/yewo"],
        help="Paths to type check (default: src/yewo)",
    )
    parser.add_argument(
        "--write-baseline",
        action="store_true",
        help="Record the current errors as the baseline instead of enforcing it",
    )
    args = parser.parse_args(argv)

    diagnostics = _diagnostics(_run_pyright(args.paths))

    if args.write_baseline:
        BASELINE_PATH.write_text(_render_baseline(diagnostics), encoding="utf-8")
        file_count = len({diagnostic.path for diagnostic in diagnostics})
        print(f"Recorded {sum(diagnostics.values())} type errors across {file_count} files.")
        return 0

    baseline = _load_baseline()
    violations = _violations(diagnostics, baseline)
    if violations:
        for violation in sorted(violations, key=lambda item: item.diagnostic.label()):
            print(violation.render(), file=sys.stderr)
        return 1

    improved = [
        (diagnostic, allowed, diagnostics.get(diagnostic, 0))
        for diagnostic, allowed in baseline.items()
        if diagnostics.get(diagnostic, 0) < allowed
    ]
    if improved:
        for diagnostic, allowed, found in sorted(
            improved, key=lambda item: item[0].label()
        ):
            print(
                f"{diagnostic.label()}: now found {found}, below its baseline of {allowed}. "
                "Rewrite the baseline.",
                file=sys.stderr,
            )
        return 1

    print(f"Type gate passed with {sum(diagnostics.values())} baselined errors.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
