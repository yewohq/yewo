#!/usr/bin/env python3
"""Enforce controlled language in the prose Yewo ships, with no dependencies.

The agent-development skill asks for short active sentences, one instruction
per sentence, and one name for one thing. Those rules are checked by the
ASD-STE100 skill's own linter, so this gate finds that linter and runs it
rather than holding a second copy of the rules that can disagree with it.

What this file owns is the extraction. A linter reading a `.py` file counts
identifiers and argv as words and reports on code nobody wrote as prose, so
only the English is passed to it: docstrings, comments, the sentences
assigned to a name or handed back from a function, the literal halves of an
f-string, and every `description=` a call carries.

The linter is vendored beside this file, so CI runs the same rules a local
check runs. A machine with the skill installed uses that copy instead, which
keeps a rule fix one edit rather than two. A checkout with neither reports
that and passes, because a gate that fails for a missing tool stops work it
cannot speak about.
"""

from __future__ import annotations

import argparse
import ast
from collections import Counter
from dataclasses import dataclass
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
import tomllib
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
BASELINE_PATH = Path(__file__).resolve().with_name("ste-baseline.toml")

LINTER_ENV = "YEWO_STE_LINT"
SKILL_LINTER = Path.home() / ".claude/skills/asd-ste100/scripts/ste-lint.py"
VENDORED_LINTER = Path(__file__).resolve().with_name("ste_lint.py")

COMMENT = re.compile(r"\s*#\s?(.+)")
MACHINERY = ("noqa", "type:", "ruff:", "pylint", "mypy", "fmt:")


def _find_linter() -> Path | None:
    """The linter to run, or None where this machine has none.

    The copy in this directory wins, so that a local run and a CI run enforce
    the same rules. A gate whose rules depend on what is installed on the
    machine reports a different answer per machine, which is not a gate.

    `YEWO_STE_LINT` overrides it. That is how a rule is tried before the
    copy here is updated, and the override is deliberate rather than
    whatever happened to be installed.
    """
    named = os.environ.get(LINTER_ENV)
    if named:
        path = Path(named).expanduser()
        return path if path.is_file() else None
    for candidate in (VENDORED_LINTER, SKILL_LINTER):
        if candidate.is_file():
            return candidate
    return None


SENTENCE_LENGTH = 40
"""How long an assigned string has to be before it counts as prose.

A name, a path and a log key are strings too. The rule has to separate those
from a prompt and a field description without holding a list of which names
carry prose, because a new prompt would not be on the list.
"""


def _assigned_prose(node: ast.AST) -> str | None:
    """A string assigned to a name, where it is long enough to be a sentence.

    A prompt lives in a module constant and a model-facing description lives
    in `Field(description=...)`. Both are read by a model, so both are prose
    this gate has to hold, and neither is a docstring or a comment.
    """
    if not isinstance(node, (ast.Assign, ast.AnnAssign)):
        return None
    if not isinstance(node.value, ast.Constant):
        return None
    value = node.value.value
    return value if isinstance(value, str) and _is_prose(value) else None


def _described(node: ast.AST) -> list[str]:
    """Every prose keyword a call passes, as a string or an f-string."""
    if not isinstance(node, ast.Call):
        return []
    found = []
    for keyword in node.keywords:
        if keyword.arg not in PROSE_KEYWORDS:
            continue
        if isinstance(keyword.value, ast.Constant) and isinstance(
            keyword.value.value, str
        ):
            found.append(keyword.value.value)
        else:
            found.extend(_formatted(keyword.value))
    return found


# A line of a file Yewo writes, rather than a line of English. A template
# for a .env or for a throwaway script is a long string too, and its own
# comments are not text this gate has anything to say about.
CODE_LINE = re.compile(r"^\s*(#|//|import\s|from\s|\w+\s*=[^=])")


def _is_prose(value: str) -> bool:
    """Whether a string is English rather than a file Yewo writes out."""
    if " " not in value or len(value) < SENTENCE_LENGTH:
        return False
    return not any(CODE_LINE.match(line) for line in value.splitlines())


LOG_METHODS = ("debug", "info", "warning", "error", "exception", "critical")
"""The logging calls whose first argument is a sentence a person reads.

A log line is read by somebody under pressure, looking for what broke. It is
the debug context the skill asks to hold to the same rules as a prompt.
"""

PROSE_KEYWORDS = (
    "description",
    "help",
    "content",
    "detail",
    "hint",
    "reason",
)
"""Keyword arguments that carry English rather than a value.

`description` reaches a model through a schema. `help` reaches a person
through `--help`. `content` is a message to a model. `detail`, `hint` and
`reason` are what a check or a report says went wrong.
"""


def _logged(node: ast.AST) -> list[str]:
    """The sentence a logging call opens with, where it is one.

    The arguments after it are values, and the format string is the English.
    """
    if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
        return []
    if node.func.attr not in LOG_METHODS or not node.args:
        return []
    first = node.args[0]
    if isinstance(first, ast.Constant) and _is_prose(first.value):
        return [first.value]
    return _formatted(first)


def _refused(node: ast.AST) -> list[str]:
    """Every sentence the exception of a `raise` was built with.

    A refusal is read twice: once by whoever reads the report, and once by
    whoever is deciding whether Yewo was right to refuse.
    """
    if not isinstance(node, ast.Raise) or not isinstance(node.exc, ast.Call):
        return []
    found = []
    for argument in node.exc.args:
        if isinstance(argument, ast.Constant) and _is_prose(argument.value):
            found.append(argument.value)
        found.extend(_formatted(argument))
        if isinstance(argument, ast.BinOp):
            found.extend(_joined(argument))
    return found


def _joined(node: ast.AST) -> list[str]:
    """One sentence built by adding or implicitly joining string literals."""
    parts = [
        part.value
        for part in ast.walk(node)
        if isinstance(part, ast.Constant) and isinstance(part.value, str)
    ]
    literal = "".join(parts)
    return [literal] if _is_prose(literal) else []


def _formatted(node: ast.AST) -> list[str]:
    """The literal halves of an f-string long enough to be a sentence.

    A prompt built with a value in it is still a prompt, and a refusal
    message is written the same way. The interpolations are left out: what
    they hold is decided at run time, and the words around them are not.
    """
    if not isinstance(node, ast.JoinedStr):
        return []
    literal = "".join(
        part.value
        for part in node.values
        if isinstance(part, ast.Constant) and isinstance(part.value, str)
    )
    return [literal] if _is_prose(literal) else []


def _returned(node: ast.AST) -> list[str]:
    """A sentence a function hands back rather than assigns.

    Several prompts are built and returned in one step, and a gate that read
    only assignments and docstrings never saw them.
    """
    if not isinstance(node, ast.Return) or not isinstance(node.value, ast.Constant):
        return []
    value = node.value.value
    return [value] if isinstance(value, str) and _is_prose(value) else []


def prose_of(path: Path) -> str:
    """Every word in one file that a person or a model reads as English.

    Docstrings and comments. The long strings assigned to a name or returned
    from a function. The literal halves of an f-string. The sentence a
    logging call opens with, and the ones a refusal was built with. And every
    prose keyword a call carries, such as `description`, `help` and `hint`.

    A prompt, a field description, a log line and a refusal are all read by
    somebody or something, so a gate that read only docstrings passed the
    text that matters most.
    """
    source = path.read_text()
    found: list[str] = []
    tree = ast.parse(source)

    # `ast.get_docstring` dedents what it returns, and the same string read off
    # the node does not. Both reach the linter, the two copies differ by their
    # indentation, and every docstring is then counted twice. Collect the
    # nodes that hold a docstring and read each one once.
    carries_docstring = set()
    for node in ast.walk(tree):
        body = getattr(node, "body", None)
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef,
                             ast.AsyncFunctionDef)) and body:
            first = body[0]
            if (
                isinstance(first, ast.Expr)
                and isinstance(first.value, ast.Constant)
                and isinstance(first.value.value, str)
            ):
                carries_docstring.add(id(first))

    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef,
                             ast.AsyncFunctionDef)):
            doc = ast.get_docstring(node)
            if doc:
                found.append(doc)
        elif isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant):
            if isinstance(node.value.value, str) and id(node) not in carries_docstring:
                found.append(node.value.value)

    for node in ast.walk(tree):
        assigned = _assigned_prose(node)
        if assigned:
            found.append(assigned)
        found.extend(_described(node))
        found.extend(_formatted(node))
        found.extend(_returned(node))
        found.extend(_logged(node))
        found.extend(_refused(node))

    for line in source.splitlines():
        match = COMMENT.match(line)
        if match and not match.group(1).startswith(MACHINERY):
            found.append(match.group(1))

    return "\n\n".join(dict.fromkeys(found))


@dataclass(frozen=True)
class HardViolation:
    """One exact hard finding, independent of its generated line number."""

    rule: str
    message: str
    match: str
    source: str

    def fingerprint(self) -> str:
        return json.dumps(
            (self.rule, self.message, self.match, self.source),
            ensure_ascii=True,
            separators=(",", ":"),
        )

    def render(self) -> str:
        return f"{self.rule}: {self.message} [{self.match}] — {self.source}"


@dataclass(frozen=True)
class Baseline:
    count: int
    digest: str


def _prose_block(lines: list[str], line_number: int) -> str:
    """The extracted prose block that owns one linter line."""
    if not 0 < line_number <= len(lines):
        return ""
    start = line_number - 1
    end = start + 1
    while start and lines[start - 1].strip():
        start -= 1
    while end < len(lines) and lines[end].strip():
        end += 1
    return " ".join(" ".join(lines[start:end]).split())


def hard_violations(linter: Path, prose: str) -> list[HardViolation]:
    """The exact hard findings in one file's extracted prose."""
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as handle:
        handle.write(prose)
        temporary = Path(handle.name)
    try:
        result = subprocess.run(
            [sys.executable, str(linter), "--json", str(temporary)],
            capture_output=True, text=True, timeout=120,
        )
    finally:
        temporary.unlink(missing_ok=True)

    try:
        report = json.loads(result.stdout)
    except json.JSONDecodeError:
        raise RuntimeError(
            f"STE linter produced no JSON (exit {result.returncode}): {result.stderr}"
        ) from None

    lines = prose.splitlines()
    found = []
    for finding in report.get("violations", []):
        if finding.get("level") != "advisory-free":
            continue
        line_number = int(finding.get("line", 0))
        found.append(
            HardViolation(
                rule=str(finding.get("rule", "unknown")),
                message=" ".join(str(finding.get("message", "")).split()),
                match=" ".join(str(finding.get("match", "")).split()),
                source=_prose_block(lines, line_number),
            )
        )
    return found


def _named(path: Path) -> str:
    """The name a baseline entry uses: relative to the repository where it can be.

    A path outside the repository keeps its own spelling. `relative_to`
    raises on one, and a gate that crashes on the file it was handed reports
    nothing about the files it could have read.
    """
    if not path.is_absolute():
        return str(path)
    try:
        return str(path.relative_to(REPOSITORY_ROOT))
    except ValueError:
        return str(path)


def _load_baseline() -> dict[str, Baseline]:
    if not BASELINE_PATH.is_file():
        return {}
    with BASELINE_PATH.open("rb") as baseline_file:
        entries = tomllib.load(baseline_file).get("files", [])
    return {
        entry["path"]: Baseline(count=entry["count"], digest=entry["digest"])
        for entry in entries
    }


def _digest(violations: list[HardViolation]) -> str:
    fingerprints = Counter(item.fingerprint() for item in violations)
    serialized = json.dumps(sorted(fingerprints.items()), separators=(",", ":"))
    return hashlib.sha256(serialized.encode()).hexdigest()


def _render_baseline(findings: dict[str, list[HardViolation]]) -> str:
    lines = [
        "# Existing hard STE findings, fingerprinted without generated line numbers.",
        "# A changed finding is new debt even when a file's total stays the same.",
    ]
    for path, violations in sorted(findings.items()):
        if not violations:
            continue
        lines.extend(
            [
                "",
                "[[files]]",
                f"path = {json.dumps(path)}",
                f"count = {len(violations)}",
                f"digest = {json.dumps(_digest(violations))}",
            ]
        )
    return "\n".join(lines) + "\n"


def _python_files(targets: list[Path]) -> list[Path]:
    files: list[Path] = []
    for target in targets:
        if target.is_dir():
            files.extend(sorted(target.rglob("*.py")))
        elif target.suffix == ".py":
            files.append(target)
    return [f for f in files if "__pycache__" not in f.parts]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="*", default=["src/yewo"])
    parser.add_argument(
        "--write-baseline",
        action="store_true",
        help="Record the current hard findings instead of enforcing the baseline",
    )
    args = parser.parse_args(argv)

    linter = _find_linter()
    if linter is None:
        print(
            f"STE gate skipped: no linter at {SKILL_LINTER} or {VENDORED_LINTER}. "
            f"Set {LINTER_ENV} to run it."
        )
        return 0

    baseline = _load_baseline()
    failures: list[str] = []
    checked = 0
    findings: dict[str, list[HardViolation]] = {}
    selected = {_named(path) for path in _python_files([Path(p) for p in args.paths])}

    for path in _python_files([Path(p) for p in args.paths]):
        prose = prose_of(path)
        if not prose.strip():
            continue
        checked += 1
        relative = _named(path)
        violations = hard_violations(linter, prose)
        findings[relative] = violations

    if args.write_baseline:
        BASELINE_PATH.write_text(_render_baseline(findings), encoding="utf-8")
        count = sum(len(violations) for violations in findings.values())
        print(f"Recorded {count} hard STE findings across {len(findings)} files.")
        return 0

    for relative, violations in findings.items():
        recorded = baseline.get(relative)
        current = Baseline(len(violations), _digest(violations))
        if violations and current != recorded:
            allowed = recorded.count if recorded else 0
            failures.append(
                f"{relative}: hard findings changed ({len(violations)} found, "
                f"{allowed} recorded)"
            )
            failures.extend(f"    {item.render()}" for item in violations[:6])
        elif not violations and recorded is not None:
            failures.append(f"{relative}: all hard findings were fixed; rewrite the baseline")

    for relative in sorted(set(baseline) & selected - set(findings)):
        failures.append(f"{relative}: no prose remains; rewrite the baseline")

    if failures:
        print("\n".join(failures))
        print(f"\nSTE gate failed across {checked} file(s).")
        return 1

    print(f"STE gate passed for {checked} file(s) with prose.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
