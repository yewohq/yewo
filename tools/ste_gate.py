#!/usr/bin/env python3
"""Pass prose from Yewo Python files to the external STE linter."""

from __future__ import annotations

import argparse
import ast
import io
import os
from pathlib import Path
import subprocess
import sys
import tokenize


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TARGETS = (REPOSITORY_ROOT / "src" / "yewo",)
LINTER_ENV = "YEWO_STE_LINT"
LINTER_LOCATIONS = (
    Path.home() / ".agents/skills/asd-ste100/scripts/ste-lint.py",
    Path.home() / ".claude/skills/asd-ste100/scripts/ste-lint.py",
)
COMMENT_MARKERS = ("noqa", "type:", "ruff:", "pylint", "mypy", "fmt:")


def _find_linter() -> Path | None:
    configured = os.environ.get(LINTER_ENV)
    if configured:
        path = Path(configured).expanduser()
        return path if path.is_file() else None
    return next((path for path in LINTER_LOCATIONS if path.is_file()), None)


def _python_files(targets: list[Path]) -> list[Path]:
    files: set[Path] = set()
    for target in targets:
        if target.is_dir():
            files.update(target.rglob("*.py"))
        elif target.is_file() and target.suffix == ".py":
            files.add(target)
    return sorted(path for path in files if "__pycache__" not in path.parts)


def _is_prose(text: str) -> bool:
    words = [
        word
        for word in text.split()
        if any(character.isalpha() for character in word)
    ]
    return len(words) >= 3


def _comments(source: str) -> list[str]:
    found: list[str] = []
    tokens = tokenize.generate_tokens(io.StringIO(source).readline)
    for token in tokens:
        if token.type != tokenize.COMMENT:
            continue
        text = token.string.removeprefix("#").strip()
        if text and not text.startswith(COMMENT_MARKERS):
            found.append(text)
    return found


def _docstring_nodes(tree: ast.AST) -> set[int]:
    found: set[int] = set()
    owners = (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
    for node in ast.walk(tree):
        body = getattr(node, "body", None)
        if not isinstance(node, owners) or not body:
            continue
        first = body[0]
        if (
            isinstance(first, ast.Expr)
            and isinstance(first.value, ast.Constant)
            and isinstance(first.value.value, str)
        ):
            found.add(id(first.value))
    return found


def _strings(tree: ast.AST) -> list[str]:
    docstring_nodes = _docstring_nodes(tree)
    formatted_parts = {
        id(part)
        for node in ast.walk(tree)
        if isinstance(node, ast.JoinedStr)
        for part in node.values
        if isinstance(part, ast.Constant)
    }
    found: list[str] = []
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Constant)
            and isinstance(node.value, str)
            and id(node) not in docstring_nodes
            and id(node) not in formatted_parts
            and _is_prose(node.value)
        ):
            found.append(node.value)
        elif isinstance(node, ast.JoinedStr):
            text = "".join(
                part.value
                for part in node.values
                if isinstance(part, ast.Constant) and isinstance(part.value, str)
            )
            if _is_prose(text):
                found.append(text)
    return found


def _prose(path: Path) -> str:
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(path))
    found = _comments(source)
    found.extend(_strings(tree))
    owners = (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
    for node in ast.walk(tree):
        if isinstance(node, owners):
            docstring = ast.get_docstring(node, clean=True)
            if docstring:
                found.append(docstring)
    return "\n\n".join(dict.fromkeys(found))


def _display_path(path: Path) -> Path:
    try:
        return path.resolve().relative_to(REPOSITORY_ROOT)
    except ValueError:
        return path.resolve()


def _lint(linter: Path, path: Path, prose: str) -> int:
    result = subprocess.run(
        [sys.executable, str(linter)],
        input=prose,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    if result.returncode:
        print(f"{_display_path(path)}:", file=sys.stderr)
        output = result.stdout or result.stderr or "STE linter failed without output."
        print(output.rstrip(), file=sys.stderr)
    return result.returncode


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="*", type=Path)
    args = parser.parse_args(argv)

    linter = _find_linter()
    if linter is None:
        locations = ", ".join(str(path) for path in LINTER_LOCATIONS)
        print(
            f"STE gate failed: set {LINTER_ENV} or install the linter at {locations}.",
            file=sys.stderr,
        )
        return 2

    requested = args.paths or list(DEFAULT_TARGETS)
    targets = [
        path if path.is_absolute() else REPOSITORY_ROOT / path
        for path in requested
    ]
    checked = 0
    failed = False
    for path in _python_files(targets):
        prose = _prose(path)
        if not prose:
            continue
        checked += 1
        failed = bool(_lint(linter, path, prose)) or failed

    if failed:
        print(f"STE gate failed across {checked} file(s).", file=sys.stderr)
        return 1
    print(f"STE gate passed for {checked} file(s) with prose.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
