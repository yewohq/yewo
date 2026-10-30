# Developing Yewo

Clone the repository. The quality gates require Python 3.12. Install the
`asd-ste100` skill so that
`~/.agents/skills/asd-ste100/scripts/ste-lint.py` is available:

```bash
git clone https://github.com/getyewo/yewo.git
cd yewo
```

Run the checks used by CI:

```bash
uv sync --group dev
uv run ruff check --ignore C90 src tests tools
uv run ruff check --select C90 src tests tools
uv run pyright
uv run pytest
uv run python tools/quality_gate.py
uv run python tools/ste_gate.py
uv run python tools/package_gate.py
```

## What the gates protect

| Gate | Contract |
|---|---|
| `quality_gate.py` | Python files under `src/yewo` and `tests` stay at or below 550 physical lines. |
| `ste_gate.py` | Passes new Python prose to the external STE linter. |
| Ruff | Rejects static defects and functions with cyclomatic complexity above 21. |
| Pyright | Checks types in the active production package. |
| pytest | Runs focused deterministic tests. |
| `package_gate.py` | Builds and installs the wheel in a clean environment, then checks its import and CLI. |

Yewo does not contain a copy of the STE linter. Set `YEWO_STE_LINT` when the
linter is installed at another location. CI checks out a pinned upstream
revision and passes its path through this variable.

The gates have no allowances inherited from the archived implementation.

## Evaluation workspace

Maintained benchmark inputs live under `eval/`. Deterministic transport and
provider-error contracts belong in `tests/`. Generated benchmark runs remain
outside version control unless a review deliberately promotes them as a
baseline.
