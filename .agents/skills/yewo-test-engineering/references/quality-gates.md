# Yewo quality gates

The clean rebuild does not inherit allowances from `old_src`, `old_tests`, or
`old_tools`.

## Active gates

Run:

```bash
uv run ruff check --ignore C90 src tests tools
uv run ruff check --select C90 src tests tools
uv run pyright
uv run pytest
uv run python tools/quality_gate.py
uv run python tools/ste_gate.py
uv run python tools/package_gate.py
```

`quality_gate.py` rejects a Python file above 550 physical lines. It reads
`src/yewo` and `tests` by default. It has no exception baseline.

`ste_gate.py` extracts human-facing and model-facing prose from Python source.
It passes that prose to the external `asd-ste100` linter. Yewo does not copy or
modify the STE rules. Set `YEWO_STE_LINT` if the skill is not installed under
`~/.agents/skills` or `~/.claude/skills`.

The gates run with Python 3.12 in checks and release verification. CI checks
out a pinned revision of the external STE skill.

Ruff enforces correctness rules separately from its `C90` cyclomatic
complexity rules. Pyright checks the active production tree. Pytest exercises
the active test tree. `package_gate.py` builds the distributions, installs the
wheel in a new environment, imports `yewo`, and starts the packaged CLI.

## Conditional checks

Add a check only when its subject and acceptance rule exist:

- A coverage threshold requires a stable test suite and a justified floor.
- Mutation testing requires a critical deterministic contract and a focused
  mutant set.
- Schema validation requires a published, versioned schema.
- An evaluation gate requires a model-driven or stochastic component, a
  representative corpus, and explicit acceptance criteria.

Write focused deterministic tests for each new contract.

## Static design rules

The absence of a gate does not authorize poor structure.

- Keep one owner for validation rules and conversions.
- Do not add dead code or duplicate domain vocabularies.
- Use explicit types at internal boundaries.
- Normalize dynamic SDK and model values at their boundary.
- Keep platform-specific mechanics in platform modules.
- Keep model-provider behavior behind LiteLLM.
- Use comments only for rationale, units, ownership, platform constraints, or
  retry safety that code cannot express.

Do not add cognitive-complexity, Halstead, or CRAP-number gates. Reconsider
them only if the active checks and focused tests miss a specific
maintainability failure.
