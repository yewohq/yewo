# Yewo quality gates

Use these gates for production Python changes. They are a ratchet: existing
debt is recorded explicitly, and a change must not add to it.

## Automated baseline

Run:

```bash
uv run python tools/quality_gate.py
uv run ruff check
uv run python tools/type_gate.py
```

The gates live in `tools/` rather than beside this skill because `.agents/` is
not tracked, and `.github/workflows/checks.yml` has to be able to run them.

`tools/quality_gate.py` is dependency-free and enforces:

- at most 550 physical lines per Python file
- no growth in explicitly baselined file-size exceptions

Cyclomatic complexity below 22 is ruff's `C901`, configured in
`pyproject.toml`. One tool owns that number. Counting it in both places meant
two numbers per function from two algorithms that disagree, and the baselined
figure came from the one CI did not enforce.

It reads `src/yewo` and `tests` by default. A test file is held to the same
size limit as production code: a file that covers several contracts at once
grows without a reason to stop, and a failure in it does not say what broke.

The baseline lives beside the script in `tools/quality-baseline.toml` and is
currently empty: every file is under the limit and every function under
complexity 22. Reducing a baselined file is progress, so lower its recorded
ceiling in the same change. Never add an entry merely to make a check pass.

## Tests and coverage

Aim for complete line and branch coverage of deterministic behavior added or
changed by the task. Treat 100% as a local change standard, not a claim that the
whole repository is already at 100%. Do not write low-value assertions or mock
the behavior under test just to improve the number.

If a changed branch can only be exercised with a real model or device, cover
the offline decision logic beneath that boundary and report the remaining
integration gap.

Repository-wide coverage is measured by `uv run coverage run -m pytest` plus
`uv run coverage report`, with line-and-branch `fail_under` in
`pyproject.toml`. The floor is 74%, just below the measured cross-platform
result. Raise it as coverage improves rather than treating it as the standard
for changed code.

For safety-critical pure logic—action validation, coordinate conversion,
status aggregation, path handling, redaction, and cleanup decisions—use the
focused mutation sweep when the subsystem has declared mutants. Its isolated
workers must catch every declared mutant. Add realistic mutants during a
hardening pass; do not run mutation tests across device or paid-model boundaries.

## Static quality

- Introduce no dead imports, unreachable branches, unused helpers, or duplicate
  implementations. Confirm ownership and callers before deleting suspected
  dead code.
- Do not add `typing.Any` merely to silence a type checker. At genuinely dynamic
  SDK or model boundaries, validate and narrow immediately to a Pydantic model,
  protocol, typed mapping, or `object`. Python has no `unknown` type.
- Apply DRY to knowledge, validation rules, and conversions—not coincidental
  syntax. Extract a shared owner when duplication can drift; avoid premature
  abstraction for two similar-looking call sites.

Ruff is configured in `pyproject.toml` with `E`, `F`, `B`, `SIM`, and `C90` at
the same complexity threshold. `E501` is off: the formatter owns code width,
and the long lines left are prompt text that must not be rewrapped. Existing
violations are recorded in `[tool.ruff.lint.per-file-ignores]`, which is a
ratchet like the baseline above — remove entries, never add them.

`ruff format` is configured but not yet enforced, because the first run
reformats most of the repository. Land that as its own mechanical commit, then
add `ruff format --check` to the workflow.

Pyright runs in `basic` mode over `src/yewo` through `tools/type_gate.py`.
`tools/type-baseline.toml` fingerprints each existing diagnostic by file, rule,
and whitespace-normalized message. This means that a different error cannot
reuse an old error's allowance merely because the per-file count stayed the
same. Every file stays checked; whole-file suppression would hide new mistakes
in the modules that already have the most. The gate also fails when an error is
fixed until the stale allowance is removed. Regenerate the baseline with
`--write-baseline` after verified improvements or intentional gate relocation,
never to absorb new errors.

The STE gate uses the same ratchet rule. Its baseline fingerprints each hard
finding by rule, message, matched text, and normalized source text. A different
finding cannot inherit an old finding's allowance because the totals match.
Rewrite the baseline only after verified prose improvements or an intentional
change to the vendored STE rules.

## Metrics not used as gates

Do not add separate cognitive-complexity, Halstead-difficulty, or CRAP-number
gates yet. They overlap substantially with cyclomatic complexity, file size,
and coverage while requiring more tooling and baselining. Use them during a
focused maintainability audit if they reveal a decision that the active gates
miss; do not optimize code solely to move a metric.
