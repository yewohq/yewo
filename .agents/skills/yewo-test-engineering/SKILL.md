---
name: yewo-test-engineering
description: Add, review, or run tests for Yewo, including offline Python checks, journey execution, device integration, reports, and usage accounting. Use when changing observable behavior or when test coverage or validation is requested.
---

# Yewo test engineering

Separate tests of Yewo from journeys run by Yewo. Python tests verify agent
logic and contracts; journeys evaluate an application on a device. Neither is
evidence for the other.

## Design valuable tests

Identify the observable contract, credible regression, and correct production
boundary before adding a test. Prefer realistic inputs and expected values that
are independent of the implementation. Do not add assertion-free probes,
source-text checks, implementation-shaped assertions, or production wrappers
created only to expose internals.

For a bug, reproduce the failure before the fix when practical. Before deleting
or weakening a test, inspect what failure it detects and what independent proof
would remain.

Use deterministic fakes at LLM, device, inference, clock, and filesystem
boundaries while exercising real Yewo logic. Ordinary tests must not require
credentials, network access, Appium, an emulator, OCR initialization, or model
downloads. Use temporary directories and avoid sleeps and test-order coupling.

Prioritize the contract at risk, especially:

- malformed or incomplete structured model responses
- action validation, coordinates, keyboard reflow, and platform differences
- bounded execution, stale observations, interruption, and partial cleanup
- journey parsing and preservation of security rules
- scenario aggregation, skipped/aborted semantics, XML escaping, and safe paths
- CLI defaults, invalid arguments, exit codes, and no-device operation
- model attribution, failed calls, unknown cost, and per-run usage isolation

## Choose the correct validation tier

Read and apply the [quality-gates reference](references/quality-gates.md) for
production Python changes. It defines the enforced complexity and file-size
ratchets, the coverage and mutation-testing scope, and the policy for types,
dead code, duplication, and deferred metrics.

Ruff, Pyright, pytest, and coverage are declared in the `dev` dependency
group and configured in `pyproject.toml`; `.github/workflows/checks.yml` runs
them. The focused mutation sweep in `tools/mutation_sweep.py` checks realistic
judge and evaluation-gate defects in isolated parallel workers. Existing tests are
`unittest`-style and run under pytest unchanged. Each file covers one
contract, named for it, and `tests/fakes.py` builds the values the boundaries
hand to the logic through their real constructors. Do not report unconfigured
tooling as available. If a task adopts additional quality tooling, declare and
configure it before relying on it.

Start with the smallest relevant offline command. Available baseline checks are:

```bash
uv run pytest tests/test_targeting.py
uv run pytest
uv run ruff check
uv run python tools/quality_gate.py
uv run python tools/type_gate.py
uv run coverage run -m pytest && uv run coverage report
uv run python -m tools.mutation_sweep
git diff --check
```

Run the narrowest one that covers the change before broadening it. `uv run
python -m unittest discover -s tests` remains equivalent to a bare `uv run
pytest`.

Live journey execution is an explicit integration tier:

```bash
uv run yewo test path/to/journey.yaml --app-path path/to/app
```

It may start device infrastructure and make paid model requests. Run it only
when the task requires that validation and the needed app, device, credentials,
and drivers are available. Record model, settings, app version, platform, and
run conditions; do not infer reliability from one stochastic success.

A rule that was not observed remains untested. Never turn missing evidence,
zero generated scenarios, aborted setup, or skipped execution into a pass.
Report exact commands, outcomes, unavailable checks, and remaining risks.
