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
production Python changes. The active gates are the 550-line limit, STE, Ruff,
cyclomatic complexity, Pyright, and pytest.

Use the lowest validation tier that proves the contract, then broaden only when
the remaining boundary matters:

1. Unit and offline integration tests exercise deterministic Yewo logic.
2. Controlled-app journeys exercise the complete stack under known conditions.
3. Third-party or release-candidate journeys measure behavior outside the
   controlled app.

One tier is not evidence for another. A source-checkout test does not prove the
wheel, and one live journey does not establish reliability.

Add a conditional check only when its subject exists:

- add a coverage threshold when a stable test suite provides a meaningful
  baseline;
- add mutation tests when a critical deterministic contract needs evidence
  that its tests detect behavioral changes;
- add schema checks when Yewo publishes a versioned schema;
- add an evaluation gate when a model-driven component makes a measurable
  quality claim against a representative corpus and explicit acceptance rule.

The active checks are:

```bash
uv run ruff check --ignore C90 src tests tools
uv run ruff check --select C90 src tests tools
uv run pyright
uv run pytest
uv run python tools/quality_gate.py
uv run python tools/ste_gate.py
```

Add focused deterministic tests for each changed production contract.

CI owns wheel build and installed-package verification. Do not reproduce that
work as a local development gate.

Use a proportional validation cadence. During implementation, run only the
focused tests and static checks that cover the changed contract. Run the full
offline pytest suite after the change is coherent and once before handoff or
commit. Do not rerun the complete suite after every small edit when a focused
test provides the needed evidence.

Live journey execution is an explicit integration tier:

```bash
uv run yewo test path/to/journey.yaml --app-path path/to/app
```

It may start device infrastructure and make paid model requests. Run it only
when the task requires that validation and the needed app, device, credentials,
and drivers are available. Record model, settings, app version, platform, and
run conditions; do not infer reliability from one stochastic success.

For comparative evaluations, change one independent variable at a time and
name any variable that remains uncontrolled. Use held-out inputs for quality
claims. Record the exact artifact versions, settings, hardware, and warm or
cold state needed to reproduce the comparison. Report correctness, latency,
and memory separately; improvement in one does not imply improvement in the
others.

A rule that was not observed remains untested. Never turn missing evidence,
zero generated scenarios, aborted setup, or skipped execution into a pass.
Report exact commands, outcomes, unavailable checks, and remaining risks.
