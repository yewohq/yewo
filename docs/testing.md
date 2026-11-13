# Testing with Yewo

This guide covers repeatable, report-producing tests. For a one-off automation
task, use `yewo run <goal>` instead.

## Journeys and workflows

A journey describes what a user does and what the app must protect. A workflow
binds one or more journeys to their environment, app, device, model, and policy.

```yaml
device:
  name: Pixel_8
  platform: android

app: build/app-release.apk

workflows:
  account:
    env:
      email: tester@example.com
      password:
        value: ${YEWO_TEST_PASSWORD}
        secret: true
    journeys:
      - id: sign_in
        name: Sign in and open settings
        journey: |
          Sign in with {{email}} and {{password}}, then open Settings.
        security_rules:
          - The password must never be shown in plain text.
```

Run every workflow or select one by name:

```bash
yewo test
yewo test --workflow account
```

The command-line flags, `YEWO_*` environment variables, the nearest
`yewo.yaml`, and built-in defaults provide configuration in that order.

## Inputs and secrets

`env` declares values used by a journey. `{{name}}` inserts a declared value
into the journey text. `${VAR}` reads it from the process environment and fails
the journey when that variable is absent.

A mapping with `secret: true` marks sensitive data. A top-level `secrets:` list
of environment names is also supported. The agent receives the real value
because it must be able to type and audit it.

What happens to that value in saved evidence depends on what is installed.
Yewo on its own does not hide it: screenshots, screen recordings, and device
logs are written and sent exactly as the device produced them. A run that
declares secret values says so before it starts.

An installed extension can hide them before they reach any artifact. Yewo Cloud
provides one, covering declared values in screenshots, in every frame of a
screen recording, and in device log text. Run `yewo doctor` to see what this
install has.

## Step and repair budgets

`max_steps` bounds the steps a scenario's own instructions ask for. Turns a
scenario spends repairing its execution are counted and bounded separately:
probing for a control perception did not report, and re-observing after a
recovery, spend a repair allowance rather than the instruction budget. A
scenario that exhausts its instructions ends as `step_limit`. One that
exhausts repair ends as `intervention_limit`. Both are reported as unverified
and attributed to the agent, not to the app.

`run.json` records both counts per scenario, as `steps_executed` and
`intervention_steps`, so a slow scenario can be priced.

## Coverage and generated scenarios

Coverage tells Yewo which kinds of scenario to generate:

```yaml
coverage:
  - category: functional
    types: [happy_path, negative]
    priority: P0
  - category: security
    types: [negative]
    priority: P1
```

Yewo generates one scenario per coverage cell. A journey with no coverage gets
one P0 functional happy-path cell. Each cell has a stable identity such as
`security.negative.0`, so it can be compared between runs.

## Test the same journey on several targets

A workflow can declare a sequential target matrix:

```yaml
workflows:
  checkout:
    targets:
      - id: android_emulator
        platform: android
        device: Pixel_8_API_35
        app: build/app-release.apk
      - id: ios_simulator
        platform: ios
        device: iPhone 16 Pro
        app: build/Runner.app
    journeys:
      - id: guest_checkout
        journey: Let a guest complete checkout.
```

Every declared target runs by default. Select targets by ID with a
comma-separated list:

```bash
yewo test
yewo test --target ios_simulator
yewo test --target android_emulator,ios_simulator
```

Targets run one at a time. Each journey and target pair gets its own run ID,
device session, state transaction, evidence, and verdict. A failure or an
unsupported capability on one target does not suppress later targets.

## App state and device preconditions

Independent scenarios clear app data by default. A journey can request a
different app state or a typed device capability:

```yaml
preconditions:
  app_state: relaunch
  capabilities:
    - kind: permission
      permission: camera
      grant: false
journey: Open the camera while camera access is denied.
```

Yewo applies and confirms preconditions before model-driven UI work, then
restores device state after the scenario. Preconditions do not create backend
data or replace visible actions in the journey. Credentials, payment details,
and other non-observable values belong in `env`.

## Evidence boundaries

Yewo can support an assertion with:

- A screenshot and the text or controls detected on it.
- A validated action and the screen change it caused.
- A supported device measurement or capability receipt.

Network traffic, HTTP headers, screen-reader behavior, focus order, and measured
contrast need separate tooling. When a scenario requires evidence the current
target cannot provide, Yewo reports `UNSUPPORTED` rather than guessing.

Accessibility data used for supported iOS measurements never reaches the
planner and never controls an action.

## Measurements

Android scenarios that restart the app record cold-process launch TTID as
diagnostic telemetry. A diagnostic sample does not affect the verdict. When a
journey explicitly declares a launch limit, the same OS-reported value becomes
required evidence and its threshold is judged deterministically.

On iOS 17 or newer, a journey can require `min_touch_target` or
`unlabelled_actionable_count`. Yewo derives these from Apple's accessibility
audit after observing the screen. Raw audit descriptions are not stored.

Every sample carries its unit, source, collection time, target context, app
state, and whether it was required or diagnostic. Missing required evidence
produces `UNVERIFIED`. iOS launch timing is not currently available because
Appium command duration is not an application first-frame measurement.

## Reports and run records

Yewo writes each run under `assets/yewo/<run-id>/` unless `output_dir` in
`yewo.yaml` selects another output root. Every run has one `run.json`;
Markdown, HTML, and JUnit outputs are views over that record. Reports are
checkpointed during a run so an interruption still leaves the evidence reached
so far.

A target-matrix execution also writes an `executions/<execution-id>/index.html`
summary that links to the independent reports. It combines JSON and JUnit
results but does not merge screenshots, measurements, or lifecycle receipts
between targets.

## Verdict semantics

| Verdict | Meaning |
|---|---|
| `PASSED` | Every declared outcome was evidenced on an independently judged screen. |
| `FAILED` | The app crashed, exposed a critical defect, or broke a declared rule. |
| `UNVERIFIED` | Execution finished but the required proof did not arrive. |
| `SKIPPED` | The planner established that the scenario could not be carried out here. |
| `UNSUPPORTED` | The target cannot provide a required state or evidence source. |
| `ERROR` | A device, model provider, or Yewo failed outside the app. |
| `ABORTED` | The run was interrupted before reaching a verdict. |

Only `PASSED` is a verified success. Only `FAILED` assigns blame to the app.
An incomplete run is not converted into an app failure.

## Useful options

```bash
yewo test --help
yewo test --workflow account,checkout
yewo test --target android_emulator,ios_simulator
```

`yewo test` selects work; it does not configure a run. `headless`, `record`,
and `output_dir` are declared in `yewo.yaml`, either at the top level or on one
workflow, device, or target.

See [`yewo-schema.json`](../yewo-schema.json) for the complete configuration
shape. Maintained applications and journeys for exercising Yewo live under
[`eval/`](../eval/README.md).
