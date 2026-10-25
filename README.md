# Yewo

Yewo is an AI-powered end-to-end testing agent for Android and iOS apps. Give it a user journey or a single goal. It reads the screen, plans actions, and drives an emulator or a device through Appium. A separate component judges what the screen then shows.

> Yewo is currently alpha software.

## Requirements

- [uv](https://docs.astral.sh/uv/) and Python 3.12+
- Node.js 18+ (required by Appium)
- Android Studio and the Android SDK for Android testing
- Xcode and its command-line tools for iOS testing (macOS only)

The Flutter SDK is only needed to build the included sample app.

## Quick start

Clone the repository and install Yewo with its vision dependencies:

```bash
git clone https://github.com/getyewo/yewo.git
cd yewo
uv sync --extra vision
```

Run the guided setup, then check your environment:

```bash
uv run yewo setup
uv run yewo doctor
uv run yewo devices
```

`yewo setup` installs Appium, the driver for your configured platform, and the required model files. Use `yewo setup --dry-run` to preview its changes or `yewo setup -y` for non-interactive setup.

Add the API key required by your chosen [LiteLLM-supported model](https://docs.litellm.ai/docs/providers) to a `.env` file or your shell environment. The default model uses OpenRouter:

```env
OPENROUTER_API_KEY=your_key_here
```

## Test a user journey

A journey file describes the flow to test and any security rules Yewo should enforce:

```yaml
id: sign_in
name: Sign in and open settings
env:
  email: test@example.com
  password:
    value: ${YEWO_TEST_PASSWORD}
    secret: true
journey: >
  Sign in with {{email}} and {{password}}, then open the settings screen.
security_rules:
  - The user's password must never be shown in plain text.
```

`env` declares the values a journey uses and `{{name}}` inserts them into its
text. `${VAR}` reads from the environment, so a real credential never has to be
committed. Write a plain value for ordinary data, or give it a `value` and
`secret: true` to mark it sensitive — a top-level `secrets:` list of names works
too. A `${VAR}` that is not set fails the journey rather than running as a
blank value.

A declared secret is masked by name in everything Yewo writes down. The
reports, the record, the logs and the scan dumps all show `{{password}}`, and
the screen regions holding it are blacked out in saved screenshots. **A screen
recording is not redacted.** A journey that declares a secret keeps no video
unless you ask for one.

The agent itself always receives the real value. It cannot type a password it
was not given. It cannot audit a rule like "the card number must never be
shown in plain text" if the card number is hidden from it.

There are two ways to run one. Point Yewo at a journey file:

```bash
uv run yewo test path/to/journey.yaml --app-path path/to/app.apk
```

Or keep the settings and the tests in one `yewo.yaml`, which is what a
project does. A workflow owns an app, a device and an environment, and the
journeys inside it are the tests. Run it from the directory that holds the
file. See [`example/testing_app/yewo.yaml`](example/testing_app/yewo.yaml).

```bash
cd example/testing_app
uv run yewo test                           # every workflow
uv run yewo test --workflow notifications  # one
```

## What Yewo generates

A journey declares its **coverage**: which kinds of test to write, and how
important each one is. Yewo enumerates the cells and asks for one scenario
per cell.

```yaml
coverage:
  - category: functional
    types: [happy_path, negative]
    priority: P0
  - category: security
    types: [negative]
    priority: P1
```

A file that declares no coverage gets one cell: a P0 functional happy path.
Such a file tells Yewo nothing about the app, so Yewo writes the most
conservative test it can.

Each cell keeps a stable name, such as `security.negative.0`, so the same
test is the same test between runs. `yewo lock` writes those scenarios to
`yewo.lock`. `yewo test --locked` runs the locked scenarios and refuses a
lock that is missing or out of date, which is what a build should do.

Yewo only writes assertions it can evidence from a screenshot, the detected
text on it, and what an action did. Network traffic, HTTP headers,
accessibility trees, screen readers, focus order and measured contrast need
separate tooling. A scenario that asks for one is reported as unsupported
rather than guessed at.

Independent scenarios start with cleared app data by default. A journey may
declare a different app state or a typed device precondition:

```yaml
preconditions:
  app_state: relaunch
  capabilities:
    - kind: permission
      permission: camera
      grant: false
journey: The user opens the camera while camera access is denied.
```

Yewo applies and confirms these conditions before model-driven UI work, then
restores device state after the scenario. Preconditions never create backend
data or replace an action in the visible user journey. Credentials, payment
details, and other non-observable values still belong in `env`.

## What a verdict means

| Verdict | Meaning |
|---|---|
| `PASSED` | Every declared outcome was evidenced on a screen judged without the plan |
| `FAILED` | The app. A crash, a critical defect, a broken rule, or a security rule violated |
| `UNVERIFIED` | The walk finished and the proof did not arrive. The cause says whose problem that is |
| `SKIPPED` | The planner reported that the scenario cannot be carried out here |
| `UNSUPPORTED` | This target cannot give the scenario the state or the evidence it declared |
| `ERROR` | Something outside the app failed: a device, a model provider, or Yewo |
| `ABORTED` | The run was interrupted and the scenario never reached a verdict |

Only `PASSED` is a verified success. `FAILED` is the only verdict that blames
the app.

Every run writes one `run.json`: the record of what was tested and what each
model call asked. The Markdown and JUnit reports are views over it. All of
them are written atomically, and the Markdown and JUnit are checkpointed
during the run, so an interrupted run still leaves what it reached.

## Run a single goal

For one-off automation without generated test scenarios or QA reports:

```bash
uv run yewo run "Sign in and open the settings screen" \
  --app-path path/to/app.apk
```

Android APKs and iOS `.app` or `.ipa` bundles are supported.

## Build the sample app

The example journey targets the Flutter app in `example/testing_app`.

Android:

```bash
cd example/testing_app
flutter build apk --debug
```

iOS Simulator:

```bash
cd example/testing_app
flutter build ios --simulator
```

Return to the repository root before running Yewo. The generated app paths are:

- Android: `example/testing_app/build/app/outputs/flutter-apk/app-debug.apk`
- iOS: `example/testing_app/build/ios/iphonesimulator/Runner.app`

## Configuration

Inspect or change the saved configuration with:

```bash
uv run yewo config
uv run yewo config set device Pixel_8a
uv run yewo config set model openrouter/google/gemini-2.5-pro
```

Configuration is resolved in this order: CLI flags, `YEWO_*` environment variables, the nearest `yewo.yaml`, then built-in defaults. Writes go to `./yewo.yaml`, which is also where the journeys live.

## Commands

| Command | Purpose |
|---|---|
| `yewo setup` | Install and configure runtime dependencies |
| `yewo doctor` | Check prerequisites and report fixes |
| `yewo devices` | List, create, and select devices |
| `yewo config` | View or change settings |
| `yewo test <file>` | Generate and run tests from a journey file |
| `yewo lock` | Write the generated scenarios to `yewo.lock` |
| `yewo run <goal>` | Drive an app toward a single goal |

Run `uv run yewo <command> --help` for all options. Add `--log-level DEBUG` before the command for detailed logs.

## Development

Install the tooling, then run the checks that CI runs:

```bash
uv sync --group dev --extra jev
uv run ruff check
uv run python tools/quality_gate.py
uv run python tools/ste_gate.py
uv run python tools/emit_schema.py --check
uv run coverage run -m pytest && uv run coverage report
uv run python -m tools.eval_gate
uv run python -m tools.mutation_sweep
```

The type gate runs in its own job, because Pyright reads imports the other
checks do not install:

```bash
uv sync --group dev --extra vision --extra jev
uv run python tools/type_gate.py
```

What each one holds:

| Gate | Holds |
|---|---|
| `quality_gate.py` | 550 lines per file, tests included. No dependencies |
| `ste_gate.py` | Controlled language in prose a model or a person under pressure reads |
| `emit_schema.py --check` | The editor schema matches the models a `yewo.yaml` is read through |
| `eval_gate.py` | The judging still answers a corpus of labelled cases as it should |
| `mutation_sweep.py` | Deliberate defects are still caught by something |
| `type_gate.py` | Pyright, against a per-file baseline |
