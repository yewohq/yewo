# Yewo

Yewo is an AI testing agent for Android and iOS apps. Describe a user journey
in plain language and Yewo operates the real app, checks what happened, and
produces an evidence-backed report.

Home page: [tryyewo.dev](https://tryyewo.dev)

Yewo looks at the same app screenshots a user sees. Visual models identify
controls and locate where to tap, so tests do not depend on accessibility IDs,
test IDs, or other app-specific selectors. During a journey test, the agent
that operates the app does not decide whether the test passed. After execution,
a separate auditor compares the saved evidence with the expected outcomes.

> Yewo is alpha software. Expect breaking changes while its public interface is
> being established.

## What it does

- Runs plain-language goals and journeys against Android and iOS builds.
- Finds and interacts with visible controls without requiring test IDs.
- Discovers a journey before generating evidence-grounded test scenarios.
- Checks functional, security, and supported device-level outcomes.
- Saves screenshots, actions, model calls, usage, and verdict evidence.
- Masks declared secrets in stored text and redacts them from saved screenshots.
- Produces JSON, HTML, Markdown, and JUnit reports for journey tests.

## Quick start

You need Python 3.12+, a Node.js runtime compatible with Appium, and the SDK for
the platform you want to test:

- Android Studio and the Android SDK for Android.
- Xcode and its command-line tools for iOS on macOS.

Install Yewo from [PyPI](https://pypi.org/project/yewo/) with
[uv](https://docs.astral.sh/uv/):

```bash
uv tool install yewo
```

If you prefer an existing virtual environment, use
`python -m pip install yewo` instead. You only need to clone the repository
when contributing to Yewo itself.

From the app project you want to test, create a starter configuration and
prepare Appium for one platform:

```bash
yewo setup --platform android
```

Use `--platform ios` for an iOS project. Setup previews each required change
and asks before creating files or installing tools. It does not overwrite an
existing `yewo.yaml`. Run `yewo setup --dry-run --platform android` to inspect
the plan without changing anything.

Add the credentials required by the models in `yewo.yaml` to the generated
`.env` file or to your shell environment. The starter configuration uses
OpenRouter:

```env
OPENROUTER_API_KEY=your_key_here
```

Check the project and host environment:

```bash
yewo doctor
```

Then give Yewo one exploratory goal. Settings come from the nearest
`yewo.yaml`, and command-line options can override its app, device, platform,
model, output directory, and execution limits:

```bash
yewo run "Open the app and sign in" --app-path path/to/app.apk
```

Use `yewo run` for one goal. Use `yewo test` when a declared journey must
produce generated scenarios, independent verdicts, and reusable reports.

## Run journey tests

Create `yewo.yaml` in the app project. A minimal Android project looks like
this:

```yaml
device:
  name: Pixel_8
  platform: android

app: build/app-release.apk

model: openrouter/deepseek/deepseek-v4.1-flash

grounder:
  model: openrouter/bytedance/ui-tars-1.5-7b
  protocol: source_pixels

workflows:
  smoke:
    journeys:
      - id: sign_in
        name: Sign in
        journey: |
          Open the app, sign in with the test account, and open Settings.
```

Run every workflow and target declared by the nearest `yewo.yaml`:

```bash
yewo test
```

Select workflows or targets with comma-separated identifiers:

```bash
yewo test --workflow smoke
yewo test --target pixel,ios_simulator
```

Each selected journey-target pair gets its own device session. Yewo discovers
the journey, records reusable app knowledge, generates scenarios grounded in
that evidence, executes them, and audits each scenario independently.

Each run stores its output under `assets/yewo/<run-id>/` by default. A
completed journey test includes:

- `reports/run.json`, the canonical machine-readable run record;
- `reports/report.html`, an interactive report;
- `reports/report.md`, a Markdown summary; and
- `reports/junit.xml`, a CI-compatible test report.

Set `output_dir` in `yewo.yaml` to replace `assets/yewo` with another output root.

## How Yewo works

```text
journey -> discover -> generate scenarios -> execute -> audit -> report
```

During discovery, Yewo analyzes screenshots with a configured visual model,
asks a planner for the next action, grounds visual targets, validates the
action, and acts through Appium. It records verified app knowledge and compiles
a reusable procedure for the journey.

For a journey test, Yewo generates scenarios only after discovery provides the
required evidence. Each scenario is executed from declared preconditions. A
separate auditor evaluates the captured evidence against the expected outcomes.

Only visible or measured evidence can prove an outcome. A planner's claim of
completion is not sufficient evidence for a passing verdict.

## Scenario verdicts

| Verdict | Meaning |
|---|---|
| `passed` | Every expected outcome was independently evidenced. |
| `failed` | At least one expected outcome was contradicted by the evidence. |
| `unverified` | Execution did not provide enough evidence for a pass or failure. |
| `error` | A capability, device, model provider, or Yewo failed. |
| `aborted` | The run was interrupted before a verdict was possible. |

Only `passed` is a verified success. Only `failed` assigns the failure to the
application. Conditions such as an unsupported capability, insufficient
discovery, a policy block, or an exhausted budget are recorded as execution
classifications and produce an `unverified` verdict rather than a false pass.

## Commands

| Command | Purpose |
|---|---|
| `yewo setup` | Create missing project files and prepare Appium tools. |
| `yewo doctor` | Check the project and host environment and explain failures. |
| `yewo run <goal>` | Drive the configured app toward one exploratory goal. |
| `yewo test` | Discover and test journeys from the nearest `yewo.yaml`. |

Run `yewo <command> --help` for complete options. When working from a source
checkout, use `uv run yewo` in place of the installed `yewo` command.

## Project configuration

Yewo finds the nearest `yewo.yaml` at or above the current directory. Settings
can be declared for the whole project or overridden by a workflow or target.
The configuration supports multiple workflows, journeys, target devices,
declared values and secrets, security rules, coverage categories, actors,
preconditions, model roles, execution budgets, and artifact settings.

See the [Yewo configuration schema](yewo-schema.json) for the complete public
configuration contract.

## Development

- [Development and quality gates](docs/development.md)
- [Controlled evaluation app](eval/controlled_app/README.md)

## License

Yewo is licensed under the [Apache License 2.0](LICENSE).
