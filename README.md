# Yewo

Yewo is an AI testing agent for Android and iOS apps. Describe a user journey
in plain language and Yewo operates the real app, checks what happened, and
produces an evidence-backed report.

It uses visual perception rather than app-specific selectors and keeps test
generation separate from the model that judges the result.

> Yewo is alpha software. Expect breaking changes while its public interface is
> being established.

## What it does

- Runs plain-language journeys against Android and iOS builds.
- Detects and interacts with visible controls without requiring test IDs.
- Generates repeatable scenario suites that can be reviewed and locked for CI.
- Checks functional, security, and supported device-level outcomes.
- Saves screenshots, actions, model calls, measurements, and verdict evidence.
- Masks declared secrets in stored text and redacts them from saved screenshots.

## Quick start

You need Python 3.12+, Node.js 18+, and the SDK for the platform you want to
test:

- Android Studio and the Android SDK for Android.
- Xcode and its command-line tools for iOS on macOS.

Install Yewo from [PyPI](https://pypi.org/project/yewo/) with
[uv](https://docs.astral.sh/uv/) and include its vision dependencies:

```bash
uv tool install "yewo[vision]"
```

If you prefer an existing virtual environment, use
`python -m pip install "yewo[vision]"` instead. You only need to clone the
repository when contributing to Yewo itself.

Install the runtime dependencies, download the perception model, and check the
machine:

```bash
yewo setup
yewo doctor
yewo devices
```

Add the API key for your chosen
[LiteLLM-supported model](https://docs.litellm.ai/docs/providers) to `.env` or
your shell environment. The default model uses OpenRouter:

```env
OPENROUTER_API_KEY=your_key_here
```

Now point Yewo at an APK, `.app`, or `.ipa` and give it a goal:

```bash
yewo run "Open the app and sign in" --app-path path/to/app.apk
```

`yewo run` is useful for exploration. Use a journey when the flow must produce
a test verdict and a reusable report.

## Run a repeatable journey

Create `yewo.yaml` beside your app project:

```yaml
device:
  name: Pixel_8
  platform: android

app: build/app-release.apk

workflows:
  smoke:
    journeys:
      - id: sign_in
        name: Sign in
        journey: |
          Open the app, sign in with the test account, and open Settings.
```

Run it:

```bash
yewo test yewo.yaml
```

Yewo generates scenarios, drives the app, and writes the run record and reports
under `assets/` by default. The CLI prints the paths it created.

When the generated scenarios are ready for review, lock them and commit
`yewo.lock` with the journey file:

```bash
yewo lock yewo.yaml
yewo test yewo.yaml --locked
```

Locked runs execute the reviewed suite and fail if its inputs are missing or
stale. This is the mode intended for CI.

## How Yewo works

```text
journey -> scenarios -> observe -> plan -> act -> verify -> report
```

Yewo reads each screen with a detector and OCR, asks a planner for the next
action, validates the selected target, and then acts through the platform
driver. A separate judge evaluates captured evidence without seeing the plan's
claim that it succeeded.

Only visible or measured evidence can prove an outcome. If the app does not
provide the required evidence, Yewo reports that explicitly instead of turning
the planner's confidence into a pass.

## Verdicts

| Verdict | Meaning |
|---|---|
| `PASSED` | Every required outcome was evidenced. |
| `FAILED` | The app crashed, broke a rule, or showed a critical defect. |
| `UNVERIFIED` | The run finished without enough evidence. |
| `SKIPPED` | The scenario could not be carried out in this context. |
| `UNSUPPORTED` | The target cannot provide a required state or measurement. |
| `ERROR` | Yewo, a device, or an external service failed. |
| `ABORTED` | The run was interrupted before a verdict was possible. |

Only `PASSED` is a verified success. Only `FAILED` assigns the failure to the
app. See [Testing with Yewo](docs/testing.md) for journeys, secrets, coverage,
target matrices, measurements, evidence, and report semantics.

## Commands

| Command | Purpose |
|---|---|
| `yewo setup` | Install and configure runtime dependencies. |
| `yewo doctor` | Check the local environment and explain failures. |
| `yewo devices` | List, create, and select test devices. |
| `yewo config` | Inspect or change saved settings. |
| `yewo run <goal>` | Drive an app toward one exploratory goal. |
| `yewo test [file]` | Generate and run a journey-based test suite. |
| `yewo lock [file]` | Generate and save the suite without running a device. |
| `yewo eval setup` | Materialize pinned evaluation apps. |

Run `yewo <command> --help` for complete options. When working from a source
checkout, use `uv run yewo` in place of the installed `yewo` command.

## Learn more

- [Testing with Yewo](docs/testing.md)
- [Evaluation corpus and maintained test apps](eval/README.md)
- [Development and quality gates](docs/development.md)
- [Yewo configuration schema](yewo-schema.json)
- [Development learnings](docs/development_learnings/README.md)

## License

Yewo is licensed under the [Apache License 2.0](LICENSE).
