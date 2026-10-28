# AndroidWorld integration

This workspace runs upstream benchmarks without copying their tasks or
evaluators. AndroidWorld owns task setup, device reset, step limits, and the
final reward. Yewo owns perception, planning, target grounding, and device
actuation.

## AndroidWorld

The integration is pinned to AndroidWorld commit
`6e4feb00702735c9a7485f4ae714528a058cb2b7`. Override
`ANDROID_WORLD_REVISION` when deliberately updating it.

Install the isolated environment:

```bash
cd yewo/eval/integrations/android_world
./setup.sh
```

Create and launch the upstream `AndroidWorldAvd` emulator by following
AndroidWorld's setup instructions. Launch it with the required gRPC port:

```bash
~/Library/Android/sdk/emulator/emulator \
  -avd AndroidWorldAvd -no-snapshot -grpc 8554
```

Install AndroidWorld's apps once and run a two-task integration check:

```bash
.venv/bin/python android_world/run.py \
  --perform_emulator_setup \
  --tasks=CameraTakePhoto,ClockStopWatchRunning \
  --n_task_combinations=1
```

After setup, omit `--perform_emulator_setup`. Run the first capability smoke
suite with three parameter combinations per task:

```bash
.venv/bin/python android_world/run.py \
  --tasks=CameraTakePhoto,ClockStopWatchRunning,ContactsAddContact,MarkorCreateNoteAndSms \
  --n_task_combinations=3 \
  --task_random_seed=30
```

The runner discovers the Android SDK, suppresses gRPC fork diagnostics, gives
each new run a fresh checkpoint directory under
`eval/runs/android_world/checkpoints/`, and
streams redacted output to both the terminal and a timestamped file under
`eval/runs/android_world/logs/`. Pass `--checkpoint_dir` only when deliberately resuming a
specific run, or `--output_path` to choose a different checkpoint root.

Yewo attaches to `emulator-<console_port>`, preserves the screen prepared by
AndroidWorld, and performs at most one agent action per AndroidWorld turn. It
does not install, relaunch, reset, or uninstall benchmark apps. AndroidWorld's
checkpoint contains the authoritative reward. Yewo's screenshots, prompts,
device logs, and usage telemetry are written to
`eval/runs/android_world/yewo/`.

The harness has offline contract coverage in the Yewo repository. A live run
still requires the AndroidWorld emulator, its apps, model credentials, Appium,
and the selected grounding dependencies. No live benchmark result is claimed
until AndroidWorld emits a reward.
