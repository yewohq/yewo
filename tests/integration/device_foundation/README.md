# Android device-foundation integration fixture

`android_1080x2400.json` exercises every action in Yewo's core Android action
vocabulary without a planner or grounding model.

This fixture is for an explicit live integration check. The ordinary pytest
suite validates the same Yewo contracts with controlled boundaries and does
not start Appium or an emulator. This check proves that the assembled external
stack can execute those contracts on a real Android runtime. It is not an
evaluation corpus or a benchmark.

The script is intentionally bound to a controlled fixture contract:

- application ID: `com.example.testing_app`;
- screenshot geometry: 1080 × 2400 pixels;
- a global Reset control at `(954, 216)` that opens the login screen;
- email, password, login, and location-denial controls at the declared script
  coordinates;
- a `yewo-controlled://foundation` deep-link intent that returns to the app.

The fixture source and dependency lock are tracked in `eval/controlled_app`.
The generated APK is not stored in Git. Build it from the repository root:

```bash
cd eval/controlled_app
flutter pub get --enforce-lockfile
flutter build apk --debug
```

Yewo rejects a different application ID or screenshot geometry before it
dispatches the first scripted action.

Example:

```bash
uv run yewo run \
  --output-root assets/device-foundation \
  --script tests/integration/device_foundation/android_1080x2400.json \
  --platform android \
  --device-id Pixel_8 \
  --app eval/controlled_app/build/app/outputs/flutter-apk/app-debug.apk
```
