# Deterministic Android device-foundation script

`android_1080x2400.json` exercises every action in Yewo's core Android action
vocabulary without a planner or grounding model.

The script is intentionally bound to a controlled fixture contract:

- application ID: `com.example.testing_app`;
- screenshot geometry: 1080 × 2400 pixels;
- a global Reset control at `(954, 216)` that opens the login screen;
- email, password, login, and location-denial controls at the declared script
  coordinates;
- a `yewo-controlled://foundation` deep-link intent that returns to the app.

The deployable APK is generated locally and is not stored in Git. Run the
script only against a reviewed build that implements this exact contract. Yewo
rejects a different application ID or screenshot geometry before dispatching
the first scripted action.

Example:

```bash
uv run yewo run \
  --output-root assets/device-foundation \
  --script eval/device_foundation/android_1080x2400.json \
  --platform android \
  --device-id Pixel_8 \
  --app /path/to/controlled-app.apk
```
