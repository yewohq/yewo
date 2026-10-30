# Yewo controlled app

This Flutter app provides deterministic Android and iOS screens for Yewo's
controlled evaluations. Its source and dependency lock are tracked. Generated
application packages remain ignored by Git.

Build Android:

```bash
cd eval/controlled_app
flutter pub get --enforce-lockfile
flutter build apk --debug
```

Build an iOS Simulator bundle on macOS:

```bash
cd eval/controlled_app
flutter pub get --enforce-lockfile
flutter build ios --simulator
```

Run the reviewed journeys from the Yewo repository root:

```bash
uv run yewo test eval/controlled_app/yewo.yaml --locked
```
