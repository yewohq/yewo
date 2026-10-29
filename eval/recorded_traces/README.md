# Recorded execution traces

This corpus contains small, reviewed fragments copied from generated runs under
`eval/runs/`. Generated runs remain ignored. Each source artifact is identified
by repository-relative path and SHA-256 digest so a fixture retains provenance
after the original local run is unavailable.

The replay runner sends each fragment through the real deterministic Yewo
component that currently owns it:

```bash
uv run python tools/recorded_trace_replay.py
```

Each case records two readings:

- `baseline` is what the current implementation must reproduce;
- `required` is the architecture contract;
- `known_gap` is true exactly when those readings differ.

This separation prevents a known defect from being mistaken for correct
behavior while keeping the Phase 1 baseline green. A later phase that fixes a
gap updates the reviewed baseline to the required reading and removes the gap.
Unexpected drift fails immediately.

The corpus is an offline Yewo regression tier. It does not prove that
perception, a model, Appium, or the application will behave the same in a live
run.
