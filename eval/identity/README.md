# Screen identity calibration

This directory owns the held-out gate for `screen-family-structural-phash-v1`.
It is separate from trainer discovery and from scenario verdicts.

`manifest.json` labels same-family and different-family pairs over frozen
Android and iOS screenshots in `eval/perception/screens`. Each sample records
the source digest, structural fingerprint key, platform, viewport, and traits.
Some same-family variants use declared deterministic perturbations for status
indicators, transient banners, animation, brightness, viewport resizing, and
script-direction stress. The keyboard and materially-different pairs use
independent screenshots.

Run:

```bash
uv run python -m tools.identity_eval.evaluate
```

The command verifies every screenshot digest, evaluates false merges, false
splits, and uncertainty independently, and rewrites
`calibration_baseline.json` with an integrity-protected result. A classifier
that abstains on too many pairs does not pass.

The corpus is an offline contract and regression gate. Deterministic
perturbations do not establish field reliability on arbitrary applications.
Operational family grouping additionally requires a passing sealed calibration
artifact and a local classification receipt for each accepted observation.
Without both, AppMap construction keeps exact states separate.
