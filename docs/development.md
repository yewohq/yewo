# Developing Yewo

Clone the repository, then use `uv` to install the development tools and the
decision-model dependency:

```bash
git clone https://github.com/getyewo/yewo.git
cd yewo
uv sync --group dev --extra jev
```

Run the checks used by CI:

```bash
uv run ruff check
uv run python tools/quality_gate.py
uv run python tools/ste_gate.py
uv run python tools/emit_schema.py --check
uv run coverage run -m pytest
uv run coverage report
uv run python -m tools.eval_gate
uv run python -m tools.mutation_sweep
```

The type gate runs in an environment that also contains the vision packages:

```bash
uv sync --group dev --extra vision --extra jev
uv run python tools/type_gate.py
```

## What the gates protect

| Gate | Contract |
|---|---|
| Ruff | Python lint and static correctness rules. |
| `quality_gate.py` | Project quality limits, including the 550-line file ceiling. |
| `ste_gate.py` | Controlled language in text read by models or people under pressure. |
| `emit_schema.py --check` | `yewo-schema.json` matches the models that load `yewo.yaml`. |
| `eval_gate.py` | The judge still answers the labelled corpus as expected. |
| `mutation_sweep.py` | Deliberate defects are still detected by at least one check. |
| `type_gate.py` | Pyright results stay within the shrinking per-file baseline. |

Run focused tests while developing, then run the complete relevant gate before
committing. Production changes need tests that prove both the intended behavior
and the important failure boundary.

## Evaluation workspace

[`eval/`](../eval/README.md) is the canonical home for maintained test apps,
journeys, labelled cases, perception fixtures, benchmark integrations, and
release baselines. Generated runs and rebuilt app bundles are intentionally not
committed.

Materialize the pinned third-party Android packages with:

```bash
uv run yewo eval setup --platform android
```

The Yewo-owned Flutter app can be rebuilt from
[`eval/controlled_app/`](../eval/controlled_app/README.md).
