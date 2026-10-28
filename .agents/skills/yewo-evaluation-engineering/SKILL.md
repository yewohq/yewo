---
name: yewo-evaluation-engineering
description: Design, maintain, review, or run Yewo evaluation corpora, model comparisons, controlled or third-party app journeys, benchmark integrations, and release baselines under eval/.
---

# Yewo evaluation engineering

Measure Yewo separately from the applications it tests. Python tests establish
Yewo contracts. Evaluation journeys and labelled corpora measure behavior under
recorded conditions. Neither substitutes for the other.

## Keep one evaluation workspace

Use `eval/` as the canonical home for maintained evaluation inputs and tools.
Keep controlled applications, third-party journeys, labelled cases, perception
fixtures, integrations, and reviewed baselines under their existing owners in
that tree.

Generated runs and rebuildable application packages are not source artifacts.
Keep them out of version control. Track the source, manifest, or build recipe
needed to reproduce them instead. Preserve a historical output only when the
task requires an archive, and keep that archive distinct from the active
corpus.

## Make inputs traceable

Record the origin, version, license, integrity value, and relevant build or
runtime settings for external models, apps, and datasets. Pin mutable remote
artifacts to immutable versions. Keep the metadata needed to interpret an
artifact with that artifact or its manifest.

Use held-out inputs for quality claims. Do not present results from known
training inputs as evidence of general behavior.

## Design comparisons that explain the result

Change one independent variable at a time. Keep subsystem conclusions
separate: a detector result does not establish OCR quality, and a complete
perception result does not establish planner quality.

Record the artifact versions, thresholds, input resolution, hardware, warm or
cold state, and other conditions needed to reproduce a comparison. Report
correctness, latency, and memory independently. Retain machine-readable results
and a human-readable view when visual inspection changes the interpretation.

Include difficult and diverse cases that represent the claimed scope. A
benchmark describes its corpus and conditions; it is not a universal product
quality claim.

## Preserve evidence

Do not turn missing labels, skipped cases, unavailable dependencies, or failed
setup into successful measurements. Report the exact command, completed cases,
failures, exclusions, and remaining uncertainty. Keep reviewed baselines stable
until a deliberate comparison updates them.

For implementation tests and validation tiers, also read the
`yewo-test-engineering` skill. For runtime changes under `src/yewo`, also read
the `yewo-agent-development` skill.
