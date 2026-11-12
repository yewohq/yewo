---
name: yewo-agent-development
description: Implement or review Python changes to Yewo's CLI, mobile-agent loop, planning, perception, actuation, device lifecycle, configuration, or reporting. Use for production changes under src/yewo.
---

# Yewo agent development

Work on Yewo as a Python 3.12+ mobile-testing agent. Use `uv` and
`pyproject.toml` for Python dependencies.

## Locate the owner

Inspect the implementation and its callers before editing. Use the current
package boundaries:

- `src/yewo/cli`: command parsing and dispatch
- `src/yewo/applications`: application requests, inspection, and resolution
- `src/yewo/artifacts`: artifact contracts, paths, policies, and persistence
- `src/yewo/automation`: Appium server, driver, and session ownership
- `src/yewo/devices`: device inventory, lifecycle, interaction, and
  platform-specific evidence services
- `src/yewo/observations`: screenshot capture, redaction, fingerprints, and
  observation persistence
- `src/yewo/models`: model profiles, LiteLLM transport, messages, usage, and
  model-call artifacts
- `src/yewo/grounding`: visual-target contracts, coordinate codecs, and the
  model-backed grounding service
- `src/yewo/agent`: `YewoAgent`, reusable app memory and route replay,
  planning, composition, settlement, and the shared observe-plan-act loop
- `src/yewo/agent/actuation`: one action's contracts, authorization, dispatch,
  and immediate observable outcome
- `src/yewo/agent/discovery`: journey-directed evidence acquisition,
  promotion, and bounded repair for YewoAgent memory
- `src/yewo/runs`: run coordination, workflow contracts, workspaces, and
  top-level terminal classification

Add CLI behavior through `src/yewo/cli` and the `yewo` entry point.

## Preserve the execution model

- Keep perception independent from actuation. Planning produces validated
  actions; device code owns platform mechanics.
- Keep observation independent from product policy. Perception describes what
  is present and which capabilities were observed. Planning and actuation
  decide whether an observation should be read, selected, or acted upon.
- Treat model responses and detected coordinates as untrusted. Validate action
  kinds, required fields, dimensions, bounds, and durations before dispatch.
- Make screenshot-to-device coordinate conversion explicit. Re-observe after an
  action before making decisions that depend on the new screen.
- Bound loops, retries, subprocesses, and timeouts. Never blindly retry a
  non-idempotent tap, submission, or text entry.
- Distinguish goal completion, failed verification, exhausted steps, app
  defects, app crashes, infrastructure failures, interruption, and skipped
  work. A model's claim of completion is not sufficient evidence.
- Own cleanup for partial startup, exceptions, and cancellation. Stop only
  resources created by the current run.
- Keep Android ADB and iOS simulator behavior separate. Run platform preflight
  before changing device state, and do not assume iOS tooling exists off macOS.
- Keep slow SDK, model, and device calls explicit at synchronous/asynchronous
  boundaries.

## Protect contracts and artifacts

Preserve public Pydantic schemas, action values, journey fields, CLI flags,
defaults, exit behavior, and report formats unless the task changes the
contract. Update callers, documentation, and tests together when a contract
changes.

Keep model-provider behavior behind the LiteLLM boundary. Do not add
provider-specific request rewriting unless the shared interface cannot express
the required behavior and the limitation is demonstrated. Keep provider names,
credentials, cache behavior, and usage normalization out of agent logic.

Keep interpretation metadata with the artifact that owns it. A model, app
artifact, or external package must carry enough versioned metadata for Yewo to
validate and interpret it. Do not replace missing artifact metadata with a
runtime compatibility constant.

Sanitize journey-derived artifact paths, isolate parallel runs, and keep
Markdown, JUnit, progress, and usage results consistent. Surface report-writing
failures. Never expose API credentials or sensitive app content in logs,
screenshots, prompts, recordings, or reports, and do not read or print `.env`.
Do not silently change providers, models, or spending defaults.

## Keep changes maintainable

Prefer focused functions, explicit types, constructor-injected boundaries, and
one owner for shared validation or conversion logic. Do not add abstraction
layers without an actual interchangeable boundary or data owner. Preserve
public names unless their callers are migrated.

### Name code by domain meaning

Review every new or changed module, class, field, function, parameter, local
variable, constant, and CSS selector before finishing a production change.

- Name stored values with noun phrases that describe their contents.
- Name operations with active verb phrases. Prefer established prefixes such
  as `build_`, `format_`, `normalize_`, `parse_`, `record_`, `resolve_`,
  `select_`, and `validate_` when they accurately describe the operation.
- Name predicates with `is_`, `has_`, `can_`, or `should_`.
- Include useful structure in collection names, such as `_count`, `_ids`,
  `_by_rule_id`, `_path`, or `_duration_ms`.
- Treat conversational, literary, sentence-shaped, and context-dependent names
  as review warnings. Examples include `said`, `asked`, `given`, `written`,
  `settled`, `held`, `seen`, `named`, `spelled`, `one`, `here`, `where`,
  `what`, `why`, and `took`. Replace them when the name does not identify the
  represented domain concept.
- Do not use single-letter production identifiers or CSS classes unless they
  are conventional mathematical coordinates or tightly scoped indices.
- Name validators `_validate_<invariant>`, normalizers `_normalize_<value>`,
  formatters `_format_<output>`, and constructors `_build_<result>` when those
  descriptions are accurate.

Name modules for the responsibility they own and classes for the domain entity
or role they represent. Avoid unqualified names such as `data`, `facts`,
`common`, `helpers`, `utils`, `manager`, `handler`, `result`, or `status` when
the package and name together still do not identify a specific responsibility.

Before defining a closed set of strings, search for its existing owner. Keep
the values, aliases, parsing, validation, and display mappings behind one enum
or typed model when they represent the same domain concept. Do not duplicate
the vocabulary across modules.

When renaming, update callers, tests, markup, styles, fixtures, and
documentation directly. Do not retain compatibility aliases unless the user
explicitly requires backward compatibility.

Before finishing, inspect the added identifiers in the diff and search the
changed subsystem for duplicated domain literals. Confirm that every name says
what the value contains or what the operation produces, not how it reads in an
English sentence.

Use controlled language for prompts, error messages, tool descriptions, and
safety text. Write short active sentences. State one instruction or condition
per sentence. Use one name for one thing. Preserve conditions and uncertainty;
do not simplify a qualified claim into an unconditional one.

For production prompt creation, review, or optimization, read and apply the
[prompt standard](references/prompt-standard.md).

Do not use `getattr`, `hasattr`, or `setattr` inside a typed internal contract
to compensate for an unclear model. Use a discriminated Pydantic model,
`isinstance`, `match`, a protocol, or an explicit adapter instead. At an SDK,
driver, or untrusted-model boundary, dynamic access is allowed only while
normalizing the value into a validated, typed contract. Keep that adaptation at
the boundary; do not pass dynamic objects into agent logic.

For production changes, apply the measurable ratchets in the
[quality-gates reference](../yewo-test-engineering/references/quality-gates.md).
Do not increase an existing exception merely because it is baselined.

Comments should explain rationale, units, ownership, platform constraints, or
retry safety that the code cannot express. Preserve Pydantic field descriptions
and prompt text that affect model behavior. Avoid narrated steps, redundant
docstrings, decorative sections, and commented-out code.

For test design and verification, also read the `yewo-test-engineering` skill.
Do not start emulators, download model weights, or make paid model calls as an
incidental check.
