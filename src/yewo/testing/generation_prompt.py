"""Stable policy and evidence sections for semantic scenario generation."""

PROMPT = """Write one test scenario from the app context and coverage cell.

Write a scenario of the requested shape, about the requested concern, and
nothing else. Yewo covers every other cell in its own call.

### The steps

Give the steps in order. Each one is:

- `enter` — a declared value, and the field it goes in.
- `capability` — a device state, such as the network going offline.
- `intent` — one user objective when the app context does not state its UI mechanics.
- `variation` — the complete journey pursued with the requested user strategy.
- `action` — one action on one control that the app context explicitly names. Open it with one of:
  {actions}.

The first step opens the app. Every scenario attempts the same complete journey.
The scenario type changes the user's strategy, not the product requirement.
Use `intent` for requirements such as creating,
adding, editing, selecting, saving, reopening, or deleting when the app context
does not explicitly state the control or route. Copy one complete objective
verbatim from the app context into both `instruction` and `evidence`. The
runtime planner will choose mechanics only from the application it observes.

Use `action` only when the app context already contains the complete executable
instruction. Copy that instruction verbatim into both `instruction` and
`evidence`; do not paraphrase it. Otherwise use `intent`. Name controls by their
visible text and never by position. Do not infer a button, menu, list, dialog,
field, screen, tab, icon, drawer, dropdown, checkbox, switch, or any future
control type from a product capability. A second model agreeing with the
inference is not evidence.

For `negative`, `boundary`, `empty`, or `exploratory`, include exactly one
`variation` step. Copy the complete app context verbatim into `goal`. Copy the
objective being varied verbatim into both `objective` and `evidence`.

- `negative`: use a declared refused `enter` value when one is available.
  Otherwise choose a concrete invalid, malformed, or extreme `probe_value`.
- `boundary`: choose a concrete `probe_value` or bounded `repetitions`.
- `empty`: omit the relevant value or item. Supply no probe parameters.
- `exploratory`: try an alternate route or action order supported by the live app.
  Supply no probe parameters and do not invent a new capability.

A probe is test input. It is not a product limit or an expected result. Do not
claim that the app must accept or reject it. Do not name UI mechanics needed to
apply it. The runtime planner discovers those from the observed application.
Use one variation step instead of repeating many identical intent steps.

Put no claim in a step. A separate component judges the outcomes, and the
agent walking the app never reads them. An agent told what must be true
steers the app until it is. The finding then goes unreported.

### The outcomes

Say what has to be visible, and give the evidence that settles each one. Copy
the journey or rule excerpt that authorizes it verbatim into `evidence`:

- "current_screen" — true or false from one screenshot. Prefer this always.
- "transition" — the run reached a screen, or moved between two.
- "measured" — only when the app description explicitly states a measurable
  requirement. Name one supported metric and copy its numeric threshold:
  `app_launch_ttid_ms`, `min_touch_target`, or
  `unlabelled_actionable_count`. Never invent a performance or accessibility
  threshold. "No unlabelled actionable controls" has threshold zero.

Each outcome has an optional `obligation` contract. Omit it when the outcome
must hold on the final screen; that is the default. Use `during_instruction`
or `after_instruction` only with the one-based `instruction_index` whose
window owns the evidence. Use `when_subject_observed` only with a `subject`
that the screenshot can establish. An activation window does not prove that a
screen, route, or control exists. Do not use it to add UI mechanics.

Yewo can read screenshots, on-screen text, and the result of an action. It
cannot read network traffic, HTTP headers, the app's own logs, an
accessibility tree, a screen reader, focus order or a contrast ratio. Write
no outcome that needs one of those. Yewo's separate iOS accessibility audit
can settle only the two named accessibility metrics; it is not available to
the planner.

Yewo writes its own outcome for every value the file says this app
refuses. A step entering one needs no claim from you.

For a `variation` step, do not assert a specific acceptance limit, rejection,
or error message. Yewo replaces the proposed outcomes with a controlled
conditional outcome: accepted input must produce a consistent completed state;
refused input must not produce false success, corruption, or an unusable app.
"""

CELL_CONTEXT = """### The app, as a user walks it

"{description}"

### The cell

Category: {category} — {category_meaning}
Scenario type: {scenario_type} — {type_meaning}

### The values you may enter

An `enter` step names one of these. Never write a value of your own, because
only the file knows what this app does with a value.

{lines}
{rules}
{app_evidence}
"""

APP_EVIDENCE = """
### Verified app evidence

AppMap: {app_map_artifact_id}

Observed capabilities:
{capabilities}

Verified route prefixes:
{routes}

Copy only listed IDs into `app_capability_ids` and `route_memory_ids`. Select
the route prefix needed to reach the test subject. Do not restate its controls
or navigation as scenario steps. App evidence establishes observed mechanics;
it does not create a requirement, product limit, or expected outcome.
{required_prefix}
"""

RULES = """
### Rules this app must not break

{rules}

Reach the screens where one of these could break. Do not write a claim about
a rule in a step, because another component judges every rule on every screen
it sees.
"""
