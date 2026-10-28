# Yewo evaluation

`eval/` is the canonical home for Yewo's application journeys, app artifact
manifest, labelled judge fixtures, perception cases, integrations, baselines,
and generated evaluation runs.

```text
eval/
├── apps/            # Android/iOS manifest and local packages
├── controlled_app/  # Yewo-owned Flutter app, workflows, and lock
├── journeys/        # Third-party app workflows
├── integrations/    # Upstream ground-truth benchmark adapters
├── perception/      # Detector and OCR corpus
├── screens/         # Labelled visual-judge cases
├── verdicts/        # Deterministic verdict cases
├── baselines/       # Reviewed release measurements
└── runs/            # Generated output; ignored by Git
```

Materialize the pinned Android packages from their official upstream releases:

```bash
uv run yewo eval setup --platform android
```

Application directories represent deployment platforms only. Compatibility
targets and known defect pairs are distinguished by metadata and journeys, not
by a separate filesystem category.

## Labelled judge corpus

Labelled cases that Yewo's own judging must answer correctly. It is a gate on
Yewo, not a command a user runs against an app. `tools/eval_gate.py` reads it.

```bash
python -m tools.eval_gate              # the deterministic corpus, free
python -m tools.eval_gate --screens    # and the auditor, which spends tokens
```

Exit codes: 0 everything answered as the corpus says or was baselined, 1 an
item answered wrongly and nothing accounts for it, or a baselined item is gone,
2 the corpus or the baseline could not be read, 3 the screen corpus was asked
for and no credentials are present, 4 the provider could not answer. Four is
kept apart from one, because an outage is not a worse auditor. A failure that
is not the provider's is not caught at all: a defect in the harness raises
where it happened.

## What it answers, and what it does not

It answers one question: **did a change to a prompt, a rule or the verdict
ranking make Yewo answer a labelled case differently?**

It does not answer whether Yewo is good, and it reports no rate. The corpus is
small, and it grows only as fast as a person labels a case. At this size a
false-pass percentage moves by whole points on one item, which reads as a
result and is noise. What an item answered is a fact. What the population does
is not knowable from here.

## The two kinds

### `verdicts/` — evidence, and the one word it adds up to

No model and no device, so the same evidence gives the same verdict every time.
One pass is enough, and an item is either correct or wrong.

A fixture holds the arguments `yewo.testing.judge.decide.determine_status`
takes, and what it must answer:

```json
{
  "id": "a-crash-outranks-an-infeasible-stop",
  "why": "A crash is something the app did. An infeasible stop describes the run.",
  "run": { "status": "hook_stop" },
  "stop_reason": "Scenario is infeasible: the delivery screen cannot be reached",
  "issues": [{ "issue_type": "crash_screen", "severity": "critical",
               "description": "A system crash dialog covers the screen." }],
  "expect": { "status": "FAILED", "cause": null, "reason_holds": "App crashed" }
}
```

`expect.status` is held exactly. `expect.reason_holds` is a fragment the reason
must contain, because the sentence is prose and prose gets rewritten. A rule id
is the part that carries meaning. `expect.cause` is checked only where the key
is present, and `null` there means the verdict must carry no cause at all.

### `screens/` — a screenshot, the rules put to it, and the answers they must get

A model answers, so each rule is asked several times and classed as correct
every time, wrong every time, or disagreeing with itself.

```json
{
  "id": "cart-one-burger",
  "why": "One item at $9.99 and a subtotal to match.",
  "screenshot": "cart_one_burger.png",
  "rules": [
    { "kind": "expected", "expect": "passed", "text": "The cart subtotal is $9.99" },
    { "kind": "expected", "expect": "failed", "inverted": true,
      "text": "The cart subtotal is $19.98" }
  ]
}
```

**Every screen fixture needs one inverted rule**, and loading fails without
one. An inverted rule is true of the same screen in the opposite direction. A
reader that answers "passed" to whatever it is shown scores full marks on a
corpus of rules that are all true, and tells you nothing. Two rules that cannot
both hold tell you whether the screen was read.

Write the inversion by hand. A rule with "not" put in front of it is bad
English and often ambiguous, and an ambiguous rule produces a wrong answer that
is the corpus's fault.

## Adding a case

1. Pick a screen or an outcome where the right answer is **not arguable**. A
   case whose truth a careful reader could dispute becomes a baselined line
   nobody can resolve.
2. Write the fixture, with `why` saying what it protects.
3. Run the gate. A new case that answers wrongly is a finding, not a defect in
   the case — decide which it is before baselining it.

A screenshot is checked before it is sent: the name must be a plain `.png`
beside the fixture, the resolved path must stay in the corpus, a link is
refused, and the bytes must start as a PNG. The file is handed to a model, so
a fixture is not a way to read something else off the machine.

## Two constraints on a screenshot

**It must carry no redaction bar.** A saved screenshot has had declared values
covered before it was written. The auditor at run time is handed the screen
before that happens. A fixture built from a covered screenshot asks the auditor
about something no run ever shows it.

**It comes from our own demo app.** Every screenshot here is a screen of
`eval/controlled_app`, with no real account and no real card. Nothing from a
user's run reaches this directory, and there is no path by which it could: a
user's `run.json` and screenshots are written to their own machine and are
never sent anywhere.

## The baseline

`tools/eval-baseline.toml` names the items that answer wrongly today. Each
entry carries the reason it is tolerated, and the hash of the fixture the
decision was made about:

```toml
[known_wrong."cart-one-burger/expected:1"]
reason = "The auditor reads the x2 banner as the total. TICKET-7."
fixture = "9f2c4a1b7d3e5086"
```

The gate reads that file and never writes it. Three things follow from binding
an entry to the fixture rather than to its name.

- **Edit the fixture and the exemption lapses.** The hash moves, the entry no
  longer covers the item, and it fails until somebody looks again. Otherwise a
  case could be edited into something else under a name already excused.
- **Delete the fixture and the gate fails.** An entry naming nothing is how a
  failing case stops failing without being fixed.
- **An item that starts answering correctly is reported**, so the line comes
  out in the commit that earned it. An item whose fixture changed is not
  repaired. It is a different case under an old name, and the build fails
  until somebody reads it again.

An entry covers the state it was written for. `known_wrong` says a rule is
answered wrongly. It does not say the auditor may stop answering it, so an
unanswered rule is never excused by one. An entry for a tier that did not run
in this sweep is left alone, which is what lets the free run read the verdict
corpus without tripping over a screen entry.

A corpus directory holding no fixtures is refused. An empty corpus answers
every question correctly and would turn the gate off without failing a build.

The first run of the screen corpus with credentials is what establishes which
of its items the auditor actually gets right. That run happened: 30 calls
against `openrouter/deepseek/deepseek-v4.1-flash`, every answered rule correct,
and three calls that named no rule. Two of those three had answered correctly
and were thrown away by Yewo because the verdicts carried no rule id. The
corpus found that, and `judge/rules.by_statement` fixes it.
