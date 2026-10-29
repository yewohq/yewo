"""Writing one scenario for one declared cell.

One call used to produce the whole suite, and the model chose the topics, the
count, the test data and the identity of every scenario. None of those are a
model's to choose. A category with no type beside it leaves the shape of a
test unsaid. A value invented for an app the model cannot inspect is a guess.

So Yewo enumerates the cells and asks for one scenario each. The model is
told which cell, handed the values it may use, and answers with the walk and
the outcomes. Yewo stamps the identity, because Yewo knows which cell it
asked about.

A journey's security rules reach a security cell only. The auditor reads
every rule on every screen of every scenario already, so a cell about
something else gains nothing from seeing them. A security cell needs them,
because reaching the screen where a rule can break is navigation that only
an instruction can ask for.
"""

import logging
from dataclasses import dataclass, field
from typing import Literal, Optional

from pydantic import ValidationError

from yewo.agent.actuation.actions import CapabilityKind
from yewo.errors import Cause, ProviderUnavailable, cause_of
from yewo.llm.failures import reportable
from yewo.llm.llm_client import LLMClient
from yewo.llm.messages import SystemMessage, TextMessage
from yewo.metrics import metrics_without_live_collectors
from yewo.testing import steps
from yewo.testing.journey_env import substitute
from yewo.testing.lifecycle import resolve_setup
from yewo.testing.models import (
    AssertionScope,
    ExpectedOutcome,
    GeneratedTestCase,
    ReportStatus,
    ScenarioDraft,
    UserJourney,
)
from yewo.testing.step_language import ACTION_VERBS
from yewo.testing import generation_policy
from yewo.testing.generation_evidence import ScenarioGenerationEvidence
from yewo.testing.generation_prompt import APP_EVIDENCE, CELL_CONTEXT, PROMPT, RULES
from yewo.testing.taxonomy import (
    SUPPORTED_CATEGORIES,
    Category,
    CoverageCell,
    ScenarioType,
    describe,
    sort_cells_by_priority,
)
from yewo.utils.secrets import SecretScrubber

logger = logging.getLogger(__name__)

GENERATION_ATTEMPTS = 3

CAPABILITY_BY_CATEGORY: dict[Category, CapabilityKind] = {
    Category.NETWORK: CapabilityKind.NETWORK,
    Category.PERMISSION: CapabilityKind.PERMISSION,
}
"""Which device state a cell about that subject has to ask for.

A network scenario that never touched the network walked the happy path and
reported as a network test. The requirement is structural: one step has to
carry the capability, and that step carries the typed capability itself.
"""


def _requirements(
    cell: CoverageCell,
    journey: UserJourney,
    generation_evidence: ScenarioGenerationEvidence | None = None,
    required_route_memory_ids: tuple[str, ...] | None = None,
) -> dict:
    """What an answer for this cell is read against, beyond its own shape.

    Reaching the validator through the client's context means a cell's own
    requirement is refused on the same path as any other mismatch. So the
    model is asked again with the reason, rather than the cell being lost.
    """
    context: dict = {
        "declared": frozenset(journey.values),
        "source_description": _scrubbed_journey_description(journey),
        "scenario_type": cell.scenario_type,
        "outcome_authority": "\n".join(
            [
                _scrubbed_journey_description(journey),
                *(journey.security_rules if cell.category is Category.SECURITY else []),
            ]
        ),
    }
    if cell.scenario_type is ScenarioType.NEGATIVE and journey.refused:
        context["refused"] = frozenset(journey.refused)
    wanted = CAPABILITY_BY_CATEGORY.get(cell.category)
    if wanted is not None:
        context["capability"] = wanted
    if generation_evidence is not None:
        context["generation_evidence"] = generation_evidence
    if required_route_memory_ids is not None:
        context["required_route_memory_ids"] = required_route_memory_ids
    return context


@dataclass(frozen=True)
class UncoveredCell:
    """A cell that produced no scenario, and whose problem that is.

    A missing declaration is the scenario's limit. A provider that refused
    is not, and neither is a response that never matched the schema. Reported
    as scenario limits, those tell a reader to rewrite a cell that was fine.
    """

    cell: CoverageCell
    status: Literal[
        ReportStatus.UNSUPPORTED,
        ReportStatus.ERROR,
        ReportStatus.UNVERIFIED,
    ]
    cause: Cause
    reason: str


@dataclass
class GeneratedSuite:
    """The scenarios a journey produced, and the cells nothing could cover."""

    scenarios: list[GeneratedTestCase] = field(default_factory=list)
    """What runs: every declared value substituted into the walk."""

    uncovered: list[UncoveredCell] = field(default_factory=list)

    locked: list[GeneratedTestCase] = field(default_factory=list)
    """What a lock stores: the same scenarios, with the names still in them.

    One scenario per entry in `scenarios`, in the same order. They differ by
    value substitution only. A lock holding substituted values would hold a
    password, and would change whenever an environment did.
    """


def _scrubbed_journey_description(journey: UserJourney) -> str:
    """The journey as the generator reads it, with declared secrets masked.

    The generator has to know that a Password field exists. It never has to
    know the password. Every other reader of a secret needs the value, so the
    masking happens here rather than where the journey is loaded.
    """
    scrubber = SecretScrubber()
    scrubber.learn_secrets(journey.secrets, journey.secret_displays)
    return scrubber.for_model(journey.description)


def _value_lines(journey: UserJourney) -> str:
    """The names a scenario may use, without the values behind them."""
    lines = []
    for name in sorted(journey.values):
        if name in journey.refused:
            value_note = "this app refuses this value"
        elif name.startswith("actors."):
            value_note = f"declared for actor '{name.split('.')[1]}'"
        else:
            value_note = "a value that is meant to work"
        lines.append(f"- {name} — {value_note}")
    return "\n".join(lines) or "- none declared"


def with_values_filled(scenario: GeneratedTestCase, values: dict[str, str]) -> GeneratedTestCase:
    """The scenario with every `{{name}}` replaced by the value it stands for.

    Done after the identity is stamped, because the sentences a planner reads
    are rendered from the steps and only exist by then.
    """
    return scenario.model_copy(
        update={
            "steps_instructions": [
                substitute(line, values) for line in scenario.steps_instructions
            ],
            "expected_outcomes": [
                outcome.model_copy(update={"statement": substitute(outcome.statement, values)})
                for outcome in scenario.expected_outcomes
            ],
        }
    )


def with_journey_preconditions(
    scenario: GeneratedTestCase, journey: UserJourney
) -> GeneratedTestCase:
    """Attach current declared capabilities to a locked scenario."""
    initial_capabilities = list(journey.preconditions.capabilities)
    kinds = [capability.kind for capability in initial_capabilities]
    kinds.extend(steps.capability_kinds_in(scenario.steps))
    return scenario.model_copy(
        update={
            "initial_capabilities": initial_capabilities,
            "required_capabilities": list(dict.fromkeys(kinds)),
        }
    )


def _with_derived_outcomes(draft: ScenarioDraft, refused: frozenset[str]) -> ScenarioDraft:
    """The draft, plus an outcome for every value confirmed the wrong shape.

    The claim was already held to the value, so the words carrying it are
    written here rather than left to free prose beside it. A variation replaces
    model-authored outcomes because an undeclared probe has only a conditional
    consistency oracle.
    """
    refusal_outcomes = [
        ExpectedOutcome(statement=steps.refusal_outcome(entered.field))
        for entered in steps.entered_values(draft.steps)
        if entered.value in refused
    ]
    variations = steps.variations_in(draft.steps)
    if variations:
        strategy = variations[0].strategy.value.replace("_", " ")
        conditional = ExpectedOutcome(
            statement=(
                f"The {strategy} attempt follows one coherent result: the app "
                "either completes the requested variation and declared journey, "
                "with the resulting accepted state observable, or refuses the probe "
                "without reporting false success, corrupting journey state, or "
                "becoming unusable."
            ),
            scope=AssertionScope.TRANSITION,
        )
        return draft.model_copy(update={"expected_outcomes": [conditional, *refusal_outcomes]})
    if not refusal_outcomes:
        return draft
    already = {outcome.statement for outcome in draft.expected_outcomes}
    added = [outcome for outcome in refusal_outcomes if outcome.statement not in already]
    return draft.model_copy(update={"expected_outcomes": [*draft.expected_outcomes, *added]})


class ScenarioGenerator:
    """Asks for one scenario per declared cell."""

    def __init__(
        self,
        llm_client: LLMClient,
        generation_evidence: ScenarioGenerationEvidence | None = None,
    ):
        self.llm = llm_client
        self.generation_evidence = generation_evidence

    def generate_suite(
        self, journey: UserJourney, cells: Optional[list[CoverageCell]] = None
    ) -> GeneratedSuite:
        """One scenario for every cell this journey covers, in priority order.

        A cell that produced nothing is recorded with the cause. The rest of
        the suite still runs, because a device is already open and one
        unanswered cell is not a reason to test nothing.

        Args:
            journey: The journey every cell is asked about.
            cells: Which of its cells to ask for. Every one of them by
                default. A lock supplies the subset it does not already hold,
                so a run pays for the cells it is missing and no others.
        """
        suite = GeneratedSuite()
        requested_cells = sort_cells_by_priority(
            list(cells if cells is not None else journey.coverage)
        )
        evidence_gap = self._evidence_gap(requested_cells)
        if evidence_gap:
            suite.uncovered.extend(evidence_gap)
            return suite
        semantic_cells: dict[tuple[tuple[str, ...], tuple[str, ...]], str] = {}
        shared_route_memory_ids: tuple[str, ...] | None = None
        for cell in requested_cells:
            missing = self._missing_declaration(journey, cell)
            if missing:
                logger.warning("Cell %s not covered. %s", cell.key, missing.reason)
                suite.uncovered.append(missing)
                continue
            try:
                draft = self._draft(journey, cell, shared_route_memory_ids)
            except ValidationError as mismatch:
                # Three answers that never matched the schema is the model
                # failing to answer, not the cell asking for the impossible.
                suite.uncovered.append(
                    self._failed(cell, Cause.PROVIDER, f"schema never matched: {mismatch}")
                )
                continue
            except ProviderUnavailable as exc:
                # The service may answer the next cell, and a cell nobody
                # covered is recorded rather than lost.
                suite.uncovered.append(self._failed(cell, Cause.PROVIDER, reportable(exc)))
                continue
            except Exception as failure:
                if cause_of(failure) is Cause.PROVIDER:
                    # A refused request is refused for every later cell, so
                    # asking for them spends calls on the same answer.
                    raise
                suite.uncovered.append(self._failed(cell, cause_of(failure), reportable(failure)))
                continue
            if self.generation_evidence is not None and shared_route_memory_ids is None:
                shared_route_memory_ids = tuple(draft.route_memory_ids)
            signature = generation_policy.semantic_signature(draft)
            duplicate = semantic_cells.get(signature)
            if duplicate is not None:
                suite.uncovered.append(
                    UncoveredCell(
                        cell,
                        ReportStatus.UNSUPPORTED,
                        Cause.SCENARIO,
                        f"The generated scenario duplicates the semantics already owned by {duplicate}.",
                    )
                )
                continue
            semantic_cells[signature] = cell.key
            unavailable_metrics = metrics_without_live_collectors(
                outcome.metric for outcome in draft.expected_outcomes if outcome.metric is not None
            )
            if unavailable_metrics:
                metric_names = ", ".join(metric.value for metric in unavailable_metrics)
                suite.uncovered.append(
                    UncoveredCell(
                        cell,
                        ReportStatus.UNVERIFIED,
                        Cause.SCENARIO,
                        f"No live collector is available for required metric(s): {metric_names}.",
                    )
                )
                continue
            setup, setup_origin = resolve_setup(journey.preconditions)
            stamped = GeneratedTestCase.of(
                cell,
                _with_derived_outcomes(draft, journey.refused),
                setup=setup,
                setup_origin=setup_origin,
                initial_capabilities=journey.preconditions.capabilities,
            )
            suite.locked.append(stamped)
            suite.scenarios.append(with_values_filled(stamped, journey.values))
        logger.info(
            "Covered %d of %d cell(s) for journey '%s'.",
            len(suite.scenarios),
            len(cells if cells is not None else journey.coverage),
            journey.id,
        )
        return suite

    def _evidence_gap(self, cells: list[CoverageCell]) -> list[UncoveredCell]:
        evidence = self.generation_evidence
        if evidence is None:
            return []
        reasons = [
            f"{item.intent} ({item.state.value})"
            for item in evidence.unavailable_required_capabilities
        ]
        if not evidence.available_capability_ids:
            reasons.append("the AppMap contains no observed capabilities")
        if not evidence.route_prefixes:
            reasons.append("no verified RouteMemory reaches a test subject")
        if not reasons:
            return []
        reason = "Required app evidence is unavailable: " + "; ".join(reasons) + "."
        return [
            UncoveredCell(
                cell,
                ReportStatus.UNVERIFIED,
                Cause.SCENARIO,
                reason,
            )
            for cell in cells
        ]

    @staticmethod
    def _missing_declaration(journey: UserJourney, cell: CoverageCell) -> UncoveredCell | None:
        """What this cell needs that the file did not declare, if anything."""
        if cell.category not in SUPPORTED_CATEGORIES:
            return UncoveredCell(
                cell,
                ReportStatus.UNSUPPORTED,
                Cause.SCENARIO,
                f"Nothing in a run can settle a {cell.category.value} cell yet. "
                f"Remove it from 'coverage' until it can.",
            )
        reason = generation_policy.unsupported_reason(journey, cell)
        if reason is not None:
            return UncoveredCell(
                cell,
                ReportStatus.UNSUPPORTED,
                Cause.SCENARIO,
                reason,
            )
        return None

    @staticmethod
    def _failed(cell: CoverageCell, cause: Cause, detail: str) -> UncoveredCell:
        """Records a cell the generator could not write, and whose fault that is."""
        reason = f"The generator could not write this cell: {detail}"
        logger.error("Cell %s not covered (%s). %s", cell.key, cause.value, reason)
        return UncoveredCell(cell, ReportStatus.ERROR, cause, reason)

    def _draft(
        self,
        journey: UserJourney,
        cell: CoverageCell,
        required_route_memory_ids: tuple[str, ...] | None = None,
    ) -> ScenarioDraft:
        """One model call for one cell.

        Raises:
            Exception: Whatever the last attempt raised, once none are left.
        """
        stable_prompt, cell_context = self._prompt_sections(
            journey, cell, required_route_memory_ids
        )
        messages = [
            SystemMessage(content=stable_prompt),
            TextMessage(content=cell_context),
        ]
        context = _requirements(
            cell,
            journey,
            self.generation_evidence,
            required_route_memory_ids,
        )
        last_error: Exception | None = None
        for attempt in range(GENERATION_ATTEMPTS):
            try:
                response = self.llm.generate_response(
                    messages=messages,
                    response_model=ScenarioDraft,
                    context=context,
                )
                if self.generation_evidence is None:
                    return response
                return ScenarioDraft.model_validate(
                    response.model_dump(mode="json"),
                    context=context,
                )
            except Exception as exc:
                # The client already decided this one. It retried what a pause
                # could fix and raised what it could not, so asking again here
                # spends provider calls on an answer that will not change.
                if cause_of(exc) is Cause.PROVIDER:
                    raise
                last_error = exc
                logger.warning(
                    "Cell %s attempt %d/%d failed: %s",
                    cell.key,
                    attempt + 1,
                    GENERATION_ATTEMPTS,
                    exc,
                )
        assert last_error is not None
        raise last_error

    def _prompt(self, journey: UserJourney, cell: CoverageCell) -> str:
        """What the model is told about one cell."""
        stable_prompt, cell_context = self._prompt_sections(journey, cell)
        return stable_prompt + cell_context

    def _prompt_sections(
        self,
        journey: UserJourney,
        cell: CoverageCell,
        required_route_memory_ids: tuple[str, ...] | None = None,
    ) -> tuple[str, str]:
        """The reusable policy and the cell-specific input it applies to."""
        rules = journey.security_rules if cell.category is Category.SECURITY else []
        stable_prompt = PROMPT.format(actions=", ".join(sorted(ACTION_VERBS)))
        cell_context = CELL_CONTEXT.format(
            description=_scrubbed_journey_description(journey),
            category=cell.category.value,
            category_meaning=describe(cell.category),
            scenario_type=cell.scenario_type.value,
            type_meaning=describe(cell.scenario_type),
            lines=_value_lines(journey),
            rules=RULES.format(rules="\n".join(f"- {rule}" for rule in rules)) if rules else "",
            app_evidence=self._app_evidence_prompt(required_route_memory_ids),
        )
        return stable_prompt, cell_context

    def _app_evidence_prompt(self, required_route_memory_ids: tuple[str, ...] | None) -> str:
        evidence = self.generation_evidence
        if evidence is None:
            return ""
        capabilities = "\n".join(
            f"- {item.capability_id}: {item.intent} [{item.state.value}]"
            for item in evidence.capabilities
            if item.is_available
        )
        routes = "\n".join(
            f"- {item.route_memory_artifact_id}: {item.semantic_instruction} "
            f"({item.segment_count} guarded segments; reaches capability IDs: "
            f"{', '.join(item.capability_ids)})"
            for item in evidence.route_prefixes
        )
        required_prefix = (
            "Use this same route_memory_ids sequence: " + ", ".join(required_route_memory_ids)
            if required_route_memory_ids is not None
            else "The first coverage variant selects the shared route prefix."
        )
        return APP_EVIDENCE.format(
            app_map_artifact_id=evidence.app_map_artifact_id,
            capabilities=capabilities or "- none",
            routes=routes or "- none",
            required_prefix=required_prefix,
        )
