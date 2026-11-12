"""Tests for evidence-grounded scenario contracts."""

from pydantic import ValidationError
import pytest

from yewo.artifacts import EvidenceApplicationIdentity
from yewo.devices import DevicePlatform
from yewo.scenarios import (
    CoverageCategory,
    DiscoveredTestSurface,
    DiscoveryProvenance,
    ExpectedOutcomeAuthority,
    JourneyRequirementEvidence,
    RequiredEvidenceOwner,
    RequirementEvidenceState,
    ScenarioCoverageObligation,
    ScenarioDefinition,
    ScenarioEvidenceReadiness,
    ScenarioEvidenceRequirement,
    ScenarioExpectedOutcome,
    ScenarioExecutionChange,
    ScenarioSource,
    ScenarioSurfaceKind,
    ScenarioSuite,
    ScenarioType,
)


APPLICATION = EvidenceApplicationIdentity(
    platform=DevicePlatform.ANDROID,
    application_id="dev.yewo.controlled",
    version="1.0",
    build="1",
    artifact_sha256="a" * 64,
)


def _provenance() -> DiscoveryProvenance:
    return DiscoveryProvenance(
        discovery_run_id="discovery_run",
        discovery_objective_id="objective",
        procedure_id="procedure",
        verification_id="verification",
        observation_ids=("observation",),
        procedure_step_indexes=(0,),
    )


def _surface(**changes: object) -> DiscoveredTestSurface:
    values: dict[str, object] = {
        "surface_id": "surface",
        "journey_id": "journey",
        "kind": ScenarioSurfaceKind.STATE_CHANGE,
        "summary": "The user can commit a visible state change.",
        "observable_outcomes": ("The changed state is visible.",),
        "state_ids": ("state",),
        "declared_value_names": ("value",),
        "provenance": _provenance(),
    }
    values.update(changes)
    return DiscoveredTestSurface.model_validate(values)


def _requirement(**changes: object) -> JourneyRequirementEvidence:
    values: dict[str, object] = {
        "requirement_id": "requirement",
        "kind": "objective",
        "statement": "Complete the requested state change.",
        "state": RequirementEvidenceState.SUPPORTED,
        "observation_ids": ("observation",),
        "procedure_step_indexes": (0,),
        "reason": "Discovery observed and independently verified the change.",
    }
    values.update(changes)
    return JourneyRequirementEvidence.model_validate(values)


def _readiness(**changes: object) -> ScenarioEvidenceReadiness:
    values: dict[str, object] = {
        "readiness_id": "readiness",
        "journey_id": "journey",
        "application": APPLICATION,
        "discovery_run_id": "discovery_run",
        "procedure_id": "procedure",
        "verification_id": "verification",
        "requirements": (_requirement(),),
        "required_declared_value_names": ("value",),
        "executable": True,
        "independently_auditable": True,
        "reason": "Accepted discovery evidence supports generation and audit.",
    }
    values.update(changes)
    return ScenarioEvidenceReadiness.model_validate(values)


def _outcome() -> ScenarioExpectedOutcome:
    return ScenarioExpectedOutcome(
        outcome_id="outcome",
        statement="The requested state change is visible.",
        authority=ExpectedOutcomeAuthority.USER_REQUIREMENT,
        required_surface_ids=("surface",),
        evidence_requirements=(
            ScenarioEvidenceRequirement(
                requirement_id="visible_state",
                owner=RequiredEvidenceOwner.APPLICATION_OBSERVATION,
                description="A settled observation must show the changed state.",
            ),
        ),
    )


def _scenario(**changes: object) -> ScenarioDefinition:
    values: dict[str, object] = {
        "scenario_id": "scenario",
        "journey_id": "journey",
        "readiness_id": "readiness",
        "execution_change": ScenarioExecutionChange.ACCEPTED_ROUTE,
        "scenario_type": ScenarioType.HAPPY_PATH,
        "name": "Persist the changed state",
        "objective": "Make the discovered state change and observe its result.",
        "variation": "Use the accepted value and complete the operation once.",
        "route_procedure_id": "procedure",
        "declared_value_names": ("value",),
        "obligations": (_obligation(),),
    }
    values.update(changes)
    return ScenarioDefinition.model_validate(values)


def _obligation(**changes: object) -> ScenarioCoverageObligation:
    values: dict[str, object] = {
        "obligation_id": "obligation",
        "category_assessment_id": "assessment",
        "source": ScenarioSource.AUTOMATIC_QA,
        "category": CoverageCategory.STATE,
        "strategy": ScenarioType.HAPPY_PATH,
        "coverage_cell_id": "state.happy_path.0",
        "required_surface_ids": ("surface",),
        "expected_outcomes": (_outcome(),),
    }
    values.update(changes)
    return ScenarioCoverageObligation.model_validate(values)


def test_outcome_surfaces_must_belong_to_the_obligation() -> None:
    with pytest.raises(ValidationError, match="belong to their coverage obligation"):
        _obligation(
            expected_outcomes=(_outcome().model_copy(update={"required_surface_ids": ("other",)}),)
        )


def test_ready_suite_links_discovery_surfaces_and_scenarios() -> None:
    suite = ScenarioSuite(
        suite_id="suite",
        journey_id="journey",
        application=APPLICATION,
        readiness=_readiness(),
        surfaces=(_surface(),),
        scenarios=(_scenario(),),
    )

    assert suite.scenarios[0].obligations[0].category is CoverageCategory.STATE
    assert suite.surfaces[0].provenance.procedure_id == "procedure"


def test_discovery_provenance_rejects_duplicate_evidence() -> None:
    with pytest.raises(ValidationError, match="Discovery observations must be unique"):
        DiscoveryProvenance(
            discovery_run_id="discovery_run",
            discovery_objective_id="objective",
            procedure_id="procedure",
            verification_id="verification",
            observation_ids=("observation", "observation"),
        )


def test_supported_requirement_needs_observations_and_procedure_steps() -> None:
    with pytest.raises(ValidationError, match="needs observations and procedure steps"):
        _requirement(procedure_step_indexes=())


def test_unresolved_requirement_cannot_claim_accepted_evidence() -> None:
    with pytest.raises(ValidationError, match="cannot claim accepted evidence"):
        _requirement(state=RequirementEvidenceState.UNRESOLVED)


def test_unresolved_readiness_cannot_be_executable() -> None:
    unresolved = _requirement(
        state=RequirementEvidenceState.UNRESOLVED,
        observation_ids=(),
        procedure_step_indexes=(),
        reason="Discovery did not reach an observable result.",
    )

    readiness = _readiness(
        requirements=(unresolved,),
        procedure_id=None,
        verification_id=None,
        executable=False,
        independently_auditable=False,
    )

    assert not readiness.executable


def test_readiness_rejects_an_executable_claim_with_a_blocking_risk() -> None:
    with pytest.raises(ValidationError, match="Executable readiness requires"):
        _readiness(blocking_action_risk_ids=("risk",))


def test_readiness_rejects_an_executable_claim_with_an_evidence_blocker() -> None:
    with pytest.raises(ValidationError, match="Executable readiness requires"):
        _readiness(readiness_blockers=("No testable surface was observed.",))


def test_unauditable_readiness_cannot_claim_independent_audit() -> None:
    with pytest.raises(ValidationError, match="Only an executable journey"):
        _readiness(
            requirements=(
                _requirement(
                    state=RequirementEvidenceState.UNRESOLVED,
                    observation_ids=(),
                    procedure_step_indexes=(),
                ),
            ),
            procedure_id=None,
            verification_id=None,
            executable=False,
            independently_auditable=True,
        )


def test_expected_outcome_needs_an_evidence_requirement() -> None:
    with pytest.raises(ValidationError):
        ScenarioExpectedOutcome(
            outcome_id="outcome",
            statement="The state changes.",
            authority=ExpectedOutcomeAuthority.USER_REQUIREMENT,
            required_surface_ids=("surface",),
            evidence_requirements=(),
        )


def test_exploratory_suggestion_cannot_be_required() -> None:
    with pytest.raises(ValidationError, match="cannot be a required obligation"):
        _obligation(source=ScenarioSource.EXPLORATORY_SUGGESTION)


def test_scenario_required_state_must_match_its_obligations() -> None:
    optional = _obligation(
        source=ScenarioSource.EXPLORATORY_SUGGESTION,
        required=False,
    )

    with pytest.raises(ValidationError, match="must match its coverage obligations"):
        _scenario(obligations=(optional,))


def test_accepted_route_uses_the_happy_path_execution_type() -> None:
    with pytest.raises(ValidationError, match="accepted-route execution"):
        _scenario(scenario_type=ScenarioType.NEGATIVE)


def test_suite_rejects_scenario_before_readiness() -> None:
    unresolved = _requirement(
        state=RequirementEvidenceState.UNRESOLVED,
        observation_ids=(),
        procedure_step_indexes=(),
    )
    readiness = _readiness(
        requirements=(unresolved,),
        procedure_id=None,
        verification_id=None,
        executable=False,
        independently_auditable=False,
    )

    with pytest.raises(ValidationError, match="only when readiness permits"):
        ScenarioSuite(
            suite_id="suite",
            journey_id="journey",
            application=APPLICATION,
            readiness=readiness,
            surfaces=(_surface(),),
            scenarios=(_scenario(),),
        )


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"journey_id": "other"}, "suite journey"),
        ({"readiness_id": "other"}, "suite readiness"),
        ({"route_procedure_id": "other"}, "readiness-approved procedure"),
        (
            {
                "obligations": (
                    _obligation(
                        required_surface_ids=("other",),
                        expected_outcomes=(
                            _outcome().model_copy(update={"required_surface_ids": ("other",)}),
                        ),
                    ),
                )
            },
            "outside the suite",
        ),
        ({"declared_value_names": ("other",)}, "outside readiness"),
    ],
)
def test_suite_rejects_scenario_references_outside_its_boundary(
    changes: dict[str, object], message: str
) -> None:
    with pytest.raises(ValidationError, match=message):
        ScenarioSuite(
            suite_id="suite",
            journey_id="journey",
            application=APPLICATION,
            readiness=_readiness(),
            surfaces=(_surface(),),
            scenarios=(_scenario(**changes),),
        )


def test_contracts_reject_unknown_fields_and_are_frozen() -> None:
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        _scenario(unknown="value")

    scenario = _scenario()
    with pytest.raises(ValidationError, match="Instance is frozen"):
        scenario.name = "Changed"  # type: ignore[misc]
