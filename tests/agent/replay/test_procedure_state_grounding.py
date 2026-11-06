"""Procedure source-state authorization and grounding tests."""

from io import BytesIO
from pathlib import Path

from PIL import Image, ImageDraw

from yewo.agent.actuation import ActionKind, ActionProposal
from yewo.agent.planning.records import (
    PlanningRequest,
)
from yewo.agent.replay.procedure.grounding import (
    ProcedureGrounder,
    build_keyboard_reflow_step,
)
from yewo.agent.replay.procedure.repair import BoundedProcedureRepairActionSource
from yewo.agent.replay.procedure.source import CompiledProcedureActionSource
from yewo.agent.replay.reconciliation.compiler import compile_replay_procedure
from yewo.agent.replay.reconciliation.records import ReplayProcedure
from yewo.agent.replay.state_matching import match_evidence_state
from yewo.agent.replay.visual_matching import (
    build_state_visual_signature,
    build_target_visual_descriptor,
)
from yewo.grounding import (
    GroundedPoint,
    GroundingResolutionSource,
    GroundingTargetOperation,
    NormalizedPoint,
    PixelPoint,
)

from agent.replay.support import (
    evidence as _evidence,
    observations as _observations,
    response as _response,
)
from agent.replay.test_procedure_repair import RepairPlanner, _outcome, _request
from agent.replay.test_state_visuals import _screen


class _RefusingFallback:
    def required_model_requests(self, request):
        del request
        return 1

    def ground(self, request):
        raise AssertionError(f"Unexpected fallback for {request.grounding_id}")


def _target_screen(width: int, height: int, target_x: int, target_y: int) -> bytes:
    image = Image.new("RGB", (width, height), "white")
    drawing = ImageDraw.Draw(image)
    drawing.rounded_rectangle(
        (target_x - 45, target_y - 25, target_x + 45, target_y + 25),
        radius=8,
        fill="#204060",
    )
    drawing.line(
        (target_x - 30, target_y, target_x + 30, target_y),
        fill="white",
        width=5,
    )
    output = BytesIO()
    image.save(output, format="PNG")
    return output.getvalue()


def test_keyboard_repair_reuses_a_locally_validated_compiled_coordinate() -> None:
    evidence = _evidence()
    first = evidence.steps[0]
    stored_grounding = first.target_grounding
    assert stored_grounding is not None
    expression = "the Password text field"
    stored_grounding = stored_grounding.model_copy(
        update={
            "request": stored_grounding.request.model_copy(
                update={
                    "expression": expression,
                    "operation": GroundingTargetOperation.TEXT_FIELD,
                }
            )
        }
    )
    focused_step = first.model_copy(
        update={
            "proposal": first.proposal.model_copy(
                update={"action": ActionKind.FOCUS, "target_expression": expression}
            ),
            "target_grounding": stored_grounding,
        }
    )
    destination_state_id = focused_step.destination_state_id
    assert destination_state_id is not None
    focused_state = evidence.states[destination_state_id].model_copy(
        update={
            "presentations": tuple(
                presentation.model_copy(update={"keyboard_visible": True})
                for presentation in evidence.states[destination_state_id].presentations
            )
        }
    )
    evidence = evidence.model_copy(
        update={
            "steps": (focused_step, *evidence.steps[1:]),
            "states": {**evidence.states, destination_state_id: focused_state},
        }
    )
    procedure = compile_replay_procedure(evidence, _response())
    assert isinstance(procedure, ReplayProcedure)
    target_grounding = procedure.steps[0].target_grounding
    assert target_grounding is not None
    point = target_grounding.result.location
    assert isinstance(point, GroundedPoint)
    screenshot = _observations()[0].redacted_screenshot
    png = _target_screen(
        screenshot.width_px,
        screenshot.height_px,
        point.pixel.x,
        point.pixel.y,
    )
    descriptor = build_target_visual_descriptor(
        png,
        point.pixel,
        screenshot.width_px,
        screenshot.height_px,
    )
    assert descriptor is not None
    first_procedure_step = procedure.steps[0].model_copy(
        update={"target_visual_descriptor": descriptor}
    )
    procedure = procedure.model_copy(update={"steps": (first_procedure_step, *procedure.steps[1:])})
    procedure_source = CompiledProcedureActionSource(procedure, evidence)
    before = _observations()[0]
    dispatched = procedure_source.plan(_request(before))
    wrong = _observations()[2].model_copy(
        update={"observation_id": "password_keyboard_closed", "keyboard_visible": False}
    )
    diverged = procedure_source.plan(_request(wrong, (_outcome(dispatched, before, wrong),)))
    assert diverged.action is ActionKind.CANNOT_PROGRESS
    repair_focus = ActionProposal(
        proposal_id="repair_password_focus",
        source_observation_id=wrong.observation_id,
        action=ActionKind.FOCUS,
        target_expression=expression,
        expected_outcome="The Password field reopens the keyboard.",
        reason="Restore the compiler-recorded focus state.",
    )
    wrong_app = wrong.model_copy(update={"foreground_application_id": "other.app"})
    assert not procedure_source.authorize_keyboard_repair_grounding(
        repair_focus,
        wrong_app,
    )
    assert procedure_source.authorize_keyboard_repair_grounding(repair_focus, wrong)
    live_request = stored_grounding.request.model_copy(
        update={
            "grounding_id": "repair_grounding",
            "screenshot": wrong.redacted_screenshot,
            "source_presentation": wrong.presentation_key,
        }
    )
    grounder = ProcedureGrounder(
        _RefusingFallback(),
        procedure_source,
        screenshot_loader=lambda reference: png,
    )

    assert grounder.required_model_requests(live_request) == 0
    result = grounder.ground(live_request)

    assert result.resolution_source is GroundingResolutionSource.PROCEDURE
    assert grounder.decisions[-1].decision == "authorized_coordinate_reused"
    assert grounder.decisions[-1].expected_phash_distance is None


def test_keyboard_reflow_uses_the_same_control_in_both_keyboard_states() -> None:
    evidence = _evidence()
    procedure = compile_replay_procedure(evidence, _response())
    assert isinstance(procedure, ReplayProcedure)
    template_grounding = procedure.steps[0].target_grounding
    assert template_grounding is not None

    def moved_grounding(y: int):
        screenshot = template_grounding.request.screenshot
        location = GroundedPoint(
            normalized=NormalizedPoint(
                x=100 / (screenshot.width_px - 1),
                y=y / (screenshot.height_px - 1),
            ),
            pixel=PixelPoint(x=100, y=y),
        )
        return template_grounding.model_copy(
            update={"result": template_grounding.result.model_copy(update={"location": location})}
        )

    evidence_steps = tuple(
        step.model_copy(
            update={
                "control": step.control.model_copy(
                    update={"identity_key": "text_field:email_address"}
                )
            }
        )
        if index < 2
        else step
        for index, step in enumerate(evidence.steps)
    )
    visible_state_id = procedure.steps[1].source_state_id
    visible_state = evidence.states[visible_state_id].model_copy(
        update={
            "presentations": tuple(
                presentation.model_copy(update={"keyboard_visible": True})
                for presentation in evidence.states[visible_state_id].presentations
            )
        }
    )
    evidence = evidence.model_copy(
        update={
            "steps": evidence_steps,
            "states": {**evidence.states, visible_state_id: visible_state},
        }
    )
    hidden_reset = procedure.steps[0].model_copy(update={"target_grounding": moved_grounding(200)})
    visible_reset = procedure.steps[1].model_copy(update={"target_grounding": moved_grounding(100)})
    required = procedure.steps[2].model_copy(update={"target_grounding": moved_grounding(150)})
    procedure = procedure.model_copy(update={"steps": (hidden_reset, visible_reset, required)})

    translated = build_keyboard_reflow_step(
        required,
        hidden_reset,
        procedure,
        evidence,
    )

    assert translated is not None and translated.target_grounding is not None
    translated_location = translated.target_grounding.result.location
    assert isinstance(translated_location, GroundedPoint)
    assert translated_location.pixel == PixelPoint(x=100, y=250)


def test_procedure_grounder_reuses_authorized_nonidentical_pixels() -> None:
    evidence = _evidence()
    procedure = compile_replay_procedure(evidence, _response())
    assert isinstance(procedure, ReplayProcedure)
    source = CompiledProcedureActionSource(procedure, evidence)
    recorded = _observations()[0]
    live_screenshot = recorded.redacted_screenshot.model_copy(
        update={
            "artifact_id": "live_screenshot",
            "sha256": "1" * 64,
            "path": Path("assets/screenshots/live_screenshot.png"),
        }
    )
    live_fingerprint = recorded.screen_fingerprint.model_copy(
        update={
            "value": "0000000000000003",
            "masked_pixels_sha256": "2" * 64,
            "source_artifact_id": "live_screenshot",
        }
    )
    live = recorded.model_copy(
        update={
            "observation_id": "live_observation",
            "redacted_screenshot": live_screenshot,
            "screen_fingerprint": live_fingerprint,
        }
    )
    proposal = source.plan(
        PlanningRequest(
            observation=live,
            objective="Replay.",
            completion_conditions=("Compiled expectation passes.",),
            available_actions=frozenset({ActionKind.TAP}),
        )
    )
    assert proposal.action is ActionKind.TAP
    stored = procedure.steps[0].target_grounding
    assert stored is not None
    request = stored.request.model_copy(
        update={
            "grounding_id": "live_grounding",
            "screenshot": live_screenshot,
            "source_presentation": live.presentation_key,
        }
    )
    grounder = ProcedureGrounder(_RefusingFallback(), source)

    result = grounder.ground(request)

    assert result.resolution_source is GroundingResolutionSource.PROCEDURE
    assert grounder.decisions[0].decision == "authorized_coordinate_reused"
    assert grounder.decisions[0].expected_phash_distance == 1


def test_compiled_source_rejects_ambiguous_known_state() -> None:
    evidence = _evidence()
    expected = evidence.states["state_001"]
    competing = evidence.states["state_002"].model_copy(
        update={
            "presentations": tuple(
                expected.representative_presentation
                for unused in evidence.states["state_002"].observation_ids
            ),
        }
    )
    evidence = evidence.model_copy(update={"states": {**evidence.states, "state_002": competing}})
    procedure = compile_replay_procedure(evidence, _response())
    assert isinstance(procedure, ReplayProcedure)
    source = CompiledProcedureActionSource(procedure, evidence)

    proposal = source.plan(
        PlanningRequest(
            observation=_observations()[0],
            objective="Replay.",
            completion_conditions=("Compiled expectation passes.",),
            available_actions=frozenset({ActionKind.TAP}),
        )
    )

    assert proposal.action is ActionKind.CANNOT_PROGRESS
    assert source.divergence is not None
    assert source.divergence.reason.endswith("source_state_ambiguous.")


def test_ambiguous_source_state_escalates_instead_of_false_rejoining() -> None:
    evidence = _evidence()
    expected = evidence.states["state_001"]
    competing = evidence.states["state_002"].model_copy(
        update={
            "presentations": tuple(
                expected.representative_presentation
                for unused in evidence.states["state_002"].observation_ids
            ),
        }
    )
    evidence = evidence.model_copy(update={"states": {**evidence.states, "state_002": competing}})
    procedure = compile_replay_procedure(evidence, _response())
    assert isinstance(procedure, ReplayProcedure)
    planner = RepairPlanner()
    source = BoundedProcedureRepairActionSource(
        CompiledProcedureActionSource(procedure, evidence),
        planner,
    )
    request = _request(_observations()[0])

    boundary = source.plan(request)
    assert boundary.action is ActionKind.WAIT
    assert source.required_model_requests(request) == 1

    repair_action = source.plan(request)

    assert repair_action.action is ActionKind.WAIT
    assert planner.request_count == 1
    assert len(source.divergences) == 1
    assert len(source.repairs) == 1
    assert not source.repairs[0].succeeded
    assert source.repairs[0].model_call_count == 1


def test_stable_regions_can_match_when_global_phash_is_too_distant() -> None:
    evidence = _evidence()
    recorded = _observations()[0]
    live = recorded.model_copy(
        update={
            "screen_fingerprint": recorded.screen_fingerprint.model_copy(
                update={"value": "000000000000ffff"}
            )
        }
    )
    state = evidence.states["state_001"].model_copy(
        update={"visual_signature": build_state_visual_signature((_screen(1), _screen(2)))}
    )
    evidence = evidence.model_copy(update={"states": {"state_001": state}})

    match = match_evidence_state(live, "state_001", evidence, lambda unused: _screen(3))

    assert match.accepted
    assert match.expected_distance is not None
    assert match.expected_distance > evidence.state_grouping_maximum_phash_distance
    assert match.visual_match_ratio == 1.0
