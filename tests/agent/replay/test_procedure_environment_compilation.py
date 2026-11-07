"""Compiler validation for discovery-proven environment transitions."""

from yewo.agent.actuation import ActionKind
from yewo.agent.replay import (
    CompilerCheck,
    CompilerRefusal,
    EvidenceStepKind,
    ReplayProcedure,
    compile_replay_procedure,
)

from agent.replay.support import checks as _checks, evidence as _evidence, response as _response

RECORDED_PERMISSION_CHOICE = "the recorded permission choice"


def _permission_evidence():
    evidence = _evidence()
    permission_state_id = evidence.steps[0].destination_state_id
    assert permission_state_id is not None
    permission_state = evidence.states[permission_state_id].model_copy(
        update={
            "presentations": tuple(
                presentation.model_copy(
                    update={
                        "foreground_application_id": ("com.google.android.permissioncontroller"),
                        "platform_context": (
                            "com.android.permissioncontroller.permission.ui."
                            "GrantPermissionsActivity"
                        ),
                    }
                )
                for presentation in evidence.states[permission_state_id].presentations
            ),
        }
    )
    environment_step = evidence.steps[1].model_copy(
        update={
            "kind": EvidenceStepKind.ENVIRONMENT,
            "proposal": evidence.steps[1].proposal.model_copy(
                update={
                    "action": ActionKind.TAP,
                    "text": None,
                    "target_expression": RECORDED_PERMISSION_CHOICE,
                }
            ),
        }
    )
    return evidence.model_copy(
        update={
            "states": {
                **evidence.states,
                permission_state_id: permission_state,
            },
            "steps": (
                evidence.steps[0],
                environment_step,
                evidence.steps[2],
            ),
        }
    )


def test_compiler_accepts_permission_transition() -> None:
    evidence = _permission_evidence()

    outcome = compile_replay_procedure(evidence, _response())

    assert isinstance(outcome, ReplayProcedure)
    assert outcome.steps[1].procedure_effect == "environment"
    requirement = outcome.setup_contract.runtime_permissions[0]
    assert requirement.trigger_source_step == 1
    assert requirement.interaction_source_step == 2
    assert requirement.prompt_state_id == evidence.steps[1].source_state_id
    assert requirement.return_state_id == evidence.steps[1].destination_state_id


def test_compiler_refuses_generic_system_surface() -> None:
    evidence = _permission_evidence()
    state_id = evidence.steps[1].source_state_id
    system_state = evidence.states[state_id].model_copy(
        update={
            "presentations": tuple(
                presentation.model_copy(
                    update={"foreground_application_id": "com.android.systemui"}
                )
                for presentation in evidence.states[state_id].presentations
            )
        }
    )
    evidence = evidence.model_copy(update={"states": {**evidence.states, state_id: system_state}})

    outcome = compile_replay_procedure(evidence, _response())

    assert isinstance(outcome, CompilerRefusal)
    assert CompilerCheck.RESULTS_REUSABLE in _checks(outcome)


def test_compiler_refuses_evidence_without_device_identity() -> None:
    evidence = _evidence()
    evidence = evidence.model_copy(update={"run": evidence.run.model_copy(update={"device": None})})

    outcome = compile_replay_procedure(evidence, _response())

    assert isinstance(outcome, CompilerRefusal)
    assert CompilerCheck.RESULTS_REUSABLE in _checks(outcome)
