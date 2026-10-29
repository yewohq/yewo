import pytest
from pydantic import ValidationError

from yewo.discovery.artifacts import seal_artifact, verify_artifact
from yewo.discovery.contracts import (
    CapabilityCandidate,
    CapabilityQuestion,
    CapabilityState,
    DiscoveryObservation,
    DiscoveryRun,
    DiscoveryTerminationSignals,
    DiscoveryTransitionEvidence,
    ObservedControl,
    SideEffectClass,
    finalize_discovery,
)
from yewo.agent.memory.replay_store import RelativeBounds
from yewo.discovery.identity_calibration import (
    IdentityCalibrationThresholds,
    ScreenFamilyClassificationReceipt,
    ScreenFamilyFingerprint,
    ScreenFamilyLabel,
    ScreenFamilyPair,
    build_calibration_result,
    seal_identity_calibration,
)
from yewo.discovery.map_builder import (
    RevalidationScope,
    ScenarioMapEvidence,
    build_app_map,
    plan_app_map_revalidation,
)


def _transition(source="observation-1", destination="observation-2"):
    return DiscoveryTransitionEvidence(
        transition_identity=f"transition-{source}-{destination}",
        source_observation_id=source,
        destination_observation_id=destination,
        control_identity="control-edit",
        action_kind="tap",
        authorization_receipt_id="authorization-1",
        dispatch_receipt_id="dispatch-1",
        settlement_receipt_id="settlement-1",
        side_effect=SideEffectClass.READ_ONLY,
        reversible=True,
    )


def _observations():
    return (
        DiscoveryObservation(
            observation_id="observation-1",
            screen_state_id="state-list",
            screen_family_candidate_id="untrusted-family",
            phash="0" * 64,
            controls=(
                ObservedControl(
                    control_identity="control-edit",
                    role="button",
                    text_digest="digest-edit",
                    bounds=RelativeBounds(left=0, top=0, right=100, bottom=100),
                    actionable=True,
                ),
            ),
        ),
        DiscoveryObservation(
            observation_id="observation-2",
            screen_state_id="state-editor",
            screen_family_candidate_id="untrusted-family",
            phash="f" * 64,
            controls=(
                ObservedControl(
                    control_identity="control-edit",
                    role="button",
                    text_digest="digest-edit",
                    bounds=RelativeBounds(left=0, top=0, right=100, bottom=100),
                    actionable=True,
                ),
            ),
        ),
    )


def _discovery_run(header, budgets):
    run = DiscoveryRun(
        header=header("discovery-run"),
        questions=(CapabilityQuestion(intent="reach the editor"),),
        observations=_observations(),
        transitions=(_transition(),),
        capability_candidates=(
            CapabilityCandidate(
                candidate_id="candidate-editor",
                capability="schedule editor",
                observation_ids=("observation-2",),
                confidence=1,
            ),
        ),
        budgets=budgets,
    )
    terminal = finalize_discovery(
        run,
        DiscoveryTerminationSignals(externally_interrupted=True),
    )
    return seal_artifact(terminal)


def _calibration():
    left = ScreenFamilyFingerprint(
        sample_id="left",
        app_scope="held-out",
        platform="android",
        viewport_class="phone-portrait",
        phash="0" * 64,
        structure_sha256="a" * 64,
        traits=frozenset({"form"}),
    )
    pair = ScreenFamilyPair(
        pair_id="same",
        left=left,
        right=left.model_copy(update={"sample_id": "right"}),
        label=ScreenFamilyLabel.SAME,
    )
    thresholds = IdentityCalibrationThresholds(
        max_false_merge_rate=0,
        max_false_split_rate=0,
        max_uncertain_rate=0,
        min_pairs=1,
        min_same_family_pairs=1,
        min_different_family_pairs=1,
        required_platforms=frozenset({"android"}),
        required_traits=frozenset({"form"}),
    )
    different = pair.model_copy(
        update={
            "pair_id": "different",
            "right": left.model_copy(update={"sample_id": "other", "structure_sha256": "b" * 64}),
            "label": ScreenFamilyLabel.DIFFERENT,
        }
    )
    return seal_identity_calibration(
        build_calibration_result((pair, different), "c" * 64, thresholds)
    )


def test_map_is_reproducible_and_ignores_uncalibrated_family_candidates(header, budgets):
    run = _discovery_run(header, budgets)

    first = build_app_map((run,))
    second = build_app_map((run,))

    assert first == second
    assert verify_artifact(first)
    assert len(first.nodes) == 2
    assert all(node.identity_scope == "exact_state" for node in first.nodes)
    assert first.edges[0].action_kind == "tap"
    assert first.identity_calibration_artifact_id is None
    assert len(first.capabilities) == 1
    assert first.capabilities[0].intent == "reach the editor"
    assert first.capabilities[0].state is CapabilityState.NOT_EVALUATED


def test_map_groups_only_classifier_receipts_backed_by_passing_calibration(header, budgets):
    run = _discovery_run(header, budgets)
    calibration = _calibration()
    receipts = tuple(
        ScreenFamilyClassificationReceipt(
            observation_id=observation.observation_id,
            screen_family_id="family-schedule",
            calibration_artifact_id=calibration.artifact_id,
            fingerprint_sha256="d" * 64,
        )
        for observation in run.observations
    )

    app_map = build_app_map(
        (run,),
        family_classifications=receipts,
        calibration=calibration,
    )

    assert len(app_map.nodes) == 1
    assert app_map.nodes[0].identity_scope == "calibrated_family"
    assert len(app_map.nodes[0].predicates) == 2
    assert app_map.identity_calibration_artifact_id == calibration.artifact_id


def test_map_refuses_unsealed_or_incompatible_evidence(header, budgets):
    sealed = _discovery_run(header, budgets)
    unsealed = sealed.model_copy(
        update={"header": sealed.header.model_copy(update={"integrity_sha256": None})}
    )

    with pytest.raises(ValueError, match="sealed source"):
        build_app_map((unsealed,))

    changed_header = sealed.header.model_copy(
        update={
            "artifact_id": "other-run",
            "compatibility": sealed.header.compatibility.model_copy(
                update={"app_artifact_sha256": "e" * 64}
            ),
            "integrity_sha256": None,
        }
    )
    incompatible = seal_artifact(sealed.model_copy(update={"header": changed_header}))
    with pytest.raises(ValueError, match="incompatible"):
        build_app_map((sealed, incompatible))


def test_map_requires_terminal_discovery_and_consistent_redaction(header, budgets):
    sealed = _discovery_run(header, budgets)
    open_run = seal_artifact(
        sealed.model_copy(
            update={
                "header": sealed.header.model_copy(update={"integrity_sha256": None}),
                "terminal": None,
            }
        )
    )
    changed_redaction = sealed.header.redaction.model_copy(
        update={"policy_version": "redaction-v2"}
    )
    scenario = seal_artifact(
        ScenarioMapEvidence(
            header=header("scenario-map-evidence").model_copy(
                update={"redaction": changed_redaction}
            ),
            observations=(sealed.observations[0],),
        )
    )

    with pytest.raises(ValueError, match="at least one discovery"):
        build_app_map(())
    with pytest.raises(ValueError, match="terminal discovery"):
        build_app_map((open_run,))
    with pytest.raises(ValueError, match="redaction policy"):
        build_app_map((sealed,), (scenario,))


def test_map_rejects_conflicting_immutable_observations(header, budgets):
    run = _discovery_run(header, budgets)
    conflicting = run.observations[0].model_copy(update={"screen_state_id": "conflicting-state"})
    scenario = seal_artifact(
        ScenarioMapEvidence(
            header=header("scenario-map-evidence"),
            observations=(conflicting,),
        )
    )

    with pytest.raises(ValueError, match="Conflicting immutable evidence"):
        build_app_map((run,), (scenario,))


def test_scenario_evidence_adds_mechanics_without_verdict_authority(header, budgets):
    run = _discovery_run(header, budgets)
    scenario_observation = DiscoveryObservation(
        observation_id="observation-3",
        screen_state_id="state-form",
        phash="a" * 64,
    )
    scenario = seal_artifact(
        ScenarioMapEvidence(
            header=header("scenario-map-evidence"),
            observations=(run.observations[1], scenario_observation),
            transitions=(_transition("observation-2", "observation-3"),),
        )
    )

    app_map = build_app_map((run,), (scenario,))

    assert len(app_map.nodes) == 3
    assert len(app_map.edges) == 2
    assert app_map.source_scenario_evidence_artifact_ids == (scenario.header.artifact_id,)
    assert "verdict" not in app_map.model_dump(mode="json")


@pytest.mark.parametrize(
    "mutation,error",
    (
        ("schema", "own schema"),
        ("duplicate_observation", "observation identifiers"),
        ("duplicate_transition", "transition identities"),
        ("unknown_transition_observation", "exact observations"),
        ("unknown_capability_observation", "capability evidence"),
    ),
)
def test_scenario_map_evidence_rejects_malformed_graph(header, mutation, error):
    observations = _observations()
    scenario = ScenarioMapEvidence(
        header=header("scenario-map-evidence"),
        observations=observations,
        transitions=(_transition(),),
        capability_candidates=(
            CapabilityCandidate(
                capability="editor",
                observation_ids=(observations[1].observation_id,),
                confidence=1,
            ),
        ),
    )
    payload = scenario.model_dump(mode="json")
    if mutation == "schema":
        payload["header"]["schema_name"] = "discovery-run"
    elif mutation == "duplicate_observation":
        payload["observations"].append(payload["observations"][0])
    elif mutation == "duplicate_transition":
        payload["transitions"].append(payload["transitions"][0])
    elif mutation == "unknown_transition_observation":
        payload["transitions"][0]["destination_observation_id"] = "missing"
    else:
        payload["capability_candidates"][0]["observation_ids"] = ["missing"]

    with pytest.raises(ValidationError, match=error):
        ScenarioMapEvidence.model_validate(payload)


def test_revalidation_preserves_nodes_only_for_edge_contract_changes(header, budgets):
    app_map = build_app_map((_discovery_run(header, budgets),))
    policy_change = app_map.header.compatibility.model_copy(
        update={"discovery_policy_version": "discovery-policy-v2"}
    )
    build_change = app_map.header.compatibility.model_copy(update={"app_artifact_sha256": "e" * 64})

    edge_plan = plan_app_map_revalidation(app_map, policy_change)
    graph_plan = plan_app_map_revalidation(app_map, build_change)

    assert edge_plan.scope is RevalidationScope.EDGES
    assert edge_plan.affected_node_ids == ()
    assert set(edge_plan.preserved_node_ids) == {node.node_id for node in app_map.nodes}
    assert graph_plan.scope is RevalidationScope.GRAPH
    assert set(graph_plan.affected_node_ids) == {node.node_id for node in app_map.nodes}
    assert graph_plan.preserved_node_ids == ()

    compatible_plan = plan_app_map_revalidation(
        app_map,
        app_map.header.compatibility,
    )
    assert compatible_plan.scope is RevalidationScope.NONE
    assert compatible_plan.affected_edge_ids == ()

    unsealed = app_map.model_copy(
        update={"header": app_map.header.model_copy(update={"integrity_sha256": None})}
    )
    with pytest.raises(ValueError, match="sealed AppMap"):
        plan_app_map_revalidation(unsealed, policy_change)

    payload = compatible_plan.model_dump(mode="json")
    payload["header"]["schema_name"] = "app-map"
    with pytest.raises(ValidationError, match="own schema"):
        type(compatible_plan).model_validate(payload)

    payload = compatible_plan.model_dump(mode="json")
    payload["incompatible_dimensions"] = ["app_artifact_sha256"]
    with pytest.raises(ValidationError, match="compatible map"):
        type(compatible_plan).model_validate(payload)

    payload = edge_plan.model_dump(mode="json")
    payload["incompatible_dimensions"] = []
    with pytest.raises(ValidationError, match="requires an incompatible"):
        type(edge_plan).model_validate(payload)


@pytest.mark.parametrize(
    "receipt_change,error",
    (
        ({"calibration_artifact_id": "another"}, "another calibration"),
        ({"observation_id": "missing"}, "accepted observation"),
    ),
)
def test_map_rejects_unowned_family_classification(header, budgets, receipt_change, error):
    run = _discovery_run(header, budgets)
    calibration = _calibration()
    receipt = ScreenFamilyClassificationReceipt(
        observation_id=run.observations[0].observation_id,
        screen_family_id="family-schedule",
        calibration_artifact_id=calibration.artifact_id,
        fingerprint_sha256="d" * 64,
    ).model_copy(update=receipt_change)

    with pytest.raises(ValueError, match=error):
        build_app_map(
            (run,),
            family_classifications=(receipt,),
            calibration=calibration,
        )


def test_map_rejects_family_receipts_without_calibration_and_duplicates(header, budgets):
    run = _discovery_run(header, budgets)
    calibration = _calibration()
    receipt = ScreenFamilyClassificationReceipt(
        observation_id=run.observations[0].observation_id,
        screen_family_id="family-schedule",
        calibration_artifact_id=calibration.artifact_id,
        fingerprint_sha256="d" * 64,
    )

    with pytest.raises(ValueError, match="require calibration"):
        build_app_map((run,), family_classifications=(receipt,))
    with pytest.raises(ValueError, match="only one family classification"):
        build_app_map(
            (run,),
            family_classifications=(receipt, receipt),
            calibration=calibration,
        )
