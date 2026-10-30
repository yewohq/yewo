"""One discovery run's doubles: the runs, batches, proposers, and runtimes.

Shared because the controller's loop and the evidence it accepts are two
contracts with two test files, and one set of doubles keeps them asking the
same thing of the same shapes.
"""

from yewo.agent.actuation.actions import Coordinates, PlannerTapAction
from yewo.agent.memory.replay_store import RelativeBounds
from yewo.discovery.control_identity import discovery_control_identity
from yewo.agent.run.settlement import SettlementReceipt, SettlementStatus
from yewo.discovery.contracts import (
    JourneyGoal,
    DiscoveryObservation,
    DiscoveryRun,
    ObservedControl,
    SideEffectClass,
)
from yewo.discovery.runtime import (
    CapabilityEvidenceReceipt,
    DiscoveryActionCandidate,
    DiscoveryDispatchReceipt,
    DiscoveryExecutionReceipt,
    DiscoveryObservationBatch,
    DiscoveryProposalReceipt,
    DiscoveryResetReceipt,
    DiscoverySetupReceipt,
)
from yewo.identity import derived_uuid


SOURCE_PHASH = "0" * 16
DESTINATION_PHASH = "f" * 16
SCREEN_SIZE = (1000, 2000)

def _run(header, budgets, goal=None):
    return DiscoveryRun(
        header=header("discovery-run"),
        goals=(goal or JourneyGoal(statement="reach the editor"),),
        budgets=budgets,
    )

def _observation(identifier, state, phash, controls=()):
    return DiscoveryObservation(
        observation_id=identifier,
        screen_state_id=state,
        phash=phash,
        controls=controls,
    )

def _candidate(
    goal_id,
    observation_id="source-observation",
    *,
    source_state="source-state",
    source_phash=SOURCE_PHASH,
    x=800,
    y=1600,
):
    action = PlannerTapAction(
        coordinates=Coordinates(x=x, y=y),
        target="observed control",
        reason="inspect the safe relevant frontier",
    )
    return DiscoveryActionCandidate(
        source_observation_id=observation_id,
        source_screen_state_id=source_state,
        control_identity=discovery_control_identity(
            source_state, action, SCREEN_SIZE
        ),
        relevant_goal_ids=(goal_id,),
        action=action,
        side_effect=SideEffectClass.READ_ONLY,
        reversible=True,
        source_phash=source_phash,
        screen_size=SCREEN_SIZE,
    )


def _source_batch(goal_id, *, candidate=None, depth=0):
    candidate = candidate or _candidate(goal_id)
    control = ObservedControl(
        control_identity=candidate.control_identity,
        role="button",
        bounds=RelativeBounds(left=750, top=750, right=900, bottom=900),
        actionable=True,
    )
    observation = _observation(
        candidate.source_observation_id,
        candidate.source_screen_state_id,
        SOURCE_PHASH,
        (control,),
    )
    return DiscoveryObservationBatch(
        observation=observation,
        candidates=(candidate,),
        state_depth=depth,
    )


class FirstCandidateProposer:
    model_calls_per_proposal = 1
    maximum_cost_micro_usd = 100

    @staticmethod
    def drain_unreported_spend():
        """This proposer reports every call on the proposal that made it."""
        return 0, 0

    def select(self, run, candidates):
        del run
        return DiscoveryProposalReceipt(
            selected_frontier_id=candidates[0].frontier_id,
            model_calls=1,
            cost_micro_usd=50,
        )


class UnknownCandidateProposer(FirstCandidateProposer):
    def select(self, run, candidates):
        del run, candidates
        return DiscoveryProposalReceipt(
            selected_frontier_id="not-offered",
            model_calls=1,
            cost_micro_usd=50,
        )


class SuccessfulRuntime:
    def __init__(
        self,
        setup_sha,
        source,
        goal_id,
        evidence_owner="local_predicate",
        transition_identity=None,
    ):
        self.setup_sha = setup_sha
        self.source = source
        self.goal_id = goal_id
        self.evidence_owner = evidence_owner
        self.claimed_transition_identity = transition_identity
        self.execute_calls = 0

    def prepare(self, setup_contract_sha256):
        del setup_contract_sha256
        return DiscoverySetupReceipt(
            success=True,
            setup_contract_sha256=self.setup_sha,
        )

    def observe(self):
        return self.source

    def execute(self, authorization, candidate):
        self.execute_calls += 1
        destination = _observation(
            "destination-observation",
            "destination-state",
            DESTINATION_PHASH,
        )
        transition_identity = derived_uuid(
            "discovery-transition-v1",
            candidate.source_observation_id,
            candidate.control_identity,
            destination.observation_id,
        )
        return DiscoveryExecutionReceipt(
            dispatch=DiscoveryDispatchReceipt(
                dispatch_id="dispatch-1",
                authorization_id=authorization.authorization_id,
                success=True,
            ),
            settlement=SettlementReceipt(
                settlement_id="settlement-1",
                control_identity=authorization.control_identity,
                source_phash=SOURCE_PHASH,
                destination_phash=DESTINATION_PHASH,
                status=SettlementStatus.DURABLE_PROGRESS,
                reason="stable destination",
                observation_ids=(
                    candidate.source_observation_id,
                    destination.observation_id,
                ),
            ),
            destination=DiscoveryObservationBatch(observation=destination, state_depth=1),
            capability_evidence=(
                CapabilityEvidenceReceipt(
                    goal_id=self.goal_id,
                    evidence_owner=self.evidence_owner,
                    observation_id=destination.observation_id,
                    kind="route_verified",
                    predicate_id="editor-visible-v1",
                    transition_identity=(
                        self.claimed_transition_identity or transition_identity
                    ),
                ),
            ),
        )

    def reset(self):
        return DiscoveryResetReceipt(success=False, reason="not expected")
