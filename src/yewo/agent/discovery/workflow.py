"""Journey-directed discovery through YewoAgent mechanics."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path
import time

from yewo.grounding import GroundingProfile
from yewo.models import ModelProfileRegistry, ModelRole
from yewo.observations import ObservationStore, RedactionRegion
from yewo.runs import (
    InterruptionSignal,
    RunExecutionContext,
    RunPreparationContext,
    RunSessionPolicy,
    RunWorkflowOutcome,
)

from yewo.agent.memory.records import build_memory_compatibility
from yewo.agent.composition import build_device_agent, build_model_client
from yewo.agent.memory.store import AppMemoryStore, load_discovery_memory
from yewo.applications import ApplicationInstallMode
from yewo.agent.replay.procedure.store import CompiledProcedureStore
from yewo.agent.runtime.records import AgentModePolicy
from yewo.agent.discovery.promotion import AppMemoryBuilder
from yewo.agent.discovery import conclusion, narration
from yewo.agent.discovery.records import DiscoveryRequest, DiscoveryResult
from yewo.agent.discovery.compilation_milestones import (
    DiscoveryCompilationMilestones,
    compile_completed_discovery,
    save_verified_compilation,
    verified_compiled_procedure,
)
from yewo.agent.discovery.procedure_handoff import prepare_procedure_replay_entry
from yewo.agent.discovery.replay_handoff import (
    exhausted_replay_budget_result,
    remaining_policy_after_replay,
    replay_model_request_count,
    replay_observation_ids,
    verify_discovery_goal,
)
from yewo.agent.discovery.task import build_discovery_agent_task
from yewo.agent.planning.goal_evidence import AdaptiveGoalEvidencePacker
from yewo.agent.planning.goal_verifier import GoalVerifier
from yewo.agent.screen_analysis import ScreenAnalysisFailureLedger
from yewo.agent.discovery import preparation as discovery_preparation
from yewo.agent.discovery.outcome import persist_discovery_outcome_and_compile


class DiscoveryWorkflow:
    """Run shared agent mechanics and promote only accepted discovery evidence."""

    def __init__(
        self,
        request: DiscoveryRequest,
        mode_policy: AgentModePolicy,
        model_profiles: ModelProfileRegistry,
        grounding_profile: GroundingProfile,
        planner_profile_id: str,
        auditor_profile_id: str,
        recovery_profile_id: str,
        action_verifier_profile_id: str,
        replay_preparation_profile_id: str,
        memory_root: Path | None,
        scrub_text: Callable[[str], str],
        *,
        screen_analyzer_profile_id: str | None = None,
        redaction_regions: Sequence[RedactionRegion] = (),
        declared_value_resolver: Callable[[str], str] | None = None,
        memory_builder: AppMemoryBuilder | None = None,
        memory_store: AppMemoryStore | None = None,
        procedure_store: CompiledProcedureStore | None = None,
        goal_verifier: GoalVerifier | None = None,
        require_reusable_evidence: bool = True,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        self._request = request
        self._mode_policy = mode_policy
        self._model_profiles = model_profiles
        self._grounding_profile = grounding_profile
        self._planner_profile_id = planner_profile_id
        self._auditor_profile_id = auditor_profile_id
        self._recovery_profile_id = recovery_profile_id
        self._action_verifier_profile_id = action_verifier_profile_id
        self._replay_preparation_profile_id = replay_preparation_profile_id
        self._screen_analyzer_profile_id = screen_analyzer_profile_id
        if memory_store is not None:
            self._memory_store = memory_store
        elif memory_root is not None:
            self._memory_store = AppMemoryStore(memory_root)
        else:
            raise ValueError("Discovery requires a Yewo memory root or memory store.")
        self._scrub_text = scrub_text
        self._redaction_regions = tuple(redaction_regions)
        if request.declared_value_names and declared_value_resolver is None:
            raise ValueError("Discovery declared values require a resolver.")
        self._declared_value_resolver = declared_value_resolver
        self._memory_builder = memory_builder or AppMemoryBuilder()
        self._goal_verifier = goal_verifier
        self._procedure_store = procedure_store
        if self._procedure_store is None:
            procedure_root = (
                memory_root
                if memory_root is not None
                else self._memory_store.root
                if isinstance(self._memory_store, AppMemoryStore)
                else None
            )
            if procedure_root is None:
                raise ValueError("Compiled replay requires a durable procedure store.")
            self._procedure_store = CompiledProcedureStore(procedure_root)
        self._require_reusable_evidence = require_reusable_evidence
        self._monotonic = monotonic

    def session_policy(self) -> RunSessionPolicy:
        """Apply declared redaction regions before automation starts."""
        return RunSessionPolicy(recording_redaction_regions=self._redaction_regions)

    def prepare(
        self,
        context: RunPreparationContext,
        interruption: InterruptionSignal,
    ) -> discovery_preparation.DiscoveryPreparation:
        """Prepare retained memory before Appium starts the application."""
        preparation_model_requests = 0
        if (
            self._procedure_store is not None
            and context.workspace.start.request.application.install_mode
            is ApplicationInstallMode.INSTALL
        ):
            prepared_procedure, preparation_model_requests = (
                discovery_preparation.prepare_compiled_procedure(
                    context,
                    self._request,
                    self._memory_store,
                    self._procedure_store,
                    self._model_profiles,
                    self._replay_preparation_profile_id,
                    self._scrub_text,
                    self._redaction_regions,
                    self._mode_policy.maximum_model_requests,
                )
            )
            if prepared_procedure is not None:
                return prepared_procedure
        return discovery_preparation.prepare_memory_identity(
            context,
            self._memory_store,
            preparation_model_requests,
        )

    def execute(
        self,
        context: RunExecutionContext,
        interruption: InterruptionSignal,
        preparation: object | None = None,
    ) -> RunWorkflowOutcome:
        """Discover, promote accepted evidence, and persist truthful coverage."""
        started_at = self._monotonic()
        narration.narrate_discovery_start(self._request)
        compatibility = build_memory_compatibility(
            context.session.application,
            context.workspace.start.request.application,
        )
        existing = load_discovery_memory(self._memory_store, compatibility)
        model_client = build_model_client(
            context.workspace.root,
            context.workspace.start.run_id,
            self._model_profiles,
            self._scrub_text,
            bool(self._redaction_regions),
        )
        screen_analysis_failures = ScreenAnalysisFailureLedger()
        compilation: DiscoveryCompilationMilestones | None = None

        def persist_outcome(result: DiscoveryResult) -> RunWorkflowOutcome:
            return persist_discovery_outcome_and_compile(
                context,
                result,
                self._require_reusable_evidence,
                model_client,
                self._replay_preparation_profile_id,
                compatibility.screen_comparison,
                self._procedure_store,
                compatibility,
                compilation.report if compilation is not None else None,
                self._memory_store,
            )

        objective = self._request.objectives[0]
        prepared = discovery_preparation.matching_preparation(preparation, existing)
        replay_entry = prepare_procedure_replay_entry(
            context,
            compatibility,
            self._request,
            self._mode_policy,
            model_client,
            self._grounding_profile,
            self._auditor_profile_id,
            self._planner_profile_id,
            self._recovery_profile_id,
            self._action_verifier_profile_id,
            interruption,
            self._procedure_store,
            self._model_profiles.resolve(self._auditor_profile_id, ModelRole.AUDITOR).images,
            existing,
            self._redaction_regions,
            self._declared_value_resolver,
            prepared,
            screen_analyzer_profile_id=self._screen_analyzer_profile_id,
            screen_analysis_failures=screen_analysis_failures,
        )
        procedure_replay = replay_entry.replay
        if replay_entry.conclusion is not None:
            return persist_outcome(replay_entry.conclusion)
        if procedure_replay is not None:
            prepared = None
        repair_policy = remaining_policy_after_replay(
            self._mode_policy,
            procedure_replay,
            self._monotonic() - started_at,
        )
        if repair_policy is None:
            result = exhausted_replay_budget_result(
                self._request,
                procedure_replay,
                existing.memory_id if existing is not None else None,
            )
            if result is not None:
                return persist_outcome(result)
            raise AssertionError("An exhausted replay budget requires replay evidence.")
        agent_result = build_device_agent(
            context,
            model_client,
            self._grounding_profile,
            self._planner_profile_id,
            recovery_profile_id=self._recovery_profile_id,
            redaction_regions=self._redaction_regions,
            action_verifier_profile_id=self._action_verifier_profile_id,
            screen_analyzer_profile_id=self._screen_analyzer_profile_id,
            screen_analysis_failures=screen_analysis_failures,
            recover_application=True,
            declared_value_resolver=self._declared_value_resolver,
            screen_comparison=compatibility.screen_comparison,
            planning_memory=existing,
            grounding_memory=existing,
            progress_context=(prepared.progress_context if prepared is not None else None),
            initial_planning_requests=(
                prepared.progress_context.preparation_planning_requests
                if prepared is not None and prepared.progress_context is not None
                else 0
            ),
        ).run(
            build_discovery_agent_task(self._request, existing),
            repair_policy,
            interruption,
        )
        compilation = compile_completed_discovery(
            procedure_replay,
            context.workspace.root,
            context.workspace.start.run_id,
            self._request,
            agent_result,
            model_client,
            self._replay_preparation_profile_id,
            compatibility.screen_comparison,
            existing,
        )
        milestone_ids = compilation.observation_ids() if compilation is not None else set()
        observations = ObservationStore(context.workspace.root).load_selected(
            replay_observation_ids(agent_result, procedure_replay) | milestone_ids
        )
        observations_by_id = {item.observation_id: item for item in observations}
        used_model_requests = (
            agent_result.usage.planning_requests
            + agent_result.usage.grounding_requests
            + replay_model_request_count(procedure_replay)
        )
        verification = verify_discovery_goal(
            objective,
            agent_result,
            observations_by_id,
            self._mode_policy,
            used_model_requests,
            self._monotonic() - started_at,
            procedure_replay,
            model_client,
            self._auditor_profile_id,
            self._goal_verifier,
            (compilation.goal_milestones(observations_by_id) if compilation is not None else ()),
            AdaptiveGoalEvidencePacker(
                context.workspace.root,
                self._model_profiles.resolve(self._auditor_profile_id, ModelRole.AUDITOR).images,
            ),
        )
        save_verified_compilation(
            compilation,
            verification,
            self._procedure_store,
            compatibility,
            context.workspace.root,
        )
        compiled_procedure = verified_compiled_procedure(compilation, verification)
        milestone_evidence_ids = tuple(sorted(milestone_ids | {agent_result.final_observation_id}))
        promotion = self._memory_builder.promote(
            self._request,
            agent_result,
            observations_by_id,
            compatibility,
            verified_objective_ids=(
                frozenset({objective.objective_id})
                if verification is not None and verification.satisfied
                else frozenset()
            ),
            source_run_id=context.workspace.start.run_id,
            updated_at=context.workspace.start.started_at,
            existing=existing,
            compiled_procedure=compiled_procedure,
            evidence_observation_ids=milestone_evidence_ids,
        )
        memory = promotion.memory
        coverage = promotion.coverage
        if memory is not None and memory != existing:
            memory = conclusion.save_discovery_memory(
                self._memory_store,
                memory,
                context.workspace.root,
                context.workspace,
                self._request,
                agent_result,
                coverage,
                verification,
                existing.memory_id if existing is not None else None,
                procedure_replay,
            )
        result = conclusion.build_discovery_result(
            self._request,
            agent_result,
            coverage,
            procedure_replay,
            verification,
            compiled_procedure,
            memory,
        )
        return persist_outcome(result)
