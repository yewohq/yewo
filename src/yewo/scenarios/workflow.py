"""Discovery-first execution workflow for ``yewo test``."""

from __future__ import annotations

from collections.abc import Callable
from typing import cast

from yewo.agent.composition import build_model_client
from yewo.agent.discovery import DiscoveryObjective, DiscoveryRequest
from yewo.agent.discovery.conclusion import load_discovery_result
from yewo.agent.discovery.workflow import DiscoveryWorkflow
from yewo.agent.memory.records import AppMemory, build_memory_compatibility
from yewo.agent.memory.store import AppMemoryStore
from yewo.agent.runtime.records import AgentResult, AgentTask
from yewo.agent.replay.procedure.execution import execute_compiled_procedure
from yewo.agent.replay.procedure.records import ProcedureReplaySetup, ProcedureResetMethod
from yewo.agent.replay.procedure.store import CompiledProcedureStore, StoredProcedureBundle
from yewo.artifacts import (
    AtomicArtifactWriter,
    EvidenceDeviceIdentity,
    new_record_identifier,
)
from yewo.devices.application_lifecycle import ApplicationState
from yewo.devices.application_preconditions import ApplicationPreconditionCoordinator
from yewo.devices.capabilities import (
    ApplicationLifecycleCapabilityAdapter,
    CapabilityPreparation,
    CapabilitySupportResult,
    CapabilitySupportState,
    CapabilityTransactionCoordinator,
)
from yewo.devices.interaction import DeviceInteraction
from yewo.models import ModelRole
from yewo.project_config import CapabilityKind
from yewo.project_config.capabilities import (
    AppStateCapabilitySpec,
    CapabilitySpec,
    DeviceCapabilitySpec,
)
from yewo.reporting import (
    ScenarioAuditRecord,
    ScenarioExecutionClassification,
)
from yewo.runs import (
    InterruptionSignal,
    RunExecutionContext,
    RunPreparationContext,
    RunSessionPolicy,
    RunTerminalCategory,
    RunWorkflowOutcome,
    RunWorkspace,
)
from yewo.runs.errors import RunWorkflowInterrupted
from yewo.scenarios.bridge import build_discovery_testing_bridge
from yewo.scenarios.auditor import ScenarioAuditorService, ScenarioAuditStore
from yewo.scenarios.budget import ScenarioBudgetBank
from yewo.scenarios.coordinator import (
    ScenarioCoordinationResult,
    ScenarioCoordinator,
    ScenarioExecutionStore,
)
from yewo.scenarios.designer import (
    TestDesignerInvalidOutputError,
    TestDesignerService,
)
from yewo.scenarios.generator import generate_scenarios
from yewo.scenarios.records import ScenarioDefinition, ScenarioGenerationResult
from yewo.scenarios.paths import SCENARIO_GENERATION_PATH, scenario_artifact_directory
from yewo.scenarios.registry import TestingRegistryContext, evaluate_testing_categories
from yewo.scenarios.shared_agent import SharedDeviceAgentTaskExecutor
from yewo.scenarios.workflow_outcome import derive_workflow_outcome
from yewo.test_config import YewoTestConfiguration


class ScenarioTestWorkflow:
    """Discover one journey, generate its suite, and execute every scenario."""

    def __init__(
        self,
        configuration: YewoTestConfiguration,
        scrub_text: Callable[[str], str],
    ) -> None:
        self._configuration = configuration
        self._scrub_text = scrub_text
        self._value_resolver = _declared_value_resolver(configuration)
        self._discovery = _discovery_workflow(configuration, scrub_text, self._value_resolver)
        self.generation: ScenarioGenerationResult | None = None
        self.executions: tuple[ScenarioCoordinationResult, ...] = ()
        self.audits: dict[str, ScenarioAuditRecord] = {}
        self.audit_model_call_count = 0
        self.device_identity: EvidenceDeviceIdentity | None = None

    def session_policy(self) -> RunSessionPolicy:
        return self._discovery.session_policy()

    def prepare(
        self,
        context: RunPreparationContext,
        interruption: InterruptionSignal,
    ) -> object | None:
        return self._discovery.prepare(context, interruption)

    def execute(
        self,
        context: RunExecutionContext,
        interruption: InterruptionSignal,
        preparation: object | None = None,
    ) -> RunWorkflowOutcome:
        discovery_outcome = self._discovery.execute(context, interruption, preparation)
        if discovery_outcome.category is not RunTerminalCategory.COMPLETED:
            return discovery_outcome
        result = load_discovery_result(context.workspace.root)
        compatibility = build_memory_compatibility(
            context.session.application,
            context.workspace.start.request.application,
        )
        memory_store = AppMemoryStore(self._configuration.memory_root)
        memory = memory_store.load(compatibility)
        procedure_store = CompiledProcedureStore(self._configuration.memory_root)
        bundle = procedure_store.load(
            compatibility,
            self._configuration.journey.journey_id,
            self._configuration.journey.objective,
        )
        if memory is None or bundle is None:
            return RunWorkflowOutcome(
                RunTerminalCategory.INCOMPLETE,
                "Discovery completed without compatible reusable testing evidence.",
            )
        self.device_identity = bundle.procedure.setup_contract.device
        procedure_store.stage_screenshots(bundle, context.workspace.root)
        suite = build_discovery_testing_bridge(
            self._configuration.journey,
            compatibility.application,
            bundle.procedure.source_run_id,
            result,
            procedure=bundle.procedure,
            evidence=bundle.evidence,
            memory=memory,
        )
        if not suite.readiness.executable:
            return RunWorkflowOutcome(RunTerminalCategory.INCOMPLETE, suite.readiness.reason)
        model_client = build_model_client(
            context.workspace.root,
            context.workspace.start.run_id,
            self._configuration.model_profiles,
            self._scrub_text,
            bool(self._configuration.redaction_regions),
        )
        try:
            design = TestDesignerService(
                model_client,
                self._configuration.test_designer_profile_id,
            ).design(
                self._configuration.journey,
                suite,
                bundle.procedure,
                bundle.evidence,
            )
        except TestDesignerInvalidOutputError as error:
            return RunWorkflowOutcome(RunTerminalCategory.INCOMPLETE, str(error))
        registry = evaluate_testing_categories(
            self._configuration.journey,
            suite,
            concerns=design.concerns,
            interpretations=design.category_interpretations,
            context=TestingRegistryContext(
                available_capabilities=frozenset({CapabilityKind.APP_STATE})
            ),
        )
        generation = generate_scenarios(
            self._configuration.journey,
            suite,
            design,
            registry,
        )
        AtomicArtifactWriter(context.workspace.root).write(
            SCENARIO_GENERATION_PATH,
            f"{generation.model_dump_json(indent=2)}\n".encode(),
        )
        self.generation = generation
        interaction = DeviceInteraction.from_session(context.session)
        application_preparer = _ProcedureApplicationPreparer(context, interaction)
        capability_preparer = _SessionCapabilityPreparer(
            self._configuration.target_id,
            context.session.device.descriptor.platform,
            interaction,
        )
        scenario_results = []
        for scenario in generation.suite.scenarios:
            scenario_result = self._execute_scenario(
                context,
                interruption,
                memory,
                bundle,
                scenario,
                generation,
                application_preparer,
                capability_preparer,
                procedure_store,
            )
            scenario_results.append(scenario_result)
            if scenario_result[0].execution.restoration_failure_reason is not None:
                break
        scenario_results_tuple = tuple(scenario_results)
        executions = tuple(result for result, unused_audit, unused_calls in scenario_results_tuple)
        self.executions = executions
        self.audits = {
            result.scenario_id: audit for result, audit, unused_calls in scenario_results_tuple
        }
        self.audit_model_call_count = sum(
            calls for unused_result, unused_audit, calls in scenario_results_tuple
        )
        return derive_workflow_outcome(executions, self.audits)

    def _execute_scenario(
        self,
        context: RunExecutionContext,
        interruption: InterruptionSignal,
        memory: AppMemory,
        bundle: StoredProcedureBundle,
        scenario: ScenarioDefinition,
        generation: ScenarioGenerationResult,
        application_preparer: _ProcedureApplicationPreparer,
        capability_preparer: _SessionCapabilityPreparer,
        procedure_store: CompiledProcedureStore,
    ) -> tuple[ScenarioCoordinationResult, ScenarioAuditRecord, int]:
        child_root = context.workspace.root / scenario_artifact_directory(scenario.scenario_id)
        child_root.mkdir(parents=True, exist_ok=False)
        child_context = RunExecutionContext(
            workspace=RunWorkspace(
                root=child_root,
                start=context.workspace.start,
                narrator=context.workspace.narrator,
            ),
            session=context.session,
        )
        procedure_store.stage_screenshots(bundle, child_root)
        scenario_model_client = build_model_client(
            child_root,
            context.workspace.start.run_id,
            self._configuration.model_profiles,
            self._scrub_text,
            bool(self._configuration.redaction_regions),
        )
        budget_bank = ScenarioBudgetBank(self._configuration.scenario_mode_policy)
        coordinator = ScenarioCoordinator(
            _CompiledRouteExecutor(
                child_context,
                bundle,
                self._configuration,
                scenario_model_client,
                interruption,
                memory,
                self._value_resolver,
                budget_bank,
            ),
            _BudgetedScenarioTaskExecutor(
                child_context,
                scenario_model_client,
                self._configuration,
                interruption,
                memory,
                self._value_resolver,
                budget_bank,
            ),
            application_preparer,
            capability_preparer,
            ScenarioExecutionStore(context.workspace.root),
        )
        result = coordinator.execute(generation.suite, scenario)
        if result.execution.classification is ScenarioExecutionClassification.INTERRUPTED:
            raise RunWorkflowInterrupted("Scenario execution was interrupted.")
        auditor_profile = self._configuration.model_profiles.resolve(
            self._configuration.auditor_profile_id,
            ModelRole.AUDITOR,
        )
        auditor = ScenarioAuditorService(
            scenario_model_client,
            self._configuration.auditor_profile_id,
            auditor_profile.images,
        )
        audit = auditor.audit(child_root, generation.suite, scenario, result)
        ScenarioAuditStore(context.workspace.root).persist(scenario.scenario_id, audit)
        return result, audit, auditor.model_call_count


class _ProcedureApplicationPreparer:
    def __init__(self, context: RunExecutionContext, interaction: DeviceInteraction) -> None:
        self._session = context.session
        self._coordinator = ApplicationPreconditionCoordinator(interaction)

    def establish(self, requested_state):
        receipt = self._coordinator.establish(requested_state)
        self._session.drain_application_setup_logs(receipt.receipt_id)
        return receipt


class _ApplicationLifecycleBoundary:
    def __init__(self, interaction: DeviceInteraction) -> None:
        self._interaction = interaction

    def state(self) -> ApplicationState:
        return self._interaction.application_state()

    def activate(self) -> None:
        self._interaction.activate()

    def background_and_resume(self, duration_seconds: int) -> None:
        self._interaction.background_and_resume(duration_seconds)


class _SessionCapabilityPreparer:
    def __init__(self, target_id, platform, interaction: DeviceInteraction) -> None:
        self._target_id = target_id
        self._platform = platform
        self._interaction = interaction

    def prepare(self, request: DeviceCapabilitySpec) -> CapabilityPreparation[CapabilitySpec]:
        if isinstance(request, AppStateCapabilitySpec):
            coordinator = CapabilityTransactionCoordinator(
                self._target_id,
                ApplicationLifecycleCapabilityAdapter(
                    self._platform,
                    _ApplicationLifecycleBoundary(self._interaction),
                ),
            )
            return cast(CapabilityPreparation[CapabilitySpec], coordinator.prepare(request))
        return CapabilityPreparation(
            support=CapabilitySupportResult(
                capability=request.kind,
                state=CapabilitySupportState.UNSUPPORTED,
                reason="This device capability is not implemented by the current test runtime.",
            )
        )


class _CompiledRouteExecutor:
    def __init__(
        self,
        context: RunExecutionContext,
        bundle: StoredProcedureBundle,
        configuration: YewoTestConfiguration,
        model_client,
        interruption: InterruptionSignal,
        memory: AppMemory,
        value_resolver: Callable[[str], str],
        budget_bank: ScenarioBudgetBank,
    ) -> None:
        self._context = context
        self._bundle = bundle
        self._configuration = configuration
        self._model_client = model_client
        self._interruption = interruption
        self._memory = memory
        self._value_resolver = value_resolver
        self._budget_bank = budget_bank

    def execute(self, scenario: ScenarioDefinition):
        if scenario.route_procedure_id != self._bundle.procedure.procedure_id:
            raise ValueError("The scenario route does not match the compiled procedure.")
        procedure = self._bundle.procedure
        auditor_profile = self._configuration.model_profiles.resolve(
            self._configuration.auditor_profile_id,
            ModelRole.AUDITOR,
        )
        result = execute_compiled_procedure(
            self._context,
            procedure,
            self._bundle.evidence,
            ProcedureReplaySetup(
                setup_id=new_record_identifier(),
                method=ProcedureResetMethod(procedure.setup_contract.reset_method),
                application=procedure.application,
                setup_contract_sha256=procedure.setup_contract.content_sha256(),
            ),
            self._budget_bank.require_policy(),
            self._model_client,
            self._configuration.grounding_profile,
            self._interruption,
            planner_profile_id=self._configuration.planner_profile_id,
            recovery_profile_id=self._configuration.recovery_profile_id,
            auditor_profile_id=self._configuration.auditor_profile_id,
            action_verifier_profile_id=self._configuration.action_verifier_profile_id,
            screen_analyzer_profile_id=self._configuration.screen_analyzer_profile_id,
            auditor_image_constraints=auditor_profile.images,
            memory=self._memory,
            redaction_regions=self._configuration.redaction_regions,
            declared_value_resolver=self._value_resolver,
        )
        self._budget_bank.record(
            result.agent_result.usage,
            result.total_elapsed_ms / 1_000,
        )
        return result


class _BudgetedScenarioTaskExecutor:
    def __init__(
        self,
        context: RunExecutionContext,
        model_client,
        configuration: YewoTestConfiguration,
        interruption: InterruptionSignal,
        memory: AppMemory,
        value_resolver: Callable[[str], str],
        budget_bank: ScenarioBudgetBank,
    ) -> None:
        self._context = context
        self._model_client = model_client
        self._configuration = configuration
        self._interruption = interruption
        self._memory = memory
        self._value_resolver = value_resolver
        self._budget_bank = budget_bank

    def execute(self, task: AgentTask) -> AgentResult:
        executor = SharedDeviceAgentTaskExecutor(
            self._context,
            self._model_client,
            self._configuration.grounding_profile,
            self._configuration.planner_profile_id,
            self._budget_bank.require_policy(),
            self._interruption,
            recovery_profile_id=self._configuration.recovery_profile_id,
            action_verifier_profile_id=self._configuration.action_verifier_profile_id,
            screen_analyzer_profile_id=self._configuration.screen_analyzer_profile_id,
            redaction_regions=self._configuration.redaction_regions,
            declared_value_resolver=self._value_resolver,
            planning_memory=self._memory,
            grounding_memory=self._memory,
        )
        return executor.execute(task)


def _discovery_workflow(
    configuration: YewoTestConfiguration,
    scrub_text: Callable[[str], str],
    value_resolver: Callable[[str], str],
) -> DiscoveryWorkflow:
    journey = configuration.journey
    request = DiscoveryRequest(
        request_id=f"discover_{journey.journey_id}"[:64],
        objectives=(
            DiscoveryObjective(
                objective_id=journey.journey_id,
                statement=journey.objective,
            ),
        ),
        declared_value_names=tuple(journey.declared_values),
    )
    return DiscoveryWorkflow(
        request,
        configuration.discovery_mode_policy,
        configuration.model_profiles,
        configuration.grounding_profile,
        configuration.planner_profile_id,
        configuration.auditor_profile_id,
        configuration.recovery_profile_id,
        configuration.action_verifier_profile_id,
        configuration.replay_preparation_profile_id,
        configuration.memory_root,
        scrub_text,
        screen_analyzer_profile_id=configuration.screen_analyzer_profile_id,
        redaction_regions=configuration.redaction_regions,
        declared_value_resolver=value_resolver,
        require_reusable_evidence=True,
    )


def _declared_value_resolver(
    configuration: YewoTestConfiguration,
) -> Callable[[str], str]:
    values = configuration.journey.declared_values

    def resolve(name: str) -> str:
        value = values.get(name)
        if value is None or value.refused:
            raise ValueError("The requested declared value is unavailable for execution.")
        return value.value

    return resolve
