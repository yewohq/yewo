#!/usr/bin/env python3
"""Build a controlled evidence-informed lock from sealed app knowledge."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from yewo.discovery.artifacts import parse_artifact
from yewo.discovery.knowledge import AppMap, RouteMemory
from yewo.testing.generation_evidence import (
    ScenarioGenerationEvidence,
    build_scenario_generation_evidence,
)
from yewo.testing.generation_startup import (
    GenerationStartupCoordinator,
    ScenarioStartupKnowledge,
)
from yewo.testing.models import ScenarioDraft
from yewo.testing.lock_file import read, write
from yewo.testing.steps import ActionStep, IntentStep, VariationStep
from yewo.testing.suite import load_workflows
from yewo.testing.taxonomy import ScenarioType


class _KnowledgeSource:
    def __init__(self, evidence: ScenarioGenerationEvidence, events: list[str]):
        self.evidence = evidence
        self.events = events

    def load_compatible(self, journey):
        self.events.append(f"knowledge:{journey.id}")
        return ScenarioStartupKnowledge(self.evidence)

    def discover(self, journey):
        del journey
        raise AssertionError("The controlled audit supplied compatible knowledge.")


class _ControlledGeneratorClient:
    """Deterministic model boundary for an offline architecture audit."""

    def __init__(self, evidence: ScenarioGenerationEvidence, events: list[str]):
        self.evidence = evidence
        self.events = events

    def generate_response(self, *, context, **request):
        del request
        scenario_type = context["scenario_type"]
        goal = context["source_description"]
        prefix = min(
            self.evidence.route_prefixes,
            key=lambda item: (item.segment_count, item.route_memory_artifact_id),
        )
        self.events.append(f"generation:{scenario_type.value}")
        if scenario_type is ScenarioType.HAPPY_PATH:
            steps = [
                ActionStep(instruction="Launch the app."),
                IntentStep(instruction=goal, evidence=goal),
            ]
        else:
            parameters: dict[str, object] = {}
            if scenario_type is ScenarioType.NEGATIVE:
                parameters["probe_value"] = "-100"
            elif scenario_type is ScenarioType.BOUNDARY:
                parameters["repetitions"] = 100
            steps = [
                ActionStep(instruction="Launch the app."),
                VariationStep(
                    goal=goal,
                    objective=goal,
                    evidence=goal,
                    strategy=scenario_type,
                    **parameters,
                ),
            ]
        return ScenarioDraft(
            scenario_name=f"{scenario_type.value} user pursues the declared journey",
            app_capability_ids=list(prefix.capability_ids),
            route_memory_ids=[prefix.route_memory_artifact_id],
            steps=steps,
            expected_outcomes=[
                {
                    "statement": "The completed journey state is visible.",
                    "evidence": goal,
                }
            ],
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--app-map", type=Path, required=True)
    parser.add_argument("--route-memory", type=Path, action="append", required=True)
    parser.add_argument("--journey", type=Path, required=True)
    parser.add_argument("--workflow", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--lock-output", type=Path)
    args = parser.parse_args()

    app_map = parse_artifact(args.app_map.read_bytes(), AppMap)
    route_memories = tuple(
        parse_artifact(path.read_bytes(), RouteMemory) for path in args.route_memory
    )
    evidence = build_scenario_generation_evidence(app_map, route_memories)
    workflow = next(item for item in load_workflows(args.journey) if item.name == args.workflow)
    journey = workflow.journeys[0]
    events: list[str] = []
    coordinator = GenerationStartupCoordinator(
        lambda journey: events.append(f"preflight:{journey.id}"),
        _KnowledgeSource(evidence, events),
    )
    args.output.mkdir(parents=True, exist_ok=True)
    lock_path = args.output / "yewo.lock"
    startup = coordinator.resolve_suite(
        journey,
        _ControlledGeneratorClient(evidence, events),
        lock_path=lock_path,
        workflow=workflow.name,
        refresh=True,
    )
    resolution = startup.suite
    if resolution is None:
        raise RuntimeError(
            f"Scenario generation was not ready: {startup.readiness.state.value}: "
            f"{startup.readiness.reason}"
        )
    summary = {
        "schema": "scenario-generation-audit-v1",
        "app_map_artifact_id": evidence.app_map_artifact_id,
        "generation_evidence_sha256": evidence.evidence_sha256,
        "generated_count": resolution.generated_count,
        "uncovered": [{"cell": gap.cell.key, "reason": gap.reason} for gap in resolution.uncovered],
        "events": events,
        "scenarios": [
            {
                "cell": scenario.cell,
                "app_capability_ids": scenario.app_capability_ids,
                "route_memory_ids": scenario.route_memory_ids,
                "steps": scenario.steps_instructions,
            }
            for scenario in resolution.scenarios
        ],
    }
    summary_path = args.output / "scenario-generation-audit.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    if args.lock_output is not None:
        generated_lock = read(lock_path)
        if generated_lock is None:
            raise RuntimeError("The controlled generation audit produced no lock.")
        write(args.lock_output, generated_lock)
    print(lock_path)
    print(summary_path)
    print(f"generated={resolution.generated_count} uncovered={len(resolution.uncovered)}")
    return 0 if not resolution.uncovered else 1


if __name__ == "__main__":
    raise SystemExit(main())
