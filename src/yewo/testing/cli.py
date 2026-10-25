import argparse
import logging
from pathlib import Path
from typing import Any, List, Optional

from yewo.app_discovery import resolve_journey_file
from yewo.config import Config, load_config
from yewo.defaults import LOG_LEVELS
from yewo.errors import UsageError
from yewo.identity import new_uuid
from yewo.schema import GROUNDINGS, PLATFORMS
from . import lock_file, lock_resolution
from .suite import load_workflows, select
from .target_matrix import (
    TargetRunResult,
    isolated_run_dir,
    restoration_failed,
    select_workflow_targets,
    uses_target_matrix,
    validate_matrix_arguments,
    write_execution_summary,
)
from .wiring import output_dir_for, runner_for

logger = logging.getLogger(__name__)


class YewoCLI:
    """Encapsulates the CLI parsing, test suite execution, and summary reporting for Yewo."""

    def __init__(self, config: Optional[Config] = None):
        self.config = config or load_config()
        self.parser = self._create_parser()

    def _create_parser(self) -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser(description="Yewo QA Journey-Based Testing CLI")
        parser.add_argument(
            "journey_file",
            nargs="?",
            help="Path to the journey YAML/JSON file. Found in this directory when omitted.",
        )
        parser.add_argument("--platform", choices=list(PLATFORMS), help="Target platform")
        parser.add_argument("--device", help="Name of emulator/device")
        parser.add_argument("--app-path", help="App binary path")
        parser.add_argument("--output-dir", help="Directory to save test outputs and reports")
        parser.add_argument("--max-steps", type=int, help="Max steps per test scenario execution")
        parser.add_argument("--settle-time", type=float, help="Settle time after actions")
        parser.add_argument("--model", help="LiteLLM model to use")
        parser.add_argument("--fast-model", help="Cheaper/faster LiteLLM model for ReAct step planning")
        parser.add_argument("--headless", action="store_true", default=None, help="Run android emulator headlessly")
        parser.add_argument(
            "--grounding",
            choices=list(GROUNDINGS),
            help="How the planner names a target. 'marks' is the default. It "
            "numbers every detected element on the screenshot, and the planner "
            "picks a number. Yewo checks that number against the detected set "
            "before anything runs. 'describe' names the target in words and "
            "uses visual grounding to place it.",
        )
        parser.add_argument(
            "--reuse-app",
            action="store_true",
            help="Skip reinstalling the app, using whatever build is on the device. "
            "Only correct when that build is the one under test: a rebuilt debug APK "
            "keeps its version, so a stale one is used without warning",
        )
        parser.add_argument(
            "--log-level",
            choices=LOG_LEVELS,
            help="Logging verbosity, when given after the subcommand",
        )
        parser.add_argument(
            "--workflow",
            help="Which workflow(s) in the file to run, comma separated. "
            "Every workflow runs when this is not given",
        )
        parser.add_argument(
            "--target",
            help="Which declared target(s) to run, comma separated. "
            "Every target runs when this is not given",
        )
        parser.add_argument(
            "--no-record",
            action="store_true",
            help="Do not record the screen. The run reports results either way, "
            "but no video is kept",
        )
        parser.add_argument(
            "--keep-app",
            action="store_true",
            help="Leave the app installed when the run ends. By default a run uninstalls "
            "the app it installed, so a rotation of apps does not fill the device. An app "
            "that was already on the device is never removed",
        )
        parser.add_argument(
            "--no-persist-inputs",
            action="store_true",
            help="Record the hash of every prompt and not its text. The run still "
            "says that two runs asked the same thing, and keeps no model input",
        )
        parser.add_argument(
            "--allow-secret-exposure",
            action=argparse.BooleanOptionalAction,
            default=None,
            help="Allow declared secrets in model input and screen recordings. "
            "Off by default. Reports, logs, traces and saved screenshots remain "
            "scrubbed or redacted",
        )
        parser.add_argument(
            "--locked",
            action="store_true",
            help="Run exactly what yewo.lock holds, and fail when it is missing or "
            "stale rather than generating a different suite. What CI uses",
        )
        parser.add_argument(
            "--replay",
            action="store_true",
            help="Re-execute navigation already worked out in an earlier scenario instead of "
            "re-planning it. Faster, but replayed screens are not perceived or audited",
        )
        parser.add_argument(
            "--no-durable-replay",
            dest="durable_replay",
            action="store_false",
            help="Plan every scenario from the start, rather than re-walking a path a "
            "passing run saved. This disables local path and verdict reuse",
        )
        parser.add_argument(
            "--update-paths",
            action="store_true",
            help="Accept changed local replay paths after a passing run. CI is always "
            "read-only, even when this flag is present",
        )
        parser.set_defaults(durable_replay=True)
        return parser

    def run(self, args_list: Optional[List[str]] = None) -> int:
        """Parses arguments and executes the suite they ask for."""
        return self.run_parsed(self.parser.parse_args(args_list))

    def run_parsed(self, args: Any) -> int:
        """Executes all loaded journeys, prints a summary, and returns a code.

        Takes a namespace rather than argv, so a caller that already parsed to
        run its own checks does not parse a second time. Two parses of one
        invocation are two answers to the same question.

        The caller owns the process. The code is:

        - 0 when every journey passed
        - 1 when any journey failed
        - 2 when Yewo could not load the suite
        - 130 when the user interrupted execution
        """
        try:
            filepath = resolve_journey_file(args.journey_file)
            workflows = select(load_workflows(filepath), args.workflow)
        except UsageError as e:
            logger.error("%s", e)
            return 2
        except Exception as e:
            logger.error("Failed to load journey file: %s", e)
            return 2

        matrix_requested = uses_target_matrix(workflows)
        if args.target and not matrix_requested:
            logger.error("--target requires a suite with a targets list.")
            return 2
        if matrix_requested:
            try:
                return self._run_target_matrix(workflows, args, filepath)
            except UsageError as error:
                logger.error("%s", error)
                return 2

        total = sum(len(workflow.journeys) for workflow in workflows)
        logger.info("==================================================")
        logger.info(
            "Yewo CLI: Executing %d journey(s) across %d workflow(s)",
            total,
            len(workflows),
        )
        logger.info("==================================================")

        all_passed = True
        interrupted = False
        results = []

        for workflow in workflows:
            if workflow.name:
                logger.info("Workflow: %s", workflow.name)
            try:
                runner = runner_for(workflow, args, filepath, self.config)
            except Exception as setup_err:
                logger.error(
                    "Could not prepare workflow '%s': %s", workflow.name, setup_err
                )
                for journey in workflow.journeys:
                    results.append((workflow.name, journey.id, journey.name, "ERROR"))
                all_passed = False
                continue

            for journey in workflow.journeys:
                logger.info("Starting run for journey: %s (%s)", journey.name, journey.id)
                try:
                    report = runner.run_journey(journey)
                    passed = (report.status == "PASSED")
                    status = str(report.status)
                    results.append(
                        (workflow.name, journey.id, journey.name, status)
                    )
                    if not passed:
                        all_passed = False
                except KeyboardInterrupt:
                    results.append((workflow.name, journey.id, journey.name, "ABORTED"))
                    all_passed = False
                    interrupted = True
                    break
                except Exception as journey_err:
                    logger.error("Error executing journey %s: %s", journey.id, journey_err)
                    results.append((workflow.name, journey.id, journey.name, "ERROR"))
                    all_passed = False
            if interrupted:
                break

        logger.info("==================================================")
        logger.info("Execution Summary:")
        for flow_name, j_id, j_name, status in results:
            status_icon = (
                "🟢" if status == "PASSED" else "🟠" if status == "ABORTED" else "🔴"
            )
            label = f"{flow_name}/{j_name}" if flow_name else j_name
            logger.info("  %s %s (%s): %s", status_icon, label, j_id, status)
        logger.info("==================================================")

        if interrupted:
            logger.info("Test execution was interrupted by the user.")
            return 130
        if all_passed:
            logger.info("All user journeys passed successfully.")
            return 0
        logger.error("One or more user journeys failed or encountered errors.")
        return 1

    def _run_target_matrix(self, workflows, args: Any, filepath: Path) -> int:
        """Run every selected workflow and target pair without sharing evidence."""
        validate_matrix_arguments(args)
        selections = select_workflow_targets(workflows, args.target)
        self._require_matrix_locks(selections, filepath)
        output_dirs = {
            output_dir_for(selection.workflow, args, self.config).resolve()
            for selection in selections
        }
        if len(output_dirs) != 1:
            raise UsageError(
                "A target matrix needs one output_dir so its execution index can own "
                "every run. Selected workflows resolve different output directories."
            )

        execution_id = new_uuid()
        execution_root = output_dirs.pop() / "executions" / execution_id
        expected_runs = sum(
            len(selection.workflow.journeys) * len(selection.targets)
            for selection in selections
        )
        results: list[TargetRunResult] = []
        quarantined_targets: set[tuple[str, str]] = set()
        summary_failed = False

        logger.info("==================================================")
        logger.info(
            "Yewo CLI: Executing %d locked target run(s) sequentially", expected_runs
        )
        logger.info("Execution ID: %s", execution_id)
        logger.info("==================================================")

        for selection in selections:
            workflow = selection.workflow
            for target in selection.targets:
                for journey in workflow.journeys:
                    run_id = new_uuid()
                    run_dir = isolated_run_dir(
                        execution_root,
                        workflow.name,
                        target.id,
                        journey.id,
                        run_id,
                    )
                    logger.info(
                        "Starting run %s: %s/%s/%s",
                        run_id,
                        workflow.name or "default",
                        target.id,
                        journey.id,
                    )
                    target_key = (
                        str(target.settings["platform"]),
                        str(target.settings["device"]),
                    )
                    if target_key in quarantined_targets:
                        result = TargetRunResult(
                            workflow=workflow.name,
                            target_id=target.id,
                            journey_id=journey.id,
                            journey_name=journey.name,
                            run_id=run_id,
                            status="ERROR",
                            platform=target_key[0],
                            device="Not recorded",
                            error="Yewo did not reuse a target that failed restoration.",
                        )
                        interrupted = False
                        reusable = False
                    else:
                        result, interrupted, reusable = self._run_target_journey(
                            workflow,
                            target,
                            journey,
                            args,
                            filepath,
                            execution_id,
                            run_id,
                            run_dir,
                        )
                    if not reusable:
                        quarantined_targets.add(target_key)
                    results.append(result)
                    try:
                        paths = write_execution_summary(
                            execution_root, execution_id, expected_runs, results
                        )
                        logger.info("Execution index: %s", paths[0])
                    except Exception as report_error:  # noqa: BLE001 - result remains usable
                        summary_failed = True
                        logger.error("Could not write execution summary: %s", report_error)
                    if interrupted:
                        logger.info("Target execution was interrupted by the user.")
                        return 130

        logger.info("==================================================")
        logger.info("Target Execution Summary:")
        for result in results:
            logger.info(
                "  %s %s/%s/%s: %s",
                "🟢" if result.status == "PASSED" else "🔴",
                result.workflow or "default",
                result.target_id,
                result.journey_name,
                result.status,
            )
        logger.info("==================================================")
        if summary_failed or any(result.status != "PASSED" for result in results):
            logger.error("One or more target runs failed or could not be reported.")
            return 1
        logger.info("All target runs passed successfully.")
        return 0

    def _run_target_journey(
        self,
        workflow,
        target,
        journey,
        args: Any,
        filepath: Path,
        execution_id: str,
        run_id: str,
        run_dir: Path,
    ) -> tuple[TargetRunResult, bool, bool]:
        """Execute one isolated matrix item and convert failures into its row."""
        runner = None
        try:
            run_dir.mkdir(parents=True, exist_ok=False)
            runner = runner_for(
                workflow,
                args,
                filepath,
                self.config,
                target=target,
                assets_dir=run_dir,
                run_id=run_id,
                execution_id=execution_id,
            )
            report = runner.run_journey(journey)
            return (
                TargetRunResult.from_report(
                    workflow.name,
                    target,
                    run_id,
                    report,
                    runner.last_report_paths.get("html_report"),
                ),
                False,
                not restoration_failed(report),
            )
        except KeyboardInterrupt:
            status = "ABORTED"
            interrupted = True
        except Exception as journey_error:  # noqa: BLE001 - later targets must still run
            logger.error(
                "Target run %s/%s/%s failed: %s",
                workflow.name or "default",
                target.id,
                journey.id,
                journey_error,
            )
            status = "ERROR"
            interrupted = False

        report_path = (
            runner.last_report_paths.get("html_report") if runner is not None else None
        )
        return (
            TargetRunResult(
                workflow=workflow.name,
                target_id=target.id,
                journey_id=journey.id,
                journey_name=journey.name,
                run_id=run_id,
                status=status,
                platform=str(target.settings["platform"]),
                device="Not recorded",
                report_path=report_path,
                error=f"Target run ended {status}.",
            ),
            interrupted,
            runner is None,
        )

    @staticmethod
    def _require_matrix_locks(selections, filepath: Path) -> None:
        """Require every journey lock before artifact or device work."""
        path = lock_file.path_beside(filepath)
        for selection in selections:
            for journey in selection.workflow.journeys:
                lock_resolution.require_lock(journey, path, selection.workflow.name)
