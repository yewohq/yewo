"""The individual things `yewo setup` installs or creates.

Steps mutate the machine, so each one asks before acting and reports what it
did as a StepResult. None of them raises. A failed step leaves a message
behind, and the remaining steps still run.
"""

import subprocess
import sys
from dataclasses import dataclass, field
from importlib import invalidate_caches
from pathlib import Path
from typing import Callable, Optional

from yewo.agent.perception.rapidocr_diagnostics import diagnose_rapidocr
from yewo.cli.console import FAIL, INFO, OK, SKIP, WARN
from yewo.defaults import DEFAULT_MODEL, DEFAULT_PERCEPTION_MODEL
from yewo.doctor import Status, check_appium_driver, is_macos
from yewo.utils.subprocess_runner import run_command
from yewo.utils.vision_deps import vision_available, vision_installation_satisfies_contract

ENV_FILENAME = ".env"

ENV_TEMPLATE = f"""\
# Credentials for the models Yewo calls through LiteLLM.
# Yewo's default model is {DEFAULT_MODEL} — get a key at https://openrouter.ai/keys
OPENROUTER_API_KEY=

# Other providers work too; pass --model to use them.
# GEMINI_API_KEY=
# ANTHROPIC_API_KEY=

# Required while the configured perception model repository is private.
# HF_TOKEN=
"""

def _detector_prefetch_script(model_reference: str | None) -> str:
    reference = model_reference or DEFAULT_PERCEPTION_MODEL
    return (
        "from yewo.agent.perception.model_package import resolve_model_package\n"
        f"resolve_model_package({reference!r})\n"
    )


@dataclass
class StepResult:
    """What a step did, in the console's reporting vocabulary."""

    level: str
    label: str
    detail: str = ""
    hint: Optional[str] = None


@dataclass
class SetupContext:
    """Policy and callbacks a step needs, kept free of any rendering concern."""

    assume_yes: bool = False
    dry_run: bool = False
    confirm: Callable[[str], bool] = field(default=lambda _prompt: True)
    log: Callable[[str], None] = field(default=lambda _message: None)

    def approved(self, prompt: str) -> bool:
        return True if self.assume_yes else self.confirm(prompt)


def _run_visible(cmd: list[str], cwd: Optional[Path] = None) -> int:
    """Runs a long install command with its output left on screen."""
    try:
        result = run_command(
            cmd,
            capture_output=False,
            text=False,
            log_command=False,
            **({"cwd": str(cwd)} if cwd else {}),
        )
        return result.returncode
    except TypeError:
        # run_command does not forward cwd, so use subprocess directly here.
        return subprocess.run(cmd, cwd=str(cwd) if cwd else None).returncode
    except (FileNotFoundError, OSError, subprocess.SubprocessError) as exc:
        print(f"  {exc}", file=sys.stderr)
        return 1


def _declined(label: str, command: list[str]) -> StepResult:
    return StepResult(WARN, label, "skipped", f"Run it yourself: {' '.join(command)}")


def _planned(label: str, command: list[str]) -> StepResult:
    return StepResult(INFO, label, f"would run: {' '.join(command)}")


def ensure_env_file(ctx: SetupContext, directory: Optional[Path] = None) -> StepResult:
    """Creates a .env template in the working directory when none exists."""
    target = (directory or Path.cwd()) / ENV_FILENAME
    if target.exists():
        return StepResult(OK, ENV_FILENAME, f"already present at {target}")
    if ctx.dry_run:
        return StepResult(INFO, ENV_FILENAME, f"would create {target}")
    try:
        target.write_text(ENV_TEMPLATE, encoding="utf-8")
    except OSError as exc:
        return StepResult(FAIL, ENV_FILENAME, f"could not write {target}: {exc}")
    return StepResult(
        WARN,
        ENV_FILENAME,
        f"created {target}",
        "Add your OPENROUTER_API_KEY to it (https://openrouter.ai/keys)",
    )


def install_appium(ctx: SetupContext) -> StepResult:
    """Installs Appium globally via npm when it is not already on PATH."""
    import shutil

    if shutil.which("appium"):
        result = run_command(["appium", "--version"], log_command=False)
        return StepResult(OK, "appium", (result.stdout or "").strip() or "installed")

    if not shutil.which("npm"):
        return StepResult(
            FAIL,
            "appium",
            "cannot install: npm not found",
            "Install Node.js 18+ first: brew install node",
        )

    command = ["npm", "install", "-g", "appium"]
    if ctx.dry_run:
        return _planned("appium", command)
    if not ctx.approved("Install Appium globally (npm install -g appium)?"):
        return _declined("appium", command)

    ctx.log("Installing Appium (this takes a minute)...")
    if _run_visible(command) != 0:
        return StepResult(
            FAIL, "appium", "npm install failed", f"Try manually: {' '.join(command)}"
        )
    return StepResult(OK, "appium", "installed")


def install_driver(ctx: SetupContext, driver: str) -> StepResult:
    """Installs one Appium driver, skipping when Appium itself is missing."""
    label = f"driver: {driver}"
    existing = check_appium_driver(driver)
    if existing.status is Status.OK:
        return StepResult(OK, label, "already installed")
    if existing.status is Status.SKIP:
        return StepResult(SKIP, label, "appium not installed")

    command = ["appium", "driver", "install", driver]
    if ctx.dry_run:
        return _planned(label, command)
    if not ctx.approved(f"Install the Appium {driver} driver?"):
        return _declined(label, command)

    ctx.log(f"Installing the {driver} driver...")
    if _run_visible(command) != 0:
        return StepResult(FAIL, label, "install failed", f"Try manually: {' '.join(command)}")
    return StepResult(OK, label, "installed")


def install_applesimutils(ctx: SetupContext) -> StepResult:
    """Installs the program that reads an iOS permission back.

    `simctl privacy` sets a decision and reports nothing, so without this a
    permission scenario is skipped rather than guessed at. Only iOS needs it,
    and it is a warning everywhere else.

    The tap is not Homebrew's own, so the formula is trusted explicitly.
    Trusting one formula is narrower than trusting the tap.
    """
    import shutil

    label = "applesimutils"
    if not is_macos():
        return StepResult(SKIP, label, "iOS automation requires macOS")
    if shutil.which(label):
        result = run_command([label, "--version"], log_command=False)
        return StepResult(OK, label, (result.stdout or "").strip() or "installed")
    if not shutil.which("brew"):
        return StepResult(
            SKIP,
            label,
            "cannot install: brew not found",
            "Install Homebrew first, then: brew tap wix/brew && brew install applesimutils",
        )

    steps = [
        ["brew", "tap", "wix/brew"],
        ["brew", "trust", "--formula", "wix/brew/applesimutils"],
        ["brew", "install", label],
    ]
    if ctx.dry_run:
        return _planned(label, steps[-1])
    if not ctx.approved(
        "Install applesimutils from the wix/brew tap? iOS permission "
        "scenarios are skipped without it"
    ):
        return _declined(label, steps[-1])

    ctx.log("Installing applesimutils...")
    for command in steps:
        if _run_visible(command) != 0:
            return StepResult(FAIL, label, "install failed", f"Try manually: {' '.join(command)}")
    return StepResult(OK, label, "installed")


def install_vision(ctx: SetupContext) -> StepResult:
    """Adds the vision extra to whichever environment Yewo is installed in."""
    from yewo.setup.environment import detect_environment

    if vision_installation_satisfies_contract():
        return StepResult(OK, "vision deps", "already installed")

    environment = detect_environment()
    command = environment.vision_install_command()
    if command is None:
        return StepResult(
            FAIL,
            "vision deps",
            f"cannot install into {environment.description}",
            "Install uv (https://docs.astral.sh/uv/) or add ultralytics, rapidocr, "
            "onnxruntime and huggingface-hub yourself",
        )

    if ctx.dry_run:
        return _planned("vision deps", command)
    if not ctx.approved(
        f"Install the vision extra into the {environment.kind.value} environment? "
        "It pulls PyTorch (hundreds of MB)"
    ):
        return _declined("vision deps", command)

    ctx.log(
        "Installing ultralytics, RapidOCR, ONNX Runtime and "
        f"huggingface-hub into the {environment.description}..."
    )
    if _run_visible(command, cwd=environment.install_cwd) != 0:
        return StepResult(
            FAIL, "vision deps", "install failed", f"Try manually: {' '.join(command)}"
        )
    invalidate_caches()
    return StepResult(OK, "vision deps", "installed")


def prefetch_detector(
    ctx: SetupContext,
    model_reference: str | None = None,
) -> StepResult:
    """Download and validate the configured detector package."""
    if not vision_available():
        return StepResult(SKIP, "detector model", "vision deps not installed")
    model_name = model_reference or DEFAULT_PERCEPTION_MODEL
    if ctx.dry_run:
        return StepResult(INFO, "detector model", f"would prefetch {model_name}")
    if not ctx.approved("Download and validate the detector model now?"):
        return StepResult(
            WARN,
            "detector model",
            "skipped",
            "It will download on the first run instead",
        )

    # Runs out-of-process: the vision packages may have been installed moments
    # ago, and importing them here would mean importing into a mid-change env.
    command = [sys.executable, "-c", _detector_prefetch_script(model_reference)]
    ctx.log(f"Downloading and validating {model_name}...")
    if _run_visible(command) != 0:
        return StepResult(
            FAIL,
            "detector model",
            "download or validation failed",
            "Check the model reference and HF_TOKEN, then run yewo setup again",
        )
    return StepResult(OK, "detector model", "cached and validated")


def verify_rapidocr(ctx: SetupContext) -> StepResult:
    """Initialize RapidOCR and validate one inference in an isolated process."""
    if not vision_available():
        return StepResult(SKIP, "RapidOCR", "vision deps not installed")
    if ctx.dry_run:
        return StepResult(INFO, "RapidOCR", "would run the isolated smoke test")
    if not ctx.approved("Initialize and test RapidOCR now?"):
        return StepResult(
            WARN,
            "RapidOCR",
            "skipped",
            "Run yewo setup again before the first marks-based test",
        )

    ctx.log("Initializing RapidOCR and running one inference...")
    diagnostic = diagnose_rapidocr(smoke_test=True)
    if not diagnostic.is_ok:
        return StepResult(FAIL, "RapidOCR", diagnostic.detail, diagnostic.hint)
    return StepResult(OK, "RapidOCR", f"{diagnostic.identity}; smoke test passed")
