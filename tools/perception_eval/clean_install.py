"""Smoke-test the built wheel and vision extra in an isolated environment."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Callable, Sequence

from yewo.defaults import DEFAULT_PERCEPTION_MODEL

from tools.perception_eval.corpus import MANIFEST_PATH, read_corpus

CommandRunner = Callable[[Sequence[str], dict[str, str]], subprocess.CompletedProcess[str]]


def run_clean_install(
    wheel: Path,
    output: Path,
    model: str = DEFAULT_PERCEPTION_MODEL,
    manifest: Path = MANIFEST_PATH,
    *,
    runner: CommandRunner | None = None,
) -> dict[str, object]:
    """Install the wheel with vision support, then exercise diagnostics and inference."""
    if wheel.is_dir():
        wheels = tuple(wheel.glob("*.whl"))
        if len(wheels) != 1:
            raise ValueError(f"expected exactly one wheel in {wheel}, found {len(wheels)}")
        wheel = wheels[0]
    if not wheel.is_file():
        raise FileNotFoundError(wheel)
    case = read_corpus(manifest).cases[0]
    execute = runner or _run
    commands: list[dict[str, object]] = []
    with tempfile.TemporaryDirectory(prefix="yewo-clean-install-") as directory:
        root = Path(directory)
        environment = root / "venv"
        _checked(execute, [sys.executable, "-m", "venv", str(environment)], commands)
        python = environment / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
        yewo = environment / ("Scripts/yewo.exe" if os.name == "nt" else "bin/yewo")
        pip = [str(python), "-m", "pip"]
        _checked(execute, [*pip, "install", f"{wheel}[vision]"], commands)
        _checked(execute, [str(yewo), "doctor", "--json"], commands)
        _checked(
            execute,
            [str(python), "-m", "yewo.agent.perception.rapidocr_diagnostics", "smoke"],
            commands,
        )
        inference = (
            "from pathlib import Path; "
            "from yewo.agent.perception.detector import ImageDescriptor; "
            f"d=ImageDescriptor({model!r}); "
            f"e,_=d.describe(Path({str(case.screenshot)!r}).read_bytes()); "
            "assert e, 'production inference returned no elements'; "
            "print(len(e))"
        )
        _checked(execute, [str(python), "-c", inference], commands)
    result = {
        "passed": True,
        "wheel": wheel.name,
        "model": model,
        "fixture": case.id,
        "commands": commands,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def _run(command: Sequence[str], environment: dict[str, str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
        timeout=900,
        env=environment,
    )


def _checked(
    runner: CommandRunner,
    command: Sequence[str],
    results: list[dict[str, object]],
) -> None:
    completed = runner(command, os.environ.copy())
    recorded_command = list(command)
    if "-c" in recorded_command:
        recorded_command[-1] = "<python expression>"
    results.append(
        {
            "command": recorded_command,
            "returncode": completed.returncode,
            "stdout_tail": completed.stdout[-1000:],
            "stderr_tail": completed.stderr[-1000:],
        }
    )
    if completed.returncode:
        raise RuntimeError(f"clean-install command failed ({completed.returncode}): {command[0]}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Smoke-test a Yewo wheel in a clean environment.")
    parser.add_argument("wheel", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--model", default=DEFAULT_PERCEPTION_MODEL)
    parser.add_argument("--manifest", type=Path, default=MANIFEST_PATH)
    arguments = parser.parse_args()
    run_clean_install(
        arguments.wheel,
        arguments.output,
        arguments.model,
        arguments.manifest,
    )
    print(f"Clean-install perception smoke passed: {arguments.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
