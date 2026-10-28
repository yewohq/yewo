"""Small process wrapper for repeatable, safely logged AndroidWorld runs."""

from __future__ import annotations

import os
import re
import subprocess
import sys
from collections.abc import Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import TextIO

from dotenv import dotenv_values

CHILD_ENV = "YEWO_ANDROID_WORLD_CHILD"
SECRET_NAME = re.compile(r"(?:API_KEY|TOKEN|SECRET|PASSWORD)$", re.IGNORECASE)
ENV_DUMP = re.compile(r"self\._os_env_vars:.*")


def configured_environment(
    benchmark_root: Path, inherited: Mapping[str, str]
) -> dict[str, str]:
    """Load benchmark credentials with exported values taking precedence."""
    environment = dict(inherited)
    candidates = (benchmark_root / ".env", benchmark_root.parents[2] / ".env")
    for path in candidates:
        if not path.is_file():
            continue
        for name, value in dotenv_values(path).items():
            if value is not None:
                environment.setdefault(name, value)
    return environment


def sdk_root(environment: Mapping[str, str]) -> Path | None:
    """Find the Android SDK without requiring a shell export."""
    configured = environment.get("ANDROID_SDK_ROOT") or environment.get("ANDROID_HOME")
    candidates = [
        Path(configured).expanduser() if configured else None,
        Path.home() / "Library" / "Android" / "sdk",
        Path.home() / "Android" / "Sdk",
    ]
    return next((path.resolve() for path in candidates if path and path.is_dir()), None)


def has_flag(arguments: Sequence[str], name: str) -> bool:
    """Accept both ``--flag=value`` and ``--flag value`` spellings."""
    return any(argument == name or argument.startswith(f"{name}=") for argument in arguments)


def redactor(environment: Mapping[str, str]):
    """Build a line scrubber before any child output can reach disk."""
    secrets = {
        value
        for name, value in environment.items()
        if SECRET_NAME.search(name) and len(value) >= 8
    }

    def redact(line: str) -> str:
        cleaned = ENV_DUMP.sub("self._os_env_vars: <redacted>", line)
        for secret in secrets:
            cleaned = cleaned.replace(secret, "<redacted>")
        return cleaned

    return redact


def child_arguments(script: Path, arguments: Sequence[str], checkpoint_root: Path) -> list[str]:
    """Give each non-resumed run a new AndroidWorld checkpoint directory."""
    forwarded = list(arguments)
    if not has_flag(forwarded, "--checkpoint_dir") and not has_flag(
        forwarded, "--output_path"
    ):
        forwarded.append(f"--output_path={checkpoint_root}")
    return [sys.executable, "-u", str(script), *forwarded]


def launch(script: Path, arguments: Sequence[str], benchmark_root: Path) -> int:
    """Run the benchmark child with SDK defaults and a redacted live log."""
    environment = configured_environment(benchmark_root, os.environ)
    environment[CHILD_ENV] = "1"
    environment.setdefault("GRPC_VERBOSITY", "ERROR")
    discovered_sdk = sdk_root(environment)
    if discovered_sdk:
        environment["ANDROID_SDK_ROOT"] = str(discovered_sdk)

    artifact_root = benchmark_root.parents[1] / "runs" / "android_world"
    log_dir = artifact_root / "logs"
    checkpoint_root = artifact_root / "checkpoints"
    log_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    log_path = log_dir / f"android-world-{timestamp}.log"
    command = child_arguments(script, arguments, checkpoint_root)
    redact = redactor(environment)

    with log_path.open("x", encoding="utf-8") as log:
        notice = f"Writing live log to {log_path}\n"
        sys.stdout.write(notice)
        sys.stdout.flush()
        log.write(notice)
        log.flush()
        process = subprocess.Popen(
            command,
            cwd=benchmark_root,
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        if process.stdout is None:
            raise RuntimeError("The AndroidWorld child has no output stream.")
        _copy_output(process.stdout, log, redact)
        return process.wait()


def _copy_output(source: TextIO, log: TextIO, redact) -> None:
    for line in source:
        cleaned = redact(line)
        sys.stdout.write(cleaned)
        sys.stdout.flush()
        log.write(cleaned)
        log.flush()
