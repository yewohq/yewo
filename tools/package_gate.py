#!/usr/bin/env python3
"""Build Yewo and verify its wheel in a clean environment."""

from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def _run(command: list[str], *, environment: dict[str, str] | None = None) -> None:
    result = subprocess.run(
        command,
        cwd=REPOSITORY_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        timeout=600,
        check=False,
    )
    if result.returncode:
        output = result.stdout + result.stderr
        raise RuntimeError(f"Command failed: {' '.join(command)}\n{output.rstrip()}")


def main() -> int:
    """Build, install, import, and start the packaged CLI."""
    uv = shutil.which("uv")
    if uv is None:
        print("Package gate failed: uv is not installed.", file=sys.stderr)
        return 2

    try:
        with tempfile.TemporaryDirectory(prefix="yewo-package-") as directory:
            root = Path(directory)
            distribution = root / "dist"
            environment = root / "venv"
            _run([uv, "build", "--out-dir", str(distribution)])

            wheels = list(distribution.glob("*.whl"))
            if len(wheels) != 1:
                raise RuntimeError(f"Expected one wheel, found {len(wheels)}.")

            _run([uv, "venv", "--python", sys.executable, str(environment)])
            python = environment / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
            yewo = environment / ("Scripts/yewo.exe" if os.name == "nt" else "bin/yewo")
            _run([uv, "pip", "install", "--python", str(python), str(wheels[0])])
            _run([str(python), "-c", "import yewo"])
            _run([str(yewo), "--help"])
    except (OSError, RuntimeError, subprocess.SubprocessError) as error:
        print(f"Package gate failed: {error}", file=sys.stderr)
        return 1

    print("Package gate passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
