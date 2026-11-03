"""Agent package import-boundary tests."""

import subprocess
import sys


def test_importing_agent_does_not_import_discovery_workflow() -> None:
    command = (
        "import sys; import yewo.agent; "
        "raise SystemExit(int('yewo.agent.discovery' in sys.modules))"
    )

    completed = subprocess.run(  # noqa: S603 - fixed interpreter and source
        [sys.executable, "-c", command],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr


def test_replay_procedure_compilation_imports_before_discovery() -> None:
    command = "import yewo.agent.replay.procedure_compilation"

    completed = subprocess.run(  # noqa: S603 - fixed interpreter and source
        [sys.executable, "-c", command],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
