"""Agent package import-boundary tests."""

import subprocess
import sys

# One interpreter for both checks. A fresh process costs over a second, and these two questions
# are answerable in the same one.
_PROGRAM = """
import sys

import yewo.agent

if "yewo.agent.discovery" in sys.modules:
    raise SystemExit("importing yewo.agent pulled in the discovery workflow")

import yewo.agent.replay.reconciliation.compilation  # noqa: F401
"""


def test_the_agent_package_holds_its_import_boundaries() -> None:
    """Two properties, one interpreter.

    Importing the agent must not drag in the discovery workflow, which costs start-up time and
    couples the runtime to a package it does not need. And procedure compilation must import on
    its own, which a cycle between the two would break.
    """
    completed = subprocess.run(  # noqa: S603 - fixed interpreter and source
        [sys.executable, "-c", _PROGRAM],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stdout + completed.stderr
