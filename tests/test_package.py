"""Verify the initial package contract."""

from yewo.cli.main import main


def test_cli_accepts_no_arguments() -> None:
    """The package entry point can start before commands exist."""
    assert main([]) == 0
