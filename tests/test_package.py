"""Verify the initial package contract."""

import pytest

from yewo.cli.main import main


def test_cli_requires_a_command() -> None:
    """The package entry point does not report an omitted command as success."""
    with pytest.raises(SystemExit) as captured:
        main([])

    assert captured.value.code == 2
