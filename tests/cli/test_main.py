"""Tests for the product command-line entry point."""

import pytest

from yewo.cli.main import main


def test_cli_rejects_execution_without_a_product_workflow(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as captured:
        main([])

    assert captured.value.code == 2
    output = capsys.readouterr()
    assert "No product workflows are available." in output.err
    assert "{run}" not in output.err
