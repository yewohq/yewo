"""Tests for YAML-backed journey testing selection."""

from argparse import Namespace
from pathlib import Path

from yewo.cli.test import configurations


def test_loads_selected_workflow_and_target_without_device_access(tmp_path: Path) -> None:
    (tmp_path / "yewo.yaml").write_text(
        """
model: openrouter/example/semantic
grounder:
  model: openrouter/example/grounder
  protocol: source_pixels
workflows:
  suite:
    targets:
      - id: pixel
        platform: android
        device: Pixel_8
        app: app.apk
    journey: Browse the catalogue.
"""
    )

    compiled = configurations(
        Namespace(workflow="suite", target="pixel"),
        tmp_path,
    )

    assert len(compiled) == 1
    assert compiled[0].workflow_name == "suite"
    assert compiled[0].target_id == "pixel"
    assert compiled[0].journey.objective == "Browse the catalogue."
