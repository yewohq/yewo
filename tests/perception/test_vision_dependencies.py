"""The packaged vision extra and runtime installer use one dependency contract."""

import tomllib
import unittest
from pathlib import Path
from unittest import mock

from yewo.utils import vision_deps


class VisionDependencyContractTests(unittest.TestCase):
    def test_pyproject_vision_extra_matches_the_runtime_installer(self):
        project_root = Path(__file__).resolve().parents[2]
        configuration = tomllib.loads((project_root / "pyproject.toml").read_text(encoding="utf-8"))

        self.assertEqual(
            tuple(configuration["project"]["optional-dependencies"]["vision"]),
            vision_deps.VISION_INSTALL_REQUIREMENTS,
        )

    @mock.patch("yewo.utils.vision_deps.missing_vision_packages", return_value=[])
    @mock.patch(
        "yewo.utils.vision_deps.mismatched_pinned_vision_packages",
        return_value=["rapidocr 4.0.0 (expected 3.10.0)"],
    )
    def test_setup_does_not_skip_a_mismatched_pinned_version(self, _mismatch, _missing):
        self.assertFalse(vision_deps.vision_installation_satisfies_contract())


if __name__ == "__main__":
    unittest.main()
