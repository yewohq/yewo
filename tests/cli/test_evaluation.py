"""The public command for preparing evaluation apps."""

import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from yewo.cli.main import main

MANIFEST = """
schema_version: 1
artifacts:
  - id: notes-android
    platform: android
    version: '1.0'
    architectures: [arm64-v8a]
    path: android/notes.apk
    size_bytes: 3
    sha256: fd3473e4479d268c96d1fa0d1e51c7bb1c62be6da0a7e2a4cebc5c76f4d2f5f7
    license: MIT
    source:
      kind: download
      project_url: https://example.test/project
      download_url: https://example.test/notes.apk
"""


class EvaluationCommandTests(unittest.TestCase):
    def test_verify_only_reports_a_missing_artifact(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.yaml"
            path.write_text(MANIFEST)
            output = io.StringIO()
            with patch("sys.stdout", output):
                code = main(["eval", "setup", "--manifest", str(path), "--verify-only"])

        self.assertEqual(code, 1)
        self.assertIn("notes-android: unavailable", output.getvalue())

    def test_unknown_apps_are_usage_errors(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.yaml"
            path.write_text(MANIFEST)
            errors = io.StringIO()
            with patch("sys.stderr", errors):
                code = main(
                    ["eval", "setup", "--manifest", str(path), "--app", "absent"]
                )

        self.assertEqual(code, 2)
        self.assertIn("Unknown evaluation app ID", errors.getvalue())
