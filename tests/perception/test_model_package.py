import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from yewo.agent.perception.model_package import (
    ModelPackageError,
    parse_model_reference,
    resolve_model_package,
)

REVISION = "a" * 40
REFERENCE = f"hf://example/yewo-ui@{REVISION}"


class ModelReferenceTests(unittest.TestCase):
    def test_accepts_only_pinned_hugging_face_reference(self):
        reference = parse_model_reference(REFERENCE)

        self.assertEqual(reference.repo_id, "example/yewo-ui")
        self.assertEqual(reference.revision, REVISION)
        self.assertEqual(reference.value, REFERENCE)

    def test_rejects_mutable_revision(self):
        with self.assertRaisesRegex(ModelPackageError, "40-character commit"):
            parse_model_reference("hf://example/yewo-ui@main")

    def test_rejects_local_path(self):
        with self.assertRaisesRegex(ModelPackageError, "hf://"):
            parse_model_reference("../models/yewo_ui_v1")


class ModelPackageResolutionTests(unittest.TestCase):
    def test_downloads_manifest_and_weights_from_exactly_one_revision(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            weights_path = root / "weights.pt"
            weights_path.write_bytes(b"model weights")
            manifest_path = root / "perception.json"
            manifest_path.write_text(
                json.dumps(self._manifest(hashlib.sha256(b"model weights").hexdigest())),
                encoding="utf-8",
            )
            calls = []

            def download(**request):
                calls.append(request)
                return str(root / request["filename"])

            package = resolve_model_package(REFERENCE, download)

        self.assertEqual(package.weights_path.name, "weights.pt")
        self.assertEqual(package.manifest.model_version, "1")
        self.assertEqual(
            calls,
            [
                {
                    "repo_id": "example/yewo-ui",
                    "filename": "perception.json",
                    "revision": REVISION,
                },
                {
                    "repo_id": "example/yewo-ui",
                    "filename": "weights.pt",
                    "revision": REVISION,
                },
            ],
        )

    def test_rejects_weights_that_do_not_match_manifest_checksum(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "weights.pt").write_bytes(b"changed")
            (root / "perception.json").write_text(
                json.dumps(self._manifest("0" * 64)), encoding="utf-8"
            )

            with self.assertRaisesRegex(ModelPackageError, "checksum"):
                resolve_model_package(
                    REFERENCE,
                    lambda **request: str(root / request["filename"]),
                )

    def test_setup_prefetches_the_configured_package(self):
        from yewo.setup.steps import _detector_prefetch_script

        script = _detector_prefetch_script(REFERENCE)

        self.assertIn("resolve_model_package", script)
        self.assertIn(repr(REFERENCE), script)
        self.assertNotIn("RapidOCRReader", script)

    @staticmethod
    def _manifest(checksum: str) -> dict:
        return {
            "schema_version": 1,
            "model_id": "yewo-ui",
            "model_version": "1",
            "weights_file": "weights.pt",
            "inference_image_size": 1280,
            "sha256": checksum,
            "classes": [{"class_id": 0, "label": "View"}],
        }


if __name__ == "__main__":
    unittest.main()
