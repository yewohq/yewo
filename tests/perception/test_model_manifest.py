import json
import tempfile
import unittest
from pathlib import Path

from pydantic import ValidationError

from yewo.agent.perception.model_manifest import (
    DetectorClass,
    PerceptionModelManifest,
    load_perception_model_manifest,
)

CHECKSUM = "5d3f73338a9525bc5792e985418cd82a95cf6d505653d27fd00f7c5a94f4b7cc"


def _detector_class(class_id: int, label: str, *roles: str) -> DetectorClass:
    return DetectorClass(class_id=class_id, label=label, semantic_roles=roles)


def _manifest(*classes: DetectorClass) -> PerceptionModelManifest:
    return PerceptionModelManifest(
        schema_version=1,
        model_id="yewo-ui",
        model_version="1",
        weights_file="weights.pt",
        inference_image_size=1280,
        sha256=CHECKSUM,
        classes=classes,
    )


class PerceptionModelManifestTests(unittest.TestCase):
    def test_records_artifact_metadata_class_table_and_optional_semantics(self):
        manifest = _manifest(
            _detector_class(0, "View"),
            _detector_class(1, "Keyboard"),
        )

        self.assertEqual(manifest.model_id, "yewo-ui")
        self.assertEqual(manifest.model_version, "1")
        self.assertEqual(manifest.weights_file, "weights.pt")
        self.assertEqual(manifest.inference_image_size, 1280)
        self.assertEqual(
            manifest.model_dump(mode="json")["classes"],
            [
                {"class_id": 0, "label": "View", "semantic_roles": []},
                {"class_id": 1, "label": "Keyboard", "semantic_roles": []},
            ],
        )

    def test_declared_semantic_roles_are_resolved_by_class_id(self):
        manifest = _manifest(
            _detector_class(4, "AnyRegionLabel", "keyboard_region"),
            _detector_class(6, "AnyKeyLabel", "keyboard_key"),
        )

        self.assertEqual(
            manifest.semantic_roles_by_class_id(),
            {4: frozenset({"keyboard_region"}), 6: frozenset({"keyboard_key"})},
        )

    def test_undeclared_roles_are_not_inferred_from_class_labels(self):
        manifest = _manifest(
            _detector_class(4, "Keyboard"),
            _detector_class(6, "Keys"),
        )

        self.assertEqual(
            manifest.semantic_roles_by_class_id(),
            {4: frozenset(), 6: frozenset()},
        )

    def test_unknown_future_roles_are_preserved_without_runtime_changes(self):
        manifest = _manifest(
            _detector_class(12, "FutureSurface", "future_model_role")
        )

        self.assertEqual(
            manifest.semantic_roles_by_class_id(),
            {12: frozenset({"future_model_role"})},
        )

    def test_empty_semantic_role_is_rejected(self):
        with self.assertRaisesRegex(ValidationError, "at least 1 character"):
            _detector_class(12, "FutureSurface", "")

    def test_twenty_class_model_requires_no_new_runtime_vocabulary(self):
        classes = tuple(
            _detector_class(class_id, f"ModelLabel{class_id}") for class_id in range(20)
        )
        manifest = _manifest(*classes)

        manifest.validate_model_classes(
            {class_id: f"ModelLabel{class_id}" for class_id in range(20)}
        )
        self.assertEqual(len(manifest.classes), 20)

    def test_duplicate_class_ids_are_rejected(self):
        with self.assertRaisesRegex(ValidationError, "class IDs must be unique"):
            _manifest(
                _detector_class(0, "First"),
                _detector_class(0, "Second"),
            )

    def test_duplicate_labels_are_rejected(self):
        with self.assertRaisesRegex(ValidationError, "labels must be unique"):
            _manifest(
                _detector_class(0, "Repeated"),
                _detector_class(1, "Repeated"),
            )

    def test_model_class_mismatch_is_rejected(self):
        manifest = _manifest(_detector_class(0, "Expected"))

        with self.assertRaisesRegex(ValueError, "do not match"):
            manifest.validate_model_classes({0: "Different"})

    def test_invalid_checksum_is_rejected(self):
        payload = self._payload()
        payload["sha256"] = "not-a-checksum"

        with self.assertRaisesRegex(ValidationError, "sha256"):
            PerceptionModelManifest.model_validate(payload)

    def test_weights_file_cannot_escape_model_package(self):
        payload = self._payload()
        payload["weights_file"] = "../weights.pt"

        with self.assertRaisesRegex(ValidationError, "plain filename"):
            PerceptionModelManifest.model_validate(payload)

    def test_sidecar_rejects_processing_policy(self):
        payload = self._payload()
        payload["classes"][0]["ocr"] = True

        with self.assertRaisesRegex(ValidationError, "Extra inputs"):
            PerceptionModelManifest.model_validate(payload)

    def test_loads_sidecar_from_model_package(self):
        payload = self._payload()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "perception.json"
            path.write_text(json.dumps(payload), encoding="utf-8")

            manifest = load_perception_model_manifest(path)

        self.assertEqual(manifest.model_version, "1")
        self.assertEqual(manifest.classes[0].label, "View")

    @staticmethod
    def _payload() -> dict:
        return {
            "schema_version": 1,
            "model_id": "yewo-ui",
            "model_version": "1",
            "weights_file": "weights.pt",
            "inference_image_size": 1280,
            "sha256": CHECKSUM,
            "classes": [{"class_id": 0, "label": "View"}],
        }


if __name__ == "__main__":
    unittest.main()
