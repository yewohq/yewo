"""Checksum-bound materialization of evaluation app packages."""

import hashlib
import io
import tempfile
import unittest
from pathlib import Path

from yewo.errors import UsageError
from yewo.evaluation.apps import (
    AppArtifact,
    AppManifest,
    default_app_root,
    directory_tree_sha256,
    load_manifest,
    materialize_apps,
)


def artifact(content: bytes = b"apk", *, source_kind: str = "download") -> AppArtifact:
    return AppArtifact.model_validate(
        {
            "id": "notes-android",
            "platform": "android",
            "version": "1.0",
            "architectures": ["arm64-v8a"],
            "path": "android/notes.apk",
            "size_bytes": len(content),
            "sha256": hashlib.sha256(content).hexdigest(),
            "license": "MIT",
            "source": {
                "kind": source_kind,
                "project_url": "https://example.test/project",
                "download_url": (
                    "https://example.test/notes.apk" if source_kind == "download" else None
                ),
                "instructions": "build notes" if source_kind == "build" else None,
            },
        }
    )


def manifest(*artifacts: AppArtifact) -> AppManifest:
    return AppManifest(schema_version=1, artifacts=list(artifacts))


class _Response(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *_args):
        self.close()


class ManifestTests(unittest.TestCase):
    def test_repository_manifests_materialize_beside_the_manifest(self):
        path = Path("project/eval/apps/manifest.yaml")

        self.assertEqual(default_app_root(path), path.parent)

    def test_packaged_manifests_materialize_in_the_current_project(self):
        path = Path("site-packages/yewo/evaluation/manifest.yaml")

        self.assertEqual(default_app_root(path), Path.cwd() / "eval" / "apps")

    def test_a_destination_cannot_escape_the_app_root(self):
        document = artifact().model_dump(mode="json")
        document["path"] = "../outside.apk"

        with self.assertRaisesRegex(ValueError, "stay beneath"):
            AppArtifact.model_validate(document)

    def test_duplicate_destinations_are_refused(self):
        first = artifact()
        second = first.model_copy(update={"id": "other-android"})

        with self.assertRaisesRegex(ValueError, "paths must be unique"):
            manifest(first, second)

    def test_a_manifest_file_is_validated(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.yaml"
            path.write_text("schema_version: 1\nartifacts: []\n")

            with self.assertRaisesRegex(UsageError, "Invalid evaluation app manifest"):
                load_manifest(path)


class MaterializationTests(unittest.TestCase):
    def test_a_matching_local_file_is_ready_without_network_access(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            destination = root / "android" / "notes.apk"
            destination.parent.mkdir()
            destination.write_bytes(b"apk")

            report = materialize_apps(
                manifest(artifact()),
                root,
                opener=lambda *_args, **_kwargs: self.fail("network was used"),
            )

        self.assertEqual(report.ready_count, 1)
        self.assertEqual(report.downloaded_count, 0)

    def test_a_missing_file_is_downloaded_and_verified_before_placement(self):
        calls = []

        def open_download(request, timeout):
            calls.append((request.full_url, timeout))
            return _Response(b"apk")

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            report = materialize_apps(manifest(artifact()), root, opener=open_download)
            written = (root / "android" / "notes.apk").read_bytes()

        self.assertEqual(written, b"apk")
        self.assertEqual(report.downloaded_count, 1)
        self.assertEqual(calls, [("https://example.test/notes.apk", 60)])

    def test_a_bad_download_never_replaces_the_destination(self):
        def open_download(_request, timeout):
            self.assertEqual(timeout, 60)
            return _Response(b"wrong")

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            report = materialize_apps(manifest(artifact()), root, opener=open_download)
            destination = root / "android" / "notes.apk"
            parts = list(destination.parent.glob("*.part"))

        self.assertEqual(report.unavailable_count, 1)
        self.assertFalse(destination.exists())
        self.assertEqual(parts, [])

    def test_a_build_only_artifact_reports_its_instructions(self):
        with tempfile.TemporaryDirectory() as directory:
            report = materialize_apps(
                manifest(artifact(source_kind="build")), Path(directory)
            )

        self.assertEqual(report.unavailable_count, 1)
        self.assertEqual(report.items[0].detail, "build notes")

    def test_verify_only_never_downloads(self):
        with tempfile.TemporaryDirectory() as directory:
            report = materialize_apps(
                manifest(artifact()),
                Path(directory),
                verify_only=True,
                opener=lambda *_args, **_kwargs: self.fail("network was used"),
            )

        self.assertEqual(report.unavailable_count, 1)

    def test_directory_hash_includes_names_and_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "a").write_bytes(b"one")
            first = directory_tree_sha256(root)
            (root / "a").write_bytes(b"two")
            second = directory_tree_sha256(root)

        self.assertNotEqual(first, second)

    def test_unknown_artifact_ids_are_usage_errors(self):
        with (
            tempfile.TemporaryDirectory() as directory,
            self.assertRaisesRegex(UsageError, "Unknown evaluation app ID"),
        ):
            materialize_apps(
                manifest(artifact()), Path(directory), app_ids=frozenset({"absent"})
            )
