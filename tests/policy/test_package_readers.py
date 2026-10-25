"""Untrusted APK and iOS package inputs fail closed at the file boundary."""

import plistlib
import subprocess
import tempfile
import unittest
import zipfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from yewo.agent.device.platform import Platform
from yewo.package_policy import (
    _resource_path,
    _safe_extract_bundle,
    _signed_entitlements,
    evaluate_package_policy,
)
from yewo.policy import PolicyCheckStatus
from yewo.utils.secrets import SecretScrubber
from yewo.utils.subprocess_runner import SubprocessRunError


class AndroidPackageReaderTests(unittest.TestCase):
    def _evaluate(self, path: str = "app.apk"):
        return evaluate_package_policy(
            Path(path), Platform.ANDROID, {}, SecretScrubber()
        )

    def test_a_non_apk_target_is_unsupported(self):
        checks = self._evaluate("app.aab")

        self.assertEqual({check.status for check in checks}, {PolicyCheckStatus.UNSUPPORTED})

    def test_an_unreadable_manifest_is_evidence_missing(self):
        with patch("yewo.package_policy.AdbControl.dump_manifest", return_value=None):
            checks = self._evaluate()

        self.assertEqual(
            {check.status for check in checks}, {PolicyCheckStatus.EVIDENCE_MISSING}
        )

    def test_a_malformed_manifest_is_evidence_missing(self):
        with patch(
            "yewo.package_policy.AdbControl.dump_manifest", return_value="E: resources"
        ):
            checks = self._evaluate()

        self.assertTrue(all("malformed" in check.evidence for check in checks))

    def test_a_logical_resource_name_is_a_safe_fallback(self):
        lines = ["resource 0x7f120001 xml/network_security_config"]

        self.assertEqual(
            _resource_path(lines, "0x7f120001"),
            "res/xml/network_security_config.xml",
        )
        self.assertIsNone(_resource_path(lines, "0x7f120002"))


class EntitlementReaderTests(unittest.TestCase):
    def test_a_host_without_codesign_is_unsupported(self):
        with patch("yewo.package_policy.shutil.which", return_value=None):
            entitlements, problem = _signed_entitlements(Path("Example.app"))

        self.assertIsNone(entitlements)
        self.assertEqual(problem[0], PolicyCheckStatus.UNSUPPORTED)

    def test_a_codesign_failure_is_evidence_missing(self):
        failure = SubprocessRunError(["codesign"], 1, "", "not signed")
        with (
            patch("yewo.package_policy.shutil.which", return_value="codesign"),
            patch("yewo.package_policy.run_command", side_effect=failure),
        ):
            entitlements, problem = _signed_entitlements(Path("Example.app"))

        self.assertIsNone(entitlements)
        self.assertEqual(problem[0], PolicyCheckStatus.EVIDENCE_MISSING)

    def test_a_codesign_timeout_is_evidence_missing(self):
        timeout = subprocess.TimeoutExpired(["codesign"], 30)
        with (
            patch("yewo.package_policy.shutil.which", return_value="codesign"),
            patch("yewo.package_policy.run_command", side_effect=timeout),
        ):
            _, problem = _signed_entitlements(Path("Example.app"))

        self.assertEqual(problem[0], PolicyCheckStatus.EVIDENCE_MISSING)

    def test_codesign_must_return_a_property_list(self):
        result = SimpleNamespace(stdout="", stderr="Executable=Example")
        with (
            patch("yewo.package_policy.shutil.which", return_value="codesign"),
            patch("yewo.package_policy.run_command", return_value=result),
        ):
            _, problem = _signed_entitlements(Path("Example.app"))

        self.assertIn("no entitlement property list", problem[1])

    def test_signed_entitlements_are_read_from_codesign_output(self):
        payload = {
            "get-task-allow": False,
            "com.apple.developer.associated-domains": ["applinks:example.test"],
        }
        result = SimpleNamespace(
            stdout=plistlib.dumps(payload).decode(),
            stderr="Executable=Example.app",
        )
        with (
            patch("yewo.package_policy.shutil.which", return_value="codesign"),
            patch("yewo.package_policy.run_command", return_value=result),
        ):
            entitlements, problem = _signed_entitlements(Path("Example.app"))

        self.assertIsNone(problem)
        self.assertEqual(entitlements, payload)

    def test_malformed_entitlement_types_are_evidence_missing(self):
        result = SimpleNamespace(
            stdout=plistlib.dumps({"get-task-allow": "false"}).decode(), stderr=""
        )
        with (
            patch("yewo.package_policy.shutil.which", return_value="codesign"),
            patch("yewo.package_policy.run_command", return_value=result),
        ):
            entitlements, problem = _signed_entitlements(Path("Example.app"))

        self.assertIsNone(entitlements)
        self.assertIn("must be a boolean", problem[1])


class IOSArchiveReaderTests(unittest.TestCase):
    def _evaluate(self, path: Path):
        return evaluate_package_policy(path, Platform.IOS, {}, SecretScrubber())

    def test_info_plist_must_be_a_dictionary(self):
        with tempfile.TemporaryDirectory() as directory:
            app = Path(directory) / "Example.app"
            app.mkdir()
            (app / "Info.plist").write_bytes(plistlib.dumps(["not", "a", "mapping"]))
            checks = self._evaluate(app)

        self.assertEqual(
            {check.status for check in checks}, {PolicyCheckStatus.EVIDENCE_MISSING}
        )

    def test_info_security_fields_have_strict_types(self):
        with tempfile.TemporaryDirectory() as directory:
            app = Path(directory) / "Example.app"
            app.mkdir()
            (app / "Info.plist").write_bytes(
                plistlib.dumps({"NSAppTransportSecurity": "allow everything"})
            )
            checks = self._evaluate(app)

        self.assertTrue(all("NSAppTransportSecurity" in check.evidence for check in checks))

    def test_an_ipa_must_contain_one_app_bundle(self):
        with tempfile.TemporaryDirectory() as directory:
            ipa = Path(directory) / "Two.ipa"
            with zipfile.ZipFile(ipa, "w") as archive:
                for name in ("One", "Two"):
                    archive.writestr(
                        f"Payload/{name}.app/Info.plist", plistlib.dumps({})
                    )
            checks = self._evaluate(ipa)

        self.assertTrue(all("exactly one" in check.evidence for check in checks))

    def test_oversized_ipa_content_is_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            ipa = Path(directory) / "Large.ipa"
            with zipfile.ZipFile(ipa, "w") as archive:
                archive.writestr("Payload/App.app/Info.plist", plistlib.dumps({}))
            with patch("yewo.package_policy.MAX_IOS_BUNDLE_BYTES", 0):
                checks = self._evaluate(ipa)

        self.assertTrue(all("inspection limit" in check.evidence for check in checks))

    def test_archive_traversal_members_are_not_extracted(self):
        with tempfile.TemporaryDirectory() as directory:
            archive_path = Path(directory) / "Traversal.ipa"
            destination = Path(directory) / "out"
            with zipfile.ZipFile(archive_path, "w") as writer:
                writer.writestr("Payload/App.app/Info.plist", plistlib.dumps({}))
                writer.writestr("Payload/App.app/../../escape", b"no")
            with zipfile.ZipFile(archive_path) as archive:
                _safe_extract_bundle(archive, "Payload/App.app", destination)

            self.assertFalse((destination / "escape").exists())
            self.assertTrue((destination / "Payload/App.app/Info.plist").is_file())
