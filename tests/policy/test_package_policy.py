"""Static package policy is cross-platform, deterministic, and paid for first."""

import plistlib
import tempfile
import unittest
import zipfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from yewo.agent.device.platform import Platform
from yewo.android_policy import evaluate_android_manifest
from yewo.ios_policy import evaluate_ios_metadata
from yewo.package_policy import evaluate_package_policy
from yewo.errors import UsageError
from yewo.policy import PolicyCheckReport, PolicyCheckStatus
from yewo.testing.models import UserJourney
from yewo.testing.runner import YewoTestRunner
from yewo.utils.secrets import SecretScrubber


ANDROID_SAFE = """
E: manifest (line=2)
  A: package="dev.example.safe" (Raw: "dev.example.safe")
  E: uses-sdk (line=4)
    A: android:targetSdkVersion(0x01010270)=(type 0x10)0x00000023
  E: application (line=7)
    A: android:debuggable(0x0101000f)=(type 0x12)0x00000000
    A: android:allowBackup(0x01010280)=(type 0x12)0x00000000
    A: android:usesCleartextTraffic(0x010104ec)=(type 0x12)0x00000000
    E: activity (line=10)
      A: android:name(0x01010003)="dev.example.Main" (Raw: "dev.example.Main")
      A: android:exported(0x01010010)=(type 0x12)0xffffffff
      E: intent-filter (line=13)
        E: action (line=14)
          A: android:name(0x01010003)="android.intent.action.MAIN" (Raw: "android.intent.action.MAIN")
        E: category (line=16)
          A: android:name(0x01010003)="android.intent.category.LAUNCHER" (Raw: "android.intent.category.LAUNCHER")
"""

ANDROID_UNSAFE = """
E: manifest (line=2)
  E: uses-sdk (line=3)
    A: android:targetSdkVersion(0x01010270)=(type 0x10)0x00000023
  E: application (line=5)
    A: android:debuggable(0x0101000f)=(type 0x12)0xffffffff
    A: android:usesCleartextTraffic(0x010104ec)=(type 0x12)0xffffffff
    E: receiver (line=8)
      A: android:name(0x01010003)="dev.example.OpenReceiver" (Raw: "dev.example.OpenReceiver")
      A: android:exported(0x01010010)=(type 0x12)0xffffffff
"""


class AndroidPolicyTests(unittest.TestCase):
    def test_release_manifest_passes_all_four_rule_families(self):
        checks = evaluate_android_manifest(ANDROID_SAFE)

        self.assertEqual(len(checks), 4)
        self.assertEqual({check.status for check in checks}, {PolicyCheckStatus.PASSED})

    def test_current_aapt2_namespace_attributes_are_parsed(self):
        xmltree = ANDROID_SAFE.replace(
            "android:", "http://schemas.android.com/apk/res/android:"
        ).replace("=(type 0x12)0x00000000", "=false")

        checks = evaluate_android_manifest(xmltree)

        self.assertEqual({check.status for check in checks}, {PolicyCheckStatus.PASSED})

    def test_insecure_manifest_names_every_finding(self):
        checks = {check.check_id: check for check in evaluate_android_manifest(ANDROID_UNSAFE)}

        self.assertEqual(checks["android.debuggable"].status, PolicyCheckStatus.FAILED)
        self.assertEqual(checks["android.allow_backup"].status, PolicyCheckStatus.FAILED)
        self.assertEqual(checks["android.cleartext_traffic"].status, PolicyCheckStatus.FAILED)
        self.assertEqual(
            checks["android.exported_components"].status, PolicyCheckStatus.FAILED
        )
        self.assertIn("OpenReceiver", checks["android.exported_components"].evidence)

    def test_a_network_security_resource_is_not_guessed_about(self):
        manifest = ANDROID_SAFE.replace(
            "    A: android:usesCleartextTraffic(0x010104ec)=(type 0x12)0x00000000",
            "    A: android:networkSecurityConfig(0x01010527)=@0x7f120001",
        )

        check = next(
            item
            for item in evaluate_android_manifest(manifest)
            if item.check_id == "android.cleartext_traffic"
        )

        self.assertEqual(check.status, PolicyCheckStatus.EVIDENCE_MISSING)

    def test_a_referenced_network_security_resource_is_evaluated(self):
        manifest = ANDROID_SAFE.replace(
            "    A: android:usesCleartextTraffic(0x010104ec)=(type 0x12)0x00000000",
            "    A: android:networkSecurityConfig(0x01010527)=@0x7f120001",
        )
        resource = """
E: network-security-config (line=2)
  E: base-config (line=3)
    A: cleartextTrafficPermitted=(type 0x12)0x00000000
  E: domain-config (line=4)
    A: cleartextTrafficPermitted=(type 0x12)0xffffffff
"""

        check = next(
            item
            for item in evaluate_android_manifest(
                manifest, {"@0x7f120001": resource}
            )
            if item.check_id == "android.cleartext_traffic"
        )

        self.assertEqual(check.status, PolicyCheckStatus.FAILED)
        self.assertIn("domain-config", check.evidence)

    def test_backup_rule_modes_and_encryption_are_interpreted(self):
        manifest = ANDROID_UNSAFE.replace(
            "  E: application (line=5)",
            """  E: application (line=5)
    A: android:dataExtractionRules=@0x7f120002
    A: android:fullBackupContent=@0x7f120003""",
        )
        extraction_rules = """
E: data-extraction-rules (line=2)
  E: cloud-backup (line=3)
    A: disableIfNoEncryptionCapabilities=true
    E: include (line=4)
      A: domain="sharedpref"
      A: path="settings.xml"
  E: cross-platform-transfer (line=8)
    A: platform="ios"
    E: include (line=9)
      A: domain="file"
      A: path="migration"
"""
        legacy_rules = """
E: full-backup-content (line=2)
  E: include (line=3)
    A: domain="file"
    A: path="exports"
    A: requireFlags="clientSideEncryption"
"""

        check = next(
            item
            for item in evaluate_android_manifest(
                manifest,
                {
                    "@0x7f120002": extraction_rules,
                    "@0x7f120003": legacy_rules,
                },
            )
            if item.check_id == "android.allow_backup"
        )

        self.assertEqual(check.status, PolicyCheckStatus.FAILED)
        self.assertIn("cloud-backup: limited to 1 include rule", check.evidence)
        self.assertIn("encryption required=true", check.evidence)
        self.assertIn("device-transfer: section absent", check.evidence)
        self.assertIn("cross-platform-transfer[ios]: limited to 1 include", check.evidence)
        self.assertIn("clientSideEncryption", check.evidence)

    def test_a_weak_custom_permission_does_not_clean_an_exported_component(self):
        manifest = ANDROID_SAFE.replace(
            "  E: application (line=7)",
            """  E: permission (line=5)
    A: android:name="dev.example.OPEN"
    A: android:protectionLevel="normal"
  E: application (line=7)""",
        ).replace(
            "    E: activity (line=10)",
            """    E: service (line=9)
      A: android:name="dev.example.OpenService"
      A: android:exported=true
      A: android:permission="dev.example.OPEN"
    E: activity (line=10)""",
        )

        check = evaluate_android_manifest(manifest)[3]

        self.assertEqual(check.status, PolicyCheckStatus.FAILED)
        self.assertIn("dev.example.OPEN (normal)", check.evidence)

    def test_a_signature_custom_permission_protects_an_exported_component(self):
        manifest = ANDROID_SAFE.replace(
            "  E: application (line=7)",
            """  E: permission (line=5)
    A: android:name="dev.example.OPEN"
    A: android:protectionLevel=(type 0x11)0x00000002
  E: application (line=7)""",
        ).replace(
            "    E: activity (line=10)",
            """    E: service (line=9)
      A: android:name="dev.example.OpenService"
      A: android:exported=true
      A: android:permission="dev.example.OPEN"
    E: activity (line=10)""",
        )

        check = evaluate_android_manifest(manifest)[3]

        self.assertEqual(check.status, PolicyCheckStatus.PASSED)
        self.assertIn("Protected exported components", check.evidence)

    def test_an_external_custom_permission_has_unknown_strength(self):
        manifest = ANDROID_SAFE.replace(
            "    E: activity (line=10)",
            """    E: service (line=9)
      A: android:name="dev.example.PartnerService"
      A: android:exported=true
      A: android:permission="partner.permission.OPEN"
    E: activity (line=10)""",
        )

        check = evaluate_android_manifest(manifest)[3]

        self.assertEqual(check.status, PolicyCheckStatus.EVIDENCE_MISSING)
        self.assertIn("definition is not packaged", check.evidence)

    def test_a_provider_must_protect_both_read_and_write_access(self):
        manifest = ANDROID_SAFE.replace(
            "    E: activity (line=10)",
            """    E: provider (line=9)
      A: android:name="dev.example.PublicProvider"
      A: android:exported=true
      A: android:readPermission="android.permission.READ_CONTACTS"
    E: activity (line=10)""",
        )

        check = evaluate_android_manifest(manifest)[3]

        self.assertEqual(check.status, PolicyCheckStatus.FAILED)
        self.assertIn("no permission for write access", check.evidence)

    def test_the_flutter_profile_receiver_has_a_strong_platform_permission(self):
        manifest = ANDROID_SAFE.replace(
            "    E: activity (line=10)",
            """    E: receiver (line=9)
      A: android:name="androidx.profileinstaller.ProfileInstallReceiver"
      A: android:exported=true
      A: android:permission="android.permission.DUMP"
    E: activity (line=10)""",
        )

        check = evaluate_android_manifest(manifest)[3]

        self.assertEqual(check.status, PolicyCheckStatus.PASSED)
        self.assertIn("signature|privileged|development", check.evidence)

    def test_an_unknown_platform_permission_strength_is_not_guessed(self):
        manifest = ANDROID_SAFE.replace(
            "    E: activity (line=10)",
            """    E: receiver (line=9)
      A: android:name="dev.example.FutureReceiver"
      A: android:exported=true
      A: android:permission="android.permission.FUTURE_PLATFORM_PERMISSION"
    E: activity (line=10)""",
        )

        check = evaluate_android_manifest(manifest)[3]

        self.assertEqual(check.status, PolicyCheckStatus.EVIDENCE_MISSING)
        self.assertIn("protection level is not packaged", check.evidence)


class IOSPolicyTests(unittest.TestCase):
    def test_hardened_metadata_passes_all_four_rule_families(self):
        checks = evaluate_ios_metadata({}, {"get-task-allow": False})

        self.assertEqual(len(checks), 4)
        self.assertEqual({check.status for check in checks}, {PolicyCheckStatus.PASSED})

    def test_insecure_metadata_names_every_finding(self):
        info = {
            "UIFileSharingEnabled": True,
            "NSAppTransportSecurity": {"NSAllowsArbitraryLoads": True},
            "CFBundleURLTypes": [{"CFBundleURLSchemes": ["private-app"]}],
        }
        checks = {
            check.check_id: check
            for check in evaluate_ios_metadata(info, {"get-task-allow": True})
        }

        self.assertEqual(checks["ios.get_task_allow"].status, PolicyCheckStatus.FAILED)
        self.assertEqual(checks["ios.file_sharing"].status, PolicyCheckStatus.FAILED)
        self.assertEqual(
            checks["ios.app_transport_security"].status, PolicyCheckStatus.FAILED
        )
        self.assertEqual(checks["ios.external_entry_points"].status, PolicyCheckStatus.FAILED)

    def test_absent_codesign_is_explicitly_unsupported(self):
        check = evaluate_ios_metadata(
            {},
            None,
            (PolicyCheckStatus.UNSUPPORTED, "codesign is unavailable"),
        )[0]

        self.assertEqual(check.status, PolicyCheckStatus.UNSUPPORTED)


class PackagePolicyTests(unittest.TestCase):
    def test_an_obfuscated_android_resource_path_is_inspected(self):
        manifest = ANDROID_SAFE.replace(
            "    A: android:usesCleartextTraffic(0x010104ec)=(type 0x12)0x00000000",
            """    A: android:networkSecurityConfig(0x01010527)=@0x7f120001
    A: android:dataExtractionRules=@0x7f120002
    A: android:fullBackupContent=@0x7f120003""",
        )
        resources = """
    resource 0x7f120001 xml/network_security_config
      () (file) res/aB.xml type=XML
    resource 0x7f120002 xml/data_extraction_rules
      () (file) res/cD.xml type=XML
    resource 0x7f120003 xml/full_backup_content
      () (file) res/eF.xml type=XML
"""
        network_resource = """
E: network-security-config (line=2)
  E: base-config (line=3)
    A: cleartextTrafficPermitted=false
"""
        extraction_resource = """
E: data-extraction-rules (line=2)
  E: cloud-backup (line=3)
    E: include (line=4)
      A: domain="file"
      A: path="exports"
  E: device-transfer (line=7)
    E: exclude (line=8)
      A: domain="database"
      A: path="private.db"
"""
        legacy_resource = """
E: full-backup-content (line=2)
  E: include (line=3)
    A: domain="file"
    A: path="exports"
"""
        outputs = [
            SimpleNamespace(stdout=resources),
            SimpleNamespace(stdout=network_resource),
            SimpleNamespace(stdout=extraction_resource),
            SimpleNamespace(stdout=legacy_resource),
        ]
        with (
            patch("yewo.package_policy.AdbControl.dump_manifest", return_value=manifest),
            patch("yewo.package_policy.AdbControl.find_build_tool", return_value="aapt2"),
            patch("yewo.package_policy.run_command", side_effect=outputs) as command,
        ):
            checks = evaluate_package_policy(
                Path("app.apk"), Platform.ANDROID, {}, SecretScrubber()
            )

        cleartext = next(
            check for check in checks if check.check_id == "android.cleartext_traffic"
        )
        backup = next(check for check in checks if check.check_id == "android.allow_backup")
        self.assertEqual(cleartext.status, PolicyCheckStatus.PASSED)
        self.assertIn("device-transfer: default exposure remains", backup.evidence)
        self.assertIn("res/aB.xml", command.call_args_list[1].args[0])
        self.assertIn("res/cD.xml", command.call_args_list[2].args[0])
        self.assertIn("res/eF.xml", command.call_args_list[3].args[0])

    def test_an_ios_app_bundle_is_read_without_a_device(self):
        with tempfile.TemporaryDirectory() as directory:
            app = Path(directory) / "Example.app"
            app.mkdir()
            (app / "Info.plist").write_bytes(plistlib.dumps({}))
            with patch(
                "yewo.package_policy._signed_entitlements",
                return_value=({"get-task-allow": False}, None),
            ):
                checks = evaluate_package_policy(
                    app, Platform.IOS, {}, SecretScrubber()
                )

        self.assertEqual({check.platform for check in checks}, {Platform.IOS})
        self.assertEqual({check.status for check in checks}, {PolicyCheckStatus.PASSED})

    def test_an_ios_ipa_is_read_without_a_device(self):
        with tempfile.TemporaryDirectory() as directory:
            ipa = Path(directory) / "Example.ipa"
            with zipfile.ZipFile(ipa, "w") as archive:
                archive.writestr("Payload/Example.app/Info.plist", plistlib.dumps({}))
                archive.writestr("Payload/Example.app/Example", b"binary")
            with patch(
                "yewo.package_policy._signed_entitlements",
                return_value=({"get-task-allow": False}, None),
            ) as entitlements:
                checks = evaluate_package_policy(
                    ipa, Platform.IOS, {}, SecretScrubber()
                )

        inspected_bundle = entitlements.call_args.args[0]
        self.assertEqual(inspected_bundle.name, "Example.app")
        self.assertEqual({check.status for check in checks}, {PolicyCheckStatus.PASSED})

    def test_a_package_for_the_wrong_platform_is_explicitly_unsupported(self):
        checks = evaluate_package_policy(
            Path("Example.apk"), Platform.IOS, {}, SecretScrubber()
        )

        self.assertEqual({check.status for check in checks}, {PolicyCheckStatus.UNSUPPORTED})
        self.assertTrue(all(".app bundle or .ipa" in check.evidence for check in checks))

    def test_a_failed_check_can_be_waived_with_a_visible_reason(self):
        with patch("yewo.package_policy.AdbControl.dump_manifest", return_value=ANDROID_UNSAFE):
            checks = evaluate_package_policy(
                Path("app.apk"),
                Platform.ANDROID,
                {"android.debuggable": "Internal diagnostic build"},
                SecretScrubber(),
            )

        check = next(item for item in checks if item.check_id == "android.debuggable")
        self.assertEqual(check.status, PolicyCheckStatus.WAIVED)
        self.assertEqual(check.waiver_reason, "Internal diagnostic build")

    def test_a_misspelled_waiver_id_is_refused(self):
        with (
            patch("yewo.package_policy.AdbControl.dump_manifest", return_value=ANDROID_SAFE),
            self.assertRaises(UsageError),
        ):
            evaluate_package_policy(
                Path("app.apk"),
                Platform.ANDROID,
                {"android.debugable": "typo"},
                SecretScrubber(),
            )


class StaticFailureOrderTests(unittest.TestCase):
    def test_a_policy_failure_opens_no_device_and_generates_no_scenario(self):
        events = []

        class Session:
            def __init__(self, *_, session_dir, **__):
                root = Path(session_dir)
                root.mkdir(parents=True, exist_ok=True)
                self.telemetry_dir = root
                self.screenshots_dir = root
                self.prompts_dir = root / "prompts"
                self.prompts_dir.mkdir(exist_ok=True)
                self.manifest_path = root / "run.json"
                self.root_log_file = root / "run.log"
                self.root_log_file.write_text("", encoding="utf-8")
                self.appium_log_path = root / "appium.log"
                self.session_id = "static"
                self.installed_by_run = False

            def start(self, **_):
                events.append("device")

            def close(self, **_):
                pass

        class Agent:
            def __init__(self, *_, **__):
                llm = SimpleNamespace(usage_callback=None)
                self._device_agent = SimpleNamespace(
                    llm=llm,
                    fast_llm=llm,
                    scrubber=SecretScrubber(),
                    record_prompts=lambda _store: None,
                )

        failed = PolicyCheckReport(
            check_id="android.debuggable",
            status=PolicyCheckStatus.FAILED,
            platform=Platform.ANDROID,
            source="manifest",
            evidence="debuggable is true",
        )
        with tempfile.TemporaryDirectory() as directory:
            runner = YewoTestRunner(
                device_name="Pixel_8",
                app_path="app.apk",
                assets_dir=Path(directory),
                durable_replay=False,
            )
            with (
                patch("yewo.testing.runner.DeviceSession", Session),
                patch("yewo.testing.runner.QAAgent", Agent),
                patch("yewo.testing.runner.evaluate_package_policy", return_value=[failed]),
                patch("yewo.testing.runner.ensure_device_ready") as ready,
                patch.object(runner, "_suite", side_effect=AssertionError("generated")),
            ):
                report = runner.run_journey(
                    UserJourney(id="j", name="J", description="Open the app.")
                )

        self.assertEqual(events, [])
        ready.assert_not_called()
        self.assertEqual(report.status, "FAILED")
        self.assertEqual(report.policy_checks[0].check_id, "android.debuggable")
