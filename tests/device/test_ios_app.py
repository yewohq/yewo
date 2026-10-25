import plistlib
import tempfile
import unittest
import zipfile
from pathlib import Path
from types import SimpleNamespace

from yewo.agent.device.ios_app import bundle_identifier
from yewo.agent.device.control import DeviceControl
from yewo.agent.device.platform import Platform


class IosAppIdentityTests(unittest.TestCase):
    def test_reads_an_app_bundle(self):
        with tempfile.TemporaryDirectory() as directory:
            app = Path(directory) / "Example.app"
            app.mkdir()
            (app / "Info.plist").write_bytes(
                plistlib.dumps({"CFBundleIdentifier": "com.example.app"})
            )

            self.assertEqual(bundle_identifier(str(app)), "com.example.app")

            control = object.__new__(DeviceControl)
            control.driver = SimpleNamespace(capabilities={})
            control.package_id = None
            control.platform = Platform.IOS
            control.app_path = str(app)
            self.assertEqual(control._resolve_app_id(), "com.example.app")

    def test_reads_the_one_app_in_an_ipa(self):
        with tempfile.TemporaryDirectory() as directory:
            ipa = Path(directory) / "Example.ipa"
            with zipfile.ZipFile(ipa, "w") as archive:
                archive.writestr(
                    "Payload/Example.app/Info.plist",
                    plistlib.dumps({"CFBundleIdentifier": "com.example.ipa"}),
                )

            self.assertEqual(bundle_identifier(str(ipa)), "com.example.ipa")

    def test_refuses_an_ipa_with_ambiguous_apps(self):
        with tempfile.TemporaryDirectory() as directory:
            ipa = Path(directory) / "Many.ipa"
            with zipfile.ZipFile(ipa, "w") as archive:
                for name in ("One", "Two"):
                    archive.writestr(
                        f"Payload/{name}.app/Info.plist",
                        plistlib.dumps({"CFBundleIdentifier": f"com.example.{name}"}),
                    )

            self.assertIsNone(bundle_identifier(str(ipa)))


if __name__ == "__main__":
    unittest.main()
