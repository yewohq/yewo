"""What produced a run, and the rule that reading it never ends one.

A report that cannot say which build it failed on is not evidence of a
defect. Every reader here is also allowed to find nothing, so these cases
cover both: what it reads when the answer is there, and what it does when it
is not.
"""

import hashlib
import plistlib
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from yewo.agent.device.platform import Platform
from yewo.identity import (
    AppIdentity,
    app_identity,
    artifact_sha256,
    ci_identity,
    derived_uuid,
    device_identity,
    git_identity,
    masked_serial,
    new_uuid,
)


class IdentifierTests(unittest.TestCase):
    def test_a_drawn_identifier_is_different_every_time(self):
        self.assertNotEqual(new_uuid(), new_uuid())

    def test_a_derived_identifier_is_the_same_every_time(self):
        self.assertEqual(derived_uuid("a", 0), derived_uuid("a", 0))

    def test_a_derived_identifier_is_specific_to_its_parts(self):
        self.assertNotEqual(derived_uuid("a", 0), derived_uuid("a", 1))
        self.assertNotEqual(derived_uuid("a", 0), derived_uuid("b", 0))


class SerialTests(unittest.TestCase):
    """A serial names hardware, and a manifest is shared."""

    def test_only_the_tail_is_kept(self):
        masked = masked_serial("R5CT90ABCDEF")

        self.assertNotIn("R5CT90", masked)
        self.assertTrue(masked.endswith("CDEF"))

    def test_two_targets_still_read_apart(self):
        self.assertNotEqual(masked_serial("aaaa1111"), masked_serial("aaaa2222"))

    def test_nothing_masks_to_nothing(self):
        self.assertIsNone(masked_serial(None))
        self.assertIsNone(masked_serial(""))

    def test_a_serial_too_short_to_mask_is_left_alone(self):
        self.assertEqual(masked_serial("ab"), "ab")


class BuildHashTests(unittest.TestCase):
    def test_a_file_is_hashed_by_its_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "app.apk"
            path.write_bytes(b"a build")

            self.assertEqual(
                artifact_sha256(path), hashlib.sha256(b"a build").hexdigest()
            )

    def test_a_bundle_is_hashed_over_its_paths_as_well(self):
        """Moving a file inside a bundle is a different build."""
        with tempfile.TemporaryDirectory() as directory:
            first = Path(directory) / "one.app"
            second = Path(directory) / "two.app"
            for root, name in ((first, "a"), (second, "b")):
                (root / "Frameworks").mkdir(parents=True)
                (root / "Frameworks" / name).write_bytes(b"same bytes")

            self.assertNotEqual(artifact_sha256(first), artifact_sha256(second))

    def test_the_same_bundle_hashes_the_same_way_twice(self):
        with tempfile.TemporaryDirectory() as directory:
            bundle = Path(directory) / "one.app"
            bundle.mkdir()
            (bundle / "binary").write_bytes(b"x" * 2048)

            self.assertEqual(artifact_sha256(bundle), artifact_sha256(bundle))

    def test_a_build_that_is_not_there_hashes_to_nothing(self):
        self.assertIsNone(artifact_sha256(Path("/no/such/build.apk")))


class AppTests(unittest.TestCase):
    def test_a_bundle_names_itself_through_its_plist(self):
        with tempfile.TemporaryDirectory() as directory:
            bundle = Path(directory) / "Example.app"
            bundle.mkdir()
            with (bundle / "Info.plist").open("wb") as handle:
                plistlib.dump(
                    {
                        "CFBundleIdentifier": "com.example.app",
                        "CFBundleShortVersionString": "2.1.0",
                        "CFBundleVersion": "417",
                    },
                    handle,
                )

            found = app_identity(str(bundle))

        self.assertEqual(found.package, "com.example.app")
        self.assertEqual(found.version, "2.1.0")
        self.assertEqual(found.build, "417")
        self.assertIsNotNone(found.artifact_sha256)

    def test_a_bundle_with_no_plist_is_read_as_nothing_rather_than_half(self):
        """A reader must never be shown a version that came from elsewhere."""
        with tempfile.TemporaryDirectory() as directory:
            bundle = Path(directory) / "Example.app"
            bundle.mkdir()
            found = app_identity(str(bundle))

        self.assertIsNone(found.package)
        self.assertIsNotNone(found.artifact_sha256)

    def test_a_malformed_plist_is_not_a_run_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            bundle = Path(directory) / "Example.app"
            bundle.mkdir()
            (bundle / "Info.plist").write_text("not a plist")

            self.assertIsNone(app_identity(str(bundle)).package)

    def test_an_app_nobody_named_is_read_as_nothing(self):
        self.assertEqual(app_identity(None), AppIdentity())
        self.assertEqual(app_identity("/no/such/app.apk"), AppIdentity())

    def test_an_apk_is_read_through_aapt(self):
        badging = (
            "package: name='com.example.app' versionCode='417' "
            "versionName='2.1.0' platformBuildVersionName='14'\n"
            "launchable-activity: name='com.example.app.Main'\n"
        )
        with tempfile.TemporaryDirectory() as directory:
            apk = Path(directory) / "app.apk"
            apk.write_bytes(b"not really an apk")
            with (
                patch("yewo.identity._read", return_value=badging),
                patch(
                    "yewo.agent.device.adb_control.AdbControl.find_aapt",
                    return_value="/sdk/aapt",
                ),
            ):
                found = app_identity(str(apk))

        self.assertEqual(found.package, "com.example.app")
        self.assertEqual(found.version, "2.1.0")
        self.assertEqual(found.build, "417")

    def test_an_absent_sdk_is_not_a_run_failure(self):
        from yewo.errors import DeviceUnavailable

        with tempfile.TemporaryDirectory() as directory:
            apk = Path(directory) / "app.apk"
            apk.write_bytes(b"not really an apk")
            with patch(
                "yewo.agent.device.adb_control.AdbControl.find_aapt",
                side_effect=DeviceUnavailable("ANDROID_HOME is not set"),
            ):
                found = app_identity(str(apk))

        self.assertIsNone(found.package)
        self.assertIsNotNone(found.artifact_sha256)


class BuildContextTests(unittest.TestCase):
    """What the job says, so a failure links back to the build that found it."""

    def test_a_github_build_names_its_commit_and_its_run(self):
        env = {
            "GITHUB_ACTIONS": "true",
            "GITHUB_SHA": "cafe1234",
            "GITHUB_REF": "refs/pull/42/merge",
            "GITHUB_HEAD_REF": "feat/lock",
            "GITHUB_SERVER_URL": "https://github.com",
            "GITHUB_REPOSITORY": "acme/app",
            "GITHUB_RUN_ID": "99",
        }
        with patch.dict("os.environ", env, clear=True):
            commit, job = git_identity(), ci_identity()

        self.assertEqual(commit.sha, "cafe1234")
        self.assertEqual(commit.branch, "feat/lock")
        self.assertEqual(commit.pull_request, "42")
        self.assertEqual(job.provider, "github_actions")
        self.assertEqual(job.run_url, "https://github.com/acme/app/actions/runs/99")

    def test_a_branch_build_carries_no_pull_request(self):
        env = {
            "GITHUB_ACTIONS": "true",
            "GITHUB_SHA": "cafe1234",
            "GITHUB_REF": "refs/heads/main",
            "GITHUB_REF_NAME": "main",
        }
        with patch.dict("os.environ", env, clear=True):
            commit = git_identity()

        self.assertIsNone(commit.pull_request)
        self.assertEqual(commit.branch, "main")

    def test_a_gitlab_build_is_read_under_its_own_names(self):
        env = {
            "GITLAB_CI": "true",
            "CI_COMMIT_SHA": "beef",
            "CI_COMMIT_REF_NAME": "main",
            "CI_MERGE_REQUEST_IID": "7",
            "CI_PIPELINE_URL": "https://gitlab.example/pipelines/3",
        }
        with patch.dict("os.environ", env, clear=True):
            commit, job = git_identity(), ci_identity()

        self.assertEqual(commit.sha, "beef")
        self.assertEqual(commit.pull_request, "7")
        self.assertEqual(job.run_url, "https://gitlab.example/pipelines/3")

    def test_an_unnamed_build_is_still_recorded_as_a_build(self):
        with patch.dict("os.environ", {"CI": "1"}, clear=True):
            self.assertEqual(ci_identity().provider, "unnamed")

    def test_a_laptop_is_not_a_build(self):
        with patch.dict("os.environ", {}, clear=True):
            self.assertIsNone(ci_identity().provider)

    def test_a_directory_that_is_not_a_repository_reads_as_no_commit(self):
        with patch.dict("os.environ", {}, clear=True), tempfile.TemporaryDirectory() as d:
            commit = git_identity(Path(d))

        self.assertIsNone(commit.sha)

    def test_a_detached_checkout_reports_no_branch_rather_than_head(self):
        """Which is what git says, and `HEAD` is not a branch name."""
        with patch.dict("os.environ", {}, clear=True), patch(
            "yewo.identity._read", side_effect=["HEAD", "abc123", "", ""]
        ):
            commit = git_identity(Path("/anywhere"))

        self.assertIsNone(commit.branch)
        self.assertEqual(commit.sha, "abc123")

    def test_a_clean_checkout_is_recorded_as_clean(self):
        with patch.dict("os.environ", {}, clear=True), patch(
            "yewo.identity._read", side_effect=["main", "abc123", "", ""]
        ):
            commit = git_identity(Path("/anywhere"))

        self.assertFalse(commit.dirty)
        self.assertIsNone(commit.diff_sha256)

    def test_an_edited_checkout_says_so_and_names_what_changed(self):
        """A run of edited files is not evidence about the commit beside it."""
        with patch.dict("os.environ", {}, clear=True), patch(
            "yewo.identity._read",
            side_effect=["main", "abc123", "diff --git a/x b/x", "new.py"],
        ):
            commit = git_identity(Path("/anywhere"))

        self.assertTrue(commit.dirty)
        self.assertEqual(len(commit.diff_sha256 or ""), 64)

    def test_two_different_edits_of_one_commit_are_told_apart(self):
        def read(*args):
            return {"rev-parse": "abc123"}.get(args[3], "main")

        shas = []
        for change in ("one change", "another change"):
            with patch.dict("os.environ", {}, clear=True), patch(
                "yewo.identity._read", side_effect=["main", "abc123", change, ""]
            ):
                shas.append(git_identity(Path("/anywhere")).diff_sha256)

        self.assertNotEqual(shas[0], shas[1])


class TargetTests(unittest.TestCase):
    def test_the_target_is_described_as_the_driver_reported_it(self):
        device = SimpleNamespace(
            platform=Platform.ANDROID,
            serial="emulator-5554",
            os_version="14",
            model="Pixel 8",
        )

        with patch(
            "yewo.testing.judge.capabilities.target_of",
            return_value=SimpleNamespace(emulator=True),
        ):
            found = device_identity(device)

        self.assertEqual(found.model, "Pixel 8")
        self.assertEqual(found.os_version, "14")
        self.assertTrue(found.emulator)
        self.assertTrue(found.serial.endswith("5554"))

    def test_a_kind_nobody_could_establish_loses_nothing_else(self):
        """The one read here that asks a command line must not take the rest."""
        device = SimpleNamespace(
            platform=Platform.IOS, serial="udid", os_version="18.2", model="iPhone"
        )

        with patch(
            "yewo.testing.judge.capabilities.target_of",
            side_effect=RuntimeError("simctl did not answer"),
        ):
            found = device_identity(device)

        self.assertEqual(found.model, "iPhone")
        self.assertEqual(found.os_version, "18.2")
        self.assertIsNone(found.emulator)

    def test_a_session_that_answers_nothing_is_not_a_run_failure(self):
        found = device_identity(SimpleNamespace())

        self.assertIsNone(found.model)

    def test_no_session_is_no_target(self):
        self.assertIsNone(device_identity(None).model)


class ReusedAppTests(unittest.TestCase):
    """What `--reuse-app` may and may not be said to have tested.

    The artifact beside the journey file is not installed under that flag.
    A rebuilt debug APK keeps its version, so the file on disk can differ
    from the app on the device with nothing to say so. A hash of a build
    that is not running is worse than no hash: a reader would trust it.
    """

    def _apk(self, directory):
        apk = Path(directory) / "app.apk"
        apk.write_bytes(b"a build that is not installed")
        return apk

    def test_nothing_is_attested_about_an_artifact_that_was_not_installed(self):
        with tempfile.TemporaryDirectory() as directory:
            found = app_identity(str(self._apk(directory)), installed=False)

        self.assertFalse(found.installed)
        self.assertIsNone(found.artifact_sha256)
        self.assertIsNone(found.version)

    def test_an_installed_artifact_is_attested_in_full(self):
        with tempfile.TemporaryDirectory() as directory:
            found = app_identity(str(self._apk(directory)), installed=True)

        self.assertTrue(found.installed)
        self.assertIsNotNone(found.artifact_sha256)

    def test_a_run_that_reuses_the_app_says_so_in_its_record(self):
        from yewo.testing.runner import YewoTestRunner

        reusing = YewoTestRunner(
            device_name="Pixel_8", app_path="app.apk", reuse_app=True
        )
        installing = YewoTestRunner(device_name="Pixel_8", app_path="app.apk")

        self.assertFalse(reusing._environment().app.installed)
        self.assertTrue(installing._environment().app.installed)


class GoalRunIdentityTests(unittest.TestCase):
    """A goal run records what it tested on the same terms as a suite does."""

    def _manifest(self, directory, reuse_app, grounding="coordinates"):
        from unittest.mock import Mock

        from yewo.agent.run import standalone

        apk = Path(directory) / "app.apk"
        apk.write_bytes(b"a build")
        session = SimpleNamespace(
            app_path=str(apk),
            reuse_app=reuse_app,
            manifest_path=Path(directory) / "run.json",
            prompts_dir=Path(directory) / "prompts",
            telemetry_dir=Path(directory),
        )
        agent = SimpleNamespace(
            llm=SimpleNamespace(model="openrouter/m"),
            grounding=grounding,
            scrubber=Mock(scrub=str),
        )
        return standalone._manifest(
            session, agent, "reach checkout", Platform.ANDROID, None
        )

    def test_an_installed_build_is_attested(self):
        with tempfile.TemporaryDirectory() as directory:
            app = self._manifest(directory, reuse_app=False).environment.app

        self.assertTrue(app.installed)
        self.assertIsNotNone(app.artifact_sha256)

    def test_a_reused_build_is_not(self):
        """The same rule as a suite run. One artifact, one answer about it."""
        with tempfile.TemporaryDirectory() as directory:
            app = self._manifest(directory, reuse_app=True).environment.app

        self.assertFalse(app.installed)
        self.assertIsNone(app.artifact_sha256)

    def test_a_marks_run_records_the_default_perception_model(self):
        from yewo.defaults import DEFAULT_PERCEPTION_MODEL

        with tempfile.TemporaryDirectory() as directory:
            manifest = self._manifest(directory, reuse_app=False, grounding="marks")

        self.assertEqual(manifest.environment.detector, DEFAULT_PERCEPTION_MODEL)
