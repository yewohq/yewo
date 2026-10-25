"""
Manages Appium server lifecycle and device connection session.
Supports starting Appium programmatically, checking platforms, and screen recording.
"""

import logging
import os
import time
import urllib.request
from pathlib import Path
from typing import Optional
from uuid import uuid4

from appium.webdriver.appium_service import AppiumService

from yewo.agent.artifacts.layout import RunLayout, attach_run_log, detach_run_log
from yewo.agent.device.adb_control import AdbControl
from yewo.agent.device.appium_runtime import resolve_appium_runtime
from yewo.agent.device.control import DeviceControl
from yewo.agent.device import device_logs
from yewo.agent.device.platform import Platform
from yewo.agent.device.recording import DeviceRecording
from yewo.agent.device.simctl_control import SimctlControl
from yewo.devices.discovery import RUNNING, ios_devices, resolve
from yewo.utils.platform_checks import run_android_preflight, run_ios_preflight

logger = logging.getLogger(__name__)


class DeviceSession(DeviceRecording):
    """Manages the startup/teardown of the Appium server and mobile device driver session."""

    def __init__(
        self,
        device_name: str,
        app_path: str,
        record_session: bool = False,
        session_dir: Path = Path("sessions"),
        platform: Platform = Platform.ANDROID,
        session_id: str = None,
        keep_device: bool = False,
        reuse_app: bool = False,
        keep_app: bool = False,
        installed_by_run: Optional[bool] = None,
    ):
        """Initializes a DeviceSession.

        Args:
            device_name: Name of the emulator/simulator/device to run against.
            app_path: Path to the target mobile app bundle (.apk, .app, or .ipa).
            record_session: If True, records the screen during execution.
            session_dir: Directory where screen recordings will be saved.
            platform: Platform name ('android' or 'ios').
            session_id: Optional existing session ID to reuse.
            keep_device: If True, leaves the emulator running when the session
                ends, so the next run skips the boot.
            reuse_app: If True, skips reinstalling the app. Only safe when the
                build has not changed — a debug rebuild keeps its version, so
                Appium would consider the stale install current.
            keep_app: If True, leaves the app installed when the session
                ends. By default a run removes the app it installed.
            installed_by_run: Whether this run installed the app, when the
                caller already knows. Session recovery passes what the failed
                session learned, because by then the app is on the device and
                a fresh look would call it preexisting.
        """

        self.session_dir = session_dir
        self.device_name = device_name
        self.app_path = app_path
        self.platform = platform
        self.device_control: Optional[DeviceControl] = None
        self.keep_device = keep_device
        self.reuse_app = reuse_app
        self.keep_app = keep_app
        self._installed_by_run: Optional[bool] = installed_by_run
        self._ios_simulator_udid: Optional[str] = None
        self._ios_simulator_was_running = False
        # Run all preflight checks before touching any hardware
        if self.platform == Platform.IOS:
            run_ios_preflight(app_path)
        else:
            run_android_preflight(app_path)
            self.adb_control = None
        self.record_session = record_session
        self.is_recording = False
        self.session_id = session_id or str(uuid4())

        self.layout = RunLayout.under(self.session_dir, self.session_id)
        self.layout.create()
        # Per session, because its segment number is what tells this
        # session's logs from those of the session that replaces it.
        self._device_logs = device_logs.DeviceLogRecorder(self.layout, self.platform)
        self._file_handler = attach_run_log(self.layout.run_log)

        host = os.environ.get("APPIUM_HOST", "127.0.0.1")
        port = int(os.environ.get("APPIUM_PORT", "4723"))
        self._appium_url = f"http://{host}:{port}"
        self._owns_appium = False

        self._appium_service = AppiumService()
        self._appium_args = {
            "host": host,
            "port": port,
            # Appium's default HTTP logging includes base64 screenshot response
            # bodies. Warning-level logs retain lifecycle failures without
            # copying screen contents into a second artifact.
            "args": [
                "--log",
                str(self.layout.appium_log),
                "--log-level",
                "warn",
                "--log-no-colors",
            ],
        }

    @property
    def screenshots_dir(self) -> Path:
        return self.layout.screenshots

    @property
    def recordings_dir(self) -> Path:
        return self.layout.recordings

    @property
    def telemetry_dir(self) -> Path:
        return self.layout.telemetry

    @property
    def manifest_path(self) -> Path:
        return self.layout.manifest

    @property
    def prompts_dir(self) -> Path:
        return self.layout.prompts

    @property
    def root_log_file(self) -> Path:
        return self.layout.run_log

    @property
    def appium_log_path(self) -> Path:
        return self.layout.appium_log

    @property
    def installed_by_run(self) -> Optional[bool]:
        """Whether this run installed the app, as far as it could tell."""
        return self._installed_by_run

    def _start_appium(self):
        try:
            with urllib.request.urlopen(
                f"{self._appium_url}/status", timeout=0.75
            ) as response:
                if response.status == 200:
                    logger.info("Using the Appium service already running at %s", self._appium_url)
                    return
        except Exception:
            pass
        if self._appium_service.is_running:
            logger.info("Appium already running")
            return
        runtime = resolve_appium_runtime()
        logger.info(
            "Starting Appium %s with Node at %s...",
            runtime.appium_version,
            runtime.node_binary,
        )
        self._appium_args.update(
            {
                "node": str(runtime.node_binary),
                "main_script": str(runtime.appium_script),
                "env": runtime.environment,
            }
        )
        self._appium_service.start(**self._appium_args)
        if not self._appium_service.is_running:
            raise RuntimeError(f"Appium failed to start. See {self.layout.appium_log}")
        logger.info("Appium started")
        self._owns_appium = True

    def _note_app_ownership(self) -> None:
        """Records whether this run is the one that installed the app.

        Teardown removes what the run put there and nothing else, so the
        answer has to be positive knowledge: the device was reachable, it
        listed its packages, and the app was not among them.

        Anything else leaves the app alone. A device that has not booted yet
        cannot be asked, and an adb call that fails answers nothing — neither
        is evidence that the app arrived with this run.
        """
        if self.platform != Platform.ANDROID or self._installed_by_run is not None:
            return
        self._installed_by_run = False
        serial = AdbControl.resolve_emulator_serial(self.device_name)
        if not serial:
            logger.info("No device to ask about %s yet. It will be left installed.", self.app_path)
            return
        packages = AdbControl.installed_packages(serial)
        if not packages:
            logger.info("Could not list packages on %s. The app will be left installed.", serial)
            return
        package, _ = AdbControl.get_apk_info(self.app_path)
        self._installed_by_run = bool(package) and package not in packages

    def start(self, is_headless: bool = False) -> DeviceControl:
        self.layout.create()
        self._note_ios_target()
        self._start_appium()
        self._note_app_ownership()
        self.device_control = DeviceControl(
            device_name=self.device_name,
            app_path=self.app_path,
            platform=self.platform,
            adb_control=None,
            is_headless=is_headless,
            keep_device=self.keep_device,
            reuse_app=self.reuse_app,
            server_url=self._appium_url,
        )
        if self.platform == Platform.ANDROID:
            serial = self.device_control.serial or self.device_name
            AdbControl.enable_feedback(
                serial=serial, pointer_location=False, tap_highlights=True
            )
        time.sleep(6)
        self._device_logs.start(self.device_control.driver)
        if self.record_session:
            self.discard_stale_recordings()
            self.start_recording()
        return self.device_control

    def _note_ios_target(self) -> None:
        """Remember whether this run is responsible for an iOS simulator boot."""
        if self.platform != Platform.IOS:
            return
        target = resolve(self.device_name, ios_devices())
        if target is None or target.kind != "simulator" or not target.identifier:
            return
        self._ios_simulator_udid = target.identifier
        self._ios_simulator_was_running = target.state == RUNNING

    def _shutdown_owned_ios_simulator(self, interrupted: bool) -> None:
        """Stop only a simulator that was shut down before this run started."""
        udid = getattr(self, "_ios_simulator_udid", None)
        if (
            self.platform != Platform.IOS
            or self.keep_device
            or getattr(self, "_ios_simulator_was_running", False)
            or not udid
        ):
            return
        try:
            SimctlControl.shutdown(udid)
            self._ios_simulator_udid = None
            logger.info("iOS simulator stopped")
        except Exception as exc:  # noqa: BLE001 - teardown must continue
            log = logger.info if interrupted else logger.warning
            log("Failed to stop iOS simulator: %s", exc)

    def _clean_device(self, final: bool) -> None:
        """Removes the app this run installed, once its artifacts are saved.

        Videos are not swept here. Saving one removes it from the device, and
        one that could not be saved is the only copy there is. What earlier
        runs abandoned is cleared at the next session's start instead.
        """
        if self.platform != Platform.ANDROID or not final:
            return
        serial = (self.device_control.serial if self.device_control else None) or (
            AdbControl.resolve_emulator_serial(self.device_name)
        )
        if not serial:
            return
        package = self.device_control.package_id if self.device_control else None
        if not package or self.keep_app or self.reuse_app or not self._installed_by_run:
            return
        try:
            AdbControl.uninstall_app(serial, package)
        except Exception as e:
            logger.warning("Failed to uninstall %s: %s", package, e)

    def checkpoint_device_logs(self) -> None:
        """Takes what the device has logged so far, at a scenario boundary."""
        driver = self._log_reading_driver()
        if driver is not None:
            # The timer may already be reading the same incremental buffer.
            # Do not hold the scenario behind that in-flight device request.
            self._device_logs.drain(driver, timeout=0)

    def _save_device_logs(self) -> None:
        """Before the quit, because a quit driver reads nothing."""
        driver = self._log_reading_driver()
        if driver is None:
            logger.warning("No driver to read the device log from.")
            return
        self._device_logs.finish(driver)

    def _log_reading_driver(self):
        if self.device_control is None:
            return None
        return self.device_control.driver

    def close(self, final: bool = True, interrupted: bool = False):
        """Ends the session, and by default removes what the run installed.

        Session recovery closes a dead session and opens another against the
        same app, so it passes final=False. Uninstalling there would take the
        app out from under the scenarios still to run.
        """
        if interrupted:
            self._device_logs.stop()
        else:
            try:
                self.save_recording()
            except Exception as e:
                logger.warning("Failed to save session recording: %s", e)
            try:
                self._save_device_logs()
            except Exception as e:  # noqa: BLE001 - evidence must not cost the teardown
                logger.warning("Failed to save the device log: %s", e)
        self._clean_device(final=final)
        if self.device_control:
            try:
                if interrupted:
                    self.device_control.quit(interrupted=True)
                else:
                    self.device_control.quit()
            except Exception as e:
                logger.warning("Failed to quit device control: %s", e)
        if final:
            self._shutdown_owned_ios_simulator(interrupted)
        if self._owns_appium and self._appium_service.is_running:
            try:
                self._appium_service.stop()
                logger.info("Appium service stopped")
            except Exception as e:
                logger.warning("Failed to stop Appium service: %s", e)
            finally:
                self._owns_appium = False
        if hasattr(self, "_file_handler"):
            detach_run_log(self._file_handler)
