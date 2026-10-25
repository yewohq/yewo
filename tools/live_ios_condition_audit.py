#!/usr/bin/env python3
"""Verify a physical iPhone condition profile and restore its prior state."""

import argparse
import tempfile
from pathlib import Path

from appium import webdriver
from appium.options.ios import XCUITestOptions
from appium.webdriver.appium_service import AppiumService

from yewo.agent.actuation.actions import NetworkCapability
from yewo.agent.device.appium_runtime import resolve_appium_runtime
from yewo.agent.device.capability_maps import active_ios_condition
from yewo.agent.device.driver_commands import DriverCommands
from yewo.agent.device.resilience import apply_ios_network_profile


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", required=True)
    parser.add_argument("--udid", required=True)
    args = parser.parse_args()

    runtime = resolve_appium_runtime()
    service = AppiumService()
    log = Path(tempfile.gettempdir()) / "yewo-ios-condition-appium.log"
    service.start(
        node=str(runtime.node_binary),
        main_script=str(runtime.appium_script),
        env=runtime.environment,
        args=["--log", str(log), "--log-level", "warn", "--log-no-colors"],
    )
    driver = None
    commands = None
    previous = None
    try:
        options = XCUITestOptions()
        options.platform_name = "iOS"
        options.automation_name = "XCUITest"
        options.device_name = args.device
        options.set_capability("appium:udid", args.udid)
        options.set_capability("appium:bundleId", "com.apple.Preferences")
        options.set_capability("appium:useNewWDA", True)
        driver = webdriver.Remote("http://127.0.0.1:4723", options=options)
        commands = DriverCommands(driver)
        before = commands.condition_inducers()
        if before is None:
            raise RuntimeError("The iPhone did not expose condition profiles")
        previous = active_ios_condition(before)
        receipt = apply_ios_network_profile(
            args.udid, NetworkCapability(mode="slow_3g"), commands
        )
        if not receipt.confirmed:
            raise RuntimeError(str(receipt))
        print(receipt)
        return 0
    finally:
        if commands is not None:
            current_conditions = commands.condition_inducers()
            current = (
                active_ios_condition(current_conditions)
                if current_conditions is not None
                else None
            )
            if current is not None and current != previous:
                commands.disable_condition_inducer()
            if previous is not None and current != previous:
                commands.enable_condition_inducer(*previous)
        if driver is not None:
            driver.quit()
        if service.is_running:
            service.stop()


if __name__ == "__main__":
    raise SystemExit(main())
