#!/usr/bin/env python3
"""Run Phase 7 resilience capabilities against one real Appium session."""

import argparse
from pathlib import Path

from yewo.agent.actuation.actions import (
    AppStateCapability,
    Capability,
    DeviceLockCapability,
    NetworkCapability,
    OrientationCapability,
)
from yewo.agent.device import restoration
from yewo.agent.device.capability_controller import CapabilityController
from yewo.agent.device.driver_commands import DriverCommands
from yewo.agent.device.platform import Platform
from yewo.agent.device.session import DeviceSession


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", required=True)
    parser.add_argument("--app", required=True, type=Path)
    parser.add_argument("--platform", choices=[item.value for item in Platform], required=True)
    args = parser.parse_args()

    platform = Platform(args.platform)
    device_session = DeviceSession(
        device_name=args.device,
        app_path=str(args.app.resolve()),
        platform=platform,
        keep_device=False,
        keep_app=True,
    )
    try:
        device = device_session.start()
        commands = DriverCommands(device.driver)
        initial = commands.orientation()
        wanted = "landscape" if initial != "landscape" else "portrait"
        capabilities: list[Capability] = [
            OrientationCapability(orientation=wanted),
            AppStateCapability(action="background_resume", duration_seconds=0.25),
            AppStateCapability(action="relaunch"),
            DeviceLockCapability(duration_seconds=0.25),
        ]
        if platform == Platform.ANDROID:
            capabilities[1:1] = [
                NetworkCapability(mode="offline"),
                NetworkCapability(mode="online"),
                NetworkCapability(mode="slow_3g"),
            ]
        controller = CapabilityController(device)
        with restoration.preserved(device, capabilities=capabilities):
            for capability in capabilities:
                receipt = controller.execute(capability)
                print(receipt or f"{capability.kind.value}: no readable receipt")
        return 0
    finally:
        device_session.close()


if __name__ == "__main__":
    raise SystemExit(main())
