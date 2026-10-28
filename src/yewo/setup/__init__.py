"""Installers behind `yewo setup`."""

from yewo.setup.environment import Environment, InstallKind, detect_environment
from yewo.setup.steps import (
    SetupContext,
    StepResult,
    ensure_env_file,
    install_appium,
    install_applesimutils,
    install_driver,
    install_vision,
    prefetch_detector,
    verify_rapidocr,
)

__all__ = [
    "Environment",
    "InstallKind",
    "SetupContext",
    "StepResult",
    "detect_environment",
    "ensure_env_file",
    "install_appium",
    "install_applesimutils",
    "install_driver",
    "install_vision",
    "prefetch_detector",
    "verify_rapidocr",
]
