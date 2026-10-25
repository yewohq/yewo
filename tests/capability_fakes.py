"""One valid instance of every capability for dispatch completeness tests."""

from yewo.agent.actuation.actions import (
    AccessibilityCapability,
    AppStateCapability,
    BatteryCapability,
    BiometricCapability,
    CapabilityKind,
    ClipboardCapability,
    DeeplinkCapability,
    DeviceLockCapability,
    InterruptCapability,
    LocaleCapability,
    LocationCapability,
    MediaInjectCapability,
    NetworkCapability,
    OrientationCapability,
    PermissionCapability,
    TimezoneCapability,
)

EVERY_CAPABILITY = {
    CapabilityKind.LOCATION: LocationCapability(latitude=1.0, longitude=2.0),
    CapabilityKind.NETWORK: NetworkCapability(mode="offline"),
    CapabilityKind.BATTERY: BatteryCapability(level=50),
    CapabilityKind.TIMEZONE: TimezoneCapability(timezone="Europe/London"),
    CapabilityKind.LOCALE: LocaleCapability(language="fr", country="FR"),
    CapabilityKind.PERMISSION: PermissionCapability(permission="camera", grant=True),
    CapabilityKind.INTERRUPT: InterruptCapability(type="low_memory"),
    CapabilityKind.BIOMETRIC: BiometricCapability(action="match"),
    CapabilityKind.CLIPBOARD: ClipboardCapability(text="hello"),
    CapabilityKind.DEEPLINK: DeeplinkCapability(url="app://cart"),
    CapabilityKind.APP_STATE: AppStateCapability(action="force_stop"),
    CapabilityKind.ORIENTATION: OrientationCapability(orientation="landscape"),
    CapabilityKind.DEVICE_LOCK: DeviceLockCapability(duration_seconds=0),
    CapabilityKind.MEDIA_INJECT: MediaInjectCapability(local_path=__file__),
    CapabilityKind.ACCESSIBILITY: AccessibilityCapability(enabled=True),
}
