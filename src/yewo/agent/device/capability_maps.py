"""How each platform names the states a capability asks for.

Shared by the code that applies a capability and the code that reads it back,
so the two cannot drift. A receipt compared against a different spelling of
the same state reads as a mismatch, and refuses a capability that worked.
"""

# Appium network connection bitmask: 1 airplane, 2 wifi, 4 data, 6 all.
APPIUM_NETWORK_TYPES = {
    "offline": 1,
    "airplane": 1,
    "wifi_only": 2,
    "data_only": 4,
    "online": 6,
}

# Which radios each mode leaves on, as `mobile: setConnectivity` takes them.
CONNECTIVITY = {
    "offline": {"wifi": False, "data": False, "airplane": True},
    "airplane": {"wifi": False, "data": False, "airplane": True},
    "online": {"wifi": True, "data": True, "airplane": False},
    "wifi_only": {"wifi": True, "data": False, "airplane": False},
    "data_only": {"wifi": False, "data": True, "airplane": False},
}

SLOW_NETWORK_MODES = frozenset({"slow_2g", "slow_3g", "high_latency"})

# Android emulator-console presets. Each profile sets throughput and latency;
# setting only one leaves an undeclared half of the network condition behind.
ANDROID_NETWORK_PROFILES = {
    "slow_2g": {
        "speed": "gsm", "delay": "gprs",
        "upload_kbps": 14.4, "download_kbps": 14.4,
        "min_latency_ms": 150, "max_latency_ms": 550,
    },
    "slow_3g": {
        "speed": "umts", "delay": "umts",
        "upload_kbps": 128.0, "download_kbps": 1920.0,
        "min_latency_ms": 35, "max_latency_ms": 200,
    },
    "high_latency": {
        "speed": "full", "delay": "gprs",
        "upload_kbps": 0.0, "download_kbps": 0.0,
        "min_latency_ms": 150, "max_latency_ms": 550,
    },
}

# Xcode names can vary by release, so these are ranked semantic fragments and
# never opaque profile identifiers. The identifiers are discovered from the
# attached physical device at run time.
IOS_NETWORK_PROFILE_TERMS = {
    "slow_2g": (("2g",), ("edge",), ("very bad", "network")),
    "slow_3g": (("3g",), ("poor", "network"), ("average", "network")),
    "high_latency": (("high", "latency"), ("lossy", "network")),
}


def ios_network_profile(conditions: list[dict], mode: str) -> tuple[str, str] | None:
    """Find the device's profile by meaning, never by a version-specific ID."""
    ranked_terms = IOS_NETWORK_PROFILE_TERMS.get(mode, ())
    for terms in ranked_terms:
        for condition in conditions:
            condition_id = condition.get("identifier")
            for profile in condition.get("profiles", ()):
                profile_id = profile.get("identifier")
                haystack = " ".join(
                    str(profile.get(key, ""))
                    for key in ("name", "description", "identifier")
                ).lower()
                if condition_id and profile_id and all(term in haystack for term in terms):
                    return str(condition_id), str(profile_id)
    return None


def active_ios_condition(conditions: list[dict]) -> tuple[str, str] | None:
    """The active condition/profile IDs, or None when the readable list has none."""
    for condition in conditions:
        condition_id = condition.get("identifier")
        profile_id = condition.get("activeProfile")
        if condition.get("isActive") and condition_id and profile_id:
            return str(condition_id), str(profile_id)
    return None

ANDROID_PERMISSIONS = {
    "camera": "android.permission.CAMERA",
    "location": "android.permission.ACCESS_FINE_LOCATION",
    "microphone": "android.permission.RECORD_AUDIO",
    "contacts": "android.permission.READ_CONTACTS",
    "storage": "android.permission.READ_EXTERNAL_STORAGE",
    "notifications": "android.permission.POST_NOTIFICATIONS",
    "photos": "android.permission.READ_MEDIA_IMAGES",
}

# simctl's own service names. 'notifications' is absent on purpose: simctl has
# no notification service, and the service that would cover it is 'all', which
# grants every other permission too.
PERMISSION_LIMITED = "limited"
"""Access to part of the photo library, as the read reports it.

A person chooses which photos in a picker, and no command sets that again.
So a scenario that would overwrite one is refused instead.
"""

PERMISSION_UNSET = "unset"
"""No decision has been made. `simctl privacy reset` is the way back to it."""

IOS_PRIVACY_SERVICES = {
    "camera": "camera",
    "microphone": "microphone",
    "photos": "photos",
    "contacts": "contacts",
}
"""The services a decision can be set on and read back from.

Location is absent. simctl splits it into `location` and `location-always`,
and the reader answers `always`, `inuse`, `never` or `unset`. A decision to
allow or deny cannot express any of that, so nothing here can put one back.
"""

TALKBACK_PACKAGE = "com.google.android.marvin.talkback"
TALKBACK = f"{TALKBACK_PACKAGE}/com.google.android.marvin.talkback.TalkBackService"
"""The screen reader, as a component. The package is the one the device
installs, so the preflight and the write name the same thing."""

# What `mobile: queryAppState` answers. 1 is installed and not running.
APP_NOT_RUNNING = 1
