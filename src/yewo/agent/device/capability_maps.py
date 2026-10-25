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
