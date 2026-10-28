"""Default values shared by every entrypoint.

Single source of truth for the flag defaults, so the CLI, the testing CLI and
`yewo doctor` cannot drift apart on what "default" means.
"""

LOG_LEVELS = ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL")
"""Verbosities every entrypoint offers, so a subcommand accepts what `yewo` does."""

DEFAULT_PLATFORM = "android"
DEFAULT_DEVICE = "Pixel_8a"
DEFAULT_MODEL = "openrouter/google/gemini-2.5-flash"
DEFAULT_PERCEPTION_MODEL = (
    "hf://BeccaSaka/yewo-ui@43ccac8703302adf4b876a055ee925ba1b444db8"
)
"""Immutable release of Yewo's default detector package."""
DEFAULT_OUTPUT_DIR = "assets"
# The ceiling for any one scenario; each scenario is given what its own
# instructions suggest it needs, below this.
DEFAULT_MAX_STEPS = 40
STEP_BUDGET_FLOOR = 15
STEP_BUDGET_PER_INSTRUCTION = 2
DEFAULT_SETTLE_TIME = 2.0
DEFAULT_APP_PATH = None
DEFAULT_GROUNDING = "marks"
"""How the planner names a target.

Marks numbers every detected element and accepts only a number back. The
point comes from the detection rather than from the model, so a tap cannot
land somewhere nothing was found.
"""
DEFAULT_RECORD = True

DEFAULT_DURABLE_REPLAY = True
"""Re-walk a path a passing run saved, rather than planning it again.

The path reuses action knowledge only. Every run audits its own screens and
earns a fresh verdict. Drift returns the affected action to the planner.
"""

# A cold emulator boot regularly exceeds Appium's 60s default.
AVD_LAUNCH_TIMEOUT_MS = 180_000

# Appium ends a session that sends no command for this long. Its default
# allows for the gap between reading a screen and acting on it; a step here
# waits on several model calls in between.
IDLE_SESSION_TIMEOUT_SECONDS = 300

# Settling waits for the screen to stop changing rather than sleeping blind.
SETTLE_POLL_INTERVAL = 0.35
SETTLE_STABLE_READINGS = 2
FOCUS_POLL_INTERVAL = 0.1
FOCUS_POLL_TIMEOUT = 1.5

# Launching is over once the screen has something on it and has stopped
# changing — an empty body means nothing is drawn yet, an unstable one means a
# splash is still giving way to the first real screen.
LAUNCH_PAINT_TIMEOUT = 15.0
LAUNCH_POLL_INTERVAL = 0.3
LAUNCH_STABLE_READINGS = 2
# How long a screen that never stops moving still counts as launched.
LAUNCH_CONTENT_GRACE = 2.0
BLANK_SCREEN_STDDEV = 2.0

# A model call takes seconds, so the target is re-checked just before acting.
# Deliberately above perception's dirty threshold: re-reading an element is
# cheap, throwing away a plan is not, so only material change should count.
STALE_TARGET_MAE = 12.0
STALE_TARGET_PATCH = 120
