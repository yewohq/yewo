"""The scaffolding a device-log test runs on: one session, one artifact tree.

Its own module rather than part of `fakes`, which builds the values the
boundaries hand to the logic. This builds the tree those values are written
into, and two test files share it.
"""

import tempfile
import unittest
from pathlib import Path

from yewo.agent.artifacts.layout import RunLayout
from yewo.agent.device import device_logs

LOG_SESSION = "s1"


class CaptureCase(unittest.TestCase):
    """A session's worth of draining, under its own artifact tree.

    Beside `LoggingDriver`, because the two are only ever used together.
    """

    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.layout = RunLayout.under(self.root, LOG_SESSION)
        self.layout.create()

    def recorder(self, platform):
        return device_logs.DeviceLogRecorder(self.layout, platform)

    def capture(self, driver, platform):
        """A session that drains once and ends."""
        return self.recorder(platform).finish(driver)

    def files(self):
        return sorted(path.name for path in self.layout.logs.iterdir())

    def by_name(self, captures):
        return {one.log_type: one for one in captures}
