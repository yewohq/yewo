"""Rules for selecting an app from a mobile launcher."""

import unittest

from yewo.agent.grounding import (
    GROUNDING_COORDINATES,
    GROUNDING_DESCRIBE,
    GROUNDING_MARKS,
)
from yewo.agent.plan.planner import DevicePlanner


class AppLaunchPromptTests(unittest.TestCase):
    def test_every_grounding_mode_refuses_unlabelled_app_identity_guesses(self):
        planner = DevicePlanner.__new__(DevicePlanner)

        for grounding in (GROUNDING_COORDINATES, GROUNDING_DESCRIBE, GROUNDING_MARKS):
            planner.grounding = grounding
            prompt = planner._action_constraints()

            self.assertIn("do not infer an app's identity", prompt)
            self.assertIn("visible name matches", prompt)
            self.assertIn("open the app drawer first", prompt)
            self.assertIn("app-drawer search only if", prompt)
            self.assertIn("Home Screen Search", prompt)
            self.assertIn("App Library", prompt)
            self.assertIn("treat that icon as disproven", prompt)


if __name__ == "__main__":
    unittest.main()
