import unittest


class FusedScreenTextPromptTests(unittest.TestCase):
    def test_unmatched_ocr_is_evidence_without_becoming_an_action_target(self):
        from yewo.agent.plan.planner import DevicePlanner

        planner = DevicePlanner.__new__(DevicePlanner)
        prompt = planner._build_prompt(
            goal_lines=["Read the status"],
            gui_elements=[],
            history="",
            trail=None,
            drift_warning=None,
            loop_nudge=None,
            screen_text=("Mostly Clear",),
        )

        self.assertIn("Independent Screen Text (evidence only, not action targets)", prompt)
        self.assertIn("- Mostly Clear", prompt)


if __name__ == "__main__":
    unittest.main()
