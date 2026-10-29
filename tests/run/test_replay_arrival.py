"""Replay waits for observed navigation, not a blanket learned delay."""

import unittest
from unittest.mock import Mock, patch

from yewo.agent.actuation.actions import TapAction, TypeAction
from yewo.agent.actuation import safety as action_safety
from yewo.agent.memory import replay_store, replay_walk
from yewo.agent.run.record import StepLog

from test_replay_walk import (
    PASSWORD,
    VALUES,
    Device,
    Orchestrator,
    centre_of,
    element,
    identity,
    scan,
    screen_read,
)


class ReplayArrivalTests(unittest.TestCase):
    def setUp(self):
        self.names = replay_store.ValueNames(VALUES)
        self.device = Device()
        self.log = StepLog()
        self._real = action_safety.ActionOrchestrator
        action_safety.ActionOrchestrator = Orchestrator
        self.addCleanup(setattr, action_safety, "ActionOrchestrator", self._real)

    def _trace(self, before, after, action):
        return replay_walk.build_trace(
            identity(),
            [action],
            [((before, scan(len(before))), (after, scan(len(after))))],
            self.names,
        )

    def test_a_tap_only_waits_while_its_destination_has_not_arrived(self):
        sign_in = element("Sign in")
        trace = self._trace(
            [sign_in],
            [element("Orders")],
            TapAction(coordinates=centre_of(sign_in), reason="r"),
        )
        transition = trace.transitions[0].model_copy(
            update={
                "arrival_ms": 5000,
                "destination": trace.transitions[0].destination.model_copy(
                    update={"phash": "destination"}
                ),
            }
        )
        settle = Mock(side_effect=["p", "destination"])

        with patch("yewo.agent.memory.replay_walk.time.sleep") as sleep:
            attempt = replay_walk.take_cached_step(
                transition,
                screen_read([sign_in]),
                self.device,
                settle,
                VALUES,
                self.names,
                self.log,
                1,
                quick_phash=lambda: "destination",
                same_build=True,
            )

        self.assertIsNotNone(attempt.step)
        sleep.assert_called_once_with(replay_walk.ARRIVAL_POLL_SECONDS)
        self.assertEqual(settle.call_count, 2)

    def test_a_non_navigating_action_never_sleeps_for_the_learned_window(self):
        field = element(PASSWORD, "field")
        trace = self._trace(
            [field],
            [field],
            TypeAction(text=PASSWORD, coordinates=centre_of(field), reason="r"),
        )
        transition = trace.transitions[0].model_copy(update={"arrival_ms": 5000})

        with patch("yewo.agent.memory.replay_walk.time.sleep") as sleep:
            attempt = replay_walk.take_cached_step(
                transition,
                screen_read([field]),
                self.device,
                lambda: "p",
                VALUES,
                self.names,
                self.log,
                1,
                quick_phash=lambda: "p",
            )

        self.assertIsNotNone(attempt.step)
        sleep.assert_not_called()


if __name__ == "__main__":
    unittest.main()
