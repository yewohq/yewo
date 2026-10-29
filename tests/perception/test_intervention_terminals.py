"""Terminal branches that must not leak an open control discovery."""

from types import SimpleNamespace
from unittest.mock import patch

from yewo.agent.grounding.grid import Candidate
from yewo.agent.perception.elements import BoundingBox, GUIElement
from yewo.agent.run.control_discovery import (
    ControlDiscoveryState,
    ControlDiscoveryTransaction,
)
from yewo.agent.step import intervention


def test_a_detection_dropped_during_numbering_closes_the_transaction():
    detection = GUIElement(
        element="Control",
        type="yolo_detection",
        confidence=0.9,
        bounding_box=BoundingBox(x_min=300, y_min=1180, x_max=800, y_max=1260),
    )
    plan = SimpleNamespace(
        intent="activate",
        candidates=(Candidate(1, (250, 1150, 850, 1290), detection),),
        description="missing control",
        region="middle_left",
        crop=(200, 1100, 900, 1300),
        scale=2,
        dropped_cells=0,
        derived=None,
    )
    snapshot = SimpleNamespace(intervention=plan, view_note=None)
    transaction = ControlDiscoveryTransaction(
        source_phash="screen",
        region="middle_left",
        description="missing control",
        intent="activate",
    )
    ctx = SimpleNamespace(control_discovery=transaction)

    with patch.object(intervention, "_renumber_planner_screen", return_value=None):
        record = intervention.resolve(snapshot, ctx, SimpleNamespace())

    assert record["directive"] is intervention.InterventionDirective.STOP
    assert transaction.state is ControlDiscoveryState.FAILED
