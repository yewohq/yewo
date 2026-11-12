"""One element the taxonomy refuses must not discard the screen it was found on."""

import pytest

from observations.factories import build_observation
from agent.planning.support import StubPlannerModel
from yewo.agent.screen_analysis import (
    ModelScreenAnalyzer,
    ScreenAnalysisError,
    ScreenSemanticRole,
)


def _element(key: str, **updates: object) -> dict[str, object]:
    element: dict[str, object] = {
        "analysis_key": key,
        "name": key.replace("-", " ").title(),
        "kind": {"visual": "text", "semantic": "text"},
        "state": None,
        "confidence": "high",
        "role_alternatives": [],
        "input_surface_key": None,
        "parent_key": None,
        "position_in_parent": None,
        "visual_cues": ["near the top"],
        "region_hint": {"left": 0.1, "top": 0.1, "right": 0.2, "bottom": 0.2},
    }
    element.update(updates)
    return element


def test_one_refused_element_does_not_discard_the_rest_of_the_screen() -> None:
    # A star rating read as text with an icon is refused by the taxonomy. Discarding the whole
    # screen leaves the planner acting on a screen it has no inventory for.
    analyzer = ModelScreenAnalyzer(
        StubPlannerModel(
            {
                "screen_summary": "Restaurant storefront with a dish list.",
                "elements": [
                    _element("unavailable-badge", name="Currently unavailable"),
                    _element(
                        "rating",
                        name="0.0",
                        kind={"visual": "text_and_icon", "semantic": "text"},
                    ),
                    _element(
                        "add-button",
                        name="Add Jollof Rice",
                        kind={"visual": "icon", "semantic": "button"},
                    ),
                ],
            }
        ),
        "screen-analyzer-primary",
        identifier_factory=lambda: "analysis-1",
    )

    analysis = analyzer.analyze(build_observation())

    assert [element.name for element in analysis.elements] == [
        "Currently unavailable",
        "Add Jollof Rice",
    ]


def test_a_screen_with_no_usable_element_is_still_a_failure() -> None:
    analyzer = ModelScreenAnalyzer(
        StubPlannerModel(
            {
                "screen_summary": "Unreadable surface.",
                "elements": [
                    _element("rating", kind={"visual": "text_and_icon", "semantic": "text"})
                ],
            }
        ),
        "screen-analyzer-primary",
        identifier_factory=lambda: "analysis-1",
    )

    with pytest.raises(ScreenAnalysisError, match="no usable element"):
        analyzer.analyze(build_observation())


def test_a_child_of_a_refused_parent_loses_its_parent_scoped_role() -> None:
    # The taxonomy refuses the drawer, so the row it held has no container left to name. A
    # menu_item requires that container, and the role survived the dropped reference.
    analyzer = ModelScreenAnalyzer(
        StubPlannerModel(
            {
                "screen_summary": "Drawer open over a dish list.",
                "elements": [
                    _element("drawer", kind={"visual": "text_and_icon", "semantic": "text"}),
                    _element(
                        "orders-row",
                        name="Orders",
                        kind={"visual": "text", "semantic": "menu_item"},
                        parent_key="drawer",
                        position_in_parent=1,
                    ),
                ],
            }
        ),
        "screen-analyzer-primary",
        identifier_factory=lambda: "analysis-1",
    )

    analysis = analyzer.analyze(build_observation())

    (row,) = analysis.elements
    assert row.name == "Orders"
    assert row.parent_ref is None
    assert row.position_in_parent is None
    assert row.kind.semantic is ScreenSemanticRole.UNKNOWN
    assert row.role_alternatives == (ScreenSemanticRole.MENU_ITEM,)
