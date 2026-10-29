"""Accessibility semantics are stable across platforms and writing systems."""

from yewo.discovery.control_semantics import (
    control_text_digest,
    parse_semantic_controls,
    semantic_control_at,
)


def test_android_nested_label_and_icon_resource_are_projected():
    source = """
    <hierarchy>
      <node class="android.widget.LinearLayout" clickable="true" enabled="true"
            bounds="[0,100][700,220]">
        <node class="android.widget.TextView" text="Edit Schedule"
              bounds="[20,100][680,220]" />
      </node>
      <node class="android.widget.ImageButton" clickable="true" enabled="true"
            resource-id="app:id/fab" bounds="[800,1800][950,1950]" />
    </hierarchy>
    """

    controls = parse_semantic_controls(source)

    assert controls[0].role == "linearlayout"
    assert controls[0].text_digest == control_text_digest("Edit Schedule")
    assert controls[1].role == "imagebutton"
    assert controls[1].text_digest == control_text_digest("app:id/fab")
    assert semantic_control_at(source, (100, 150)) == controls[0]


def test_ios_and_non_latin_accessibility_names_use_the_same_contract():
    source = """
    <AppiumAUT>
      <XCUIElementTypeButton type="XCUIElementTypeButton" label="فتح القائمة"
            enabled="true" hittable="true" x="10" y="20" width="80" height="40" />
      <XCUIElementTypeButton type="XCUIElementTypeButton" name="編集"
            enabled="true" hittable="true" x="10" y="80" width="80" height="40" />
    </AppiumAUT>
    """

    controls = parse_semantic_controls(source)

    assert [control.role for control in controls] == [
        "xcuielementtypebutton",
        "xcuielementtypebutton",
    ]
    assert controls[0].text_digest == control_text_digest("فتح القائمة")
    assert controls[1].text_digest == control_text_digest("編集")
    assert controls[1].bounds == (10, 80, 90, 120)


def test_missing_semantic_name_is_not_promoted_to_reusable_control():
    source = """
    <hierarchy>
      <node class="android.view.View" clickable="true" enabled="true"
            bounds="[0,0][100,100]" />
    </hierarchy>
    """

    assert parse_semantic_controls(source) == ()
    assert semantic_control_at(source, (50, 50)) is None
