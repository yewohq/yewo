"""Reaching a screen is not evidence that its capability can be exercised."""

from yewo.discovery.capability_surface import CapabilitySurfacePredicate
from yewo.discovery.control_semantics import build_control_surface

VIEWPORT = (1080, 2400)

ENTRY_FORM = """<hierarchy>
  <node class="android.widget.LinearLayout" bounds="[0,0][1080,2400]">
    <node class="android.widget.TextView" text="Name" bounds="[40,300][300,360]" />
    <node class="android.widget.EditText" bounds="[40,380][1040,470]" focusable="true" />
    <node class="android.widget.Button" text="Save" bounds="[40,560][1040,660]" clickable="true" />
  </node>
</hierarchy>"""

CONTAINING_SCREEN = """<hierarchy>
  <node class="android.widget.LinearLayout" bounds="[0,0][1080,2400]">
    <node class="android.widget.TextView" text="Schedule" bounds="[40,100][600,160]" />
    <node class="android.widget.Button" text="Add" bounds="[40,300][1040,400]" clickable="true" />
  </node>
</hierarchy>"""


def _entry_predicate() -> CapabilitySurfacePredicate:
    return CapabilitySurfacePredicate(
        predicate_id="entry-form-surface-v1",
        minimum_editable_controls=1,
        minimum_activatable_controls=1,
    )


def test_a_form_offering_its_controls_satisfies_the_capability():
    surface = build_control_surface(ENTRY_FORM, VIEWPORT)

    assert _entry_predicate().is_satisfied_by(surface)


def test_reaching_the_containing_screen_alone_does_not_satisfy_it():
    surface = build_control_surface(CONTAINING_SCREEN, VIEWPORT)

    predicate = _entry_predicate()

    assert not predicate.is_satisfied_by(surface)
    assert "0 editable control(s)" in str(predicate.shortfalls(surface)[0])


def test_a_control_nothing_can_name_again_blocks_the_capability():
    anonymous = ENTRY_FORM.replace(
        '<node class="android.widget.Button" text="Save"',
        '<node class="android.widget.ImageButton"',
    )
    surface = build_control_surface(anonymous, VIEWPORT)

    predicate = _entry_predicate()

    assert not predicate.is_satisfied_by(surface)
    assert "no name and no anchor" in str(predicate.shortfalls(surface)[-1])


def test_a_capability_may_accept_an_unnameable_control_when_it_declares_so():
    anonymous = ENTRY_FORM.replace(
        '<node class="android.widget.Button" text="Save"',
        '<node class="android.widget.ImageButton"',
    )
    surface = build_control_surface(anonymous, VIEWPORT)

    tolerant = CapabilitySurfacePredicate(
        predicate_id="entry-form-surface-v1",
        minimum_editable_controls=1,
        minimum_activatable_controls=1,
        requires_reacquirable_controls=False,
    )

    assert tolerant.is_satisfied_by(surface)


def test_every_shortfall_is_reported_rather_than_the_first():
    surface = build_control_surface(CONTAINING_SCREEN, VIEWPORT)

    demanding = CapabilitySurfacePredicate(
        predicate_id="entry-form-surface-v1",
        minimum_editable_controls=2,
        minimum_activatable_controls=5,
    )

    assert len(demanding.shortfalls(surface)) == 2


def test_a_predicate_demanding_nothing_is_satisfied_by_any_surface():
    surface = build_control_surface(CONTAINING_SCREEN, VIEWPORT)

    assert CapabilitySurfacePredicate(predicate_id="any-v1").is_satisfied_by(surface)
