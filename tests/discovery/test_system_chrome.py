"""Platform-owned regions are normalized without app-specific coordinates."""

from yewo.discovery.system_chrome import android_system_chrome, ios_system_chrome


def test_android_uses_every_valid_system_bar_rectangle():
    bars = {
        "statusBar": {"x": 0, "y": 0, "width": 400, "height": 24},
        "navigationBar": {"x": 0, "y": 776, "width": 400, "height": 24},
    }

    assert android_system_chrome(bars, (400, 800)) == (
        (0, 0, 400, 24),
        (0, 776, 400, 800),
    )


def test_invalid_android_bar_evidence_adds_no_exclusion():
    assert android_system_chrome({"statusBar": {"height": 24}}, (400, 800)) == ()


def test_ios_excludes_only_the_area_outside_its_reported_viewport():
    viewport = {"x": 0, "y": 44, "width": 400, "height": 722}

    assert ios_system_chrome(viewport, (400, 800)) == (
        (0, 0, 400, 44),
        (0, 766, 400, 800),
    )


def test_a_full_ios_viewport_excludes_no_application_pixels():
    viewport = {"x": 0, "y": 0, "width": 400, "height": 800}

    assert ios_system_chrome(viewport, (400, 800)) == ()
