"""The reading a screenshot came from, written beside it."""

import json
from numbers import Integral, Real
from pathlib import Path
from typing import Any, Optional

from yewo.agent.perception.elements import GUIElement, ScanInfo
from yewo.agent.perception.fusion import PerceptionFrame


def _to_json_serializable(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _to_json_serializable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_json_serializable(v) for v in value]
    if isinstance(value, Integral) and not isinstance(value, bool):
        return int(value)
    if isinstance(value, Real) and not isinstance(value, bool):
        return float(value)
    return value


def element_to_dict(el: GUIElement, index: int, scrubber=None) -> dict[str, Any]:
    """Serializes one element, masking declared secrets in its detected text.

    A scan dump is a file, so a password the app happens to be showing would
    otherwise be written out verbatim. The planner reads the element itself,
    never this.
    """
    cx = (el.bounding_box.x_min + el.bounding_box.x_max) // 2
    cy = (el.bounding_box.y_min + el.bounding_box.y_max) // 2
    text = el.text or ""
    return {
        "index": index,
        "element": el.element,
        "type": el.type,
        "confidence": round(el.confidence, 4),
        "text": scrubber.scrub(text) if scrubber else text,
        "bounding_box": el.bounding_box.model_dump(),
        "center": {"x": cx, "y": cy},
    }


def perception_to_dict(frame: PerceptionFrame, scrubber=None) -> dict[str, Any]:
    """Serialize independent model evidence while masking declared secrets."""
    return {
        "schema_version": frame.schema_version,
        "image_digest": frame.image_digest,
        "detector_identity": frame.detector_identity,
        "ocr_identity": frame.ocr_identity,
        "ocr_lines": [
            {
                "index": index,
                "text": scrubber.scrub(line.text) if scrubber else line.text,
                "confidence": round(line.confidence, 4),
                "box": [point.model_dump() for point in line.box],
            }
            for index, line in enumerate(frame.text_lines)
        ],
        "associations": [association.model_dump() for association in frame.associations],
    }


def build_screen_scan_record(
    *,
    step: int,
    timestamp: str,
    scan_info: ScanInfo,
    gui_elements: list[GUIElement],
    screenshot_paths: dict[str, str],
    prev_phash: Optional[str] = None,
    scrubber=None,
    perception_frame: Optional[PerceptionFrame] = None,
) -> dict[str, Any]:
    record = {
        "step": step,
        "timestamp": timestamp,
        "screenshots": screenshot_paths,
        "phash": scan_info.phash,
        "prev_phash": prev_phash,
        "phash_distance": scan_info.phash_distance,
        "screen": {
            "width": scan_info.screen_width,
            "height": scan_info.screen_height,
        },
        "scan": {
            "mode": scan_info.mode,
            "reason": scan_info.reason,
            "element_count": scan_info.element_count,
            "matched": scan_info.matched,
            "new": scan_info.new,
            "removed": scan_info.removed,
            "match_rate": scan_info.match_rate,
            "dirty_ocr_count": scan_info.dirty_ocr_count,
            "uncovered_mae": scan_info.uncovered_mae,
            "keyboard": {
                "native_visible": scan_info.keyboard_native_visible,
                "visual_visible": scan_info.keyboard_visual_visible,
                "effective_visible": scan_info.keyboard_effective_visible,
                "owned_count": scan_info.keyboard_owned_count,
                "state_disagrees": scan_info.keyboard_state_disagrees,
            },
            "incremental": {
                "detector_reused": scan_info.detector_reused,
                "ocr_input_regions": scan_info.ocr_input_regions,
                "ocr_input_pixels": scan_info.ocr_input_pixels,
                "changed_pixel_fraction": scan_info.changed_pixel_fraction,
                "surface_roles": list(scan_info.incremental_surface_roles),
                "full_scan_fallback": scan_info.full_scan_fallback,
            },
            "thresholds": scan_info.thresholds,
        },
        "gui_elements": [element_to_dict(el, idx, scrubber) for idx, el in enumerate(gui_elements)],
    }
    if perception_frame is not None:
        record["perception"] = perception_to_dict(perception_frame, scrubber)
    return record


def write_screen_scan_dump(
    path: Path,
    record: dict[str, Any],
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(_to_json_serializable(record), f, indent=2)
    return path
