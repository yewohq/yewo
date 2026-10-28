import io
import logging

import imagehash
import numpy as np
from PIL import Image

from yewo.agent.device.session import DeviceSession
from yewo.agent.perception.elements import GUIElement
from yewo.agent.perception.detector import ImageDescriptor
from yewo.agent.perception.elements import NAVIGATION_THRESHOLD, ScanInfo
from yewo.agent.perception.fusion import PerceptionFrame
from yewo.agent.perception.incremental import (
    MAX_CHANGED_FRACTION,
    MAX_DIRTY_REGIONS,
    PIXEL_DELTA_THRESHOLD,
    plan_incremental_refresh,
)
from yewo.agent.perception.keyboard import KEYBOARD_REGION_ROLE

logger = logging.getLogger(__name__)

# --- Tunable thresholds (adjust per device/app after calibration) ---
# Per-element mean absolute pixel error (0–255) above which an element is dirty.
DIRTY_THRESHOLD = 4.0
# Minimum IoU to consider two same-class elements the same element across frames.
IOU_MATCH_THRESHOLD = 0.70
# If fewer than this fraction of previous elements match, treat as layout shift.
LAYOUT_SHIFT_THRESHOLD = 0.50
# MAE on non-YOLO screen regions above which an uncovered visual change is flagged.
UNCOVERED_THRESHOLD = 2.0


class ScreenPerception:
    """Manages screen perception using YOLO, OCR, and per-element pixel diff.

    Three tiers decide whether the screen changed:

      - a 256-bit pHash, as a coarse navigation detector only
      - per-element bounding-box MAE, for subtle element-level changes
      - raw MAE on regions YOLO did not cover, for surfaces it cannot see into
    """

    def __init__(
        self,
        device_session: DeviceSession,
        model_reference: str | None = None,
    ):
        self.device_session = device_session
        self.image_descriptor = ImageDescriptor(model_reference)

        self._last_hash: imagehash.ImageHash | None = None
        self._last_screenshot: Image.Image | None = None
        self._last_elements: list[GUIElement] | None = None
        self._last_frame: PerceptionFrame | None = None

        self.last_scan_info: ScanInfo | None = None

    @property
    def current_hash(self) -> "imagehash.ImageHash | None":
        return self._last_hash

    @property
    def current_frame(self) -> PerceptionFrame | None:
        """The latest independent detector/OCR evidence and associations."""
        return self.image_descriptor.last_frame

    def reset(self):
        """Resets all perception state. Call between scenarios."""
        self._last_hash = None
        self._last_screenshot = None
        self._last_elements = None
        self._last_frame = None
        self.image_descriptor.clear_cache()
        self.last_scan_info = None
        logger.info("Perception state reset.")

    def perceive(self) -> tuple[list[GUIElement], Image.Image, Image.Image]:
        """Captures the current screen and returns up-to-date GUI elements.

        Returns:
            (gui_elements, raw_img, annotated_img)
            - gui_elements: detected elements with fresh text where changed
            - raw_img: clean screenshot with no overlays — send this to the LLM
            - annotated_img: YOLO bounding boxes drawn on screenshot — save to disk
        """
        device_control = self.device_session.device_control
        if device_control is None:
            raise RuntimeError("screen perception requires an active device control")
        png_bytes = device_control.get_screenshot()
        raw_img = Image.open(io.BytesIO(png_bytes))
        current_hash = imagehash.phash(raw_img, hash_size=16)
        phash_hex = str(current_hash)
        w, h = raw_img.size
        distance = current_hash - self._last_hash if self._last_hash is not None else None

        if self._last_screenshot is None or self._last_hash is None:
            logger.info("First perception pass — running full YOLO + OCR.")
            return self._full_scan(
                png_bytes,
                raw_img,
                current_hash,
                reason="first_pass",
                phash_distance=distance,
            )

        distance = current_hash - self._last_hash
        logger.debug("pHash distance: %d (threshold: %d)", distance, NAVIGATION_THRESHOLD)
        if distance >= NAVIGATION_THRESHOLD:
            logger.info(
                "pHash distance %d ≥ %d — navigation detected, full re-scan.",
                distance,
                NAVIGATION_THRESHOLD,
            )
            return self._full_scan(
                png_bytes,
                raw_img,
                current_hash,
                reason="navigation",
                phash_distance=distance,
            )

        previous_elements = self._last_elements or []
        previous_frame = self._last_frame
        plan = plan_incremental_refresh(
            self._last_screenshot,
            raw_img,
            previous_elements,
            frozenset({KEYBOARD_REGION_ROLE}),
        )
        partial_refresh = (
            plan.full_scan_reason is None
            and bool(plan.regions)
            and previous_frame is not None
        )
        if partial_refresh:
            assert previous_frame is not None
            curr_elements, annotated_img = self.image_descriptor.describe_incremental(
                png_bytes,
                previous_frame,
                plan.regions,
                plan.reuse_detector,
            )
            mode = "incremental"
            ocr_input_pixels = plan.ocr_input_pixels
            ocr_input_regions = len(plan.regions)
        else:
            curr_elements, annotated_img = self.image_descriptor.describe(png_bytes)
            mode = "cached" if not plan.changed_pixels else "full"
            ocr_input_pixels = 0 if mode == "cached" else w * h
            ocr_input_regions = 0 if mode == "cached" else 1

        matched_pairs, new_elements, removed = self._match_elements(
            previous_elements, curr_elements
        )
        match_rate = len(matched_pairs) / max(len(previous_elements), 1)
        logger.debug(
            "Element diff: %d matched, %d new, %d removed (match rate %.0f%%).",
            len(matched_pairs),
            len(new_elements),
            len(removed),
            match_rate * 100,
        )

        if match_rate < LAYOUT_SHIFT_THRESHOLD:
            logger.info(
                "IoU match rate %.0f%% < %.0f%% — layout shift detected, full re-scan.",
                match_rate * 100,
                LAYOUT_SHIFT_THRESHOLD * 100,
            )
            if partial_refresh:
                curr_elements, annotated_img = self.image_descriptor.describe(png_bytes)
                matched_pairs, new_elements, removed = self._match_elements(
                    previous_elements, curr_elements
                )
                match_rate = len(matched_pairs) / max(len(previous_elements), 1)
                mode = "full"
                ocr_input_pixels = w * h
                ocr_input_regions = 1
                plan = plan.__class__(
                    regions=plan.regions,
                    changed_pixels=plan.changed_pixels,
                    changed_fraction=plan.changed_fraction,
                    full_scan_reason="detector_structure_changed",
                )
            if match_rate < LAYOUT_SHIFT_THRESHOLD:
                self._update_state(raw_img, curr_elements, current_hash)
                self.last_scan_info = ScanInfo(
                    mode="full",
                    reason="layout_shift",
                    phash=phash_hex,
                    phash_distance=distance,
                    screen_width=w,
                    screen_height=h,
                    element_count=len(curr_elements),
                    matched=len(matched_pairs),
                    new=len(new_elements),
                    removed=len(removed),
                    match_rate=round(match_rate, 4),
                    ocr_input_regions=1,
                    ocr_input_pixels=w * h,
                    changed_pixel_fraction=round(plan.changed_fraction, 6),
                    full_scan_fallback=(
                        plan.full_scan_reason or "layout_structure_changed"
                    ),
                    thresholds=self._scan_thresholds(),
                )
                return curr_elements, raw_img, annotated_img

        dirty_elements = []
        for prev_el, curr_el in matched_pairs:
            mae = self._element_mae(self._last_screenshot, raw_img, prev_el.bounding_box)
            if mae > DIRTY_THRESHOLD:
                dirty_elements.append(curr_el)
                logger.debug("Element dirty (MAE=%.2f): %s", mae, self._element_key(prev_el))

        uncovered_mae = self._uncovered_region_mae(self._last_screenshot, raw_img, curr_elements)
        if uncovered_mae > UNCOVERED_THRESHOLD:
            logger.warning(
                "Visual change (MAE=%.2f) in region not covered by YOLO elements. "
                "Check the screenshot for canvas/WebView changes.",
                uncovered_mae,
            )

        if not plan.changed_pixels:
            logger.info("Screen stable — no dirty or newly detected elements.")

        self._update_state(raw_img, curr_elements, current_hash)
        self.last_scan_info = ScanInfo(
            mode=mode,
            reason="ocr_refresh" if plan.changed_pixels else "stable",
            phash=phash_hex,
            phash_distance=distance,
            screen_width=w,
            screen_height=h,
            element_count=len(curr_elements),
            matched=len(matched_pairs),
            new=len(new_elements),
            removed=len(removed),
            match_rate=round(match_rate, 4),
            dirty_ocr_count=len(dirty_elements),
            uncovered_mae=round(uncovered_mae, 4),
            detector_reused=partial_refresh and plan.reuse_detector,
            ocr_input_regions=ocr_input_regions,
            ocr_input_pixels=ocr_input_pixels,
            changed_pixel_fraction=round(plan.changed_fraction, 6),
            incremental_surface_roles=tuple(sorted(plan.surface_roles)),
            full_scan_fallback=plan.full_scan_reason,
            thresholds=self._scan_thresholds(),
        )
        return curr_elements, raw_img, annotated_img

    def _scan_thresholds(self) -> dict[str, float | int]:
        return {
            "navigation_phash": NAVIGATION_THRESHOLD,
            "dirty_mae": DIRTY_THRESHOLD,
            "iou_match": IOU_MATCH_THRESHOLD,
            "layout_shift_match_rate": LAYOUT_SHIFT_THRESHOLD,
            "uncovered_mae": UNCOVERED_THRESHOLD,
            "pixel_delta": PIXEL_DELTA_THRESHOLD,
            "max_changed_fraction": MAX_CHANGED_FRACTION,
            "max_dirty_regions": MAX_DIRTY_REGIONS,
        }

    def _full_scan(
        self,
        png_bytes: bytes,
        raw_img: Image.Image,
        current_hash: imagehash.ImageHash,
        reason: str,
        phash_distance: int | None = None,
        matched: int | None = None,
        new: int | None = None,
        removed: int | None = None,
        match_rate: float | None = None,
    ) -> tuple[list[GUIElement], Image.Image, Image.Image]:
        """Runs the complete YOLO + OCR pipeline and updates state."""
        gui_elements, annotated_img = self.image_descriptor.describe(png_bytes)
        logger.info("Full scan complete: %d elements detected.", len(gui_elements))
        self._update_state(raw_img, gui_elements, current_hash)
        w, h = raw_img.size
        self.last_scan_info = ScanInfo(
            mode="full",
            reason=reason,
            phash=str(current_hash),
            phash_distance=phash_distance,
            screen_width=w,
            screen_height=h,
            element_count=len(gui_elements),
            matched=matched,
            new=new,
            removed=removed,
            match_rate=round(match_rate, 4) if match_rate is not None else None,
            ocr_input_regions=1,
            ocr_input_pixels=w * h,
            thresholds=self._scan_thresholds(),
        )
        return gui_elements, raw_img, annotated_img

    def _update_state(
        self,
        raw_img: Image.Image,
        elements: list[GUIElement],
        current_hash: imagehash.ImageHash,
    ) -> None:
        self._last_screenshot = raw_img
        self._last_elements = elements
        self._last_hash = current_hash
        self._last_frame = (
            self.image_descriptor.last_frame.model_copy(deep=True)
            if self.image_descriptor.last_frame is not None
            else None
        )

    def _match_elements(
        self,
        prev: list[GUIElement],
        curr: list[GUIElement],
    ) -> tuple[list[tuple[GUIElement, GUIElement]], list[GUIElement], list[GUIElement]]:
        """Greedy IoU-based matching of elements across frames by class + position.

        Returns:
            matched_pairs: (prev_el, curr_el) for elements in both frames
            new_elements:  current elements with no match before
            removed:       previous elements with no match now
        """
        matched_pairs = []
        unmatched_curr = list(curr)
        unmatched_prev = list(prev)

        for prev_el in prev:
            best_iou, best_match = 0.0, None
            for curr_el in unmatched_curr:
                if prev_el.element != curr_el.element:
                    continue
                iou = self._compute_iou(prev_el.bounding_box, curr_el.bounding_box)
                if iou > best_iou:
                    best_iou, best_match = iou, curr_el

            if best_iou >= IOU_MATCH_THRESHOLD and best_match is not None:
                matched_pairs.append((prev_el, best_match))
                unmatched_curr.remove(best_match)
                unmatched_prev.remove(prev_el)

        return matched_pairs, unmatched_curr, unmatched_prev

    def _compute_iou(self, a, b) -> float:
        x_left = max(a.x_min, b.x_min)
        y_top = max(a.y_min, b.y_min)
        x_right = min(a.x_max, b.x_max)
        y_bottom = min(a.y_max, b.y_max)
        if x_right <= x_left or y_bottom <= y_top:
            return 0.0
        intersection = (x_right - x_left) * (y_bottom - y_top)
        area_a = (a.x_max - a.x_min) * (a.y_max - a.y_min)
        area_b = (b.x_max - b.x_min) * (b.y_max - b.y_min)
        union = area_a + area_b - intersection
        return intersection / union if union > 0 else 0.0

    def _element_mae(self, prev_img: Image.Image, curr_img: Image.Image, bbox) -> float:
        """Mean absolute pixel error within an element's bounding box crop."""
        box = (bbox.x_min, bbox.y_min, bbox.x_max, bbox.y_max)
        prev_arr = np.array(prev_img.crop(box)).astype(float)
        curr_arr = np.array(curr_img.crop(box)).astype(float)
        if prev_arr.size == 0:
            return 0.0
        return float(np.mean(np.abs(prev_arr - curr_arr)))

    def _uncovered_region_mae(
        self, prev_img: Image.Image, curr_img: Image.Image, elements: list[GUIElement]
    ) -> float:
        """MAE across screen pixels not covered by any YOLO bounding box (Option D)."""
        prev_arr = np.array(prev_img).astype(float)
        curr_arr = np.array(curr_img).astype(float)
        h, w = prev_arr.shape[:2]
        mask = np.ones((h, w), dtype=bool)
        for el in elements:
            bb = el.bounding_box
            mask[bb.y_min : bb.y_max, bb.x_min : bb.x_max] = False
        if not mask.any():
            return 0.0
        diff = np.abs(prev_arr - curr_arr)
        return float(np.mean(diff[mask]))

    def _element_key(self, el: GUIElement) -> str:
        return f"{el.element}_{el.bounding_box.x_min}_{el.bounding_box.y_min}"
