"""Covering declared values in persisted and model-bound screenshots."""

import io
import logging
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
from yewo.brand import CLAY, MOSS, SAND, draw_mark
from yewo.agent.perception.elements import NAVIGATION
from yewo.utils.secrets import ScreenshotMaskType


logger = logging.getLogger(__name__)

SECRET_MASK_BACKGROUND = MOSS
SECRET_MASK_FOREGROUND = SAND
SECRET_MASK_ACCENT = CLAY


def _mask_font(height: int):
    """Use a readable bundled font when Pillow supplies one."""
    size = max(10, min(18, height - 8))
    try:
        return ImageFont.truetype("DejaVuSansMono.ttf", size=size)
    except OSError:
        return ImageFont.load_default()


def _fit_label(draw: ImageDraw.ImageDraw, label: str, font, width: int) -> str:
    """Shorten a declaration token to the available mask width."""
    if width <= 0:
        return ""
    if draw.textlength(label, font=font) <= width:
        return label
    suffix = "..."
    shortened = label
    while shortened and draw.textlength(shortened + suffix, font=font) > width:
        shortened = shortened[:-1]
    return shortened + suffix if shortened else ""


def draw_secret_mask(
    draw: ImageDraw.ImageDraw,
    bounds: tuple[int, int, int, int],
    labels: tuple[str, ...],
    mask_type: ScreenshotMaskType,
) -> None:
    """Draw the one branded secret overlay used by images and video frames."""
    if mask_type is not ScreenshotMaskType.SECRET:
        raise ValueError(f"Unsupported screenshot mask type: {mask_type}")
    left, top, right, bottom = bounds
    draw.rectangle(bounds, fill=SECRET_MASK_BACKGROUND)
    width = max(0, right - left)
    height = max(0, bottom - top)
    if width < 18 or height < 12:
        return

    seal_size = min(24, height - 6, width // 4)
    seal_left = left + 4
    seal_top = top + (height - seal_size) // 2
    draw_mark(
        draw,
        seal_left,
        seal_top,
        seal_size,
        foreground=SECRET_MASK_FOREGROUND,
        accent=SECRET_MASK_ACCENT,
    )

    font = _mask_font(height)
    text_left = seal_left + seal_size + 5
    label = _fit_label(draw, ", ".join(labels), font, right - text_left - 4)
    if not label:
        return
    text_box = draw.textbbox((0, 0), label, font=font)
    text_y = top + (height - (text_box[3] - text_box[1])) / 2 - text_box[1]
    draw.text(
        (text_left, text_y),
        label,
        fill=SECRET_MASK_FOREGROUND,
        font=font,
    )


class RedactedImages:
    """Every screenshot this run writes, with its declared secrets covered."""

    def __init__(self, scrubber=None):
        """`scrubber` holds the journey's declared secrets; without one, nothing is masked."""
        self.scrubber = scrubber

    def clear_after_navigation(self, scan_info) -> None:
        """Stop covering a field after the run leaves its screen."""
        if self.scrubber is not None and scan_info and scan_info.reason == NAVIGATION:
            self.scrubber.clear_protected_regions()

    def redact_artifact_image(self, image: Image.Image, elements: list) -> Image.Image:
        """Covers declared secret values before a screenshot is persisted."""
        if self.scrubber is None:
            return image
        safe = image.copy()
        draw = ImageDraw.Draw(safe)
        for element in elements:
            if not self.scrubber.contains_secret(element.text):
                continue
            box = element.bounding_box
            draw_secret_mask(
                draw,
                (box.x_min, box.y_min, box.x_max, box.y_max),
                tuple(self.scrubber.declared_labels_in(element.text)),
                ScreenshotMaskType.SECRET,
            )
        self._draw_protected_regions(draw, image.size)
        return safe

    def _draw_protected_regions(
        self, draw: ImageDraw.ImageDraw, size, *, for_model: bool = False
    ) -> bool:
        """Covers fields Yewo typed into even when OCR cannot read their value."""
        regions = self.scrubber.protected_regions() if self.scrubber is not None else ()
        drawn = False
        for region in regions:
            labels = region.model_labels if for_model else region.artifact_labels
            if not labels:
                continue
            bounds = region.bounds or (0, 0, size[0], size[1])
            draw_secret_mask(draw, bounds, labels, region.mask_type)
            drawn = True
        return drawn

    def redact_model_image(self, data: bytes, elements: list) -> bytes:
        """Covers declared values before screenshot bytes leave the process.

        The label preserves which declared value occupied the field without
        disclosing its characters. If OCR found no declared value, the image
        is returned byte-for-byte so ordinary calls do not pay an image
        re-encode or lose metadata.
        """
        if self.scrubber is None:
            return data
        labelled = [
            (element, self.scrubber.model_labels_in(element.text))
            for element in elements or []
        ]
        labelled = [(element, labels) for element, labels in labelled if labels]
        protected = any(
            region.model_labels for region in self.scrubber.protected_regions()
        )
        if not labelled and not protected:
            return data
        try:
            image = Image.open(io.BytesIO(data)).convert("RGB")
        except Exception as exc:  # noqa: BLE001 - transport redaction must fail closed
            raise ValueError("Could not redact a screenshot containing a secret") from exc
        draw = ImageDraw.Draw(image)
        for element, labels in labelled:
            box = element.bounding_box
            draw_secret_mask(
                draw,
                (box.x_min, box.y_min, box.x_max, box.y_max),
                labels,
                ScreenshotMaskType.SECRET,
            )
        self._draw_protected_regions(draw, image.size, for_model=True)
        output = io.BytesIO()
        image.save(output, format="PNG")
        return output.getvalue()

    def save_redacted_image(self, image: Image.Image, elements: list, path: Path) -> None:
        self.redact_artifact_image(image, elements).save(path)

    def save_redacted_bytes(self, data: bytes, elements: list, path: Path) -> None:
        image = Image.open(io.BytesIO(data))
        self.save_redacted_image(image, elements, path)

    def save_redacted_pair(
        self,
        annotated: Image.Image,
        raw: bytes,
        elements: list,
        annotated_path: Path,
        raw_path: Path,
    ) -> None:
        self.save_redacted_image(annotated, elements, annotated_path)
        self.save_redacted_bytes(raw, elements, raw_path)

    def save_redacted_action_annotation(
        self, annotated: Image.Image, elements: list, action, path: Path
    ) -> bool:
        return self.save_action_annotation(
            self.redact_artifact_image(annotated, elements), action, path
        )

    def annotate(self, png_bytes, x, y, label, color="red"):
        image = Image.open(io.BytesIO(png_bytes))
        draw = ImageDraw.Draw(image)
        px, py = int(x), int(y)
        radius = 40
        draw.ellipse(
            [px - radius, py - radius, px + radius, py + radius], outline=color, width=6
        )
        draw.line([px - radius, py, px + radius, py], fill=color, width=3)
        draw.line([px, py - radius, px, py + radius], fill=color, width=3)
        draw.text((px + radius + 5, py - 10), f"{label} ({x:.0f}, {y:.0f})", fill=color)
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        return buf.getvalue()

    def save_action_annotation(
        self,
        annotated_img: Image.Image,
        action_plan,
        filepath: Path,
    ) -> bool:
        """Annotates the planned action (tap, type, swipe, drag) on the image and saves it.

        Returns:
            True if the image was annotated and saved, False otherwise.
        """
        if not action_plan:
            return False

        action_img = annotated_img.copy()
        draw = ImageDraw.Draw(action_img)
        first_action = action_plan
        action_type_str = first_action.type.value
        highlighted = False

        if hasattr(first_action, "coordinates") and first_action.coordinates:
            px, py = int(first_action.coordinates.x), int(first_action.coordinates.y)
            radius = 25
            draw.ellipse(
                [px - radius, py - radius, px + radius, py + radius],
                outline="red",
                width=5,
            )
            draw.line([px - radius, py, px + radius, py], fill="red", width=3)
            draw.line([px, py - radius, px, py + radius], fill="red", width=3)
            draw.text(
                (px + radius + 5, py - 10),
                f"{action_type_str.upper()} ({px}, {py})",
                fill="red",
            )
            highlighted = True
        elif action_type_str == "swipe" and hasattr(first_action, "start_x"):
            sx, sy = int(first_action.start_x), int(first_action.start_y)
            ex, ey = int(first_action.end_x), int(first_action.end_y)
            draw.ellipse([sx - 10, sy - 10, sx + 10, sy + 10], fill="red")
            draw.line([sx, sy, ex, ey], fill="red", width=5)
            draw.text((sx + 15, sy - 10), f"SWIPE START ({sx}, {sy})", fill="red")
            highlighted = True
        elif action_type_str == "drag_and_drop" and hasattr(first_action, "source_x"):
            sx, sy = int(first_action.source_x), int(first_action.source_y)
            ex, ey = int(first_action.destination_x), int(first_action.destination_y)
            draw.ellipse([sx - 10, sy - 10, sx + 10, sy + 10], fill="red")
            draw.line([sx, sy, ex, ey], fill="red", width=5)
            draw.text((sx + 15, sy - 10), f"DRAG START ({sx}, {sy})", fill="red")
            highlighted = True

        if highlighted:
            filepath.parent.mkdir(parents=True, exist_ok=True)
            action_img.save(filepath)
            logger.info("Saved action location annotation to %s", filepath)
            return True

        return False
