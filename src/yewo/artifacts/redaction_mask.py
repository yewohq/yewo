"""One branded mask renderer for every visual artifact."""

from PIL import ImageDraw, ImageFont

from yewo.brand import MOSS, SAND, draw_mark


YEWO_REDACTION_POLICY_ID = "yewo-branded-region-mask"
_MASK_LABEL = "YEWO REDACTED"


def _font(height: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    size = max(10, min(18, height - 8))
    try:
        return ImageFont.truetype("DejaVuSansMono.ttf", size=size)
    except OSError:
        return ImageFont.load_default()


def draw_yewo_redaction_mask(
    drawing: ImageDraw.ImageDraw,
    bounds: tuple[int, int, int, int],
) -> None:
    """Draw an opaque Yewo mask within inclusive pixel bounds."""
    left, top, right, bottom = bounds
    drawing.rectangle(bounds, fill=MOSS)
    width = right - left + 1
    height = bottom - top + 1
    if width < 18 or height < 12:
        return

    mark_size = min(24, height - 6, width // 4)
    mark_left = left + 4
    mark_top = top + (height - mark_size) // 2
    draw_mark(drawing, mark_left, mark_top, mark_size)

    text_left = mark_left + mark_size + 5
    available_width = right - text_left - 3
    font = _font(height)
    if available_width <= 0 or drawing.textlength(_MASK_LABEL, font=font) > available_width:
        return
    text_box = drawing.textbbox((0, 0), _MASK_LABEL, font=font)
    text_y = top + (height - (text_box[3] - text_box[1])) / 2 - text_box[1]
    drawing.text((text_left, text_y), _MASK_LABEL, fill=SAND, font=font)


__all__ = ["YEWO_REDACTION_POLICY_ID", "draw_yewo_redaction_mask"]
