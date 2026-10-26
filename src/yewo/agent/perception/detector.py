"""
Vision perception subsystem mapping screenshots to semantic GUI elements.

This module processes device screenshots using:
1. YOLOv8 object detection (using the `deki-yolo` model from Hugging Face) to find element bounding boxes.
2. EasyOCR running in parallel via a ThreadPoolExecutor to perform OCR only on relevant cropped regions (e.g., Text, Button, and EditText).
3. PIL to generate annotated images with overlaid bounding boxes and detected labels.
"""

import io
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from yewo.agent.perception.elements import BoundingBox, GUIElement
from yewo.utils.vision_deps import require_vision

logger = logging.getLogger(__name__)


OCR_RELEVANT_TYPES = {"Text", "Button", "EditText"}

YOLO_REPO_ID = "orasul/deki-yolo"
YOLO_FILENAME = "best.pt"
YOLO_REVISION = "287f12d1ee2413bbe217fc321483a9931e0e5317"
OCR_LANGUAGES = ["en"]

DETECTION_IMAGE_SIZE = 768
"""Longest side the screenshot is scaled to before detection.

Ultralytics defaults to 640. A tall phone screenshot scaled to that leaves a
30px control about 23px wide, and a grid of them is then found only in part.
Lowering the confidence threshold does not recover them, because the misses
are not low-confidence detections. The model was trained at 640, so this is
raised only as far as small controls require.
"""


class ImageDescriptor:
    def __init__(self):
        require_vision()
        # Deferred: these pull torch, which lives behind the 'vision' extra.
        import easyocr
        from huggingface_hub import hf_hub_download
        from ultralytics import YOLO

        logger.info("Loading deki-yolo model...")
        model_path = hf_hub_download(
            repo_id=YOLO_REPO_ID, filename=YOLO_FILENAME, revision=YOLO_REVISION
        )
        self.yolo = YOLO(model_path)
        self.names = self.yolo.names
        logger.info("✓ Model loaded")
        logger.info("Loading EasyOCR...")
        self.reader = easyocr.Reader(OCR_LANGUAGES, gpu=False)
        logger.info("✓ OCR loaded")

    def describe(
        self,
        png_bytes: bytes,
        box_color: tuple = (0, 255, 0),
        thickness: int = 2,
    ) -> tuple[list[GUIElement], Image.Image]:
        image = Image.open(io.BytesIO(png_bytes)).convert("RGB")
        result = self.yolo(image, imgsz=DETECTION_IMAGE_SIZE)[0]
        names = result.names if hasattr(result, "names") else self.names
        gui_elements = self._get_yolo_gui_elements(result, names)

        # Run OCR in parallel
        ocr_targets = [el for el in gui_elements if el.element in OCR_RELEVANT_TYPES]
        if ocr_targets:
            with ThreadPoolExecutor() as executor:
                futures = {
                    executor.submit(self._extract_text, image, el): el
                    for el in ocr_targets
                }
                for future in as_completed(futures):
                    el = futures[future]
                    el.text = future.result()

        return gui_elements, self._draw_gui_elements(
            image, gui_elements, box_color, thickness
        )

    def run_ocr_on_elements(self, image: Image.Image, elements: list[GUIElement]) -> None:
        """Re-runs OCR on the provided elements in-place without re-running YOLO.

        Used after per-element pixel diff identifies dirty elements — bounding boxes
        are already known from the previous detection pass.
        """
        targets = [el for el in elements if el.element in OCR_RELEVANT_TYPES]
        if not targets:
            return
        with ThreadPoolExecutor() as executor:
            futures = {
                executor.submit(self._extract_text, image, el): el
                for el in targets
            }
            for future in as_completed(futures):
                futures[future].text = future.result()

    def _extract_text(self, image: Image.Image, element: GUIElement) -> str:
        x1, y1, x2, y2 = element.bounding_box
        pad, (w, h) = 4, image.size
        crop = image.crop(
            (max(0, x1 - pad), max(0, y1 - pad), min(w, x2 + pad), min(h, y2 + pad))
        )
        results = self.reader.readtext(np.array(crop), detail=0)
        return " ".join(results).strip()

    def _get_yolo_gui_elements(self, result, names) -> list[GUIElement]:
        if result.boxes is None:
            return []
        xyxy = result.boxes.xyxy.cpu().numpy().astype(int)
        clss = result.boxes.cls.cpu().numpy().astype(int)
        confs = result.boxes.conf.cpu().numpy()
        elements = []
        for box, cls_id, conf in zip(xyxy, clss, confs):
            x_min, y_min, x_max, y_max = box
            class_name = (
                names.get(cls_id, str(cls_id))
                if isinstance(names, dict)
                else names[cls_id]
            )
            elements.append(
                GUIElement(
                    element=class_name,
                    bounding_box=BoundingBox(
                        x_min=int(x_min),
                        y_min=int(y_min),
                        x_max=int(x_max),
                        y_max=int(y_max),
                    ),
                    confidence=float(conf),
                    type="yolo_detection",
                )
            )
        return elements

    def _draw_gui_elements(self, image, elements, box_color, thickness) -> Image.Image:
        image = image.copy()
        draw = ImageDraw.Draw(image)
        w, h = image.size
        scale_ref = max(w, h)
        t = max(1, int(thickness * (scale_ref / 1000.0)))
        font_size = max(10, int(14 * (scale_ref / 1200.0)))
        try:
            font = ImageFont.truetype("arial.ttf", font_size)
        except IOError:
            font = ImageFont.load_default()

        for item in elements:
            x1, y1, x2, y2 = item.bounding_box
            label_text = f"{item.element} {item.confidence:.2f}"
            draw.rectangle([x1, y1, x2, y2], outline=box_color, width=t)
            bbox = draw.textbbox((0, 0), label_text, font=font)
            tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
            text_x, text_y = max(0, x2 - tw - 3), max(3, y1 - th - 5)
            draw.rectangle(
                [text_x - 2, text_y - 2, text_x + tw + 2, text_y + th + 2],
                fill=(0, 0, 0),
            )
            draw.text((text_x, text_y), label_text, fill=(255, 255, 255), font=font)
        return image
