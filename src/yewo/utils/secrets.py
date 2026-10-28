"""Keeping declared secrets out of what Yewo writes down.

The planner writes free text that can land in logs, trails, and reports. Those
records outlive the run. A journey says which values are sensitive, and Yewo
removes those values from persisted content.

Persisted text uses the journey's `{{name}}` spelling. Text sent to an external
model uses `$NAME`. The device resolves that model token only when a type
action is dispatched. Exact exposure checks stay local so masking model input
does not hide a security failure.
"""

from dataclasses import dataclass
from typing import Callable, Optional
import re
from enum import Enum

MIN_LENGTH = 3


def model_token(name: str) -> str:
    """The placeholder an external model sees for one declared name."""
    normalized = re.sub(r"[^A-Za-z0-9]+", "_", name).strip("_").upper()
    return "$" + (normalized or "SECRET")


class SecretDisplay(str, Enum):
    """How much of a declared value an external model may receive."""

    HIDE = "hide"
    LAST4 = "last4"
    SHOW = "show"


class ScreenshotMaskType(str, Enum):
    """The meaning of an overlay that Yewo adds to a screenshot."""

    SECRET = "yewo_secret_mask"


class AppValueRendering(str, Enum):
    """What local device evidence says the app rendered for a typed value."""

    MASKED = "masked"
    EXPOSED = "exposed"
    UNKNOWN = "unknown"


SECRET_MASK_GUIDANCE = """A `yewo_secret_mask` is a moss overlay with a Yewo seal and a declaration
token such as `$DECLARATION_NAME`. Yewo adds this overlay before an image
leaves the process. It is transport metadata, not app content. Do not treat it
as a button, field value, placeholder, overlap, layout defect, blank screen, or
crash. Do not reconstruct the covered characters. Local device evidence can
state how the app rendered the covered value. Use that evidence only for the
declaration token it names. Without local evidence, the overlay does not prove
whether the app masked or exposed the value."""


@dataclass(frozen=True)
class ProtectedRegion:
    """A screen region known to have received a declared secret."""

    mask_type: ScreenshotMaskType
    bounds: Optional[tuple[int, int, int, int]]
    artifact_labels: tuple[str, ...]
    model_labels: tuple[str, ...]
    app_rendering: AppValueRendering = AppValueRendering.UNKNOWN


class SecretScrubber:
    """Replaces a journey's declared secret values with the name that declared them.

    `{{access_code}}` keeps the declared name in the report. It does not write the
    value itself.
    """

    def __init__(self):
        self._named: dict[str, str] = {}
        self._values_by_token: dict[str, str] = {}
        self._display_by_value: dict[str, SecretDisplay] = {}
        self._allow_model_exposure = False
        self._protected_regions: list[ProtectedRegion] = []
        self.protected_region_callback: Optional[Callable[[ProtectedRegion], None]] = None

    def learn_secrets(
        self,
        secrets: Optional[dict],
        displays: Optional[dict[str, str]] = None,
        *,
        allow_model_exposure: bool = False,
    ) -> None:
        """Takes the name-to-value pairs a journey declared sensitive."""
        self._allow_model_exposure = allow_model_exposure
        for name, value in (secrets or {}).items():
            if value and len(value) >= MIN_LENGTH:
                self._named[value] = name
                display = SecretDisplay((displays or {}).get(name, SecretDisplay.HIDE))
                self._display_by_value[value] = display
                self._values_by_token[model_token(name)] = value
                self._values_by_token[self._model_label(name, value, display)] = value

    def _effective_display(self, value: str) -> SecretDisplay:
        """The policy in force after the command-line safety gate."""
        if not self._allow_model_exposure:
            return SecretDisplay.HIDE
        return self._display_by_value.get(value, SecretDisplay.HIDE)

    @classmethod
    def _model_label(cls, name: str, value: str, display: SecretDisplay) -> str:
        token = model_token(name)
        if display is SecretDisplay.LAST4:
            return f"{token}[…{value[-4:]}]"
        return token

    def _masks(self) -> list[tuple[str, str]]:
        pairs = [(value, "{{" + name + "}}") for value, name in self._named.items()]
        return sorted(pairs, key=lambda pair: len(pair[0]), reverse=True)

    def scrub(self, text: Optional[str]) -> str:
        """Masks every declared secret value found in persisted text."""
        if not text:
            return text or ""
        for value, mask in self._masks():
            text = text.replace(value, mask)
        return text

    def for_model(self, text: Optional[str]) -> str:
        """Replaces secret values with tokens before text leaves the process."""
        if not text:
            return text or ""
        masked = text
        for value, name in sorted(
            self._named.items(), key=lambda pair: len(pair[0]), reverse=True
        ):
            display = self._effective_display(value)
            replacement = (
                value
                if display is SecretDisplay.SHOW
                else self._model_label(name, value, display)
            )
            masked = masked.replace(value, replacement)
        return masked

    def resolve_model_text(self, text: Optional[str]) -> str:
        """Resolves placeholders in a type action immediately before device input."""
        resolved = text or ""
        for token, value in sorted(
            self._values_by_token.items(), key=lambda pair: len(pair[0]), reverse=True
        ):
            resolved = resolved.replace(token, value)
        return resolved

    def model_labels_in(self, text: Optional[str]) -> list[str]:
        """The placeholder labels for declared values present in detected text."""
        if not text:
            return []
        return [
            self._model_label(
                name,
                value,
                self._effective_display(value),
            )
            for value, name in self._named.items()
            if value in text
            and self._effective_display(value) is not SecretDisplay.SHOW
        ]

    def labels_for_typed_text(self, text: Optional[str]) -> list[str]:
        """Safe labels for secrets a type action is about to put on screen."""
        if not text:
            return []
        labels = []
        for value, name in self._named.items():
            display = self._effective_display(value)
            if display is SecretDisplay.SHOW:
                continue
            tokens = [
                token for token, token_value in self._values_by_token.items()
                if token_value == value
            ]
            if value in text or any(token in text for token in tokens):
                labels.append(self._model_label(name, value, display))
        return list(dict.fromkeys(labels))

    def declared_labels_for_typed_text(self, text: Optional[str]) -> list[str]:
        """Labels for every declared secret in typed text, under any policy."""
        if not text:
            return []
        labels = []
        for value, name in self._named.items():
            tokens = [
                token for token, token_value in self._values_by_token.items()
                if token_value == value
            ]
            if value in text or any(token in text for token in tokens):
                labels.append(model_token(name))
        return list(dict.fromkeys(labels))

    def protect_typed_region(
        self, text: Optional[str], point, elements, bounds=None
    ) -> bool:
        """Remember where a declared value is typed, without depending on OCR.

        Nested detections are normal. The smallest box containing the action
        point is the field itself. If perception found no containing box, the
        next model-bound image is masked in full rather than risking exposure.
        """
        artifact_labels = tuple(self.declared_labels_for_typed_text(text))
        if not artifact_labels:
            return False
        model_labels = tuple(self.labels_for_typed_text(text))
        resolved_bounds = None
        if bounds is not None:
            resolved_bounds = (bounds.x_min, bounds.y_min, bounds.x_max, bounds.y_max)
        else:
            candidates = []
            for element in elements or ():
                box = element.bounding_box
                if box.x_min <= point.x <= box.x_max and box.y_min <= point.y <= box.y_max:
                    area = max(0, box.x_max - box.x_min) * max(
                        0, box.y_max - box.y_min
                    )
                    candidates.append((area, box))
            if candidates:
                box = min(candidates, key=lambda candidate: candidate[0])[1]
                resolved_bounds = (box.x_min, box.y_min, box.x_max, box.y_max)
        region = ProtectedRegion(
            mask_type=ScreenshotMaskType.SECRET,
            bounds=resolved_bounds,
            artifact_labels=artifact_labels,
            model_labels=model_labels,
        )
        if region not in self._protected_regions:
            self._protected_regions.append(region)
            if self.protected_region_callback is not None:
                self.protected_region_callback(region)
        return True

    def protect_typed_bounds(
        self,
        text: Optional[str],
        bounds: tuple[int, int, int, int],
        app_rendering: AppValueRendering = AppValueRendering.UNKNOWN,
    ) -> bool:
        """Add the active field's settled bounds after keyboard reflow."""
        artifact_labels = tuple(self.declared_labels_for_typed_text(text))
        if not artifact_labels:
            return False
        region = ProtectedRegion(
            mask_type=ScreenshotMaskType.SECRET,
            bounds=bounds,
            artifact_labels=artifact_labels,
            model_labels=tuple(self.labels_for_typed_text(text)),
            app_rendering=app_rendering,
        )
        if region not in self._protected_regions:
            self._protected_regions.append(region)
            if self.protected_region_callback is not None:
                self.protected_region_callback(region)
        return True

    def protected_regions(self) -> tuple[ProtectedRegion, ...]:
        """Regions that must remain covered for the rest of this scenario."""
        return tuple(self._protected_regions)

    def app_renderings(self) -> dict[str, AppValueRendering]:
        """Return the strongest local rendering evidence for each model token."""
        priority = {
            AppValueRendering.UNKNOWN: 0,
            AppValueRendering.MASKED: 1,
            AppValueRendering.EXPOSED: 2,
        }
        renderings: dict[str, AppValueRendering] = {}
        for region in self._protected_regions:
            for label in region.model_labels:
                renderings.setdefault(label, AppValueRendering.UNKNOWN)
                current = renderings.get(label, AppValueRendering.UNKNOWN)
                if priority[region.app_rendering] > priority[current]:
                    renderings[label] = region.app_rendering
        return renderings

    def declared_names_in(self, text: Optional[str]) -> tuple[str, ...]:
        """Return declared names explicitly named by natural-language text."""
        if not text:
            return ()
        normalized = re.sub(r"[^a-z0-9]+", " ", text.casefold()).strip()
        padded = f" {normalized} "
        found = []
        for name in self._named.values():
            words = re.sub(r"[^a-z0-9]+", " ", name.casefold()).strip()
            if words and f" {words} " in padded:
                found.append(name)
        return tuple(dict.fromkeys(found))

    def model_label_for_name(self, name: str) -> Optional[str]:
        """Return the safe external label for one declared value name."""
        for value, declared_name in self._named.items():
            if declared_name == name:
                return self._model_label(
                    declared_name,
                    value,
                    self._effective_display(value),
                )
        return None

    def clear_protected_regions(self) -> None:
        """Forget screen locations when a new scenario starts."""
        self._protected_regions.clear()

    def declared_labels_in(self, text: Optional[str]) -> list[str]:
        """Placeholder names for exact declared values, regardless of display policy."""
        if not text:
            return []
        return [
            model_token(name)
            for value, name in self._named.items()
            if value in text
        ]

    def contains_secret(self, text: Optional[str]) -> bool:
        """Whether any declared secret appears in the text, for image masking."""
        if not text:
            return False
        return any(value in text for value in self._named)
