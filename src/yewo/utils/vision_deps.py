"""Access to the optional vision dependencies.

YOLO detection and OCR pull in torch, which is large enough that it is not
part of the base install. These helpers keep the rest of the codebase
importable without them, and turn a missing install into an actionable
message instead of a bare ``ModuleNotFoundError`` surfacing mid-run.
"""

from importlib import metadata
from importlib.util import find_spec

VISION_REQUIREMENTS = (
    ("ultralytics", "ultralytics>=8.0.0"),
    ("huggingface_hub", "huggingface-hub>=0.16.0"),
    ("rapidocr", "rapidocr==3.10.0"),
    ("onnxruntime", "onnxruntime==1.31.1"),
)
VISION_PACKAGES = tuple(module for module, _requirement in VISION_REQUIREMENTS)
VISION_INSTALL_REQUIREMENTS = tuple(requirement for _module, requirement in VISION_REQUIREMENTS)
PINNED_VISION_VERSIONS = {"rapidocr": "3.10.0", "onnxruntime": "1.31.1"}

INSTALL_HINT = (
    "Vision dependencies are not installed. Install them with:\n"
    "  uv sync --extra vision          (from a clone)\n"
    "  pip install 'yewo-ai[vision]'  (installed)"
)


class VisionDependencyError(ImportError):
    """Raised when YOLO/OCR are needed but the vision extra is not installed."""


def missing_vision_packages() -> list[str]:
    """Returns the vision packages that are not importable, in declaration order."""
    return [name for name in VISION_PACKAGES if find_spec(name) is None]


def vision_available() -> bool:
    """True when every optional vision dependency is importable."""
    return not missing_vision_packages()


def mismatched_pinned_vision_packages() -> list[str]:
    """Return pinned packages whose installed versions do not satisfy Yewo."""
    mismatches = []
    for package, expected in PINNED_VISION_VERSIONS.items():
        try:
            installed = metadata.version(package)
        except metadata.PackageNotFoundError:
            continue
        if installed != expected:
            mismatches.append(f"{package} {installed} (expected {expected})")
    return mismatches


def vision_installation_satisfies_contract() -> bool:
    """Return whether setup may safely skip the vision installation command."""
    return vision_available() and not mismatched_pinned_vision_packages()


def require_vision() -> None:
    """Raises VisionDependencyError if the vision extra is not installed."""
    missing = missing_vision_packages()
    if missing:
        raise VisionDependencyError(f"{INSTALL_HINT}\n\nMissing: {', '.join(missing)}")
