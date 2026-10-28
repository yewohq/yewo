"""Read-only installation checks and an isolated RapidOCR smoke test."""

from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import dataclass
from importlib import metadata
from importlib.util import find_spec
from pathlib import Path
from typing import Callable

from yewo.agent.perception.rapidocr_adapter import RAPIDOCR_MODEL_FILES
from yewo.utils.subprocess_runner import run_command
from yewo.utils.vision_deps import INSTALL_HINT, PINNED_VISION_VERSIONS

DIAGNOSTIC_PREFIX = "YEWO_RAPIDOCR_DIAGNOSTIC="
DIAGNOSTIC_TIMEOUT_SECONDS = 45
CPU_PROVIDER = "CPUExecutionProvider"


@dataclass(frozen=True)
class RapidOCRDiagnostic:
    """One isolated diagnosis suitable for setup, doctor, and preflight."""

    category: str
    detail: str
    hint: str = ""
    versions: tuple[tuple[str, str], ...] = ()
    providers: tuple[str, ...] = ()
    identity: str = ""
    recognized_text: tuple[str, ...] = ()

    @property
    def is_ok(self) -> bool:
        return self.category == "ok"


CommandRunner = Callable[..., subprocess.CompletedProcess]


def diagnose_rapidocr(
    *, smoke_test: bool = False, runner: CommandRunner | None = None
) -> RapidOCRDiagnostic:
    """Inspect RapidOCR in a child process and contain native runtime failures."""
    command_runner = runner or run_command
    mode = "smoke" if smoke_test else "inspect"
    try:
        completed = command_runner(
            [sys.executable, "-m", __name__, mode],
            timeout=DIAGNOSTIC_TIMEOUT_SECONDS,
            log_command=False,
        )
    except subprocess.TimeoutExpired:
        return _diagnostic("timeout", "RapidOCR diagnostic timed out")
    except (FileNotFoundError, OSError, subprocess.SubprocessError) as exc:
        return _diagnostic("process_failure", f"RapidOCR diagnostic could not run: {exc}")

    payload = _read_payload(completed.stdout or "")
    if payload is not None:
        diagnostic = _from_payload(payload)
        if not diagnostic.is_ok or completed.returncode == 0:
            return diagnostic
    error = _short_error(completed.stderr or completed.stdout or "")
    detail = f"RapidOCR diagnostic exited with code {completed.returncode}"
    if error:
        detail = f"{detail}: {error}"
    return _diagnostic("native_failure", detail)


def _diagnostic(category: str, detail: str) -> RapidOCRDiagnostic:
    return RapidOCRDiagnostic(category, detail, _hint_for(category))


def _hint_for(category: str) -> str:
    if category == "ok":
        return ""
    if category == "packages_missing":
        return INSTALL_HINT
    if category == "provider_missing":
        return "Reinstall the vision extra with a CPU-compatible ONNX Runtime build."
    if category == "models_missing":
        return "Reinstall the vision extra. The packaged RapidOCR model files are incomplete."
    if category == "timeout":
        return "Run yewo setup again. Check security software if ONNX initialization hangs."
    return "Run yewo setup again to reinstall and test the vision dependencies."


def _read_payload(output: str) -> dict | None:
    for line in reversed(output.splitlines()):
        if not line.startswith(DIAGNOSTIC_PREFIX):
            continue
        try:
            payload = json.loads(line.removeprefix(DIAGNOSTIC_PREFIX))
        except json.JSONDecodeError:
            return None
        return payload if isinstance(payload, dict) else None
    return None


def _from_payload(payload: dict) -> RapidOCRDiagnostic:
    category = str(payload.get("category") or "invalid_diagnostic")
    versions = payload.get("versions") or {}
    return RapidOCRDiagnostic(
        category=category,
        detail=str(payload.get("detail") or "RapidOCR diagnostic returned no detail"),
        hint=_hint_for(category),
        versions=tuple(sorted((str(name), str(value)) for name, value in versions.items())),
        providers=tuple(str(provider) for provider in payload.get("providers") or ()),
        identity=str(payload.get("identity") or ""),
        recognized_text=tuple(str(text) for text in payload.get("recognized_text") or ()),
    )


def _short_error(output: str) -> str:
    lines = [line.strip() for line in output.splitlines() if line.strip()]
    return lines[-1][:300] if lines else ""


def _inspect_installation() -> dict:
    versions: dict[str, str] = {}
    missing = []
    for package in PINNED_VISION_VERSIONS:
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            missing.append(package)
    if missing:
        return {
            "category": "packages_missing",
            "detail": f"missing: {', '.join(missing)}",
            "versions": versions,
        }

    mismatches = [
        f"{package} {versions[package]} (expected {expected})"
        for package, expected in PINNED_VISION_VERSIONS.items()
        if versions[package] != expected
    ]
    if mismatches:
        return {
            "category": "version_mismatch",
            "detail": "; ".join(mismatches),
            "versions": versions,
        }

    try:
        import onnxruntime

        providers = list(onnxruntime.get_available_providers())
    except Exception as exc:  # noqa: BLE001 - this executes in the crash-isolation subprocess
        return {
            "category": "onnx_load_failure",
            "detail": f"ONNX Runtime could not load: {type(exc).__name__}: {exc}",
            "versions": versions,
        }
    if CPU_PROVIDER not in providers:
        return {
            "category": "provider_missing",
            "detail": f"{CPU_PROVIDER} is unavailable",
            "versions": versions,
            "providers": providers,
        }

    spec = find_spec("rapidocr")
    package_dir = Path(spec.origin).parent if spec is not None and spec.origin else None
    model_dir = package_dir / "models" if package_dir is not None else None
    missing_models = [
        filename
        for filename in RAPIDOCR_MODEL_FILES
        if model_dir is None or not (model_dir / filename).is_file()
    ]
    if missing_models:
        return {
            "category": "models_missing",
            "detail": f"missing packaged models: {', '.join(missing_models)}",
            "versions": versions,
            "providers": providers,
        }
    return {
        "category": "ok",
        "detail": "packages, CPU provider, and packaged models are available",
        "versions": versions,
        "providers": providers,
    }


def _run_smoke_test(inspection: dict) -> dict:
    if inspection.get("category") != "ok":
        return inspection
    try:
        from PIL import Image, ImageDraw

        from yewo.agent.perception.rapidocr_adapter import RapidOCRReader

        image = Image.new("RGB", (128, 32), "white")
        ImageDraw.Draw(image).text((4, 4), "Yewo 123", fill="black")
        image = image.resize((512, 128), Image.Resampling.NEAREST)
        result = RapidOCRReader(max_side_len=960).recognize(image)
    except Exception as exc:  # noqa: BLE001 - this executes in the crash-isolation subprocess
        return {
            **inspection,
            "category": "engine_failure",
            "detail": f"RapidOCR inference failed: {type(exc).__name__}: {exc}",
        }
    recognized_text = [line.text for line in result.lines if line.text.strip()]
    if not recognized_text:
        return {
            **inspection,
            "category": "empty_inference",
            "detail": "RapidOCR ran but returned no text for the smoke fixture",
            "identity": result.engine_identity,
        }
    return {
        **inspection,
        "detail": f"smoke test read {len(recognized_text)} text line(s)",
        "identity": result.engine_identity,
        "recognized_text": recognized_text,
    }


def _main() -> int:
    mode = sys.argv[1] if len(sys.argv) > 1 else "inspect"
    inspection = _inspect_installation()
    payload = _run_smoke_test(inspection) if mode == "smoke" else inspection
    print(f"{DIAGNOSTIC_PREFIX}{json.dumps(payload, sort_keys=True)}")
    return 0 if payload.get("category") == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(_main())
