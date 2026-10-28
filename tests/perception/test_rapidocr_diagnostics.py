"""RapidOCR installation failures remain isolated and actionable."""

import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from yewo.agent.perception import rapidocr_diagnostics as diagnostics
from yewo.agent.perception.rapidocr_adapter import RAPIDOCR_MODEL_FILES


def completed(payload: dict, returncode: int = 0) -> subprocess.CompletedProcess:
    output = f"library log\n{diagnostics.DIAGNOSTIC_PREFIX}{json.dumps(payload)}\n"
    return subprocess.CompletedProcess(["python"], returncode, output, "")


class ParentProcessTests(unittest.TestCase):
    def test_a_successful_smoke_test_returns_runtime_identity(self):
        payload = {
            "category": "ok",
            "detail": "smoke test read 1 text line(s)",
            "versions": {"rapidocr": "3.10.0", "onnxruntime": "1.31.1"},
            "providers": ["CPUExecutionProvider"],
            "identity": "rapidocr@test",
            "recognized_text": ["Yewo 123"],
        }

        result = diagnostics.diagnose_rapidocr(
            smoke_test=True,
            runner=lambda *_args, **_kwargs: completed(payload),
        )

        self.assertTrue(result.is_ok)
        self.assertEqual(result.identity, "rapidocr@test")
        self.assertEqual(result.recognized_text, ("Yewo 123",))
        self.assertEqual(result.hint, "")

    def test_a_child_failure_keeps_its_specific_repair_hint(self):
        payload = {
            "category": "provider_missing",
            "detail": "CPUExecutionProvider is unavailable",
            "providers": ["CoreMLExecutionProvider"],
        }

        result = diagnostics.diagnose_rapidocr(
            runner=lambda *_args, **_kwargs: completed(payload, returncode=1)
        )

        self.assertFalse(result.is_ok)
        self.assertIn("CPU-compatible", result.hint)

    def test_a_native_crash_is_contained_as_a_diagnostic(self):
        result = diagnostics.diagnose_rapidocr(
            runner=lambda *_args, **_kwargs: subprocess.CompletedProcess(
                ["python"], -11, "", "native crash"
            )
        )

        self.assertEqual(result.category, "native_failure")
        self.assertIn("code -11", result.detail)

    def test_a_hung_runtime_is_bounded(self):
        def timeout(*_args, **_kwargs):
            raise subprocess.TimeoutExpired(["python"], 45)

        result = diagnostics.diagnose_rapidocr(runner=timeout)

        self.assertEqual(result.category, "timeout")


class ChildInspectionTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.package_root = Path(directory.name) / "rapidocr"
        self.package_root.mkdir()
        self.origin = self.package_root / "__init__.py"
        self.origin.write_text("", encoding="utf-8")
        model_dir = self.package_root / "models"
        model_dir.mkdir()
        for filename in RAPIDOCR_MODEL_FILES:
            (model_dir / filename).write_bytes(b"model")

    def _inspect(self, versions=None, providers=None):
        installed = versions or {"rapidocr": "3.10.0", "onnxruntime": "1.31.1"}
        onnxruntime = SimpleNamespace(
            get_available_providers=lambda: providers or ["CPUExecutionProvider"]
        )
        with (
            mock.patch.object(diagnostics.metadata, "version", side_effect=installed.__getitem__),
            mock.patch.object(
                diagnostics, "find_spec", return_value=SimpleNamespace(origin=str(self.origin))
            ),
            mock.patch.dict("sys.modules", {"onnxruntime": onnxruntime}),
        ):
            return diagnostics._inspect_installation()

    def test_exact_versions_provider_and_models_pass(self):
        result = self._inspect()

        self.assertEqual(result["category"], "ok")

    def test_a_version_mismatch_is_named(self):
        result = self._inspect(versions={"rapidocr": "4.0.0", "onnxruntime": "1.31.1"})

        self.assertEqual(result["category"], "version_mismatch")
        self.assertIn("expected 3.10.0", result["detail"])

    def test_a_missing_cpu_provider_is_named(self):
        result = self._inspect(providers=["CoreMLExecutionProvider"])

        self.assertEqual(result["category"], "provider_missing")

    def test_a_missing_packaged_model_is_named(self):
        (self.package_root / "models" / RAPIDOCR_MODEL_FILES[0]).unlink()

        result = self._inspect()

        self.assertEqual(result["category"], "models_missing")
        self.assertIn(RAPIDOCR_MODEL_FILES[0], result["detail"])


if __name__ == "__main__":
    unittest.main()
