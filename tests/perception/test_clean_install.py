import subprocess
import tempfile
import unittest
from pathlib import Path

from tools.perception_eval.clean_install import run_clean_install


class CleanInstallTests(unittest.TestCase):
    def test_isolates_installs_diagnoses_and_runs_production_inference(self):
        commands = []

        def runner(command, _environment):
            commands.append(list(command))
            return subprocess.CompletedProcess(command, 0, "ok", "")

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            wheel = root / "yewo-1-py3-none-any.whl"
            wheel.write_bytes(b"wheel")
            result = run_clean_install(wheel, root / "result.json", runner=runner)

        self.assertTrue(result["passed"])
        self.assertEqual(len(commands), 5)
        self.assertEqual(commands[0][1:3], ["-m", "venv"])
        self.assertEqual(commands[1][-2:], ["install", f"{wheel}[vision]"])
        self.assertIn("doctor", commands[2])
        self.assertIn("yewo.agent.perception.rapidocr_diagnostics", commands[3])
        self.assertIn("ImageDescriptor", commands[4][-1])


if __name__ == "__main__":
    unittest.main()
