"""The type ratchet distinguishes errors, not only their total."""

import unittest
from collections import Counter

from tools.type_gate import Diagnostic, _diagnostics, _violations


def report(message: str, rule: str = "reportArgumentType") -> dict:
    return {
        "generalDiagnostics": [
            {
                "severity": "error",
                "file": "/repo/src/yewo/example.py",
                "rule": rule,
                "message": message,
            }
        ]
    }


class DiagnosticBaselineTests(unittest.TestCase):
    def test_another_error_cannot_take_an_old_errors_allowance(self):
        old = Diagnostic("/repo/src/yewo/example.py", "reportArgumentType", "old error")
        baseline = Counter({old: 1})

        violations = _violations(_diagnostics(report("new error")), baseline)

        self.assertEqual(len(violations), 1)
        self.assertEqual(violations[0].diagnostic.message, "new error")

    def test_whitespace_does_not_create_a_different_diagnostic(self):
        compact = _diagnostics(report("One line with detail"))
        wrapped = _diagnostics(report("One line\n  with detail"))

        self.assertEqual(compact, wrapped)


if __name__ == "__main__":
    unittest.main()
