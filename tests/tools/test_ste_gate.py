"""What the controlled-language gate reads, and what it counts once.

The gate is only as good as its extraction. Two defects it had made it
report on prose nobody wrote and stay silent about prose a model reads: it
counted every docstring twice, and it read neither a prompt constant nor a
model-facing field description.
"""

import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

TOOLS = Path(__file__).resolve().parents[2] / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

import ste_gate  # noqa: E402


def _written(tmp: Path, body: str) -> Path:
    path = tmp / "sample.py"
    path.write_text(body)
    return path


class ExtractionTests(unittest.TestCase):
    def setUp(self):
        import tempfile

        self.tmp = Path(tempfile.mkdtemp())

    def test_a_docstring_is_read_once(self):
        """`get_docstring` dedents and the raw literal does not, so both reach
        the linter and every sentence is counted twice."""
        path = _written(self.tmp, '"""One sentence about one thing."""\n')

        prose = ste_gate.prose_of(path)

        self.assertEqual(prose.count("One sentence about one thing."), 1)

    def test_a_nested_docstring_is_read_once(self):
        path = _written(
            self.tmp,
            'class Thing:\n'
            '    """What this thing is for."""\n\n'
            '    def act(self):\n'
            '        """What acting does."""\n',
        )

        prose = ste_gate.prose_of(path)

        self.assertEqual(prose.count("What this thing is for."), 1)
        self.assertEqual(prose.count("What acting does."), 1)

    def test_a_prompt_constant_is_read(self):
        """A prompt reaches a model, so it is the prose that matters most."""
        path = _written(
            self.tmp,
            'PROMPT = """You are a reader. Answer with one sentence about the '
            'screen in front of you."""\n',
        )

        self.assertIn("You are a reader.", ste_gate.prose_of(path))

    def test_a_model_facing_description_is_read(self):
        path = _written(
            self.tmp,
            'from pydantic import Field\n'
            'x = Field(description="What must be true for this to pass.")\n',
        )

        self.assertIn("What must be true for this to pass.", ste_gate.prose_of(path))

    def test_a_short_assigned_string_is_not_prose(self):
        """A name, a path and a log key are strings too."""
        path = _written(self.tmp, 'LINTER_ENV = "YEWO_STE_LINT"\nSEP = ", "\n')

        self.assertEqual(ste_gate.prose_of(path).strip(), "")

    def test_a_comment_is_read_and_machinery_is_not(self):
        path = _written(
            self.tmp,
            "# A real comment about a real reason.\n"
            "import os  # noqa: F401\n",
        )

        prose = ste_gate.prose_of(path)

        self.assertIn("A real comment about a real reason.", prose)
        self.assertNotIn("noqa", prose)

    def test_a_file_with_no_prose_yields_nothing(self):
        path = _written(self.tmp, "VALUE = 1\n")

        self.assertEqual(ste_gate.prose_of(path).strip(), "")


class NamingTests(unittest.TestCase):
    def test_a_path_outside_the_repository_keeps_its_own_spelling(self):
        """`relative_to` raises on one, and a gate that crashes on the file it
        was handed reports nothing about the files it could have read."""
        outside = Path("/tmp/elsewhere/sample.py")

        self.assertEqual(ste_gate._named(outside), str(outside))

    def test_a_path_inside_the_repository_is_named_relative_to_it(self):
        inside = ste_gate.REPOSITORY_ROOT / "src" / "yewo" / "errors.py"

        self.assertEqual(ste_gate._named(inside), "src/yewo/errors.py")

    def test_a_relative_path_is_left_alone(self):
        self.assertEqual(ste_gate._named(Path("src/yewo/errors.py")), "src/yewo/errors.py")


class BaselineTests(unittest.TestCase):
    def test_the_baseline_names_only_files_that_exist(self):
        """An entry for a deleted file hides nothing and reads as live debt."""
        for name in ste_gate._load_baseline():
            self.assertTrue((ste_gate.REPOSITORY_ROOT / name).is_file(), name)

    def test_no_baseline_entry_allows_nothing(self):
        """A zero entry is the same as absence, so it is noise."""
        for name, allowed in ste_gate._load_baseline().items():
            self.assertGreater(allowed.count, 0, name)

    def test_a_different_finding_cannot_reuse_the_same_count(self):
        before = ste_gate.HardViolation(
            "long-sentence", "Sentence has 26 words.", "26 words", "Old sentence."
        )
        after = ste_gate.HardViolation(
            "long-sentence", "Sentence has 26 words.", "26 words", "New sentence."
        )

        self.assertNotEqual(ste_gate._digest([before]), ste_gate._digest([after]))


if __name__ == "__main__":
    unittest.main()


class LinterResolutionTests(unittest.TestCase):
    """Which linter runs, and why a gate that skips is worth nothing.

    The gate used to pass on a machine without the skill installed. That is
    right for a checkout that cannot speak about the rules, and wrong for CI,
    which would have reported a green prose gate that ran on no file.
    """

    def test_the_copy_beside_the_gate_is_what_ci_runs(self):
        self.assertTrue(
            ste_gate.VENDORED_LINTER.is_file(), ste_gate.VENDORED_LINTER
        )

    def test_the_copy_wins_over_an_installed_skill(self):
        """A gate whose rules depend on the machine answers differently per machine."""
        with mock.patch.object(ste_gate.Path, "is_file", return_value=True), \
             mock.patch.dict(os.environ, {}, clear=True):
            self.assertEqual(ste_gate._find_linter(), ste_gate.VENDORED_LINTER)

    def test_a_named_linter_overrides_the_copy(self):
        """Trying a rule before the copy is updated is a deliberate act."""
        with mock.patch.object(ste_gate.Path, "is_file", return_value=True), \
             mock.patch.dict(os.environ, {ste_gate.LINTER_ENV: "/tmp/other.py"}):
            self.assertEqual(ste_gate._find_linter(), Path("/tmp/other.py"))

    def test_the_vendored_linter_checks_its_own_rules(self):
        """Vendored unchanged, so its self-test is the proof it still works."""
        done = subprocess.run(
            [sys.executable, str(ste_gate.VENDORED_LINTER), "--selftest"],
            capture_output=True,
            text=True,
        )

        self.assertEqual(done.returncode, 0, done.stderr)


class WiderExtractionTests(unittest.TestCase):
    """Two kinds of prose the gate used to walk straight past.

    A prompt built with a value in it, and a prompt handed back rather than
    assigned. Both reach a model, and neither is a docstring, a comment or an
    assignment, so the gate read none of them.
    """

    def _prose(self, body: str) -> str:
        with tempfile.TemporaryDirectory() as directory:
            return ste_gate.prose_of(_written(Path(directory), body))

    def test_the_words_around_a_value_are_read(self):
        prose = self._prose(
            'def refuse(name):\n'
            '    return f"The capability {name} could not be applied on this target."\n'
        )

        self.assertIn("could not be applied on this target", prose)

    def test_what_a_value_holds_is_not_read(self):
        """It is decided at run time, and the words around it are not."""
        prose = self._prose(
            'def refuse(name):\n'
            '    return f"The capability {name} could not be applied on this target."\n'
        )

        self.assertNotIn("name", prose)

    def test_a_sentence_handed_back_is_read(self):
        prose = self._prose(
            'def rendered():\n'
            '    return "Set the device network to offline before the next step."\n'
        )

        self.assertIn("Set the device network to offline", prose)

    def test_a_file_yewo_writes_out_is_not_prose(self):
        """A template carries its own comments, and they are not text to hold."""
        prose = self._prose(
            'TEMPLATE = """# Credentials Yewo reads\n'
            'OPENROUTER_API_KEY=\n'
            '"""\n'
        )

        self.assertNotIn("Credentials Yewo reads", prose)

    def test_a_name_is_still_not_a_sentence(self):
        self.assertEqual(self._prose('NAME = "a short label"\n').strip(), "")


class CallSiteExtractionTests(unittest.TestCase):
    """Prose that reaches a person or a model without being assigned first.

    A log line, a refusal and a `--help` string are read under pressure by
    somebody working out what broke. The gate passed on every one of them,
    so "STE gate passed" said nothing about the debug context or the errors.
    """

    def _prose(self, body: str) -> str:
        with tempfile.TemporaryDirectory() as directory:
            return ste_gate.prose_of(_written(Path(directory), body))

    def test_the_sentence_a_log_line_opens_with_is_read(self):
        prose = self._prose(
            'def warn(count):\n'
            '    logger.warning("The device answered %d times fewer than '
            'asked.", count)\n'
        )

        self.assertIn("The device answered", prose)

    def test_an_f_string_log_line_is_read_too(self):
        prose = self._prose(
            'def warn(name):\n'
            '    logger.error(f"The capability {name} could not be applied here.")\n'
        )

        self.assertIn("could not be applied here", prose)

    def test_the_values_after_a_log_line_are_not_prose(self):
        """They are decided at run time, and the format string is the English."""
        prose = self._prose(
            'def warn(path):\n'
            '    logger.info("Wrote the report.", "/some/long/path/to/a/file")\n'
        )

        self.assertNotIn("/some/long/path", prose)

    def test_the_sentence_a_refusal_was_built_with_is_read(self):
        prose = self._prose(
            'def refuse():\n'
            '    raise ValueError("The scenario asked for a state nothing sets.")\n'
        )

        self.assertIn("asked for a state nothing sets", prose)

    def test_a_refusal_joined_from_two_literals_is_read_as_one(self):
        """Split across lines to fit, it is still one sentence to a reader."""
        prose = self._prose(
            'def refuse():\n'
            '    raise ValueError(\n'
            '        "The capability could not be applied on this target; "\n'
            '        "nothing read the device as being in that state."\n'
            '    )\n'
        )

        self.assertIn("nothing read the device", prose)

    def test_a_command_line_help_string_is_read(self):
        prose = self._prose(
            'def add(parser):\n'
            '    parser.add_argument("--grounding", help="How the planner '
            'names a target on the screen.")\n'
        )

        self.assertIn("How the planner names a target", prose)

    def test_a_message_to_a_model_is_read(self):
        prose = self._prose(
            'def ask():\n'
            '    return TextMessage(content="Answer again as JSON matching '
            'the schema exactly.")\n'
        )

        self.assertIn("Answer again as JSON", prose)

    def test_what_a_check_reports_is_read(self):
        prose = self._prose(
            'def checked():\n'
            '    return CheckResult("node", OK, detail="Installed, and new '
            'enough to run the driver.")\n'
        )

        self.assertIn("new enough to run the driver", prose)

    def test_a_label_is_still_not_a_sentence(self):
        """Most short log lines are labels, and holding them to prose is noise."""
        for line in ('"done"', '"Screen: %s", size', '"Goal reached: %s", goal'):
            with self.subTest(line=line):
                prose = self._prose(f"def log():\n    logger.info({line})\n")

                self.assertEqual(prose.strip(), "")
