"""The runtime brand resource stays aligned with the design source."""

from pathlib import Path
import unittest

from yewo.brand import mark_definition, mark_source, mark_svg


class BrandTests(unittest.TestCase):
    def test_packaged_mark_matches_the_brand_repository(self):
        source = Path(__file__).resolve().parents[2]
        design_mark = source / "yewo-brand/svg/yewo-mark-for-dark-bg.svg"

        self.assertEqual(
            mark_source().rstrip("\n"),
            design_mark.read_text(encoding="utf-8").rstrip("\n"),
        )

    def test_inline_mark_uses_the_packaged_paths(self):
        definition = mark_definition()

        inline = mark_svg(
            21,
            19,
            foreground="var(--logo)",
            accent="var(--clay)",
            stroke_width=7,
        )

        self.assertIn(definition.paths[0], inline)
        self.assertIn(definition.paths[1], inline)
        self.assertIn('stroke="var(--logo)"', inline)
