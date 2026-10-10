from __future__ import annotations

import math
import tomllib
import unittest
from pathlib import Path

import numpy as np

import spinor_mobius as sm

HERE = Path(__file__).resolve().parent


class SpinorTest(unittest.TestCase):
    def test_strip_vector_flips_after_one_turn_and_returns_after_two(self) -> None:
        for u in (0.0, 0.7, 2.1, 5.9):
            np.testing.assert_allclose(sm.across(u + 2 * math.pi), -sm.across(u), atol=1e-12)
            np.testing.assert_allclose(sm.across(u + 4 * math.pi), sm.across(u), atol=1e-12)

    def test_vector_lies_across_the_strip(self) -> None:
        # the vector is tangent to the strip and orthogonal to the core circle direction
        for u in (0.3, 1.9, 4.4):
            tangent_core = np.array([-math.sin(u), math.cos(u), 0.0])
            self.assertAlmostEqual(float(sm.across(u) @ tangent_core), 0.0, places=12)
            self.assertAlmostEqual(float(np.linalg.norm(sm.across(u))), 1.0, places=12)

    def test_vector_is_the_derivative_of_the_surface_in_v(self) -> None:
        u = 1.3
        h = 1e-6
        d = (sm.strip_point(np.array(u), np.array(h)) - sm.strip_point(np.array(u), np.array(-h))) / (2 * h)
        np.testing.assert_allclose(d, sm.across(u), atol=1e-8)

    def test_spinor_component(self) -> None:
        self.assertAlmostEqual(sm.spinor_component(0.0), 1.0)
        self.assertAlmostEqual(sm.spinor_component(2 * math.pi), -1.0)
        self.assertAlmostEqual(sm.spinor_component(4 * math.pi), 1.0)

    def test_angle_schedule(self) -> None:
        total = 20.0
        self.assertEqual(sm.angle_at(0.0, total), 0.0)
        self.assertAlmostEqual(sm.angle_at(total, total), 4 * math.pi)
        values = [sm.angle_at(t, total) for t in np.linspace(0, total, 400)]
        self.assertTrue(all(b >= a - 1e-12 for a, b in zip(values, values[1:])))
        self.assertTrue(any(abs(v - 2 * math.pi) < 1e-9 for v in values))


class ConfigTest(unittest.TestCase):
    def test_texts_have_the_same_keys_in_both_languages_and_balanced_formulas(self) -> None:
        texts = tomllib.loads((HERE / "texts.toml").read_text(encoding="utf-8"))
        self.assertEqual(set(texts), {"en", "ru"})
        self.assertEqual(set(texts["en"]), set(texts["ru"]))
        for lang, table in texts.items():
            for key, value in table.items():
                self.assertEqual(value.count("$") % 2, 0, f"{lang}.{key}: unbalanced $")
            self.assertIn("{deg}", table["phi_line"])

    def test_schedule_uses_the_configured_pauses(self) -> None:
        total = sm.CFG.timeline.film_length
        hold, delay = sm.CFG.timeline.hold_s, sm.CFG.timeline.start_delay_s
        move = (total - 2 * hold - sm.CFG.timeline.slack_s) / 2
        self.assertAlmostEqual(sm.angle_at(delay + move + hold / 2, total), 2 * math.pi)   # the hold after the first turn
        self.assertAlmostEqual(sm.angle_at(delay + 2 * move + hold, total), 4 * math.pi)

    def test_video_section_is_complete(self) -> None:
        v = sm.CFG.video
        self.assertEqual(v.width * 9, v.height * 16)
        self.assertGreater(v.width, v.preview_width)
        self.assertGreater(v.fps, 0)
        self.assertEqual(sm.HALF_WIDTH, sm.CFG.model.half_width)

    def test_the_script_has_no_hard_coded_video_settings(self) -> None:
        src = (HERE / "spinor_mobius.py").read_text(encoding="utf-8")
        self.assertIn("load_config", src)
        for forbidden in ('"libx264", "-preset", "slow"', "(1280, 720)", "dpi = 100\n"):
            self.assertNotIn(forbidden, src)


if __name__ == "__main__":
    unittest.main()
