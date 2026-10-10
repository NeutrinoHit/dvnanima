from __future__ import annotations

import math
import tomllib
import unittest
from pathlib import Path

import numpy as np

import electron_kick as ek

HERE = Path(__file__).resolve().parent


class FluxContinuityTest(unittest.TestCase):
    def test_inner_and_outer_fluxes_agree(self) -> None:
        for beta in (0.1, 0.5, 0.9):
            theta = np.linspace(0.05, math.pi - 0.05, 50)
            psi = ek.psi_of_theta(theta, beta)
            np.testing.assert_allclose(ek.flux_fraction_moving(psi, beta), 0.5 * (1 - np.cos(theta)), atol=1e-12)

    def test_tan_relation(self) -> None:
        beta = 0.8
        gamma = 1 / math.sqrt(1 - beta ** 2)
        theta = np.linspace(0.1, 3.0, 40)
        psi = ek.psi_of_theta(theta, beta)
        np.testing.assert_allclose(np.tan(theta), np.tan(psi) / gamma, rtol=1e-9)

    def test_inverse_functions(self) -> None:
        theta = np.linspace(0.1, 3.0, 40)
        np.testing.assert_allclose(ek.theta_of_psi(ek.psi_of_theta(theta, 0.7), 0.7), theta, atol=1e-9)

    def test_total_flux_is_one(self) -> None:
        self.assertAlmostEqual(float(ek.flux_fraction_moving(math.pi, 0.95)), 1.0, places=12)
        self.assertAlmostEqual(float(ek.flux_fraction_moving(0.0, 0.95)), 0.0, places=12)


class GeometryTest(unittest.TestCase):
    def test_inner_line_ends_on_the_shell(self) -> None:
        beta, t = 0.6, 3.0
        for theta in np.linspace(0.1, 3.0, 20):
            g = ek.line_geometry(float(theta), beta, t)
            self.assertAlmostEqual(float(np.hypot(*g["hit"])), t, places=9)
            # the line leaves the present position of the charge
            self.assertAlmostEqual(float(np.arctan2(g["hit"][1], g["hit"][0] - beta * t)), g["psi"], places=9)

    def test_arcs_vanish_for_a_tiny_kick(self) -> None:
        g = ek.line_geometry(1.0, 1e-6, 2.0)
        self.assertAlmostEqual(g["theta_in"], g["theta_out"], places=5)

    def test_front_is_brighter_for_larger_beta(self) -> None:
        # total arc length (the transverse field of the front) grows with beta
        def arc_sum(beta: float) -> float:
            theta = np.arccos(1 - 2 * (np.arange(22) + 0.5) / 22)
            return float(sum(abs(ek.line_geometry(float(th), beta, 3.0)["theta_in"] - th) for th in theta))
        self.assertGreater(arc_sum(0.85), arc_sum(0.3))


class ConfigTest(unittest.TestCase):
    def test_texts_have_the_same_keys_in_both_languages_and_balanced_formulas(self) -> None:
        texts = tomllib.loads((HERE / "texts.toml").read_text(encoding="utf-8"))
        self.assertEqual(set(texts), {"en", "ru"})
        self.assertEqual(set(texts["en"]), set(texts["ru"]))
        for lang, table in texts.items():
            for key, value in table.items():
                self.assertEqual(value.count("$") % 2, 0, f"{lang}.{key}: unbalanced $")
        self.assertIn("{beta}", texts["ru"]["beta_line"])

    def test_the_runs_follow_each_other_and_cover_the_film(self) -> None:
        runs = ek.CFG.timeline.runs
        self.assertEqual(runs[0][1], 0.0)
        self.assertEqual(runs[-1][2], ek.CFG.timeline.film_length)
        for a, b in zip(runs, runs[1:]):
            self.assertEqual(a[2], b[1])
        for beta, start, end in runs:
            self.assertTrue(0.0 < beta < 1.0 and start < end)

    def test_video_section_is_complete(self) -> None:
        v = ek.CFG.video
        self.assertEqual(v.width * 9, v.height * 16)
        self.assertGreater(v.width, v.preview_width)
        self.assertGreater(v.fps, 0)

    def test_decimal_comma_in_russian_formulas(self) -> None:
        self.assertEqual(ek.num(0.3, ".2f", "ru"), "0{,}30")
        self.assertEqual(ek.num(0.3, ".2f", "en"), "0.30")

    def test_the_script_has_no_hard_coded_video_settings(self) -> None:
        src = (HERE / "electron_kick.py").read_text(encoding="utf-8")
        self.assertIn("load_config", src)
        for forbidden in ('"libx264", "-preset", "slow"', "(1280, 720)", "dpi = 100\n"):
            self.assertNotIn(forbidden, src)


if __name__ == "__main__":
    unittest.main()
