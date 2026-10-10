from __future__ import annotations

import math
import tomllib
import unittest
from pathlib import Path

import numpy as np

import rutherford as ru

# a single nucleus, no screening effect for impact parameters well below SCREEN
SINGLE = np.zeros((1, 3))


def deflection(b: float, r_min: float) -> float:
    """Final deflection angle (degrees) for one nucleus at the origin and impact parameter b."""
    from scipy.integrate import solve_ivp
    k = r_min * 0.5
    big = 1e12                                   # effectively unscreened: use a huge screening length
    old = ru.SCREEN
    ru.SCREEN = big
    try:
        def rhs(_t, s):
            return np.concatenate([s[3:], ru.acceleration(s[:3], SINGLE, k)])
        s0 = np.array([-3000.0, b, 0.0, 1.0, 0.0, 0.0])
        sol = solve_ivp(rhs, (0, 6000), s0, method="DOP853", rtol=1e-11, atol=1e-11)
    finally:
        ru.SCREEN = old
    v = sol.y[3:, -1]
    return math.degrees(math.acos(v[0] / np.linalg.norm(v)))


class RutherfordTest(unittest.TestCase):
    def test_deflection_angle_matches_rutherford_formula(self) -> None:
        r_min = 1.0
        for b in (0.3, 0.5, 1.0, 3.0):
            expected = 2 * math.degrees(math.atan(r_min / (2 * b)))
            self.assertAlmostEqual(deflection(b, r_min), expected, delta=0.05 * expected)

    def test_head_on_particle_turns_back(self) -> None:
        self.assertGreater(deflection(0.01, 1.0), 170.0)

    def test_foil_has_expected_nuclei(self) -> None:
        nuclei = ru.make_foil(np.random.default_rng(0), 4)
        n = int(round(2 * ru.HALF / ru.SPACING))
        self.assertEqual(len(nuclei), 4 * n * n)

    def test_screening_removes_far_field(self) -> None:
        r = 10 * ru.SCREEN
        screened = np.linalg.norm(ru.acceleration(np.array([r, 0, 0]), SINGLE, 0.5))
        coulomb = 0.5 / r ** 2
        self.assertLess(screened / coulomb, 1e-3)


class ConfigTest(unittest.TestCase):
    def test_texts_have_the_same_keys_in_both_languages(self) -> None:
        texts = tomllib.loads((Path(ru.HERE) / "texts.toml").read_text(encoding="utf-8"))
        self.assertEqual(set(texts), {"en", "ru"})
        self.assertEqual(set(texts["en"]), set(texts["ru"]))

    def test_the_placeholders_of_the_texts_can_be_filled(self) -> None:
        words = dict(turned="90", bin="2", layers=4, one_in=8000, n=7)
        for lang in ("en", "ru"):
            tx = ru.TEXT[lang]
            for key in ("fired", "turned", "ylabel", "turned_label", "note"):
                self.assertNotIn("{", tx[key].format(**words))

    def test_defaults_come_from_the_config(self) -> None:
        self.assertEqual((ru.LAYERS, ru.N_ALPHA, ru.R_MIN), (ru.CFG.foil.layers, ru.CFG.foil.n_alpha, ru.CFG.foil.r_min))

    def test_tracks_are_reproducible_from_the_seed(self) -> None:
        a = ru.make_foil(np.random.default_rng(ru.CFG.simulation.seed), ru.LAYERS)
        b = ru.make_foil(np.random.default_rng(ru.CFG.simulation.seed), ru.LAYERS)
        self.assertTrue(np.array_equal(a, b))

    def test_decimal_comma_in_russian_formulas_only(self) -> None:
        self.assertEqual(ru.num(2.5, "g", "en"), "2.5")
        self.assertEqual(ru.num(2.5, "g", "ru"), "2{,}5")
        self.assertEqual(ru.num(2.0, "g", "ru"), "2")

    def test_rutherford_curve_scales_with_the_number_of_particles(self) -> None:
        th = np.array([20.0, 60.0, 120.0])
        one = ru.rutherford_curve(th, 100, 1.0, 4)
        self.assertTrue(np.allclose(ru.rutherford_curve(th, 200, 1.0, 4), 2 * one))
        self.assertTrue(np.all(np.diff(one) < 0))


if __name__ == "__main__":
    unittest.main()
