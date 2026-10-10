from __future__ import annotations

import math
import tomllib
import unittest
from pathlib import Path

import numpy as np

import running_charge as rc


class ChargeTest(unittest.TestCase):
    def test_far_charge_is_the_classical_one(self) -> None:
        self.assertAlmostEqual(float(rc.charge_ratio(100.0)), 1.0, places=3)

    def test_short_distance_limit_is_the_logarithm(self) -> None:
        r = 1e-3
        self.assertAlmostEqual(float(rc.charge_ratio(r, 0.05)), 1.0 / (1.0 - 0.05 * math.log(1.0 / r)), places=4)

    def test_charge_grows_towards_the_centre(self) -> None:
        r = np.geomspace(0.1, 3.3, 100)
        q = rc.charge_ratio(r)
        self.assertTrue(np.all(np.diff(q) < 0))
        self.assertTrue(np.all(q > 1.0))

    def test_real_coupling(self) -> None:
        # about one percent at r = 1e-3 lambda_C for the electron
        self.assertAlmostEqual(float(rc.charge_ratio(1e-3, rc.B_REAL)) - 1.0, 0.0108, delta=0.002)


class PairsTest(unittest.TestCase):
    def mean_radial_projection(self, strength: float) -> float:
        pairs = rc.make_pairs(1, 30.0)
        vals = []
        for t in np.linspace(2, 28, 60):
            pos, ele, op, age, life = rc.pair_ends(pairs, t, lambda _t: strength)
            if len(pos):
                mid = 0.5 * (pos + ele)
                near = np.hypot(mid[:, 0], mid[:, 1]) < 1.0
                if near.any():
                    u = mid[near] / np.hypot(mid[near, 0], mid[near, 1])[:, None]
                    dip = ele[near] - pos[near]
                    vals.extend(np.sum(dip * u, axis=1) / np.hypot(dip[:, 0], dip[:, 1]))
        return float(np.mean(vals))

    def test_pairs_are_polarized_near_the_charge(self) -> None:
        # the electron end of a pair lies farther from the charge than the positron end
        self.assertGreater(self.mean_radial_projection(1.0), 0.6)

    def test_pairs_are_random_without_charge(self) -> None:
        self.assertLess(abs(self.mean_radial_projection(0.0)), 0.15)


class ConfigTest(unittest.TestCase):
    def test_texts_have_the_same_keys_in_both_languages(self) -> None:
        texts = tomllib.loads((Path(rc.HERE) / "texts.toml").read_text(encoding="utf-8"))
        self.assertEqual(set(texts), {"en", "ru"})
        self.assertEqual(set(texts["en"]), set(texts["ru"]))

    def test_decimal_comma_in_russian_only(self) -> None:
        self.assertEqual(rc.num(1.234, ".2f", "en"), "1.23")
        self.assertEqual(rc.num(1.234, ".2f", "ru"), "1,23")

    def test_real_coupling_comes_from_the_fine_structure_constant(self) -> None:
        self.assertAlmostEqual(rc.B_REAL, 2 / (3 * math.pi * rc.CFG.model.alpha_inverse), places=12)
        self.assertAlmostEqual(rc.B_REAL, 0.00155, places=5)

    def test_timeline_scales_with_the_film_length(self) -> None:
        a, b = rc.timeline(26.0), rc.timeline(13.0)
        for name in a:
            self.assertAlmostEqual(b[name][0], 0.5 * a[name][0])
            self.assertAlmostEqual(b[name][1], 0.5 * a[name][1])

    def test_probe_flies_from_far_to_near(self) -> None:
        tl = rc.timeline(26.0)
        self.assertAlmostEqual(rc.probe_radius(tl["probe"][0], tl), rc.CFG.probe.r_start)
        self.assertAlmostEqual(rc.probe_radius(tl["probe"][1], tl), rc.CFG.probe.r_end)

    def test_pairs_are_reproducible_from_the_seed(self) -> None:
        a, b = rc.make_pairs(rc.CFG.pairs.seed, 10.0), rc.make_pairs(rc.CFG.pairs.seed, 10.0)
        self.assertTrue(np.array_equal(a["t0"], b["t0"]))
        self.assertTrue(np.array_equal(a["vm_seed"], b["vm_seed"]))


if __name__ == "__main__":
    unittest.main()
