from __future__ import annotations

import math
import unittest

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


if __name__ == "__main__":
    unittest.main()
