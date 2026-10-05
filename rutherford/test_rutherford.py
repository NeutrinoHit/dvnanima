from __future__ import annotations

import math
import unittest

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


if __name__ == "__main__":
    unittest.main()
