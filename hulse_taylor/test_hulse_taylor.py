from __future__ import annotations

import math
import unittest

import numpy as np

import hulse_taylor as ht


class OrbitTest(unittest.TestCase):
    def test_semi_major_axis_from_keplers_third_law(self) -> None:
        # about 1.95e9 m (2.8 solar radii) for PSR B1913+16
        self.assertAlmostEqual(ht.semi_major_axis() / 1e9, 1.949, delta=0.005)

    def test_kepler_equation(self) -> None:
        m = np.linspace(0, 2 * math.pi, 50, endpoint=False)
        E = ht.kepler(m)
        np.testing.assert_allclose(E - ht.ECC * np.sin(E), m, atol=1e-10)

    def test_periastron_and_apastron_separations(self) -> None:
        _, _, r = ht.relative_orbit(np.array([0.0, math.pi]), 1.0)
        self.assertAlmostEqual(float(r[0]), 1 - ht.ECC, places=10)
        self.assertAlmostEqual(float(r[1]), 1 + ht.ECC, places=10)

    def test_orbit_is_an_ellipse(self) -> None:
        x, y, _ = ht.relative_orbit(np.linspace(0, 2 * math.pi, 200), 1.0)
        b = math.sqrt(1 - ht.ECC ** 2)
        np.testing.assert_allclose(((x + ht.ECC) / 1.0) ** 2 + (y / b) ** 2, 1.0, atol=1e-9)


class ShiftTest(unittest.TestCase):
    def test_periastron_shift_after_thirty_years(self) -> None:
        # about -38.6 s, as in the figure of the book (about -40 s in 2005)
        self.assertAlmostEqual(float(ht.periastron_shift(30.0)), -38.6, delta=0.1)

    def test_shift_is_quadratic_and_negative(self) -> None:
        self.assertAlmostEqual(float(ht.periastron_shift(20.0) / ht.periastron_shift(10.0)), 4.0, places=9)
        self.assertLess(float(ht.periastron_shift(5.0)), 0.0)

    def test_measured_derivative_agrees_with_gr(self) -> None:
        self.assertAlmostEqual(ht.PDOT_INTR / ht.PDOT_GR, 0.9983, delta=0.0005)

    def test_orbit_shrink_in_thirty_years(self) -> None:
        self.assertAlmostEqual(ht.fractional_shrink(30.0), -5.4e-8, delta=0.1e-8)

    def test_period_change_in_microseconds_per_year(self) -> None:
        self.assertAlmostEqual(ht.PDOT_GR * ht.YEAR * 1e6, -75.82, delta=0.02)


if __name__ == "__main__":
    unittest.main()
