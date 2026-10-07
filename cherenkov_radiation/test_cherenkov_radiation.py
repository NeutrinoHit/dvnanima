from __future__ import annotations

import math
import unittest

import numpy as np

import cherenkov_radiation as cr


class CherenkovTest(unittest.TestCase):
    def test_threshold(self) -> None:
        self.assertAlmostEqual(cr.threshold_beta(), 0.7519, places=3)
        self.assertIsNone(cr.cherenkov_angle(0.74))
        self.assertIsNotNone(cr.cherenkov_angle(0.76))

    def test_maximum_angle_in_water(self) -> None:
        self.assertAlmostEqual(math.degrees(math.acos(1 / cr.N_INDEX)), 41.2, places=1)
        self.assertAlmostEqual(math.degrees(cr.cherenkov_angle(0.99999)), 41.2, places=1)

    def test_angle_and_mach_angle_are_complementary(self) -> None:
        for b in (0.8, 0.9, 0.99):
            self.assertAlmostEqual(cr.cherenkov_angle(b) + cr.mach_half_angle(b), math.pi / 2, places=12)

    def test_threshold_energies(self) -> None:
        self.assertAlmostEqual(cr.threshold_kinetic_energy(cr.M_ELECTRON), 0.26, places=2)
        self.assertAlmostEqual(cr.threshold_kinetic_energy(cr.M_MUON), 54.6, delta=0.1)

    def test_cone_is_the_common_tangent_of_the_wavelets(self) -> None:
        """For constant beta the distance from each wavelet centre to the cone line equals its radius."""
        beta = 0.9
        psi = cr.mach_half_angle(beta)
        xp = 10.0
        for lag in (0.3, 1.0, 2.7):
            te_to_t = lag / beta
            radius = te_to_t / cr.N_INDEX
            distance = lag * math.sin(psi)
            self.assertAlmostEqual(distance, radius, places=12)

    def test_wavelets_expand_at_c_over_n(self) -> None:
        ts, xs = cr.trajectory(15.0)
        xe, r = cr.wavelets(14.0, ts, xs)
        self.assertAlmostEqual(r[0], 14.0 / cr.N_INDEX, places=9)
        self.assertAlmostEqual(r[1] - r[0], -cr.EMIT_DT / cr.N_INDEX, places=9)

    def test_ring_radius(self) -> None:
        self.assertIsNone(cr.ring_radius(0.7))
        self.assertAlmostEqual(cr.ring_radius(0.99999), math.tan(math.acos(1 / cr.N_INDEX)), places=3)

    def test_beta_profile_is_continuous_except_at_the_sweep_start(self) -> None:
        for t in (5.0, 10.75, 15.0, 19.75, 24.0, 35.0):
            self.assertGreater(cr.beta_of_t(t), 0.5)
        self.assertLess(cr.beta_of_t(30.0), cr.beta_of_t(38.0))


if __name__ == "__main__":
    unittest.main()
