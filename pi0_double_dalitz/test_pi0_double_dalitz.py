from __future__ import annotations

import math
import unittest

import numpy as np

import pi0_double_dalitz as pd


class DistributionTest(unittest.TestCase):
    def test_density_is_normalized(self) -> None:
        phi = np.linspace(0, 2 * math.pi, 20001)
        for mode in ("ps", "s"):
            self.assertAlmostEqual(float(np.trapezoid(pd.density(phi, mode=mode), phi)), 1.0, places=6)

    def test_pseudoscalar_has_humps_at_orthogonal_planes(self) -> None:
        self.assertGreater(pd.density(math.pi / 2), pd.density(0.0))
        self.assertGreater(pd.density(3 * math.pi / 2), pd.density(math.pi))
        self.assertAlmostEqual(float(pd.density(math.pi / 2)), (1 + pd.A_AMP) / (2 * math.pi))

    def test_scalar_has_humps_at_parallel_planes(self) -> None:
        self.assertGreater(pd.density(0.0, mode="s"), pd.density(math.pi / 2, mode="s"))
        self.assertGreater(pd.density(math.pi, mode="s"), pd.density(3 * math.pi / 2, mode="s"))

    def test_sampling_reproduces_the_second_harmonic(self) -> None:
        for mode, sign in (("ps", -1.0), ("s", +1.0)):
            phi = pd.sample_phi(np.random.default_rng(1), 200_000, mode=mode)
            # <cos 2 phi> = (-+ a) / 2 for the density (1 -+ a cos 2 phi) / (2 pi)
            self.assertAlmostEqual(float(np.mean(np.cos(2 * phi))), sign * pd.A_AMP / 2, delta=0.005)
            self.assertAlmostEqual(float(np.mean(np.cos(phi))), 0.0, delta=0.01)

    def test_events_are_distinct_and_ordered(self) -> None:
        events = pd.make_events(3)
        times = [e["t"] for e in events]
        self.assertEqual(times, sorted(times))
        for mode in ("ps", "s"):
            self.assertGreater(sum(e["mode"] == mode for e in events), 55)
        self.assertEqual(len({round(e["azim1"], 9) for e in events}), len(events))


class FieldTest(unittest.TestCase):
    def test_e_is_perpendicular_to_b_and_to_the_photon(self) -> None:
        for direction in (+1, -1):
            for phi in (0.0, 0.7, 2.1):
                e, b = pd.field_vectors(phi, direction)
                self.assertAlmostEqual(float(e @ b), 0.0, places=12)
                self.assertAlmostEqual(float(e[2]), 0.0, places=12)
                self.assertAlmostEqual(float(np.linalg.norm(b)), 1.0, places=12)

    def test_pseudoscalar_amplitude_is_sine_of_the_angle(self) -> None:
        for phi in np.linspace(0.0, 2 * math.pi, 13):
            ps, _ = pd.triple_product_check(phi)
            self.assertAlmostEqual(abs(ps), 2 * abs(math.sin(phi)), places=12)

    def test_scalar_amplitude_is_cosine_of_the_angle(self) -> None:
        for phi in np.linspace(0.0, 2 * math.pi, 13):
            _, s = pd.triple_product_check(phi)
            self.assertAlmostEqual(abs(s), 2 * abs(math.cos(phi)), places=12)

    def test_squared_amplitudes_match_the_event_rates(self) -> None:
        for phi in np.linspace(0.0, math.pi, 9):
            self.assertAlmostEqual(pd.amplitude_squared(phi, "ps"), (1 - math.cos(2 * phi)) / 2, places=12)
            self.assertAlmostEqual(pd.amplitude_squared(phi, "s"), (1 + math.cos(2 * phi)) / 2, places=12)

    def test_width_formula_matches_the_book(self) -> None:
        alpha, m, f = 1 / 137.036, 134.977e6, 92.2e6            # eV
        gamma = alpha ** 2 * m ** 3 / (64 * math.pi ** 3 * f ** 2)
        self.assertAlmostEqual(gamma, 7.76, delta=0.2)


if __name__ == "__main__":
    unittest.main()
