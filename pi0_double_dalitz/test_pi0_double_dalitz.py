from __future__ import annotations

import math
import unittest

import numpy as np

import pi0_double_dalitz as pd


class DistributionTest(unittest.TestCase):
    def test_density_is_normalized(self) -> None:
        phi = np.linspace(0, 2 * math.pi, 20001)
        self.assertAlmostEqual(float(np.trapezoid(pd.density(phi), phi)), 1.0, places=6)

    def test_pseudoscalar_has_humps_at_orthogonal_planes(self) -> None:
        self.assertGreater(pd.density(math.pi / 2), pd.density(0.0))
        self.assertGreater(pd.density(3 * math.pi / 2), pd.density(math.pi))
        self.assertAlmostEqual(float(pd.density(math.pi / 2)), (1 + pd.A_AMP) / (2 * math.pi))

    def test_sampling_reproduces_the_second_harmonic(self) -> None:
        phi = pd.sample_phi(np.random.default_rng(1), 200_000)
        # <cos 2 phi> = -a / 2 for the density (1 - a cos 2 phi) / (2 pi)
        self.assertAlmostEqual(float(np.mean(np.cos(2 * phi))), -pd.A_AMP / 2, delta=0.005)
        self.assertAlmostEqual(float(np.mean(np.cos(phi))), 0.0, delta=0.01)

    def test_events_are_distinct_and_ordered(self) -> None:
        events = pd.make_events(3, 34.0)
        times = [e["t"] for e in events]
        self.assertEqual(times, sorted(times))
        self.assertGreater(len(events), 60)
        self.assertEqual(len({round(e["azim1"], 9) for e in events}), len(events))


if __name__ == "__main__":
    unittest.main()
