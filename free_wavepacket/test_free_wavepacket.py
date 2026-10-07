from __future__ import annotations

import math
import unittest

import numpy as np

import free_wavepacket as fw


class PacketTest(unittest.TestCase):
    def setUp(self) -> None:
        self.x, self.p = fw.grid()

    def test_norm_is_conserved(self) -> None:
        dx = self.x[1] - self.x[0]
        psi0 = fw.initial_state(self.x, fw.S1)
        for t in (0.0, 5.0, 15.0):
            psi = fw.evolve(psi0, self.p, t)
            self.assertAlmostEqual((np.abs(psi) ** 2).sum() * dx, 1.0, places=6)

    def test_initial_width_and_uncertainty(self) -> None:
        for s0 in (fw.S1, fw.S2):
            psi0 = fw.initial_state(self.x, s0)
            _, sig = fw.moments(self.x, psi0)
            self.assertAlmostEqual(sig, s0, places=5)
            self.assertAlmostEqual(sig * (1 / (2 * s0)), 0.5, places=5)

    def test_width_follows_the_analytic_formula(self) -> None:
        for s0 in (fw.S1, fw.S2):
            psi0 = fw.initial_state(self.x, s0)
            for t in (3.0, 8.0, 15.0):
                _, sig = fw.moments(self.x, fw.evolve(psi0, self.p, t))
                self.assertAlmostEqual(sig / fw.sigma_x_exact(s0, t), 1.0, places=4)

    def test_centre_moves_with_the_group_velocity(self) -> None:
        psi0 = fw.initial_state(self.x, fw.S1)
        mean, _ = fw.moments(self.x, fw.evolve(psi0, self.p, 10.0))
        self.assertAlmostEqual(mean, fw.K0 * 10.0, places=3)

    def test_momentum_distribution_is_static(self) -> None:
        psi0 = fw.initial_state(self.x, fw.S1)
        _, a = fw.momentum_density(psi0, self.p)
        _, b = fw.momentum_density(fw.evolve(psi0, self.p, 12.0), self.p)
        self.assertTrue(np.allclose(np.abs(a), np.abs(b), atol=1e-9))

    def test_narrow_packet_spreads_faster(self) -> None:
        self.assertGreater(fw.sigma_x_exact(fw.S1, 15.0) / fw.S1, fw.sigma_x_exact(fw.S2, 15.0) / fw.S2)


if __name__ == "__main__":
    unittest.main()
