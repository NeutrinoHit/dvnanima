from __future__ import annotations

import math
import unittest

import numpy as np

import electron_kick as ek


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


if __name__ == "__main__":
    unittest.main()
