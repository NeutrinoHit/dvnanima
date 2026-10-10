from __future__ import annotations

import math
import unittest

import numpy as np
from scipy.special import jv

import oam_beams as ob


class FieldTest(unittest.TestCase):
    def test_phase_winds_m_times(self) -> None:
        a = np.linspace(0, 2 * math.pi, 400, endpoint=False)
        for m in (-3, -1, 1, 2, 4):
            x, y = 0.6 * np.cos(a), 0.6 * np.sin(a)
            self.assertEqual(ob.winding_number(ob.bessel_beam(x, y, m)), m)
            self.assertEqual(ob.winding_number(ob.lg_beam(x, y, m)), m)

    def test_orbital_angular_momentum_is_m_hbar(self) -> None:
        rho = np.linspace(0.05, 1.5, 40)[:, None]
        phi = np.linspace(0, 2 * math.pi, 256, endpoint=False)[None, :]
        for m in (-2, 1, 3):
            psi = ob.bessel_beam(rho * np.cos(phi), rho * np.sin(phi), m)
            self.assertAlmostEqual(ob.lz_expectation(psi, phi[0]), m, delta=0.01 * abs(m))

    def test_dark_core_for_m_not_zero(self) -> None:
        for m in (1, 2, 3):
            self.assertLess(abs(ob.bessel_beam(np.array([1e-4]), np.array([0.0]), m)[0]), 1e-3)
            self.assertLess(abs(ob.lg_beam(np.array([1e-4]), np.array([0.0]), m)[0]), 1e-3)
        self.assertAlmostEqual(abs(ob.bessel_beam(np.array([0.0]), np.array([0.0]), 0)[0]), 1.0)

    def test_bessel_beam_solves_the_helmholtz_equation(self) -> None:
        h = 1e-3
        for m in (0, 1, 3):
            for (x0, y0) in ((0.4, 0.2), (-0.7, 0.5)):
                f = lambda x, y: ob.bessel_beam(np.array([x]), np.array([y]), m)[0]
                lap = (f(x0 + h, y0) + f(x0 - h, y0) + f(x0, y0 + h) + f(x0, y0 - h) - 4 * f(x0, y0)) / h ** 2
                self.assertAlmostEqual(abs(lap + ob.KAPPA ** 2 * f(x0, y0)), 0.0, delta=1e-3)

    def test_lg_ring_radius_is_the_intensity_maximum(self) -> None:
        r = np.linspace(1e-3, 2.0, 40001)
        for m in (1, 2, 3, -2):
            i = np.abs(ob.lg_beam(r, np.zeros_like(r), m)) ** 2
            self.assertAlmostEqual(float(r[np.argmax(i)]), ob.lg_ring_radius(m), delta=2e-3)

    def test_spiral_plate_turns_gauss_into_the_winding_beam(self) -> None:
        a = np.linspace(0, 2 * math.pi, 400, endpoint=False)
        x, y = 0.4 * np.cos(a), 0.4 * np.sin(a)
        for m in (1, 2, 3, -1):
            psi = ob.gauss_beam(x, y) * ob.spiral_plate(x, y, m)
            self.assertEqual(ob.winding_number(psi), m)

    def test_first_bessel_ring(self) -> None:
        r = np.linspace(1e-3, 2.0, 40001)
        for m in (1, 2, 3):
            i = jv(m, ob.KAPPA * r) ** 2
            maxima = np.flatnonzero((i[1:-1] > i[:-2]) & (i[1:-1] > i[2:])) + 1
            peak = r[maxima[0]]
            self.assertAlmostEqual(float(peak), ob.bessel_ring_radius(m), delta=3e-3)


class WavefrontTest(unittest.TestCase):
    def test_points_lie_on_the_constant_phase_surfaces(self) -> None:
        for m in (1, 2, 3, -2):
            for n in range(abs(m)):
                z = np.linspace(0, 2.4, 25)
                phi = ob.wavefront_phi(m, n, z, 0.7)
                total = m * phi + ob.KZ * z - ob.OMEGA * 0.7
                self.assertTrue(np.allclose(total, 2 * math.pi * n, atol=1e-9))

    def test_m_threads(self) -> None:
        # at a fixed height the m helicoids sit at m azimuths separated by 2 pi / |m|
        z = np.array([0.5])
        for m in (2, 3):
            phis = sorted(float(ob.wavefront_phi(m, n, z, 0.0)[0]) % (2 * math.pi) for n in range(m))
            gaps = np.diff(phis + [phis[0] + 2 * math.pi])
            self.assertTrue(np.allclose(gaps, 2 * math.pi / m, atol=1e-9))


if __name__ == "__main__":
    unittest.main()
