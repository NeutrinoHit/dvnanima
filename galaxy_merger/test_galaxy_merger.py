from __future__ import annotations

import math
import unittest

import numpy as np

import galaxy_merger as gm


M1, M2 = 1.0, 0.8
A1, A2 = 0.25, 0.25 * math.sqrt(M2)


def integrate_centres(lnl: float, t_end: float, dt: float = 0.004):
    """Leapfrog integration of the two halo centres, as used by the film."""
    c1, c2, v1, v2 = gm.orbit_start(M1, M2, 1.0, 0.6, -2.6)
    k1, k2 = gm.core_acc(c1, c2, v1, v2, M1, M2, A1, A2, lnl)
    history = [(c1.copy(), c2.copy(), v1.copy(), v2.copy())]
    for _ in range(int(round(t_end / dt))):
        v1 = v1 + 0.5 * dt * k1
        v2 = v2 + 0.5 * dt * k2
        c1 = c1 + dt * v1
        c2 = c2 + dt * v2
        k1, k2 = gm.core_acc(c1, c2, v1, v2, M1, M2, A1, A2, lnl)
        v1 = v1 + 0.5 * dt * k1
        v2 = v2 + 0.5 * dt * k2
        history.append((c1.copy(), c2.copy(), v1.copy(), v2.copy()))
    return history


def total_energy(state) -> float:
    c1, c2, v1, v2 = state
    eps2 = A1 * A1 + A2 * A2
    kinetic = 0.5 * M1 * (v1 @ v1) + 0.5 * M2 * (v2 @ v2)
    potential = -gm.G * M1 * M2 / math.sqrt((c2 - c1) @ (c2 - c1) + eps2)
    return kinetic + potential


class DynamicsTest(unittest.TestCase):
    def test_start_orbit_has_requested_pericentre_and_angular_momentum(self) -> None:
        q, e = 1.0, 0.6
        c1, c2, v1, v2 = gm.orbit_start(M1, M2, q, e, 0.0)
        rel, vrel = c2 - c1, v2 - v1
        self.assertAlmostEqual(float(np.linalg.norm(rel)), q, places=12)
        self.assertAlmostEqual(float(rel @ vrel), 0.0, places=12)
        h = np.linalg.norm(np.cross(rel, vrel))
        self.assertAlmostEqual(float(h), math.sqrt(gm.G * (M1 + M2) * q * (1 + e)), places=12)

    def test_start_orbit_centre_of_mass_is_at_rest(self) -> None:
        c1, c2, v1, v2 = gm.orbit_start(M1, M2, 1.0, 0.6, -2.6)
        np.testing.assert_allclose(M1 * c1 + M2 * c2, 0.0, atol=1e-12)
        np.testing.assert_allclose(M1 * v1 + M2 * v2, 0.0, atol=1e-12)

    def test_energy_is_conserved_without_friction(self) -> None:
        history = integrate_centres(lnl=0.0, t_end=30.0)
        energy = np.array([total_energy(s) for s in history])
        self.assertLess(np.ptp(energy) / abs(energy[0]), 1e-4)

    def test_friction_conserves_momentum_and_removes_energy(self) -> None:
        history = integrate_centres(lnl=0.6, t_end=30.0)
        for c1, c2, v1, v2 in history[::50]:
            np.testing.assert_allclose(M1 * v1 + M2 * v2, 0.0, atol=1e-9)
        self.assertLess(total_energy(history[-1]), total_energy(history[0]))

    def test_friction_opposes_the_motion(self) -> None:
        r = np.array([0.4, 0.1, 0.0])
        v = np.array([0.3, -0.2, 0.1])
        a = gm.df_accel(r, v, 0.8, 1.0, 0.25, 0.6)
        self.assertLess(float(a @ v), 0.0)
        np.testing.assert_allclose(np.cross(a, v), 0.0, atol=1e-12)

    def test_friction_vanishes_for_a_body_at_rest(self) -> None:
        a = gm.df_accel(np.array([0.2, 0.0, 0.0]), np.zeros(3), 0.8, 1.0, 0.25, 0.6)
        np.testing.assert_allclose(a, 0.0, atol=1e-6)


class DiskTest(unittest.TestCase):
    def test_stars_move_on_circular_orbits_without_scatter(self) -> None:
        rng = np.random.default_rng(1)
        pos, vel = gm.make_disk(2000, 1.0, 0.25, 0.9, 0.0, 0.0, rng, sigma_v=0.0)
        r = np.hypot(pos[:, 0], pos[:, 1])
        self.assertLessEqual(float(r.max()), 0.9 + 1e-12)
        speed = np.hypot(vel[:, 0], vel[:, 1])
        expected = np.sqrt(gm.G * 1.0 * r**2 / (r**2 + 0.25**2) ** 1.5)
        np.testing.assert_allclose(speed, expected, rtol=1e-12)
        np.testing.assert_allclose(np.sum(pos[:, :2] * vel[:, :2], axis=1), 0.0, atol=1e-12)

    def test_tilted_disk_keeps_radii_and_speeds(self) -> None:
        rng = np.random.default_rng(2)
        flat = gm.make_disk(500, 1.0, 0.25, 0.9, 0.0, 0.0, rng, sigma_v=0.0)
        rng = np.random.default_rng(2)
        tilted = gm.make_disk(500, 1.0, 0.25, 0.9, math.radians(55), math.radians(30), rng, sigma_v=0.0)
        np.testing.assert_allclose(np.linalg.norm(tilted[0], axis=1), np.linalg.norm(flat[0], axis=1), rtol=1e-12)
        np.testing.assert_allclose(np.linalg.norm(tilted[1], axis=1), np.linalg.norm(flat[1], axis=1), rtol=1e-12)


class RenderTest(unittest.TestCase):
    def test_blurred_density_keeps_the_star_count(self) -> None:
        npx = 128
        rng = np.random.default_rng(3)
        x = rng.normal(0, 0.2, 5000)
        y = rng.normal(0, 0.2, 5000)
        sel = np.ones(5000, dtype=bool)
        half = 2.0
        kernel = gm.blur_kernel(npx)
        dens = gm.surface_density(x, y, sel, half, npx, kernel)
        area = (2 * half / npx) ** 2
        # the kernel is a sum of three Gaussians with total weight 12
        self.assertAlmostEqual(float(dens.sum() * area) / (12 * 5000), 1.0, places=6)

    def test_camera_never_gets_narrower_than_the_minimum(self) -> None:
        class Args:
            reach = 1.15
            rdisk = 0.9
            min_width = 1.3
            cam_smooth = 5.0

        cores = np.zeros((100, 2, 2))
        half = gm.camera(cores, Args)
        self.assertGreaterEqual(float(half.min()), 1.3 - 1e-9)


if __name__ == "__main__":
    unittest.main()
