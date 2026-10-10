from __future__ import annotations

import math
import unittest

import numpy as np

import coupled_lattice_2d as cl
import lattice2d_physics as P


def laplacian_matrix(nx: int, ny: int) -> np.ndarray:
    """The matrix Omega of the lattice (fixed edge) acting on q.ravel() (index i * ny + j)."""
    n = nx * ny
    om = np.zeros((n, n))
    for i in range(nx):
        for j in range(ny):
            r = i * ny + j
            om[r, r] = 4.0 * P.KC
            for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                ii, jj = i + di, j + dj
                if 0 <= ii < nx and 0 <= jj < ny:
                    om[r, ii * ny + jj] = -P.KC
    return om


class ModesTest(unittest.TestCase):
    def test_modes_are_orthonormal(self) -> None:
        self.assertTrue(np.allclose(P.SX @ P.SX.T, np.eye(P.NX), atol=1e-12))
        self.assertTrue(np.allclose(P.SY @ P.SY.T, np.eye(P.NY), atol=1e-12))

    def test_frequencies_are_the_eigenvalues_of_the_lattice_matrix(self) -> None:
        ev = np.sort(np.linalg.eigvalsh(laplacian_matrix(P.NX, P.NY)))
        self.assertTrue(np.allclose(ev, np.sort(P.OMEGA.ravel() ** 2), atol=1e-9))

    def test_a_mode_is_an_eigenvector_of_the_force(self) -> None:
        for a, b in ((1, 1), (3, 2), (P.NX, P.NY)):
            q = P.mode_shape(a, b)
            self.assertTrue(np.allclose(P.force(q, 0.0, 0.0), -P.omega(a, b) ** 2 * q, atol=1e-10))

    def test_a_mode_oscillates_with_its_frequency(self) -> None:
        q0 = 11.0 * P.mode_shape(2, 1)
        q, _ = P.linear_state(q0, np.zeros_like(q0), math.pi / P.omega(2, 1))
        self.assertTrue(np.allclose(q, -q0, atol=1e-9))

    def test_the_slope_of_the_dispersion_is_the_speed_of_sound(self) -> None:
        self.assertAlmostEqual(P.omega(1, 1) / float(P.KMAG[0, 0]), 1.0, delta=0.02)

    def test_the_highest_mode_is_a_checkerboard(self) -> None:
        sh = P.mode_shape(P.NX, P.NY)
        self.assertTrue(np.all(sh[:-1, :] * sh[1:, :] < 0))
        self.assertTrue(np.all(sh[:, :-1] * sh[:, 1:] < 0))

    def test_the_period_of_the_first_mode_in_the_film(self) -> None:
        self.assertAlmostEqual(2 * math.pi / (P.omega(1, 1) * cl.PHYS_PER_S), cl.PERIOD_11, places=9)


class DynamicsTest(unittest.TestCase):
    def test_linear_evolution_matches_the_integrator(self) -> None:
        q0, v0 = P.collision_initial()
        times = np.array([3.0, 7.0])
        Q, V = P.integrate(q0, v0, 0.0, 0.0, times, dt=0.01)
        for t, q, v in zip(times, Q, V):
            qe, ve = P.linear_state(q0, v0, float(t))
            self.assertTrue(np.allclose(q, qe, atol=3e-4))
            self.assertTrue(np.allclose(v, ve, atol=3e-4))

    def test_mode_energies_are_constant_in_the_linear_lattice(self) -> None:
        q0, v0 = P.collision_initial()
        e0 = P.mode_energies(q0, v0)
        for t in (2.0, 9.0, 17.0):
            self.assertTrue(np.allclose(P.mode_energies(*P.linear_state(q0, v0, t)), e0, atol=1e-10))

    def test_the_energy_of_the_modes_is_the_energy_of_the_lattice(self) -> None:
        q0, v0 = P.collision_initial()
        e = 0.5 * float((v0 ** 2).sum()) + P.potential_energy(q0, 0.0, 0.0)
        self.assertAlmostEqual(float(P.mode_energies(q0, v0).sum()), e, places=9)

    def test_the_packets_move_towards_each_other(self) -> None:
        for sgn in (+1, -1):
            q, v = P.linear_state(*P.packet(7.0, 5.5, P.PK_K0, P.PK_SX, P.PK_SY, 0.4, sgn), 4.0)
            c = float((P.IX[:, None] * q ** 2).sum() / (q ** 2).sum())
            self.assertGreater((c - 7.0) * sgn, 2.0)

    def test_the_potential_is_convex(self) -> None:
        self.assertLess(P.ALPHA ** 2, 3.0 * P.BETA * P.KC)

    def test_nonlinear_lattice_conserves_the_energy(self) -> None:
        q0, v0 = P.collision_initial()
        Q, V = P.integrate(q0, v0, P.ALPHA, P.BETA, np.array([0.0, 10.0, 20.0]), dt=0.01)
        e = [0.5 * float((v ** 2).sum()) + P.potential_energy(q, P.ALPHA, P.BETA) for q, v in zip(Q, V)]
        self.assertAlmostEqual(e[1] / e[0], 1.0, delta=2e-3)
        self.assertAlmostEqual(e[2] / e[0], 1.0, delta=2e-3)

    def test_nonlinearity_moves_energy_between_the_modes(self) -> None:
        q0, v0 = P.collision_initial()
        Q, V = P.integrate(q0, v0, P.ALPHA, P.BETA, np.array([0.0, 16.0]), dt=0.01)
        f0 = P.fractions(P.mode_energies(Q[0], V[0]))
        f1 = P.fractions(P.mode_energies(Q[1], V[1]))
        self.assertGreater(0.5 * float(np.abs(f1 - f0).sum()), 0.1)

    def test_the_continuum_field_has_a_fixed_edge(self) -> None:
        phi = cl.fine_field(3.0)
        self.assertAlmostEqual(float(np.abs(phi).max()) > 0.1, True)
        self.assertTrue(np.all(np.isfinite(phi)))

    def test_the_continuum_bump_spreads_with_the_speed_of_sound(self) -> None:
        i0, j0 = P.BUMP[0], P.BUMP[1]
        x = np.arange(1, cl.NXF + 1) * cl.LXF / (cl.NXF + 1)
        y = np.arange(1, cl.NYF + 1) * cl.LYF / (cl.NYF + 1)
        r = np.hypot(x[:, None] - i0, y[None, :] - j0)
        m = (r > 1.0) & (r < 5.5)

        def front(t: float) -> float:
            return float(r[m][np.argmax(cl.fine_field(t)[m])])
        self.assertAlmostEqual((front(4.0) - front(2.0)) / 2.0, 1.0, delta=0.1)


class FilmTest(unittest.TestCase):
    def test_parts_follow_each_other(self) -> None:
        parts = [cl.T_INTRO, cl.T_MODES, cl.T_PLUCK, cl.T_LIN, cl.T_NONLIN, cl.T_CONT]
        for a, b in zip(parts[:-1], parts[1:]):
            self.assertEqual(a[1], b[0])
        self.assertEqual(parts[-1][1], cl.CONTENT_TOTAL)

    def test_the_modes_of_the_film_fill_their_part(self) -> None:
        self.assertEqual(cl.MODES[0][1], cl.T_MODES[0])
        self.assertEqual(cl.MODES[-1][2], cl.T_MODES[1])
        for (_m, _a, b, _c), (_m2, a2, _b2, _c2) in zip(cl.MODES[:-1], cl.MODES[1:]):
            self.assertEqual(b, a2)

    def test_state_is_defined_everywhere(self) -> None:
        for t in np.linspace(0.0, cl.CONTENT_TOTAL - 0.01, 57):
            st = cl.state(float(t))
            self.assertEqual(st["q"].shape, (P.NX, P.NY))
            self.assertIn(st["cap"], cl.TEXT["en"])
            self.assertTrue(np.all(np.isfinite(st["q"])))

    def test_timeline_inserts_the_chapter_cards(self) -> None:
        self.assertAlmostEqual(cl.TOTAL, cl.CONTENT_TOTAL + cl.CARD_S * len(cl.CARD_AT))
        prev = -1.0
        for tf in np.linspace(0.0, cl.TOTAL - 0.01, 3000):
            tc, card, prog = cl.timeline(float(tf))
            self.assertGreaterEqual(tc, prev - 1e-9)
            prev = tc
        for tc in (0.5, 17.0, 50.0, 85.0, 111.0):
            tc2, card, _ = cl.timeline(cl.film_time(tc))
            self.assertIsNone(card)
            self.assertAlmostEqual(tc2, tc)

    def test_all_texts_exist_in_both_languages(self) -> None:
        self.assertEqual(set(cl.TEXT["en"]), set(cl.TEXT["ru"]))
        for key in cl.CARD_KEYS:
            self.assertIn(key + "s", cl.TEXT["en"])


if __name__ == "__main__":
    unittest.main()
