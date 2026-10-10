from __future__ import annotations

import math
import unittest

import numpy as np

import coupled_lattice_3d as cl
import lattice3d_physics as P


def lattice_matrix(nx: int, ny: int, nz: int) -> np.ndarray:
    n = nx * ny * nz
    om = np.zeros((n, n))
    idx = lambda i, j, k: (i * ny + j) * nz + k
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                r = idx(i, j, k)
                om[r, r] = 6.0 * P.KC
                for d in ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)):
                    ii, jj, kk = i + d[0], j + d[1], k + d[2]
                    if 0 <= ii < nx and 0 <= jj < ny and 0 <= kk < nz:
                        om[r, idx(ii, jj, kk)] = -P.KC
    return om


class ModesTest(unittest.TestCase):
    def test_modes_are_orthonormal(self) -> None:
        for s in (P.SX, P.SY, P.SZ):
            self.assertTrue(np.allclose(s @ s.T, np.eye(len(s)), atol=1e-12))

    def test_frequencies_are_the_eigenvalues_of_the_lattice_matrix(self) -> None:
        ev = np.sort(np.linalg.eigvalsh(lattice_matrix(P.NX, P.NY, P.NZ)))
        self.assertTrue(np.allclose(ev, np.sort(P.OMEGA.ravel() ** 2), atol=1e-9))

    def test_a_mode_is_an_eigenvector_of_the_force(self) -> None:
        for m in ((1, 1, 1), (2, 2, 1), (P.NX, P.NY, P.NZ)):
            q = P.mode_shape(*m)
            self.assertTrue(np.allclose(P.force(q, 0.0, 0.0), -P.omega(*m) ** 2 * q, atol=1e-10))

    def test_the_transform_is_its_own_inverse(self) -> None:
        q, _ = P.collision_initial()
        self.assertTrue(np.allclose(P.from_modes(P.to_modes(q)), q, atol=1e-12))

    def test_a_mode_oscillates_with_its_frequency(self) -> None:
        q0 = 15.0 * P.mode_shape(2, 1, 1)
        q, _ = P.linear_state(q0, np.zeros_like(q0), math.pi / P.omega(2, 1, 1))
        self.assertTrue(np.allclose(q, -q0, atol=1e-9))

    def test_the_slope_of_the_dispersion_is_the_speed_of_sound(self) -> None:
        self.assertAlmostEqual(P.omega(1, 1, 1) / float(P.KMAG[0, 0, 0]), 1.0, delta=0.03)

    def test_the_highest_frequency(self) -> None:
        self.assertLess(P.omega(P.NX, P.NY, P.NZ), P.OMEGA_MAX)
        self.assertGreater(P.omega(P.NX, P.NY, P.NZ), 0.97 * P.OMEGA_MAX)

    def test_the_period_of_the_first_mode_in_the_film(self) -> None:
        self.assertAlmostEqual(2 * math.pi / (P.omega(1, 1, 1) * cl.PHYS_PER_S), cl.PERIOD_111, places=9)


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
            q, _ = P.linear_state(*P.packet(7.0, P.PK_K0, P.PK_SX, P.PK_SY, 0.4, sgn), 4.0)
            c = float((P.IX[:, None, None] * q ** 2).sum() / (q ** 2).sum())
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
        Q, V = P.integrate(q0, v0, P.ALPHA, P.BETA, np.array([0.0, 8.0]), dt=0.01)
        f0 = P.fractions(P.mode_energies(Q[0], V[0]))
        f1 = P.fractions(P.mode_energies(Q[1], V[1]))
        self.assertGreater(0.5 * float(np.abs(f1 - f0).sum()), 0.08)

    def test_the_spectrum_adds_up_to_one(self) -> None:
        q0, v0 = P.collision_initial()
        self.assertAlmostEqual(float(cl.spectrum(P.mode_energies(q0, v0)).sum()), 1.0, places=9)

    def test_the_continuum_bump_spreads_with_the_speed_of_sound(self) -> None:
        i0, j0, k0 = P.BUMP[0], P.BUMP[1], P.BUMP[2]
        x = np.arange(1, cl.NXF + 1) * cl.LXF / (cl.NXF + 1)
        y = np.arange(1, cl.NYF + 1) * cl.LYF / (cl.NYF + 1)
        z = np.arange(1, cl.NZF + 1) * cl.LZF / (cl.NZF + 1)
        r = np.sqrt((x[:, None, None] - i0) ** 2 + (y[None, :, None] - j0) ** 2 + (z[None, None, :] - k0) ** 2)
        m = (r > 1.2) & (r < 3.7)                    # inside the box: the sphere does not reach a wall (4.5 to the side walls)

        def front(t: float) -> float:
            return float(r[m][np.argmax(cl.fine_field(t)[m])])
        self.assertAlmostEqual((front(3.0) - front(1.5)) / 1.5, 1.0, delta=0.2)


class ConfigTest(unittest.TestCase):
    def test_every_font_is_positive_and_the_layout_fits_the_frame(self) -> None:
        for name in cl.CFG.fonts.keys():
            self.assertGreater(getattr(cl.CFG.fonts, name), 0.0)
        for name in ("lattice_axes", "spectrum_axes", "dispersion_axes"):
            x, y, w, h = getattr(cl.CFG.layout, name)
            self.assertTrue(0.0 <= x and 0.0 <= y and x + w <= 1.0 and y + h <= 1.0, name)

    def test_the_parts_cover_the_content_and_the_cards_precede_parts(self) -> None:
        pb = list(cl.CFG.timeline.part_bounds)
        self.assertEqual(pb, sorted(pb))
        for ca in cl.CFG.timeline.card_at:
            self.assertIn(ca, pb)

    def test_the_levels_of_the_limit_refine_the_lattice(self) -> None:
        lv = [tuple(x) for x in cl.CFG.limit.levels]
        for a, b in zip(lv[:-1], lv[1:]):
            self.assertTrue(all(q > p for p, q in zip(a, b)))
        self.assertEqual(lv[0], (cl.NX, cl.NY, cl.NZ))


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
            self.assertEqual(st["q"].shape, (P.NX, P.NY, P.NZ))
            self.assertIn(st["cap"], cl.TEXT["en"])
            self.assertTrue(np.all(np.isfinite(st["q"])))

    def test_timeline_inserts_the_chapter_cards(self) -> None:
        self.assertAlmostEqual(cl.TOTAL, cl.CONTENT_TOTAL + cl.CARD_S * len(cl.CARD_AT))
        prev = -1.0
        for tf in np.linspace(0.0, cl.TOTAL - 0.01, 3000):
            tc, _card, _prog = cl.timeline(float(tf))
            self.assertGreaterEqual(tc, prev - 1e-9)
            prev = tc
        for tc in (0.5, 13.0, 45.0, 70.0, 107.0):
            tc2, card, _ = cl.timeline(cl.film_time(tc))
            self.assertIsNone(card)
            self.assertAlmostEqual(tc2, tc)

    def test_all_texts_exist_in_both_languages(self) -> None:
        self.assertEqual(set(cl.TEXT["en"]), set(cl.TEXT["ru"]))
        for key in cl.CARD_KEYS:
            self.assertIn(key + "s", cl.TEXT["en"])


if __name__ == "__main__":
    unittest.main()
