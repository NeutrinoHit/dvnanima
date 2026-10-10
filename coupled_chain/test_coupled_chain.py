from __future__ import annotations

import math
import unittest

import numpy as np

import coupled_chain as cc


def omega_matrix() -> np.ndarray:
    om = np.zeros((cc.N, cc.N))
    for i in range(cc.N):
        om[i, i] = 2.0 * cc.KC
        if i > 0:
            om[i, i - 1] = -cc.KC
        if i < cc.N - 1:
            om[i, i + 1] = -cc.KC
    return om


class ModesTest(unittest.TestCase):
    def test_modes_are_orthonormal(self) -> None:
        self.assertTrue(np.allclose(cc.S @ cc.S.T, np.eye(cc.N), atol=1e-12))

    def test_frequencies_are_the_eigenvalues_of_the_matrix(self) -> None:
        ev = np.sort(np.linalg.eigvalsh(omega_matrix()))
        self.assertTrue(np.allclose(ev, np.sort(cc.OMEGA ** 2), atol=1e-10))

    def test_modes_diagonalize_the_matrix(self) -> None:
        d = cc.S @ omega_matrix() @ cc.S.T
        self.assertTrue(np.allclose(d, np.diag(cc.OMEGA ** 2), atol=1e-10))

    def test_a_mode_oscillates_with_its_frequency(self) -> None:
        s = 4
        x0 = 3.2 * cc.mode_shape(s)
        x, _ = cc.linear_state(x0, np.zeros(cc.N), math.pi / cc.omega(s))
        self.assertTrue(np.allclose(x, -x0, atol=1e-9))

    def test_the_slope_of_the_dispersion_gives_the_speed_of_sound(self) -> None:
        for n in (100, 1000):
            self.assertAlmostEqual(cc.omega(1, n) * (n + 1) / math.pi, 1.0, delta=2.0 / n ** 2 * 100)

    def test_the_highest_mode_has_alternating_signs(self) -> None:
        sh = cc.mode_shape(cc.N)
        self.assertTrue(np.all(sh[:-1] * sh[1:] < 0))

    def test_the_mode_periods_in_the_film(self) -> None:
        self.assertAlmostEqual(2 * math.pi / (cc.omega(1) * cc.PHYS_PER_S), cc.PERIOD_S1, places=9)


class DynamicsTest(unittest.TestCase):
    def test_linear_evolution_matches_the_integrator(self) -> None:
        x0, v0 = cc.collision_initial()
        times = np.array([3.0, 7.5])
        X, V = cc.integrate(x0, v0, 0.0, 0.0, times)
        for t, x, v in zip(times, X, V):
            xe, ve = cc.linear_state(x0, v0, float(t))
            self.assertTrue(np.allclose(x, xe, atol=2e-4))
            self.assertTrue(np.allclose(v, ve, atol=2e-4))

    def test_mode_energies_are_constant_in_the_linear_chain(self) -> None:
        x0, v0 = cc.collision_initial()
        e0 = cc.mode_energies(x0, v0)
        for t in (2.0, 9.0, 17.0):
            self.assertTrue(np.allclose(cc.mode_energies(*cc.linear_state(x0, v0, t)), e0, atol=1e-10))

    def test_the_packets_move_towards_each_other(self) -> None:
        x0, v0 = cc.packet(6.0, cc.PK_K0, cc.PK_SIG, 0.4, +1.0)
        for t, sign in ((4.0, 1), (4.0, -1)):
            x, v = cc.linear_state(*cc.packet(12.0, cc.PK_K0, cc.PK_SIG, 0.4, sign), t)
            c = float((cc.J * x ** 2).sum() / (x ** 2).sum())
            self.assertGreater((c - 12.0) * sign, 2.0)

    def test_the_potential_is_convex(self) -> None:
        self.assertLess(cc.ALPHA ** 2, 3.0 * cc.BETA * cc.KC)

    def test_nonlinear_chain_conserves_the_energy(self) -> None:
        x0, v0 = cc.collision_initial()
        X, V = cc.integrate(x0, v0, cc.ALPHA, cc.BETA, np.array([0.0, 10.0, 20.0]))
        e = [0.5 * float(v @ v) + cc.potential_energy(x, cc.ALPHA, cc.BETA) for x, v in zip(X, V)]
        self.assertAlmostEqual(e[1] / e[0], 1.0, delta=2e-3)
        self.assertAlmostEqual(e[2] / e[0], 1.0, delta=2e-3)

    def test_nonlinearity_moves_energy_between_the_modes(self) -> None:
        x0, v0 = cc.collision_initial()
        X, V = cc.integrate(x0, v0, cc.ALPHA, cc.BETA, np.array([0.0, 20.0]))
        f0 = cc.fractions(cc.mode_energies(X[0], V[0]))
        f1 = cc.fractions(cc.mode_energies(X[1], V[1]))
        self.assertGreater(0.5 * float(np.abs(f1 - f0).sum()), 0.1)

    def test_the_bump_splits_into_two_waves(self) -> None:
        x, _ = cc.linear_state(cc.bump(cc.BUMP_J0, cc.BUMP_SIG, cc.BUMP_A), np.zeros(cc.N), 6.0)
        self.assertLess(float(x[int(cc.BUMP_J0) - 1]), 0.35 * cc.BUMP_A)         # the middle is empty
        self.assertGreater(float(x[int(cc.BUMP_J0) + 5]), 0.25 * cc.BUMP_A)        # a wave went to the right

    def test_the_continuum_field_has_fixed_ends(self) -> None:
        for t in (0.0, 3.0, 11.0):
            f = cc.continuum_field(np.array([0.0, cc.N + 1.0]), t, cc.N + 1.0)
            self.assertTrue(np.allclose(f, 0.0, atol=1e-9))


class FilmTest(unittest.TestCase):
    def test_parts_follow_each_other(self) -> None:
        parts = [cc.T_INTRO, cc.T_MODES, cc.T_PLUCK, cc.T_LIN, cc.T_NONLIN, cc.T_CONT]
        for a, b in zip(parts[:-1], parts[1:]):
            self.assertEqual(a[1], b[0])
        self.assertEqual(parts[-1][1], cc.CONTENT_TOTAL)

    def test_the_modes_of_the_film_fill_their_part(self) -> None:
        self.assertEqual(cc.MODES[0][1], cc.T_MODES[0])
        self.assertEqual(cc.MODES[-1][2], cc.T_MODES[1])
        for (_s, _a, b, _c), (_s2, a2, _b2, _c2) in zip(cc.MODES[:-1], cc.MODES[1:]):
            self.assertEqual(b, a2)

    def test_state_is_defined_everywhere(self) -> None:
        for t in np.linspace(0.0, cc.CONTENT_TOTAL - 0.01, 57):
            st = cc.state(float(t))
            self.assertEqual(st["x"].shape, (cc.N,))
            self.assertIn(st["cap"], cc.TEXT["en"])
            self.assertTrue(np.all(np.isfinite(st["x"])))

    def test_timeline_inserts_the_chapter_cards(self) -> None:
        self.assertAlmostEqual(cc.TOTAL, cc.CONTENT_TOTAL + cc.CARD_S * len(cc.CARD_AT))
        prev = -1.0
        for tf in np.linspace(0.0, cc.TOTAL - 0.01, 3000):
            tc, card, prog = cc.timeline(float(tf))
            self.assertGreaterEqual(tc, prev - 1e-9)                     # content time never runs backwards
            prev = tc
            if card is not None:
                self.assertTrue(0.0 <= prog < 1.0)
        self.assertEqual(cc.timeline(0.5)[1], 0)                          # the film title card first
        self.assertEqual(cc.timeline(cc.CARD_S + 0.5)[1], 1)
        self.assertIsNone(cc.timeline(2 * cc.CARD_S + 0.5)[1])
        self.assertAlmostEqual(cc.timeline(2 * cc.CARD_S + 0.5)[0], 0.5)

    def test_film_time_is_the_inverse_of_the_timeline(self) -> None:
        for tc in (0.5, 17.0, 50.0, 85.0, 111.0):
            tc2, card, _ = cc.timeline(cc.film_time(tc))
            self.assertIsNone(card)
            self.assertAlmostEqual(tc2, tc)

    def test_every_card_has_a_title_in_both_languages(self) -> None:
        for lang in ("en", "ru"):
            for key in cc.CARD_KEYS:
                self.assertIn(key, cc.TEXT[lang])
                self.assertIn(key + "s", cc.TEXT[lang])

    def test_all_texts_exist_in_both_languages(self) -> None:
        self.assertEqual(set(cc.TEXT["en"]), set(cc.TEXT["ru"]))


if __name__ == "__main__":
    unittest.main()
