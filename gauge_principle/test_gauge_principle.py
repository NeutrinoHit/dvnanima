"""Tests of the physics, of the timeline, of the configuration and of the layout of the film "The gauge principle: two friends and a field".

    python -m pytest test_gauge_principle.py
"""

from __future__ import annotations

import math
import re
import subprocess
import sys
import tempfile
import tomllib
import unittest
from pathlib import Path

import numpy as np

import gauge_physics as gp
import gauge_principle as film

HERE = Path(__file__).resolve().parent
CFG = film.CFG
PI = math.pi


# ================================================================================================ part 1: two friends and a phase

class FriendsTest(unittest.TestCase):
    def test_the_difference_depends_on_the_conventions(self) -> None:
        n = film.part1_numbers()
        d = CFG.part1.phase_difference
        self.assertAlmostEqual(n["same"], 2 * math.sin(d / 2), places=12)
        a = n["alpha"]
        self.assertAlmostEqual(n["different"], 2 * abs(math.sin((d + a[1] - a[0]) / 2)), places=12)
        self.assertGreater(abs(n["different"] - n["same"]), 0.5)

    def test_a_common_phase_changes_neither_the_modulus_nor_the_difference(self) -> None:
        psi = np.exp(1j * np.array([-0.3, 0.8]))
        for al in (0.0, 1.0, 2.5, -4.0):
            self.assertAlmostEqual(abs(np.exp(1j * al) * psi[0] - np.exp(1j * al) * psi[1]), abs(psi[0] - psi[1]), places=12)
            self.assertTrue(np.allclose(np.abs(np.exp(1j * al) * psi), np.abs(psi)))

    def test_the_wilson_line_makes_the_difference_well_defined(self) -> None:
        """psi(y) - U(y,x) psi(x) does not change its modulus under psi -> e^{i alpha} psi, U -> e^{i alpha(y)} U e^{-i alpha(x)}."""
        rng = np.random.default_rng(3)
        psi_x, psi_y = rng.normal(size=2) + 1j * rng.normal(size=2)
        u = np.exp(-1j * 0.7)
        ax, ay = 0.9, -2.1
        before = abs(psi_y - u * psi_x)
        after = abs(np.exp(1j * ay) * psi_y - np.exp(1j * ay) * u * np.exp(-1j * ax) * np.exp(1j * ax) * psi_x)
        self.assertAlmostEqual(before, after, places=12)
        self.assertGreater(abs(abs(np.exp(1j * ay) * psi_y - np.exp(1j * ax) * psi_x) - abs(psi_y - psi_x)), 1e-3)     # without U it is ambiguous


# ================================================================================================ part 2: two paths

class InterferenceTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.itf = film.itf()

    def test_the_beam_solves_the_paraxial_equation(self) -> None:
        """2 i k d_x psi + d_y^2 psi = 0 for the envelope e^{-ikx} beam (finite differences)."""
        k, sg = self.itf.k, self.itf.sigma
        h = 1e-4
        for x0, y0 in ((1.0, 0.3), (4.0, -0.7), (8.0, 1.1)):
            env = lambda x, y: gp.beam(x, y, 0.2, k, sg) * np.exp(-1j * k * x)            # noqa: E731
            dx = (env(x0 + h, y0) - env(x0 - h, y0)) / (2 * h)
            dyy = (env(x0, y0 + h) - 2 * env(x0, y0) + env(x0, y0 - h)) / h ** 2
            self.assertLess(abs(2j * k * dx + dyy), 1e-4 * (k * abs(dx) + abs(dyy)) + 1e-6)

    def test_the_power_of_a_beam_is_conserved(self) -> None:
        y = np.linspace(-30, 30, 20001)
        p = [np.trapezoid(np.abs(gp.beam(x, y, 0.5, self.itf.k, self.itf.sigma)) ** 2, y) for x in (0.0, 3.0, 8.0)]
        self.assertTrue(np.allclose(p, p[0], rtol=1e-6))

    def test_a_common_phase_does_not_change_the_pattern(self) -> None:
        ref = self.itf.screen_intensity([0.0, 0.0])
        for th in (0.3, 1.7, PI, 5.0, -2.0):
            self.assertLess(np.max(np.abs(self.itf.screen_intensity([th, th]) - ref)), 1e-12)

    def test_the_pattern_depends_only_on_the_relative_phase(self) -> None:
        for d in (0.4, 1.0, 2.2, 4.0):
            a = self.itf.screen_intensity([d, 0.0])
            b = self.itf.screen_intensity([d + 1.3, 1.3])
            self.assertLess(np.max(np.abs(a - b)), 1e-12)

    def test_a_relative_phase_moves_the_fringes_by_delta_over_two_pi_periods(self) -> None:
        for d in (0.3, 0.7 * PI, 1.0 * PI, 1.4 * PI, 2.0 * PI, 2.6 * PI):
            ints = self.itf.screen_intensity([d, 0.0])
            shift = self.itf.central_shift(ints, expected=d / (2 * PI))
            with self.subTest(delta=d):
                self.assertAlmostEqual(shift, d / (2 * PI), delta=0.01)

    def test_after_two_pi_the_pattern_returns(self) -> None:
        self.assertLess(np.max(np.abs(self.itf.screen_intensity([2 * PI, 0.0]) - self.itf.screen_intensity([0.0, 0.0]))), 1e-12)

    def test_the_fringe_period_is_lambda_L_over_d(self) -> None:
        I = CFG.interf
        ff = gp.fringe_period_farfield(I.wavelength, I.screen_x - I.barrier_x, I.slit_separation)
        self.assertAlmostEqual(self.itf.period, ff, delta=0.02 * ff)

    def test_the_central_fringe_is_on_the_axis_without_a_phase(self) -> None:
        self.assertAlmostEqual(self.itf.peak0, 0.0, delta=1e-3)

    def test_the_intensity_is_normalised_to_one_at_the_centre(self) -> None:
        self.assertAlmostEqual(float(self.itf.screen_intensity([0.0, 0.0]).max()), 1.0, places=9)

    def test_the_map_has_the_same_pattern_as_the_screen_profile(self) -> None:
        y = self.itf.y_screen
        x = np.full_like(y, CFG.interf.screen_x)
        psi = self.itf.field_map(x, y, [1.0, 1.0 - 0.8])
        self.assertTrue(np.allclose(np.abs(psi) ** 2, self.itf.screen_intensity([1.0, 0.2]), atol=1e-12))


# ================================================================================================ parts 3 and 4: the particle on a line

class ChainTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.sim = film.chain()

    def test_the_hamiltonians_are_hermitian(self) -> None:
        sim = self.sim
        for th in (None, sim.theta_a):
            h = sim.ch.hamiltonian(th)
            self.assertTrue(np.allclose(h, h.conj().T))

    def test_the_evolution_is_unitary_and_conserves_the_energy(self) -> None:
        sim = self.sim
        h = sim.ch.hamiltonian(sim.theta_a)
        e0 = np.vdot(sim.psi0_new, h @ sim.psi0_new).real
        for tau in (0.5, 2.0, sim.t_end):
            _, _, psi = sim.states(tau)
            self.assertAlmostEqual(sim.ch.norm(psi), 1.0, places=10)
            self.assertAlmostEqual(np.vdot(psi, h @ psi).real, e0, places=8)

    def test_the_free_packet_moves_with_the_group_velocity(self) -> None:
        sim = self.sim
        C = CFG.chain
        t = sim.t_end
        ptrue, _, _ = sim.states(t)
        self.assertAlmostEqual((sim.ch.centroid(ptrue) - C.packet_x0) / t, C.packet_k0 / C.mass, delta=0.01 * C.packet_k0)

    def test_the_new_convention_looks_like_a_force_on_the_free_packet(self) -> None:
        """The packet of the experimenter, evolved with the FREE equation, drifts with (k_0 + <alpha'>)/m instead of k_0/m."""
        sim = self.sim
        C = CFG.chain
        t = sim.t_end
        _, pnaive, _ = sim.states(t)
        v = (sim.ch.centroid(pnaive) - C.packet_x0) / t
        self.assertAlmostEqual(v, sim.packet_speed("naive"), delta=0.01 * v)
        self.assertGreater(abs(v - C.packet_k0 / C.mass), 1.0)

    def test_the_naive_packet_is_not_the_rephased_true_packet(self) -> None:
        sim = self.sim
        t = 1.5
        ptrue, pnaive, _ = sim.states(t)
        self.assertGreater(float(np.max(np.abs(np.abs(pnaive) ** 2 - np.abs(ptrue) ** 2))), 0.05)

    def test_the_covariant_evolution_equals_the_rephased_true_evolution(self) -> None:
        sim = self.sim
        for t in (0.0, 0.8, 1.6, 3.2):
            ptrue, _, pcov = sim.states(t)
            with self.subTest(t=t):
                self.assertLess(float(np.max(np.abs(np.abs(pcov) ** 2 - np.abs(ptrue) ** 2))), 1e-9)
                self.assertLess(float(np.max(np.abs(pcov - np.exp(1j * sim.alpha) * ptrue))), 1e-8)

    def test_the_covariant_derivative_transforms_like_psi_and_the_naive_one_does_not(self) -> None:
        sim = self.sim
        ptrue, _, pcov = sim.states(1.3)
        dn, dc = sim.derivative_defects(ptrue, pcov)
        self.assertLess(dc, 1e-9)
        self.assertGreater(dn, 0.3)

    def test_links_transform_like_wilson_lines(self) -> None:
        """U(x_{j+1}, x_j) = exp(-i q theta_j) -> e^{i alpha_{j+1}} U e^{-i alpha_j} when A -> A - d alpha / q."""
        sim = self.sim
        q = sim.q
        rng = np.random.default_rng(5)
        a_func = lambda x: 0.4 * np.sin(1.3 * x) + 0.2                 # noqa: E731
        theta = sim.ch.links(a_func)
        alpha = rng.normal(size=len(sim.ch.x))
        theta_new = theta + sim.ch.gauge_links(alpha)
        u, u_new = np.exp(-1j * q * theta), np.exp(-1j * q * theta_new)
        self.assertTrue(np.allclose(u_new, np.exp(1j * alpha[1:]) * u * np.exp(-1j * alpha[:-1]), atol=1e-12))

    def test_a_general_field_is_covariant_for_a_general_alpha(self) -> None:
        """D' psi' = e^{i alpha} D psi with a non-trivial A and a random alpha (independent of the film's alpha)."""
        sim = self.sim
        ch = sim.ch
        rng = np.random.default_rng(11)
        a_func = lambda x: 0.7 * np.cos(0.9 * x) - 0.3                 # noqa: E731
        theta = ch.links(a_func)
        alpha = np.cumsum(rng.normal(size=len(ch.x))) * 0.05
        psi = np.exp(-(ch.x ** 2) / 8.0) * (1 + 0.3j * np.sin(ch.x))
        theta_new = theta + ch.gauge_links(alpha)
        lhs = ch.covariant_derivative(np.exp(1j * alpha) * psi, theta_new)
        rhs = np.exp(1j * alpha) * ch.covariant_derivative(psi, theta)
        self.assertLess(float(np.max(np.abs(lhs - rhs))), 1e-10)
        h_new, h = ch.hamiltonian(theta_new), ch.hamiltonian(theta)
        self.assertLess(float(np.max(np.abs(h_new @ (np.exp(1j * alpha) * psi) - np.exp(1j * alpha) * (h @ psi)))), 1e-9)

    def test_the_lattice_derivative_converges_to_the_continuum_covariant_derivative(self) -> None:
        errs = []
        for n in (512, 1024):
            ch = gp.Chain(n, 6.0, 1.0, 1.0)
            a_func = lambda x: 0.8 * np.sin(x) + 0.3                       # noqa: E731
            psi = np.exp(-(ch.x ** 2) / 4.0 + 0.7j * ch.x)
            dpsi = (-ch.x / 2.0 + 0.7j) * psi
            exact = dpsi + 1j * a_func(ch.x) * psi
            num = ch.covariant_derivative(psi, ch.links(a_func))
            errs.append(float(np.max(np.abs(num[2:-2] - exact[2:-2]))))
        self.assertLess(errs[1], errs[0] / 3.5)                             # second order in the lattice spacing
        self.assertLess(errs[1], 1e-3)

    def test_the_lattice_hamiltonian_converges_to_the_continuum_one(self) -> None:
        ch = gp.Chain(2048, 6.0, 1.0, 1.0)
        a_func = lambda x: 0.8 * np.sin(x) + 0.3                           # noqa: E731
        psi = np.exp(-(ch.x ** 2) / 4.0 + 0.7j * ch.x)
        d1 = (-ch.x / 2.0 + 0.7j) * psi
        d2 = ((-ch.x / 2.0 + 0.7j) ** 2 - 0.5) * psi
        a = a_func(ch.x)
        da = 0.8 * np.cos(ch.x)
        # (d + i a)^2 psi = d2 + 2 i a d1 + i a' psi - a^2 psi
        exact = -0.5 * (d2 + 2j * a * d1 + 1j * da * psi - a ** 2 * psi)
        num = ch.hamiltonian(ch.links(a_func)) @ psi
        self.assertLess(float(np.max(np.abs(num[3:-3] - exact[3:-3]))), 2e-3)

    def test_the_mean_phase_gradient_is_the_exact_drift(self) -> None:
        sim = self.sim
        self.assertAlmostEqual(sim.mean_dalpha, sim.ch.mean_of(sim.dalpha(sim.ch.x), sim.psi0), places=12)
        self.assertGreater(sim.mean_dalpha, 1.0)


# ================================================================================================ part 5: the field strength and the loop

class PlaneTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.pl = film.plane()

    def grid(self, h: float = 0.02, half=(2.0, 1.0)):
        fx = np.arange(-half[0], half[0] + 1e-9, h)
        fy = np.arange(-half[1], half[1] + 1e-9, h)
        return np.meshgrid(fx, fy)

    def test_the_tube_potential_has_the_right_curl_and_flux(self) -> None:
        pl = self.pl
        h = 0.01
        X, Y = np.meshgrid(np.arange(-1.5, 1.5, h), np.arange(-1.5, 1.5, h))
        ax, ay = gp.tube_potential(X, Y, pl.flux, 0.25)
        f = gp.curl_fd(ax, ay, h)
        self.assertLess(float(np.max(np.abs(f - gp.tube_field(X[1:-1, 1:-1], Y[1:-1, 1:-1], pl.flux, 0.25)))) / gp.tube_field(0.0, 0.0, pl.flux, 0.25), 2e-3)
        r = 1.4
        th = np.linspace(0, 2 * PI, 4001)[:-1]
        ax, ay = gp.tube_potential(r * np.cos(th), r * np.sin(th), pl.flux, 0.25)
        circ = float(np.sum(-ax * np.sin(th) * r + ay * np.cos(th) * r) * (th[1] - th[0]))
        self.assertAlmostEqual(circ, pl.flux * (1 - math.exp(-r * r / (2 * 0.25 ** 2))), delta=1e-9)      # the flux inside the circle

    def test_a_pure_gauge_field_has_no_field_strength(self) -> None:
        pl = self.pl
        X, Y = self.grid()
        ax, ay = pl.field.a_field(X, Y, 0.0, 1.0)
        self.assertGreater(float(np.max(np.hypot(ax, ay))), 1.0)                       # arrows everywhere ...
        fpk = float(pl.field.b_field(np.array(-1.8), np.array(0.0), 1.0))
        self.assertLess(float(np.max(np.abs(gp.curl_fd(ax, ay, 0.02)))) / fpk, 1e-4)   # ... and F = 0 (finite-difference error only)

    def test_the_field_strength_is_gauge_invariant(self) -> None:
        pl = self.pl
        X, Y = self.grid()
        fpk = float(pl.field.b_field(np.array(-1.8), np.array(0.0), 1.0))
        a0 = pl.field.a_field(X, Y, 1.0, 0.0)
        f0 = gp.curl_fd(*a0, 0.02)
        for g in (0.5, 1.0, 1.5):
            a1 = pl.field.a_field(X, Y, 1.0, g)
            with self.subTest(g=g):
                self.assertGreater(float(np.max(np.abs(np.hypot(a1[0] - a0[0], a1[1] - a0[1])))), 0.5)       # A changes a lot
                self.assertLess(float(np.max(np.abs(gp.curl_fd(*a1, 0.02) - f0))) / fpk, 1e-4)             # F does not

    def test_the_loop_integral_is_the_flux_stokes(self) -> None:
        pl = self.pl
        for loop in (pl.loop1, pl.loop2):
            line = pl.loop_phase(loop, 1.0, 0.0)
            area = pl.loop_flux(loop, 1.0)
            with self.subTest(n=len(loop)):
                self.assertAlmostEqual(line, area, delta=1e-6 * pl.q * pl.flux)

    def test_the_loop_around_the_tube_collects_q_phi_and_the_other_loop_nothing(self) -> None:
        pl = self.pl
        self.assertAlmostEqual(pl.loop_phase(pl.loop1, 1.0, 0.0), pl.q * pl.flux, delta=2e-3 * pl.q * pl.flux)
        self.assertAlmostEqual(pl.loop_phase(pl.loop2, 1.0, 0.0), 0.0, delta=1e-9)

    def test_the_loop_is_gauge_invariant(self) -> None:
        pl = self.pl
        for loop in (pl.loop1, pl.loop2):
            base = pl.loop_phase(loop, 1.0, 0.0)
            for g in (0.5, 1.0, 1.5):
                self.assertAlmostEqual(pl.loop_phase(loop, 1.0, g), base, delta=1e-9)

    def test_a_pure_gauge_loop_is_zero(self) -> None:
        pl = self.pl
        for loop in (pl.loop1, pl.loop2):
            self.assertAlmostEqual(pl.loop_phase(loop, 0.0, 1.0), 0.0, delta=1e-9)

    def test_the_flux_is_additive_and_linear(self) -> None:
        pl = self.pl
        self.assertAlmostEqual(pl.loop_flux(pl.loop1, 0.5), 0.5 * pl.loop_flux(pl.loop1, 1.0), delta=1e-9)

    def test_the_orientation_of_the_triangle_is_counter_clockwise(self) -> None:
        self.assertGreater(gp.polygon_area(self.pl.loop1), 0.0)
        self.assertGreater(gp.polygon_area(self.pl.loop2), 0.0)

    def test_the_fringes_shift_by_q_phi_over_two_pi_in_any_gauge(self) -> None:
        pl = self.pl
        ref = pl.screen_pattern(0.0, 0.0)
        shifted = pl.screen_pattern(1.0, 0.0)
        expect = pl.q * pl.flux / (2 * PI)
        self.assertAlmostEqual(pl.peak_shift(shifted, expect), expect, delta=0.005)
        self.assertGreater(float(np.max(np.abs(shifted - ref))), 0.3)
        for g in (0.4, 1.0, 1.5):
            with self.subTest(g=g):
                self.assertLess(float(np.max(np.abs(pl.screen_pattern(1.0, g) - shifted))), 1e-9)
        for g in (0.4, 1.0, 1.5):                                                        # a pure gauge field does not move the fringes
            self.assertLess(float(np.max(np.abs(pl.screen_pattern(0.0, g) - ref))), 1e-9)

    def test_the_dials_turn_with_the_convention(self) -> None:
        pl = self.pl
        d0 = pl.dial_angles(0.0, 0.0)
        d1 = pl.dial_angles(0.0, 1.0)
        for j, s in enumerate(pl.slits):
            want = float(pl.field.alpha(*s))
            self.assertAlmostEqual(math.remainder(d1[j] - d0[j] - want, 2 * PI), 0.0, delta=1e-9)

    def test_the_sense_of_the_shift_follows_the_sign_of_the_flux(self) -> None:
        pl = self.pl
        pat = pl.screen_pattern(1.0, 0.0)
        pat_neg = pl.screen_pattern(-1.0, 0.0)
        self.assertAlmostEqual(pl.peak_shift(pat_neg, -pl.q * pl.flux / (2 * PI)), -pl.q * pl.flux / (2 * PI), delta=0.005)
        self.assertGreater(pl.peak_shift(pat, 0.35), 0.0)


# ================================================================================================ timeline, configuration, layout

class TimelineTest(unittest.TestCase):
    def test_parts_follow_each_other(self) -> None:
        pb = film.PB
        self.assertEqual(pb[0], 0.0)
        self.assertTrue(all(b > a for a, b in zip(pb, pb[1:])))
        self.assertEqual(film.CARD_AT, [0.0, *pb[:-1]])
        self.assertEqual(len(film.CARD_D), len(film.CARD_AT))

    def test_the_length_is_between_60_and_110_seconds(self) -> None:
        self.assertTrue(60.0 <= film.TOTAL <= 110.0, film.TOTAL)

    def test_timeline_inserts_the_chapter_cards(self) -> None:
        t, card, prog = film.timeline(0.5 * film.CARD_D[0])
        self.assertEqual((card, t), (0, 0.0))
        self.assertAlmostEqual(prog, 0.5)
        t, card, _ = film.timeline(film.CARD_D[0] + film.CARD_D[1] + 1.0)
        self.assertIsNone(card)
        self.assertAlmostEqual(t, 1.0)

    def test_film_time_is_the_inverse_of_the_timeline(self) -> None:
        for tc in (0.3, 10.0, 16.2, 31.0, 45.5, 60.0, 78.0, 85.0):
            tf = film.film_time(tc)
            t, card, _ = film.timeline(tf)
            self.assertIsNone(card)
            self.assertAlmostEqual(t, tc, places=9)

    def test_every_part_has_its_events_inside_it(self) -> None:
        pb = film.PB
        dur = [b - a for a, b in zip(pb, pb[1:])]
        P1, P2, P3, P4, P5, P6 = (CFG.part1, CFG.part2, CFG.part3, CFG.part4, CFG.part5, CFG.part6)

        def flat(x):
            return [v for item in x for v in (flat(item) if isinstance(item, list) else [item])]

        def ends(*intervals):
            return max(flat(list(intervals)))
        self.assertLessEqual(ends(P1.fly, P1.dials_in, P1.hands_in, P1.turn, P1.signals, P1.rows_in, P1.question_in), dur[0])
        self.assertLessEqual(ends(P2.common_turn, P2.relative_turn, P2.formula_in), dur[1])
        self.assertLessEqual(ends(P3.run, P3.formula_in), dur[2])
        self.assertLessEqual(ends(P4.run, P4.formula_in, P4.morph), dur[3])
        self.assertLessEqual(ends(P5.gauge1, P5.tube_on, P5.gauge2, P5.formula_in), dur[4])
        self.assertLessEqual(max(max(t) for t in P6.times), dur[5])

    def test_the_packet_runs_for_the_whole_simulated_time(self) -> None:
        for P in (CFG.part3, CFG.part4):
            self.assertGreater(P.run[1] - P.run[0], 5.0)

    def test_the_packet_stays_away_from_the_walls(self) -> None:
        sim = film.chain()
        for psi in sim.states(sim.t_end):
            rho = np.abs(psi) ** 2
            self.assertLess(float(rho[:5].max() + rho[-5:].max()), 1e-3)


class ConfigTest(unittest.TestCase):
    """The numbers live in config.toml and the words in texts.toml (see ../test_dvconfig.py for the same rules)."""

    def setUp(self) -> None:
        self.cfg = tomllib.loads((HERE / "config.toml").read_text(encoding="utf-8"))
        self.texts = tomllib.loads((HERE / "texts.toml").read_text(encoding="utf-8"))

    def test_both_languages_have_the_same_keys(self) -> None:
        self.assertEqual(set(self.texts), {"en", "ru"})
        self.assertEqual(set(self.texts["en"]), set(self.texts["ru"]))

    def test_formulas_are_balanced_and_use_plain_latex(self) -> None:
        for lang, table in self.texts.items():
            for key, text in table.items():
                with self.subTest(lang=lang, key=key):
                    self.assertEqual(text.count("$") % 2, 0)
                    for bad in ("\\bm", "\\Box", "\\text"):
                        self.assertNotIn(bad, text)

    def test_the_same_placeholders_in_both_languages(self) -> None:
        for key in self.texts["en"]:
            with self.subTest(key=key):
                self.assertEqual(sorted(re.findall(r"@[a-z0-9_]+@", self.texts["en"][key])), sorted(re.findall(r"@[a-z0-9_]+@", self.texts["ru"][key])))

    def test_placeholders_stand_inside_formulas(self) -> None:
        for lang, table in self.texts.items():
            for key, text in table.items():
                outside = re.sub(r"\$[^$]*\$", "", text)
                with self.subTest(lang=lang, key=key):
                    self.assertIsNone(re.search(r"@[a-z0-9_]+@", outside))

    def test_no_russian_words_inside_formulas_and_no_english_in_russian(self) -> None:
        for key, text in self.texts["ru"].items():
            for formula in re.findall(r"\$([^$]*)\$", text):
                with self.subTest(key=key):
                    self.assertIsNone(re.search(r"[А-Яа-яЁё]", formula))
        words = re.compile(r"\b(the|and|of|with|field|phase|convention)\b")
        for key, text in self.texts["ru"].items():
            with self.subTest(key=key):
                self.assertIsNone(words.search(re.sub(r"\$[^$]*\$", "", text)))

    def test_no_traces_of_discussions_in_the_texts(self) -> None:
        for lang, table in self.texts.items():
            for key, text in table.items():
                low = text.lower()
                for bad in ("author", "audit", "borrow", "lend", "please check", "автор", "аудит", "занимает", "одалж"):
                    with self.subTest(lang=lang, key=key, bad=bad):
                        self.assertNotIn(bad, low)

    def test_video_section_is_complete_and_16_to_9(self) -> None:
        v = self.cfg["video"]
        keys = {"width", "height", "fps", "dpi", "crf", "preset", "preview_width", "preview_height", "preview_fps", "fade_s", "reference_height"}
        self.assertTrue(keys <= set(v), keys - set(v))
        self.assertEqual(v["width"] * 9, v["height"] * 16)
        self.assertGreater(v["width"], v["preview_width"])
        self.assertIn("style", self.cfg)

    def test_script_reads_the_config(self) -> None:
        src = (HERE / "gauge_principle.py").read_text(encoding="utf-8")
        self.assertIn("load_config", src)
        for forbidden in ('"libx264", "-preset", "slow"', "(1280, 720)", "dpi = 100\n"):
            self.assertNotIn(forbidden, src)

    def test_set_overrides_a_value(self) -> None:
        out = subprocess.run([sys.executable, "-c", "import gauge_principle as m; print(m.CFG.video.crf)", "--set", "video.crf=23"],
                             cwd=HERE, capture_output=True, text=True, check=True).stdout.strip()
        self.assertEqual(out, "23")

    def test_a_misspelt_set_is_an_error(self) -> None:
        r = subprocess.run([sys.executable, "-c", "import gauge_principle", "--set", "video.crff=23"], cwd=HERE, capture_output=True, text=True)
        self.assertNotEqual(r.returncode, 0)

    def test_every_text_key_used_by_the_script_exists(self) -> None:
        src = (HERE / "gauge_principle.py").read_text(encoding="utf-8")
        used = set(re.findall(r'tx\["([a-zA-Z0-9_]+)"\]', src)) | set(re.findall(r'self\.(?:caption|formula)\(\s*"([a-zA-Z0-9_]+)"', src))
        used |= {"c1a", "c1b", "c1c", "c2a", "c2b", "c2c", "c2d", "c3a", "c3b", "c3c", "c4a", "c4b", "c4c", "c5a", "c5b", "c5c", "f3a", "f3b", "f4a", "f4b", "path1", "path2",
                 "experimenter_x", "experimenter_y", "psi_x", "psi_y", "psi_x_new", "psi_y_new", "loop1", "loop2", "row_exp_free", "row_exp_cov"}
        used |= set(CFG.part6.lines)
        missing = used - set(self.texts["en"])
        self.assertFalse(missing, missing)

    def test_every_card_has_a_title_in_both_languages(self) -> None:
        for lang in ("en", "ru"):
            for key in film.CARD_KEYS:
                self.assertIn(key, film.TEXT[lang])
                self.assertIn(key + "s", film.TEXT[lang])

    def test_no_numbers_hidden_in_the_script_that_belong_to_the_physics(self) -> None:
        """The values of the physics and of the pictures are read from the configuration (spot check of the keys)."""
        for sect, keys in (("chain", ["n", "alpha_slope", "packet_k0", "t_end"]), ("plane", ["flux_over_pi", "tube_sigma"]), ("interf", ["wavelength", "slit_separation"])):
            for k in keys:
                self.assertIn(k, self.cfg[sect])


class LayoutTest(unittest.TestCase):
    """Rendered frames: every text is inside the frame and no two texts overlap (measured with get_window_extent)."""

    TIMES = [0.5, 2.5, 4.5, 6.0, 8.0, 10.0, 12.0, 14.5, 16.5, 18.0, 20.5, 23.0, 25.0, 27.0, 29.5,                     # cards, parts 1 and 2
             33.5, 36.0, 38.5, 40.5, 42.5, 45.5, 48.0, 50.0, 52.0, 54.5, 57.0, 59.5,                                  # parts 3 and 4
             62.0, 64.5, 66.5, 68.5, 71.0, 73.5, 75.5, 77.0, 79.5, 81.5, 83.5, 85.0]                                  # parts 5 and 6 (content time)

    @classmethod
    def setUpClass(cls) -> None:
        cls.canvas = {lang: film.Canvas((film.CFG.video.width, film.CFG.video.height), lang) for lang in ("en", "ru")}

    def film_times(self):
        return [film.film_time(t) for t in self.TIMES] + [0.5 * film.CARD_D[0]]

    def test_texts_are_inside_the_frame_and_do_not_overlap(self) -> None:
        W, H = film.CFG.video.width, film.CFG.video.height
        for lang, cv in self.canvas.items():
            for tf in self.film_times():
                cv.draw(tf, film.TOTAL)
                boxes = cv.text_boxes()
                for s, b in boxes:
                    with self.subTest(lang=lang, t=tf, text=s):
                        self.assertGreaterEqual(b.x0, -1.0)
                        self.assertGreaterEqual(b.y0, -1.0)
                        self.assertLessEqual(b.x1, W + 1.0)
                        self.assertLessEqual(b.y1, H + 1.0)
                for i, (s1, b1) in enumerate(boxes):
                    for s2, b2 in boxes[i + 1:]:
                        ox = min(b1.x1, b2.x1) - max(b1.x0, b2.x0)
                        oy = min(b1.y1, b2.y1) - max(b1.y0, b2.y0)
                        with self.subTest(lang=lang, t=tf, a=s1, b=s2):
                            self.assertFalse(ox > 2.0 and oy > 2.0, f"{s1!r} overlaps {s2!r}")

    def test_russian_numbers_use_the_decimal_comma(self) -> None:
        cv = self.canvas["ru"]
        for tf in self.film_times():
            cv.draw(tf, film.TOTAL)
            for s, _ in cv.text_boxes():
                with self.subTest(t=tf, text=s):
                    self.assertIsNone(re.search(r"\d\.\d", s))

    def test_the_numbers_on_the_screen_are_the_computed_ones(self) -> None:
        cv = self.canvas["en"]
        n = film.part1_numbers()
        cv.draw(film.film_time(film.CFG.part1.rows_in[1][1] + 1.0), film.TOTAL)
        strings = " ".join(s for s, _ in cv.text_boxes())
        self.assertIn(f"{n['same']:.2f}", strings)
        self.assertIn(f"{n['different']:.2f}", strings)
        t0 = film.PB[4]
        cv.draw(film.film_time(t0 + film.CFG.part5.tube_on[1] + 0.5), film.TOTAL)
        strings = " ".join(s for s, _ in cv.text_boxes())
        pl = film.plane()
        self.assertIn(f"{pl.loop_phase(pl.loop1, 1.0, 0.0) / PI:.3f}", strings)


class FilmSmokeTest(unittest.TestCase):
    def test_a_snapshot_in_each_language(self) -> None:
        from PIL import Image
        with tempfile.TemporaryDirectory() as d:
            sizes = {}
            for lang in ("en", "ru"):
                out = Path(d) / f"s_{lang}.mp4"
                subprocess.run([sys.executable, "gauge_principle.py", "--lang", lang, "--snapshot", "30", "--out", str(out)], cwd=HERE, check=True, capture_output=True)
                sizes[lang] = Image.open(out.with_suffix(".png")).size
            self.assertEqual(sizes["en"], (film.CFG.video.width, film.CFG.video.height))
            self.assertEqual(sizes["en"], sizes["ru"])


if __name__ == "__main__":
    unittest.main()
