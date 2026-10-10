"""Physics tests of the film: explicit Dirac spinors, spherical harmonics and numerical angular integrals against the closed forms.

    python -m unittest test_ffbar
"""

from __future__ import annotations

import math
import unittest

import numpy as np
from scipy.optimize import brentq
from scipy.special import sph_harm_y, spherical_jn

import ffbar as fb

E_H, M_F, V_EV = 62.5, 4.18, 246.0                       # a b quark in the decay of the Higgs boson (GeV)
P_F = math.sqrt(E_H ** 2 - M_F ** 2)
BETA = P_F / E_H
UNIT_N = M_F * P_F * E_H / (2.0 * math.pi * V_EV)         # the unit N of the amplitudes
I2, Z2 = np.eye(2), np.zeros((2, 2))
G0 = np.block([[I2, Z2], [Z2, -I2]])                      # Dirac representation
G5 = np.block([[Z2, I2], [I2, Z2]])
RNG = np.random.default_rng(20260219)
QUAD_PTS, QUAD_W = fb.gauss_sphere(40, 80)


def rand_unit() -> np.ndarray:
    return fb.unit(RNG.normal(size=3))


def u_spinor(n, ch):
    return np.concatenate([math.sqrt(E_H + M_F) * ch, math.sqrt(E_H - M_F) * fb.sdot(n) @ ch])


def v_spinor(n, et):                                      # v(-p n): Eq. (12) of the paper
    return np.concatenate([-math.sqrt(E_H - M_F) * fb.sdot(n) @ et, math.sqrt(E_H + M_F) * et])


def bilinears(n, st: fb.PairState) -> tuple[complex, complex]:
    uu, vv = u_spinor(n, fb.chi(st.s1, st.rho1)), v_spinor(n, fb.eta(st.s2, st.rho2))
    ub = uu.conj() @ G0
    return complex(ub @ vv), complex(ub @ G5 @ vv)


def psi_numeric(rvec, st: fb.PairState) -> tuple[complex, complex]:
    """psi_S, psi_P in the literal conventions of the paper, Eqs. (5)-(7), (19): i m_f p / (16 pi^2 v) Int dOmega e^{i p n.r} (...)."""
    sm = pm = 0j
    for n, w in zip(QUAD_PTS, QUAD_W):
        a, b = bilinears(n, st)
        ph = np.exp(1j * P_F * float(n @ rvec))
        sm += w * ph * a
        pm += w * ph * b
    c0 = 1j * M_F * P_F / (16.0 * math.pi ** 2 * V_EV)
    return c0 * sm, c0 * pm


def random_state() -> fb.PairState:
    return fb.PairState(tuple(rand_unit()), tuple(rand_unit()), int(RNG.choice([-1, 1])), int(RNG.choice([-1, 1])))


class SpinorTest(unittest.TestCase):
    def test_eigenvalues_and_projectors(self) -> None:
        for _ in range(5):
            s = rand_unit()
            for rho in (1, -1):
                ch, et = fb.chi(s, rho), fb.eta(s, rho)
                self.assertTrue(np.allclose(fb.sdot(s) @ ch, rho * ch))
                self.assertTrue(np.allclose(fb.sdot(s) @ et, -rho * et))                  # the antifermion spinor has the opposite eigenvalue
                self.assertAlmostEqual(float(np.vdot(ch, ch).real), 1.0)
                self.assertAlmostEqual(float(np.vdot(et, et).real), 1.0)
                self.assertTrue(np.allclose(np.outer(ch, ch.conj()), 0.5 * (np.eye(2) + rho * fb.sdot(s))))      # Eq. (A1)
                self.assertTrue(np.allclose(np.outer(et, et.conj()), 0.5 * (np.eye(2) - rho * fb.sdot(s))))

    def test_eta_is_i_sigma2_chi_conjugate_up_to_a_sign(self) -> None:
        isig2 = 1j * fb.SIGMA[1]
        for _ in range(5):
            s = rand_unit()
            for rho in (1, -1):
                ref = isig2 @ fb.chi(s, rho).conj()
                et = fb.eta(s, rho)
                self.assertTrue(np.allclose(et, ref) or np.allclose(et, -ref))

    def test_dirac_bilinears(self) -> None:
        """ubar v = -2 p chi^dag (sigma.n) eta,  ubar gamma5 v = 2 E chi^dag eta  (Eqs. (10), (20))."""
        for _ in range(6):
            n, st = rand_unit(), random_state()
            sv, pv = bilinears(n, st)
            ch, et = fb.chi(st.s1, st.rho1), fb.eta(st.s2, st.rho2)
            self.assertAlmostEqual(abs(sv - (-2.0 * P_F * ch.conj() @ fb.sdot(n) @ et)), 0.0, places=9)
            self.assertAlmostEqual(abs(pv - 2.0 * E_H * ch.conj() @ et), 0.0, places=9)

    def test_spin_summed_matrix_element_is_the_book_formula(self) -> None:
        """Sum over the four spin states of |ubar v|^2 (m_f/v)^2 = 2 m_H^2 m_f^2 beta^2 / v^2 (the book, Eq. (higgsdecays_3))."""
        n, s = rand_unit(), rand_unit()
        tot = sum(abs(bilinears(n, fb.PairState(tuple(s), tuple(s), r1, r2))[0]) ** 2 for r1 in (1, -1) for r2 in (1, -1))
        self.assertAlmostEqual(tot * (M_F / V_EV) ** 2 / (2.0 * (2 * E_H) ** 2 * M_F ** 2 * BETA ** 2 / V_EV ** 2), 1.0, places=9)

    def test_matrix_element_squared_has_the_book_spin_correlation(self) -> None:
        """|ubar v|^2 = 2 p^2 [1 + xi1.xi2 - 2 (xi1.n)(xi2.n)] for xi_i = rho_i s_i  (the book, Eq. (higgsdecays_5))."""
        for _ in range(8):
            n, st = rand_unit(), random_state()
            sv, _ = bilinears(n, st)
            ref = 2.0 * P_F ** 2 * (1.0 + st.xi1 @ st.xi2 - 2.0 * (st.xi1 @ n) * (st.xi2 @ n))
            self.assertAlmostEqual(abs(sv) ** 2, ref, places=7)

    def test_pseudoscalar_matrix_element_has_no_direction(self) -> None:
        st = random_state()
        vals = [abs(bilinears(rand_unit(), st)[1]) for _ in range(5)]
        self.assertTrue(np.allclose(vals, vals[0]))
        self.assertAlmostEqual(vals[0] ** 2, 2.0 * E_H ** 2 * (1.0 - st.xi1 @ st.xi2), places=7)

    def test_width_ratio_is_beta_squared(self) -> None:
        """Gamma_S / Gamma_P = beta^2: P-wave beta^3 against S-wave beta."""
        n, s = rand_unit(), rand_unit()
        sums = [sum(abs(bilinears(n, fb.PairState(tuple(s), tuple(s), r1, r2))[k]) ** 2 for r1 in (1, -1) for r2 in (1, -1)) for k in (0, 1)]
        self.assertAlmostEqual(sums[0] / sums[1], BETA ** 2, places=9)


class AngularIntegralTest(unittest.TestCase):
    def test_plane_wave_integrals(self) -> None:
        """Int dOmega e^{i p n.r} = 4 pi j_0(pr) and Int dOmega e^{i p n.r} n = 4 pi i j_1(pr) rhat  (Eq. (13))."""
        for x in (0.3, 1.7, 4.0, 7.5):
            rhat = rand_unit()
            ph = np.exp(1j * x * (QUAD_PTS @ rhat))
            self.assertAlmostEqual(abs(np.sum(QUAD_W * ph) - 4.0 * math.pi * spherical_jn(0, x)), 0.0, places=9)
            vec = np.sum((QUAD_W * ph)[:, None] * QUAD_PTS, axis=0)
            self.assertTrue(np.allclose(vec, 4.0 * math.pi * 1j * spherical_jn(1, x) * rhat, atol=1e-9))

    def test_closed_form_bessel_functions(self) -> None:
        x = np.linspace(0.0, 15.0, 301)
        self.assertTrue(np.allclose(fb.j0(x), spherical_jn(0, x), atol=1e-12))
        self.assertTrue(np.allclose(fb.j1(x), spherical_jn(1, x), atol=1e-12))
        self.assertAlmostEqual(float(fb.j0(0.0)), 1.0)
        self.assertAlmostEqual(float(fb.j1(0.0)), 0.0)

    def test_first_maximum_of_j1(self) -> None:
        x = fb.j1_first_maximum()
        xs = np.linspace(0.1, 4.0, 400001)
        self.assertAlmostEqual(x, float(xs[np.argmax(spherical_jn(1, xs))]), places=3)
        self.assertAlmostEqual(x, 2.0816, places=3)

    def test_the_sphere_quadrature_weights(self) -> None:
        self.assertAlmostEqual(float(QUAD_W.sum()), 4.0 * math.pi)
        self.assertTrue(np.allclose(np.linalg.norm(QUAD_PTS, axis=1), 1.0))


class WaveFunctionTest(unittest.TestCase):
    def test_literal_amplitudes_give_the_closed_forms(self) -> None:
        """psi_S = N beta j_1 chi^dag (sigma.rhat) eta and psi_P = +i N j_0 chi^dag eta for iM = i (m_f/v) ubar (1, gamma5) v."""
        for _ in range(6):
            st = random_state()
            rvec = rand_unit() * float(RNG.uniform(0.5, 7.0)) / P_F
            ps, pp = psi_numeric(rvec, st)
            x = np.array(rvec) * P_F                                # x = p r
            self.assertAlmostEqual(abs(ps - UNIT_N * fb.psi_s(x, st, BETA)), 0.0, delta=1e-8 * UNIT_N)
            self.assertAlmostEqual(abs(pp - UNIT_N * fb.psi_p(x, st, 1j)), 0.0, delta=1e-8 * UNIT_N)

    def test_p_wave_amplitude_vanishes_at_the_origin_and_s_wave_does_not(self) -> None:
        st = fb.common_axis_state([0, 0, 1], 1, -1)
        self.assertAlmostEqual(abs(fb.psi_s(np.zeros(3), st, BETA)), 0.0)
        self.assertAlmostEqual(abs(fb.psi_p(np.zeros(3), st, -1j)), 1.0)

    def test_density_formulas_scalar_and_pseudoscalar(self) -> None:
        """Eqs. (15) and (22) from the explicit spinors: the squared moduli of psi_S and psi_P (unit N^2)."""
        for _ in range(10):
            st = random_state()
            x = rand_unit() * float(RNG.uniform(0.3, 8.0))
            r = float(np.linalg.norm(x))
            rhat = x / r
            self.assertAlmostEqual(float(abs(fb.psi_s(x, st, BETA)) ** 2), BETA ** 2 * float(fb.j1(r)) ** 2 * float(fb.angular_scalar(rhat, st.xi1, st.xi2)), places=12)
            self.assertAlmostEqual(float(abs(fb.psi_p(x, st, -1j)) ** 2), float(fb.j0(r)) ** 2 * fb.angular_pseudoscalar(st.xi1, st.xi2), places=12)

    def test_mixed_density_from_the_spinors_matches_the_closed_form(self) -> None:
        """|eps1 psi_S + eps2 psi_P|^2 against the three-term closed form, for complex couplings."""
        for _ in range(20):
            st = random_state()
            x = rand_unit() * float(RNG.uniform(0.3, 8.0))
            e1, e2 = complex(*RNG.normal(size=2)), complex(*RNG.normal(size=2))
            direct = abs(fb.psi_mixed(x, st, e1, e2, BETA, -1j)) ** 2
            closed = fb.density_closed(x, st.xi1, st.xi2, e1, e2, BETA)
            self.assertAlmostEqual(float(direct), float(closed), places=10)

    def test_the_cp_odd_term_follows_the_literal_amplitudes_with_the_opposite_sign_of_eps2(self) -> None:
        """With the literal phase +i the real-coupling interference has the sign opposite to Eq. (B4): eps2 -> -eps2 (gamma5 -> -gamma5)."""
        st = fb.PairState((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), 1, 1)
        x = np.array([0.3, -0.4, 1.9])
        film = abs(fb.psi_mixed(x, st, 1.0, 1.0, BETA, -1j)) ** 2 - abs(fb.psi_s(x, st, BETA)) ** 2 - abs(fb.psi_p(x, st, -1j)) ** 2
        literal = abs(fb.psi_mixed(x, st, 1.0, 1.0, BETA, 1j)) ** 2 - abs(fb.psi_s(x, st, BETA)) ** 2 - abs(fb.psi_p(x, st, 1j)) ** 2
        self.assertAlmostEqual(float(film), -float(literal), places=12)
        self.assertNotAlmostEqual(float(film), 0.0, places=3)


class OrbitalMomentumTest(unittest.TestCase):
    @staticmethod
    def rotation_to_axis(s):
        """Orthogonal matrix with columns (e1, e2, s): the coordinates in which the spin axis is z."""
        s = fb.unit(s)
        t = fb.unit(np.cross(s, [0.3, 0.5, 0.8]))
        return np.column_stack([t, np.cross(s, t), s])

    def test_identity_16_needs_the_conjugate_harmonics(self) -> None:
        """sigma.rhat = sqrt(4 pi / 3) sum_m Y_1m^*(rhat) sigma_m^(1): holds with Y^* (not with Y: the labels m <-> -m are swapped)."""
        sig_m = {0: fb.SIGMA[2], 1: -(fb.SIGMA[0] + 1j * fb.SIGMA[1]) / math.sqrt(2), -1: (fb.SIGMA[0] - 1j * fb.SIGMA[1]) / math.sqrt(2)}
        th, ph = 0.7, 1.9
        rhat = np.array([math.sin(th) * math.cos(ph), math.sin(th) * math.sin(ph), math.cos(th)])
        Y = {m: sph_harm_y(1, m, th, ph) for m in (-1, 0, 1)}
        lhs = fb.sdot(rhat)
        self.assertTrue(np.allclose(lhs, math.sqrt(4 * math.pi / 3) * sum(np.conj(Y[m]) * sig_m[m] for m in Y)))
        self.assertFalse(np.allclose(lhs, math.sqrt(4 * math.pi / 3) * sum(Y[m] * sig_m[m] for m in Y)))

    def m_content(self, psi_angular, axis) -> dict[int, float]:
        """Weights |<Y_1m|psi>|^2 / <psi|psi> of an angular function on the unit sphere, m about the given axis."""
        R = self.rotation_to_axis(axis)
        pts = QUAD_PTS @ R                                    # coordinates of the quadrature points in the frame where the axis is z
        th, ph = np.arccos(np.clip(pts[:, 2], -1, 1)), np.arctan2(pts[:, 1], pts[:, 0])
        f = psi_angular(QUAD_PTS)
        tot = float(np.sum(QUAD_W * np.abs(f) ** 2))
        return {m: float(abs(np.sum(QUAD_W * np.conj(sph_harm_y(1, m, th, ph)) * f)) ** 2 / tot) for m in (-1, 0, 1)}

    def test_common_axis_selects_a_definite_m(self) -> None:
        """J_s = 0: m = -(rho1 + rho2)/2.  rho1 = -rho2: m = 0; rho1 = rho2 = +: m = -1; rho1 = rho2 = -: m = +1 (any direction of the axis)."""
        for axis in ([0, 0, 1], list(rand_unit()), list(rand_unit())):
            for r1 in (1, -1):
                for r2 in (1, -1):
                    st = fb.common_axis_state(axis, r1, r2)
                    c = self.m_content(lambda P: fb.psi_s(P, st, 1.0), axis)
                    m = -(r1 + r2) // 2
                    self.assertAlmostEqual(c[m], 1.0, places=8, msg=f"{axis} {(r1, r2)} {c}")

    def test_pseudoscalar_is_a_pure_s_wave(self) -> None:
        st = fb.common_axis_state([0, 0, 1], 1, -1)
        f = fb.psi_p(QUAD_PTS, st, -1j) / float(fb.j0(1.0))      # the points have |x| = 1
        self.assertTrue(np.allclose(f, f[0]))                 # no dependence on the direction
        self.assertAlmostEqual(float(np.sum(QUAD_W * abs(f) ** 2)), 4.0 * math.pi)       # a pure Y_00 = 1/sqrt(4 pi)

    def test_normalisation_of_the_angular_functions(self) -> None:
        """int dOmega |chi^dag (sigma.rhat) eta|^2: 8 pi / 3 for rho1 = rho2 and 4 pi / 3 for rho1 = -rho2; the four outcomes sum to 8 pi."""
        tot = 0.0
        for r1 in (1, -1):
            for r2 in (1, -1):
                st = fb.common_axis_state([0, 0, 1], r1, r2)
                ang = np.abs(QUAD_PTS @ st.vector()) ** 2
                val = float(np.sum(QUAD_W * ang))
                self.assertAlmostEqual(val, 8 * math.pi / 3 if r1 == r2 else 4 * math.pi / 3, places=9)
                tot += val
        self.assertAlmostEqual(tot, 8.0 * math.pi, places=9)


class SpecialAngleTest(unittest.TestCase):
    def test_zero_and_right_angle(self) -> None:
        """s1 = s2 = s: at theta = 0 only rho1 = -rho2 survives, |psi|^2 ~ (1 - rho1 rho2); at theta = pi/2 only rho1 = rho2, ~ (1 + rho1 rho2)."""
        s = rand_unit()
        perp = fb.unit(np.cross(s, rand_unit()))
        for r1 in (1, -1):
            for r2 in (1, -1):
                st = fb.common_axis_state(s, r1, r2)
                a0 = float(abs(fb.psi_s(2.0 * s, st, 1.0)) ** 2 / float(fb.j1(2.0)) ** 2)
                a90 = float(abs(fb.psi_s(2.0 * perp, st, 1.0)) ** 2 / float(fb.j1(2.0)) ** 2)
                self.assertAlmostEqual(a0, 0.5 * (1 - r1 * r2), places=12)            # (1/2)(1 - rho1 rho2)
                self.assertAlmostEqual(a90, 0.5 * (1 + r1 * r2), places=12)
                for th in (0.3, 0.9, 1.3):
                    rh = math.cos(th) * s + math.sin(th) * perp
                    a = float(abs(fb.psi_s(2.0 * rh, st, 1.0)) ** 2 / float(fb.j1(2.0)) ** 2)
                    self.assertAlmostEqual(a, 0.5 * (1 + r1 * r2 * (1 - 2 * math.cos(th) ** 2)), places=12)

    def test_the_four_outcomes_sum_to_a_direction_independent_value(self) -> None:
        """Summed over the spins |psi_S|^2 = 2 beta^2 j_1^2 and |psi_P|^2 = 2 j_0^2: no dependence on rhat."""
        for _ in range(4):
            s = rand_unit()
            x = rand_unit() * 3.3
            ss = sum(abs(fb.psi_s(x, fb.common_axis_state(s, a, b), BETA)) ** 2 for a in (1, -1) for b in (1, -1))
            pp = sum(abs(fb.psi_p(x, fb.common_axis_state(s, a, b), -1j)) ** 2 for a in (1, -1) for b in (1, -1))
            self.assertAlmostEqual(float(ss), 2.0 * BETA ** 2 * float(fb.j1(3.3)) ** 2, places=12)
            self.assertAlmostEqual(float(pp), 2.0 * float(fb.j0(3.3)) ** 2, places=12)

    def test_pseudoscalar_common_axis_only_antiparallel(self) -> None:
        s = rand_unit()
        for r1, r2 in ((1, 1), (-1, -1)):
            self.assertAlmostEqual(float(abs(fb.psi_p(2.0 * rand_unit(), fb.common_axis_state(s, r1, r2), -1j))), 0.0, places=12)
        for r1, r2 in ((1, -1), (-1, 1)):
            vals = [float(abs(fb.psi_p(2.0 * rand_unit(), fb.common_axis_state(s, r1, r2), -1j))) for _ in range(4)]
            self.assertTrue(np.allclose(vals, vals[0]))
            self.assertAlmostEqual(vals[0], float(fb.j0(2.0)), places=12)


class CPOddTermTest(unittest.TestCase):
    def interference(self, x, st, eps1=1.0, eps2=1.0) -> float:
        return float(abs(fb.psi_mixed(x, st, eps1, eps2, BETA, -1j)) ** 2 - abs(fb.psi_s(x, st, BETA) * eps1) ** 2 - abs(fb.psi_p(x, st, -1j) * eps2) ** 2)

    def test_term_equals_the_triple_product(self) -> None:
        for _ in range(10):
            st = random_state()
            x = rand_unit() * float(RNG.uniform(0.5, 6.0))
            r = float(np.linalg.norm(x))
            rhat = x / r
            ref = BETA * float(fb.j0(r)) * float(fb.j1(r)) * float(fb.triple_product(rhat, st.xi1, st.xi2))
            self.assertAlmostEqual(self.interference(x, st), ref, places=10)

    def test_sign_flips_under_exchange_of_the_spins_and_under_parity(self) -> None:
        for _ in range(6):
            st = random_state()
            sw = fb.PairState(st.s2, st.s1, st.rho2, st.rho1)                     # the spin labels of the fermion and the antifermion exchanged
            x = rand_unit() * 2.2
            self.assertAlmostEqual(self.interference(x, st), -self.interference(x, sw), places=12)
            self.assertAlmostEqual(self.interference(x, st), -self.interference(-x, st), places=12)       # r -> -r
            flip = fb.PairState(st.s1, st.s2, -st.rho1, st.rho2)                  # rho1 -> -rho1 reverses xi1
            self.assertAlmostEqual(self.interference(x, st), -self.interference(x, flip), places=12)

    def test_term_vanishes_for_a_common_axis(self) -> None:
        for _ in range(5):
            s = rand_unit()
            for r1 in (1, -1):
                for r2 in (1, -1):
                    st = fb.common_axis_state(s, r1, r2)
                    self.assertAlmostEqual(self.interference(rand_unit() * 2.0, st), 0.0, places=12)

    def test_term_changes_sign_with_the_relative_sign_of_the_couplings(self) -> None:
        st = fb.PairState((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), 1, 1)
        x = np.array([0.5, 0.2, 1.4])
        self.assertAlmostEqual(self.interference(x, st, 1.0, 1.0), -self.interference(x, st, 1.0, -1.0), places=12)

    def test_on_the_axis_the_two_waves_add_on_one_side_and_cancel_on_the_other(self) -> None:
        """xi1 = x, xi2 = y: on the axis +-(xi2 x xi1) |psi|^2 = (1/2)(a +- b)^2 with a = eps1 beta j_1, b = eps2 j_0."""
        st = fb.PairState((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), 1, 1)
        axis = fb.unit(np.cross(st.xi2, st.xi1))
        for r in (0.7, 2.04, 3.6, 5.1):
            for sign in (1, -1):
                d = float(abs(fb.psi_mixed(sign * r * axis, st, 0.8, 0.6, BETA, -1j)) ** 2)
                self.assertAlmostEqual(d, float(fb.density_on_axis(r, 0.8, 0.6, BETA, sign)), places=12)

    def test_complete_cancellation_where_the_two_amplitudes_are_equal(self) -> None:
        st = fb.PairState((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), 1, 1)
        axis = fb.unit(np.cross(st.xi2, st.xi1))
        r_eq = brentq(lambda r: float(fb.j0(r)) - BETA * float(fb.j1(r)), 1.5, 2.6, xtol=1e-14)        # the root of beta j_1(r) = j_0(r)
        self.assertAlmostEqual(float(fb.j0(r_eq)) - BETA * float(fb.j1(r_eq)), 0.0, places=9)
        self.assertAlmostEqual(float(abs(fb.psi_mixed(-r_eq * axis, st, 1.0, 1.0, BETA, -1j)) ** 2), 0.0, places=18)
        self.assertGreater(float(abs(fb.psi_mixed(r_eq * axis, st, 1.0, 1.0, BETA, -1j)) ** 2), 0.5 * float(fb.j0(r_eq)) ** 2)

    def test_pure_scalar_and_pure_pseudoscalar_are_mirror_symmetric(self) -> None:
        """Without the interference the density does not change under the exchange of the spins (no handedness)."""
        for _ in range(5):
            st = random_state()
            sw = fb.PairState(st.s2, st.s1, st.rho2, st.rho1)
            x = rand_unit() * 3.0
            for e1, e2 in ((1.0, 0.0), (0.0, 1.0)):
                self.assertAlmostEqual(float(abs(fb.psi_mixed(x, st, e1, e2, BETA, -1j)) ** 2), float(abs(fb.psi_mixed(x, sw, e1, e2, BETA, -1j)) ** 2), places=12)

    def test_transverse_spins_give_a_rotated_cosine_whose_shift_changes_sign_with_the_exchange(self) -> None:
        """r perpendicular to xi1, xi2 (angle Delta between them): |psi|^2 = A + R cos(Delta + delta); the exchange of the spins reverses delta."""
        rhat = np.array([0.0, 0.0, 1.0])
        x = 2.0 * rhat
        a, b = BETA * float(fb.j1(2.0)), float(fb.j0(2.0))
        A, B, C = 0.5 * (a * a + b * b), 0.5 * (a * a - b * b), a * b
        for delta in np.linspace(0.0, 2 * math.pi, 9):
            xi1 = np.array([1.0, 0.0, 0.0])
            xi2 = np.array([math.cos(delta), math.sin(delta), 0.0])
            st = fb.PairState(tuple(xi1), tuple(xi2), 1, 1)
            d = float(abs(fb.psi_mixed(x, st, 1.0, 1.0, BETA, -1j)) ** 2)
            self.assertAlmostEqual(d, A + B * math.cos(delta) - C * math.sin(delta), places=12)
            sw = fb.PairState(tuple(xi2), tuple(xi1), 1, 1)
            dsw = float(abs(fb.psi_mixed(x, sw, 1.0, 1.0, BETA, -1j)) ** 2)
            self.assertAlmostEqual(dsw, A + B * math.cos(delta) + C * math.sin(delta), places=12)


class PlaneWaveSumTest(unittest.TestCase):
    def test_halton_points_are_on_the_sphere_and_every_prefix_is_spread(self) -> None:
        pts = fb.halton_sphere(512)
        self.assertTrue(np.allclose(np.linalg.norm(pts, axis=1), 1.0))
        for n in (16, 64, 256):
            self.assertLess(float(np.abs(pts[:n].mean(axis=0)).max()), 0.35 / math.sqrt(n) * 4)      # the centroid of a prefix is near zero

    def test_partial_sums_converge_to_the_vortex(self) -> None:
        half = 8.0
        pts = fb.halton_sphere(2048)
        s = fb.PlaneWaveSum(pts, pts[:, 0] + 1j * pts[:, 1], 80, half)
        exact = s.exact()
        mask = np.hypot(s.X, s.Y) < half
        peak = 4.0 * math.pi * 0.4362
        errs = []
        for n in (8, 64, 512, 2048):
            errs.append(float(np.abs(s.get(n) - exact)[mask].max() / peak))
        self.assertTrue(all(b < a for a, b in zip(errs[:-1], errs[1:])), errs)
        self.assertLess(errs[-1], 0.03)

    def test_the_exact_limit_is_the_p_wave_vortex(self) -> None:
        pts = fb.halton_sphere(4)
        s = fb.PlaneWaveSum(pts, pts[:, 0] + 1j * pts[:, 1], 41, 6.0)
        ex = s.exact()
        c = 20                                                # the grid centre
        self.assertAlmostEqual(abs(ex[c, c]), 0.0, places=6)                  # the dark core: j_1(0) = 0
        # the phase winds once around the origin: arg psi = pi/2 + phi (j_1 > 0 inside the first lobe)
        ang = np.linspace(0, 2 * math.pi, 9)[:-1]
        i = (c - np.round(2.0 * np.sin(ang) / 0.3)).astype(int)
        j = (c + np.round(2.0 * np.cos(ang) / 0.3)).astype(int)
        ph = np.unwrap(np.angle(ex[i, j]))
        self.assertAlmostEqual(float((ph[-1] - ph[0]) / (len(ang) - 1) * len(ang) / (2 * math.pi)), 1.0, delta=0.2)


if __name__ == "__main__":
    unittest.main()
