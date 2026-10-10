from __future__ import annotations

import math
import re
import subprocess
import sys
import tomllib
import unittest
from pathlib import Path

import numpy as np

import z_decay_asymmetry as za

HERE = Path(__file__).resolve().parent

# ---------------------------------------------------------------------------- explicit Dirac algebra (chiral representation)

I2, Z2 = np.eye(2), np.zeros((2, 2))
SIG = [np.array([[0, 1], [1, 0]], complex), np.array([[0, -1j], [1j, 0]]), np.array([[1, 0], [0, -1]], complex)]
G0 = np.block([[Z2, I2], [I2, Z2]]).astype(complex)
GI = [np.block([[Z2, s], [-s, Z2]]) for s in SIG]
GAM = [G0] + GI
G5 = 1j * G0 @ GI[0] @ GI[1] @ GI[2]
I4 = np.eye(4)
PL, PR = (I4 - G5) / 2, (I4 + G5) / 2
MET = np.diag([1.0, -1.0, -1.0, -1.0])


def slash(p) -> np.ndarray:
    return sum(MET[m, m] * p[m] * GAM[m] for m in range(4))


def _msqrt(a: np.ndarray) -> np.ndarray:
    w, v = np.linalg.eigh(a)
    return v @ np.diag(np.sqrt(np.clip(w, 0.0, None))) @ v.conj().T


def spinors(p, kind: str) -> list[np.ndarray]:
    """Massless u (kind 'u') or v ('v') spinors; the sum over the two spin states of u ubar (v vbar) is p-slash."""
    e, n = p[0], np.array(p[1:])
    ps = e * I2 - sum(n[i] * SIG[i] for i in range(3))
    pb = e * I2 + sum(n[i] * SIG[i] for i in range(3))
    sgn = 1.0 if kind == "u" else -1.0
    return [np.concatenate([_msqrt(ps) @ xi, sgn * _msqrt(pb) @ xi]) for xi in (np.array([1, 0], complex), np.array([0, 1], complex))]


def bar(u: np.ndarray) -> np.ndarray:
    return u.conj() @ G0


def summed_square(a: str, b: str, c: float, e: float = 1.0) -> float:
    """Sum over all spins of |vbar(p2) gamma^mu P_a u(p1) ubar(k1) gamma_mu P_b v(k2)|^2 for e- (p1) and e+ (p2) along the z axis."""
    p1, p2 = [e, 0, 0, e], [e, 0, 0, -e]
    st = math.sqrt(1.0 - c * c)
    k1, k2 = [e, e * st, 0, e * c], [e, -e * st, 0, -e * c]
    pa, pb = (PL if a == "L" else PR), (PL if b == "L" else PR)
    tot = 0.0
    for u1 in spinors(p1, "u"):
        for v2 in spinors(p2, "v"):
            j1 = [bar(v2) @ GAM[m] @ pa @ u1 for m in range(4)]
            for ub in spinors(k1, "u"):
                for vv in spinors(k2, "v"):
                    j2 = [bar(ub) @ GAM[m] @ pb @ vv for m in range(4)]
                    tot += abs(sum(MET[m, m] * j1[m] * j2[m] for m in range(4))) ** 2
    return tot


class DiracAlgebraTest(unittest.TestCase):
    def test_spin_sums_are_the_momentum_slash(self) -> None:
        p = [1.0, 0.3, 0.4, math.sqrt(0.75)]
        self.assertTrue(np.allclose(sum(np.outer(u, bar(u)) for u in spinors(p, "u")), slash(p)))
        self.assertTrue(np.allclose(sum(np.outer(v, bar(v)) for v in spinors(p, "v")), slash(p)))

    def test_chiral_channels_have_the_angular_distributions_of_the_film(self) -> None:
        """The book's eq. for |M|^2: LL and RR ~ (s + t)^2 ~ (1 + cos)^2, LR and RL ~ t^2 ~ (1 - cos)^2, with the common factor 4."""
        s = 4.0
        for c in (-0.9, -0.3, 0.0, 0.45, 0.8):
            t = -0.5 * s * (1.0 - c)
            for ab in ("LL", "RR"):
                self.assertAlmostEqual(summed_square(ab[0], ab[1], c), 4.0 * (s + t) ** 2, places=9)
                self.assertAlmostEqual(summed_square(ab[0], ab[1], c), s * s * (1.0 + c) ** 2, places=9)
            for ab in ("LR", "RL"):
                self.assertAlmostEqual(summed_square(ab[0], ab[1], c), 4.0 * t * t, places=9)
                self.assertAlmostEqual(summed_square(ab[0], ab[1], c), s * s * (1.0 - c) ** 2, places=9)

    def test_z_decay_rate_is_proportional_to_gl2_plus_gr2_and_the_chiralities_add_up(self) -> None:
        """sum over the Z polarizations and the spins of |ubar gamma_mu eps^mu (g_L P_L + g_R P_R) v|^2 = 2 m_Z^2 (g_L^2 + g_R^2):
        the book's Gamma ~ g_v^2 + g_a^2, and Gamma(Z -> f_L fbar_R) : Gamma(Z -> f_R fbar_L) = g_L^2 : g_R^2."""
        m = 1.0
        n = np.array([0.36, 0.48, 0.8])
        p1 = [m / 2, *(m / 2 * n)]
        p2 = [m / 2, *(-m / 2 * n)]
        eps = [np.array([0, 1, 1j, 0]) / math.sqrt(2), np.array([0, 1, -1j, 0]) / math.sqrt(2), np.array([0, 0, 0, 1.0 + 0j])]

        def total(gl: float, gr: float) -> float:
            op = [GAM[mu] @ (gl * PL + gr * PR) for mu in range(4)]
            out = 0.0
            for e in eps:
                for u in spinors(p1, "u"):
                    for v in spinors(p2, "v"):
                        amp = sum(e[mu] * (bar(u) @ op[mu] @ v) for mu in range(4))
                        out += abs(amp) ** 2
            return out

        for gl, gr in ((1.0, 0.0), (0.0, 1.0), (0.7, -0.4)):
            self.assertAlmostEqual(total(gl, gr), 2.0 * m * m * (gl * gl + gr * gr), places=9)
        self.assertAlmostEqual(total(1.0, 0.0) / total(0.0, 1.0), 1.0, places=9)


# ---------------------------------------------------------------------------- the couplings and the asymmetries


def forward_backward_numeric(fn, n: int = 100001) -> float:
    """(N_F - N_B)/(N_F + N_B) from the numerical integration of fn(cos theta) over [0, 1] and [-1, 0]."""
    cf, cb = np.linspace(0.0, 1.0, n), np.linspace(-1.0, 0.0, n)
    nf, nb = np.trapezoid(fn(cf), cf), np.trapezoid(fn(cb), cb)
    return float((nf - nb) / (nf + nb))


class CouplingsTest(unittest.TestCase):
    def test_book_definitions(self) -> None:
        s2 = 0.2316
        c = za.couplings("e", s2)
        self.assertAlmostEqual(c["gv"], -0.5 + 2 * s2)
        self.assertAlmostEqual(c["ga"], -0.5)
        self.assertAlmostEqual(c["L"], c["gv"] + c["ga"])
        self.assertAlmostEqual(c["R"], c["gv"] - c["ga"])
        self.assertAlmostEqual(c["L"], 2 * (-0.5) - 2 * (-1) * s2)
        self.assertAlmostEqual(c["R"], -2 * (-1) * s2)
        d = za.couplings("d", s2)
        self.assertAlmostEqual(d["R"], 2.0 / 3.0 * s2)
        u = za.couplings("u", s2)
        self.assertAlmostEqual(u["gv"], 0.5 - 4.0 / 3.0 * s2)

    def test_the_neutrino_is_purely_left_handed(self) -> None:
        c = za.couplings("nu")
        self.assertAlmostEqual(c["R"], 0.0)
        self.assertAlmostEqual(c["L"], 1.0)
        self.assertAlmostEqual(za.a_lr("nu"), 1.0)

    def test_limits_of_the_asymmetry(self) -> None:
        self.assertAlmostEqual(za.a_lr_from(0.7, 0.0), 1.0)                  # g_R = 0
        self.assertAlmostEqual(za.a_lr_from(0.0, -0.3), -1.0)                # g_L = 0
        self.assertAlmostEqual(za.a_lr_from(0.5, -0.5), 0.0)                 # g_v = 0: g_L = -g_R
        self.assertAlmostEqual(za.a_lr("e", 0.25), 0.0)                      # sin^2 = 1/4 makes g_v of the charged leptons vanish
        self.assertAlmostEqual(za.couplings("e", 0.25)["gv"], 0.0)

    def test_a_lr_is_two_gv_ga_over_the_sum_of_squares(self) -> None:
        for f in za.FERMIONS:
            c = za.couplings(f)
            self.assertAlmostEqual(za.a_lr(f), 2 * c["gv"] * c["ga"] / (c["gv"] ** 2 + c["ga"] ** 2))

    def test_the_numbers_of_the_film(self) -> None:
        self.assertAlmostEqual(za.a_lr("e"), 0.146407, places=5)
        self.assertAlmostEqual(za.a_lr("u"), 0.667231, places=5)
        self.assertAlmostEqual(za.a_lr("d"), 0.935472, places=5)
        self.assertAlmostEqual(za.a_fb("e"), 0.016077, places=5)
        self.assertAlmostEqual(za.a_fb("d"), 0.102720, places=5)

    def test_the_width_of_the_book_is_reproduced_within_a_third_of_a_percent(self) -> None:
        """Gamma(Z -> l l) = g^2 (g_v^2 + g_a^2) m_Z / (48 pi cos^2 theta_W) = 83.48 MeV in the book."""
        c = za.couplings("e")
        gam = za.G2 * (c["gv"] ** 2 + c["ga"] ** 2) * za.MZ / (48.0 * math.pi * (1.0 - za.S2W))
        self.assertAlmostEqual(gam * 1000.0, 83.48, delta=0.3)

    def test_lepton_asymmetries_are_precision_probes(self) -> None:
        for s2 in (0.2316, 0.235, 0.2285):
            d = 0.25 - s2
            self.assertAlmostEqual(za.a_lr("e", s2) / (8.0 * d), 1.0, delta=0.03)                 # A_LR^e ~ 8 (1/4 - s^2)
            self.assertAlmostEqual(za.a_fb("e", s2) / (48.0 * d * d), 1.0, delta=0.05)            # A_FB^l ~ 48 (1/4 - s^2)^2


# ---------------------------------------------------------------------------- the channels, the sum and A_FB


class ChannelsTest(unittest.TestCase):
    def test_shares_sum_to_one_and_equal_products_of_the_squared_couplings(self) -> None:
        for f in za.FERMIONS:
            sh = za.channel_shares(f)
            self.assertAlmostEqual(sum(sh.values()), 1.0)
            ce, cf = za.couplings("e"), za.couplings(f)
            norm = (ce["L"] ** 2 + ce["R"] ** 2) * (cf["L"] ** 2 + cf["R"] ** 2)
            self.assertAlmostEqual(sh["LR"], ce["L"] ** 2 * cf["R"] ** 2 / norm)
            self.assertAlmostEqual(sh["RL"], ce["R"] ** 2 * cf["L"] ** 2 / norm)

    def test_the_sum_of_the_channels_is_one_plus_cos2_plus_the_asymmetry_term(self) -> None:
        c = np.linspace(-1, 1, 41)
        for f in za.FERMIONS:
            sh = za.channel_shares(f)
            self.assertTrue(np.allclose(za.distribution(c, sh), za.shape(c, za.a_fb_from_shares(sh))))

    def test_equal_weights_give_the_symmetric_one_plus_cos2(self) -> None:
        c = np.linspace(-1, 1, 41)
        sh = {ab: 0.25 for ab in za.CHANNELS}
        self.assertTrue(np.allclose(za.distribution(c, sh), 1.0 + c * c))
        self.assertAlmostEqual(za.a_fb_from_shares(sh), 0.0)

    def test_afb_formula_matches_numerical_integration(self) -> None:
        for f in za.FERMIONS:
            for s2 in (0.2316, 0.25, 0.21):
                sh = za.channel_shares(f, s2)
                numeric = forward_backward_numeric(lambda c: za.distribution(c, sh))
                self.assertAlmostEqual(numeric, za.a_fb(f, s2), places=8)
                self.assertAlmostEqual(numeric, za.a_fb_from_shares(sh), places=8)

    def test_afb_limits(self) -> None:
        # a purely left-handed electron and fermion: only the LL channel, A_FB = 3/4
        self.assertAlmostEqual(za.a_fb_from_shares({"LL": 1.0, "LR": 0.0, "RL": 0.0, "RR": 0.0}), 0.75)
        self.assertAlmostEqual(za.a_fb_from_shares({"LL": 0.0, "LR": 1.0, "RL": 0.0, "RR": 0.0}), -0.75)
        # the neutrino has g_R = 0: A_LR^nu = 1 and A_FB = (3/4) A_LR^e
        self.assertAlmostEqual(za.a_fb("nu"), 0.75 * za.a_lr("e"))
        # sin^2(theta_W) = 1/4: no asymmetry for any fermion
        for f in za.FERMIONS:
            self.assertAlmostEqual(za.a_fb(f, 0.25), 0.0)
            self.assertAlmostEqual(za.a_fb_from_shares(za.channel_shares(f, 0.25)), 0.0)

    def test_lobes_and_the_zero_in_the_backward_direction(self) -> None:
        self.assertAlmostEqual(float(za.lobe("LL", -1.0)), 0.0)          # f_L from e_L never goes backward
        self.assertAlmostEqual(float(za.lobe("LL", 1.0)), 4.0)
        self.assertAlmostEqual(float(za.lobe("LR", 1.0)), 0.0)
        self.assertAlmostEqual(float(za.lobe("RL", -1.0)), 4.0)
        for c in (-0.5, 0.2, 0.9):
            self.assertAlmostEqual(float(za.lobe("LL", c)), float(za.lobe("RR", c)))
            self.assertAlmostEqual(float(za.lobe("LR", c)), float(za.lobe("LL", -c)))


class CrossSectionTest(unittest.TestCase):
    def test_total_cross_section_of_the_book_for_the_photon_only(self) -> None:
        """eq. (54): sigma = 4 pi alpha^2 N_c / (3 s) (e_f/e)^2, from the sum of the channels with g = 0 (no Z)."""
        for f in ("e", "u", "d"):
            for sq in (20.0, 60.0):
                s = sq * sq
                expect = 4 * math.pi * za.ALPHA ** 2 * za.FERMIONS[f]["nc"] / (3 * s) * za.FERMIONS[f]["q"] ** 2
                self.assertAlmostEqual(za.sigma_total(sq, f, z=False) / expect, 1.0, places=9)
                c = np.linspace(-1, 1, 20001)
                self.assertAlmostEqual(np.trapezoid(za.dsigma_dcos(sq, f, c, z=False), c) / expect, 1.0, places=6)

    def test_total_cross_section_of_the_book_for_the_z_only(self) -> None:
        """eq. (55): sigma = N_c (gv_e^2 + ga_e^2)(gv_f^2 + ga_f^2) / (12 pi) (g / 2 cos)^4 s / ((s - m^2)^2 + m^2 Gamma^2)."""
        for f in za.FERMIONS:
            for sq in (60.0, 91.1876, 120.0):
                s = sq * sq
                ce, cf = za.couplings("e"), za.couplings(f)
                expect = (za.FERMIONS[f]["nc"] * (ce["gv"] ** 2 + ce["ga"] ** 2) * (cf["gv"] ** 2 + cf["ga"] ** 2) / (12 * math.pi)
                          * (za.G2 / (4 * (1 - za.S2W))) ** 2 * s / ((s - za.MZ ** 2) ** 2 + za.MZ ** 2 * za.GZ ** 2))
                self.assertAlmostEqual(za.sigma_total(sq, f, photon=False) / expect, 1.0, places=9)
                c = np.linspace(-1, 1, 20001)
                self.assertAlmostEqual(np.trapezoid(za.dsigma_dcos(sq, f, c, photon=False), c) / expect, 1.0, places=6)

    def test_the_peak_cross_section_equals_the_width_formula_of_the_problem(self) -> None:
        """sigma(m_Z) = 12 pi Gamma(Z -> ee) Gamma(Z -> f f) / (m_Z^2 Gamma_Z^2) (tree level, Z only, colour factor included in Gamma)."""
        def width(f: str) -> float:
            c = za.couplings(f)
            return za.FERMIONS[f]["nc"] * za.G2 * (c["gv"] ** 2 + c["ga"] ** 2) * za.MZ / (48 * math.pi * (1 - za.S2W))
        for f in ("e", "u", "d"):
            expect = 12 * math.pi * width("e") * width(f) / (za.MZ ** 2 * za.GZ ** 2)
            self.assertAlmostEqual(za.sigma_total(za.MZ, f, photon=False) / expect, 1.0, places=9)
        # a number to compare with the textbooks: about 2 nb for mu+mu- (tree level)
        nb = za.sigma_total(za.MZ, "e") * za.CFG.physics.gev2_to_nb
        self.assertTrue(1.8 < nb < 2.2, nb)

    def test_channel_weights_are_the_z_only_squares_of_rho(self) -> None:
        for f in za.FERMIONS:
            w = za.channel_weights(f)
            r = {ab: abs(za.rho(za.MZ, f, ab[0], ab[1], photon=False)) ** 2 for ab in za.CHANNELS}
            self.assertTrue(np.allclose(list(za.normalise(w).values()), list(za.normalise(r).values())))

    def test_the_book_integration_of_the_massless_distribution(self) -> None:
        """the distribution in cos(theta) of the book's eq. (49)+(52): the forward and backward parts of the full gamma/Z cross section."""
        for f in ("e", "d"):
            for sq in (50.0, 89.0, 91.1876, 95.0, 130.0):
                numeric = forward_backward_numeric(lambda c: za.dsigma_dcos(sq, f, c))
                self.assertAlmostEqual(numeric, za.a_fb_scan(sq, f), places=8)


class PoleScanTest(unittest.TestCase):
    def test_at_the_pole_the_full_asymmetry_is_the_pole_value_to_better_than_a_percent(self) -> None:
        for f in ("e", "u", "d"):
            self.assertAlmostEqual(za.a_fb_scan(za.MZ, f), za.a_fb(f), delta=3e-4)

    def test_sign_pattern_of_the_muon_asymmetry(self) -> None:
        self.assertLess(za.a_fb_scan(50.0, "e"), 0.0)
        self.assertLess(za.a_fb_scan(80.0, "e"), 0.0)
        self.assertLess(za.a_fb_scan(89.0, "e"), 0.0)
        self.assertGreater(za.a_fb_scan(za.MZ, "e"), 0.0)
        self.assertGreater(za.a_fb_scan(100.0, "e"), 0.3)
        self.assertGreater(za.a_fb_scan(150.0, "e"), 0.3)
        # the zero of A_FB lies a little below the pole
        grid = np.linspace(85.0, 92.0, 7001)
        vals = np.array([za.a_fb_scan(g, "e") for g in grid])
        zero = grid[np.where(np.diff(np.sign(vals)) != 0)[0][0]]
        self.assertTrue(89.5 < zero < za.MZ, zero)

    def test_photon_only_is_symmetric_and_has_no_asymmetry(self) -> None:
        for sq in (30.0, 91.1876):
            self.assertAlmostEqual(za.a_fb_scan(sq, "e", z=False), 0.0)

    def test_the_cross_section_peaks_at_the_mass(self) -> None:
        grid = np.linspace(80.0, 100.0, 4001)
        vals = [za.sigma_total(g, "e") for g in grid]
        self.assertAlmostEqual(grid[int(np.argmax(vals))], za.MZ, delta=0.1)


# ---------------------------------------------------------------------------- the Monte Carlo


class SamplingTest(unittest.TestCase):
    def test_sampled_asymmetry_agrees_with_the_formula(self) -> None:
        for f in ("e", "d", "u"):
            afb = za.a_fb(f)
            rng = np.random.default_rng(za.CFG.sampling.seed_test)
            c = za.sample_cos(rng, 2_000_000, afb)
            a, err, nf, nb = za.estimate(c)
            self.assertLess(abs(a - afb), 4.0 * err)
            self.assertEqual(nf + nb, c.size)
            self.assertAlmostEqual(err, math.sqrt((1 - a * a) / c.size))

    def test_histogram_follows_the_curve(self) -> None:
        afb = za.a_fb("d")
        rng = np.random.default_rng(za.CFG.sampling.seed_test + 1)
        n, bins = 400_000, 40
        c = za.sample_cos(rng, n, afb)
        counts, edges = np.histogram(c, bins=bins, range=(-1, 1))
        # expected counts: the integral of the normalised shape (area 8/3) over every bin
        exp = np.array([np.trapezoid(za.shape(np.linspace(a, b, 51), afb), np.linspace(a, b, 51)) for a, b in zip(edges[:-1], edges[1:])]) * n * 3.0 / 8.0
        chi2 = float(((counts - exp) ** 2 / exp).sum())
        self.assertTrue(0.55 < chi2 / bins < 1.6, chi2 / bins)

    def test_the_shape_is_normalised_and_positive(self) -> None:
        c = np.linspace(-1, 1, 20001)
        for afb in (0.0, 0.1, 0.75, -0.75):
            self.assertAlmostEqual(np.trapezoid(za.shape(c, afb), c), 8.0 / 3.0, places=6)
            self.assertTrue(np.all(za.shape(c, afb) >= -1e-12))

    def test_the_error_shrinks_as_one_over_sqrt_n(self) -> None:
        rng = np.random.default_rng(5)
        errs = [za.estimate(za.sample_cos(rng, n, 0.1))[1] for n in (10_000, 1_000_000)]
        self.assertAlmostEqual(errs[0] / errs[1], 10.0, delta=0.2)

    def test_the_film_samples_are_reproducible(self) -> None:
        a = za.sample_cos(np.random.default_rng(za.CFG.sampling.seed_b), 1000, za.a_fb("d"))
        b = za.sample_cos(np.random.default_rng(za.CFG.sampling.seed_b), 1000, za.a_fb("d"))
        self.assertTrue(np.array_equal(a, b))


# ---------------------------------------------------------------------------- the film: configuration, texts, timeline, frames


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
                    for bad in ("\\bm", "\\Box", "\\text{"):
                        self.assertNotIn(bad, text)

    def test_the_same_placeholders_in_both_languages(self) -> None:
        for key in self.texts["en"]:
            with self.subTest(key=key):
                self.assertEqual(sorted(re.findall(r"@[a-z]+@", self.texts["en"][key])), sorted(re.findall(r"@[a-z]+@", self.texts["ru"][key])))

    def test_russian_has_decimal_commas_and_no_words_in_formulas(self) -> None:
        for key, text in self.texts["ru"].items():
            with self.subTest(key=key):
                self.assertIsNone(re.search(r"\d\.\d", text), text)
                for math in re.findall(r"\$([^$]*)\$", text):
                    self.assertIsNone(re.search(r"[\u0400-\u04FF]", math), math)

    def test_english_has_no_cyrillic_and_russian_has_cyrillic_words(self) -> None:
        for key, text in self.texts["en"].items():
            self.assertIsNone(re.search(r"[\u0400-\u04FF]", text), key)
        for key in ("title", "ch1", "c1a", "c4c"):
            self.assertIsNotNone(re.search(r"[\u0400-\u04FF]", self.texts["ru"][key]), key)

    def test_every_text_key_used_by_the_script_exists(self) -> None:
        src = (HERE / "z_decay_asymmetry.py").read_text(encoding="utf-8")
        used = set(re.findall(r'tx\["([a-zA-Z0-9_]+)"\]', src)) | set(re.findall(r'"(c[1-5][a-g])"', src))
        used |= {f"ch{i}" for i in range(1, 6)} | {f"ch{i}s" for i in range(1, 6)}
        for key in sorted(used):
            self.assertIn(key, self.texts["en"], key)
        for ab in za.CHANNELS:
            for pre in ("cell_", "w_"):
                self.assertIn(pre + ab, self.texts["en"])
        for f in za.CFG.couplings.groups:
            for pre in ("n_", "q_", "a_"):
                self.assertIn(pre + f, self.texts["en"])

    def test_video_section_is_complete_and_16_to_9(self) -> None:
        v = self.cfg["video"]
        keys = {"width", "height", "fps", "dpi", "crf", "preset", "preview_width", "preview_height", "preview_fps", "fade_s", "reference_height"}
        self.assertTrue(keys <= set(v), keys - set(v))
        self.assertEqual(v["width"] * 9, v["height"] * 16)
        self.assertGreater(v["width"], v["preview_width"])
        self.assertIn("style", self.cfg)

    def test_script_reads_the_config(self) -> None:
        src = (HERE / "z_decay_asymmetry.py").read_text(encoding="utf-8")
        self.assertIn("load_config", src)
        for forbidden in ('"libx264", "-preset", "slow"', "(1280, 720)", "dpi = 100\n"):
            self.assertNotIn(forbidden, src)

    def test_set_overrides_a_value(self) -> None:
        out = subprocess.run([sys.executable, "-c", "import z_decay_asymmetry as m; print(m.CFG.video.crf, m.S2W)", "--set", "video.crf=23",
                              "--set", "physics.sin2_theta_w=0.25"], cwd=HERE, capture_output=True, text=True, check=True).stdout.split()
        self.assertEqual(out, ["23", "0.25"])

    def test_the_numbers_of_the_book_are_in_the_config(self) -> None:
        self.assertEqual(za.S2W, 0.2316)                                   # sin^2(theta_W)(m_Z), eq. (Z_decay_sinW) of the chapter
        self.assertAlmostEqual(1.0 / za.ALPHA, 128.95)                     # alpha_em^{-1}(m_Z)
        self.assertAlmostEqual(za.GZ, 2.4955)                              # the experimental Gamma_Z of the chapter


class FilmTest(unittest.TestCase):
    def test_parts_follow_each_other_and_cards_precede_them(self) -> None:
        for a, b in zip(za.PARTS[:-1], za.PARTS[1:]):
            self.assertEqual(a[1], b[0])
        self.assertEqual(za.PARTS[-1][1], za.CONTENT_TOTAL)
        self.assertEqual(len(za.CARD_AT), len(za.CARD_S))
        self.assertEqual(za.CARD_AT[0], 0.0)
        self.assertEqual(za.CARD_AT[1:], [p[0] for p in za.PARTS[1:]])
        self.assertTrue(60.0 <= za.TOTAL <= 110.0, za.TOTAL)

    def test_timeline_inserts_the_cards_and_is_monotone(self) -> None:
        prev = -1.0
        for tf in np.linspace(0.0, za.TOTAL - 0.01, 4000):
            tc, card, prog = za.timeline(float(tf))
            self.assertGreaterEqual(tc, prev - 1e-9)
            prev = tc
            if card is not None:
                self.assertTrue(0.0 <= prog < 1.0)
        self.assertEqual(za.timeline(0.5)[1], 0)
        self.assertIsNone(za.timeline(za.CARD_S[0] + 0.1)[1])
        for tc in (0.5, 14.0, 40.0, 60.0, 80.0):
            tc2, card, _ = za.timeline(za.film_time(tc))
            self.assertIsNone(card)
            self.assertAlmostEqual(tc2, tc)

    def test_every_part_has_a_caption_at_every_time(self) -> None:
        for lang in ("en", "ru"):
            P = za.Painter(320, 180)
            for tf in np.linspace(0.0, za.TOTAL - 0.05, 60):
                za.draw_frame(P, float(tf), lang)
            P.plt.close(P.fig)

    def test_texts_stay_inside_the_frame_and_do_not_overlap(self) -> None:
        """Every visible text of sample frames of both languages lies inside the frame and no two visible texts overlap."""
        w, h = za.CFG.video.width, za.CFG.video.height
        times = []
        for k, (a, b) in enumerate(za.PARTS):
            times += [a + (b - a) * f for f in (0.03, 0.12, 0.22, 0.32, 0.42, 0.52, 0.62, 0.72, 0.82, 0.92, 0.985)]
        bad = []
        for lang in ("en", "ru"):
            P = za.Painter(w, h)
            for tc in times:
                za.draw_frame(P, za.film_time(tc), lang)
                P.finish()
                boxes = [(s, bb, a) for s, bb, a in P.text_boxes() if a > 0.35]
                for s, bb, a in boxes:
                    if bb[0] < -1.0 or bb[2] > w + 1.0 or bb[1] < -1.0 or bb[3] > h + 1.0:
                        bad.append(("outside", lang, round(tc, 2), s))
                for i in range(len(boxes)):
                    for j in range(i + 1, len(boxes)):
                        a_, b_ = boxes[i][1], boxes[j][1]
                        ox = min(a_[2], b_[2]) - max(a_[0], b_[0])
                        oy = min(a_[3], b_[3]) - max(a_[1], b_[1])
                        if ox > 2.0 and oy > 2.0:
                            bad.append(("overlap", lang, round(tc, 2), boxes[i][0], boxes[j][0]))
            P.plt.close(P.fig)
        self.assertEqual(bad, [])

    def test_the_numbers_on_the_screen_are_the_computed_ones(self) -> None:
        w, h = za.CFG.video.width, za.CFG.video.height
        P = za.Painter(w, h)
        tcs = {1: za.PARTS[0][1] - 0.1, 2: za.PARTS[1][1] - 0.1, 3: za.PARTS[2][1] - 0.1}
        for lang, dec in (("en", "."), ("ru", "{,}")):
            seen: dict = {}
            for part, tc in tcs.items():
                za.draw_frame(P, za.film_time(tc), lang)
                P.finish()
                seen[part] = " ".join(s for s, _bb, _a in P.text_boxes())
            fmt = lambda x, d=3: f"{x:.{d}f}".replace(".", dec)
            self.assertIn(fmt(za.a_lr("e")), seen[1])
            self.assertIn(fmt(za.a_lr("d")), seen[1])
            self.assertIn(fmt(za.a_fb("d")), seen[2])
            self.assertIn(fmt(za.a_lr("e")), seen[3])
            self.assertIn(fmt(za.a_fb("e")), seen[3])
            self.assertIn(fmt(za.a_fb("d")), seen[3])
        P.plt.close(P.fig)

    def test_the_film_events_agree_with_the_model(self) -> None:
        ev = za.event_data()
        for key, f in (("cb", "d"), ("cm", "e")):
            a, err, _nf, _nb = za.estimate(ev[key])
            self.assertLess(abs(a - za.a_fb(f)), 3.0 * err, (key, a, za.a_fb(f), err))
        self.assertEqual(len(ev["cb"]), za.CFG.sampling.n_b)
        self.assertEqual(len(ev["cm"]), za.CFG.sampling.n_mu)
        self.assertTrue(np.all(np.abs(ev["cb"]) <= 1.0))

    def test_the_event_counts_grow_monotonically_to_the_final_values(self) -> None:
        E = za.CFG.events
        last = 0
        for t in np.linspace(0.0, za.PARTS[3][1] - za.PARTS[3][0], 600):
            n = za.n_b_at(float(t))
            self.assertGreaterEqual(n, last)
            last = n
        self.assertEqual(za.n_b_at(E.ramp_b[1] + 1.0), za.CFG.sampling.n_b)
        self.assertEqual(za.n_mu_at(E.ramp_mu[1] + 1.0), za.CFG.sampling.n_mu)
        self.assertEqual(za.n_b_at(E.slow_events[-1] + 1e-6), len(E.slow_events))

    def test_the_energy_sweep_is_smooth_monotone_and_visits_the_pole(self) -> None:
        sc = za.CFG.scan
        ts = np.linspace(sc.path_t[0], sc.path_t[-1], 500)
        s = np.array([za.sqrt_s_at(float(t)) for t in ts])
        self.assertTrue(np.all(np.diff(s) > 0))
        self.assertAlmostEqual(s[0], sc.x_range[0])
        self.assertAlmostEqual(s[-1], sc.x_range[1])
        self.assertAlmostEqual(za.sqrt_s_at(sc.path_t[3]), za.MZ)

    def test_snapshot_renders_in_both_languages(self) -> None:
        for lang in ("en", "ru"):
            out = Path(__import__("tempfile").mkdtemp()) / f"snap_{lang}.mp4"
            subprocess.run([sys.executable, "z_decay_asymmetry.py", "--lang", lang, "--snapshot", "31", "--content", "--out", str(out)], cwd=HERE, check=True,
                           capture_output=True)
            png = out.with_suffix(".png")
            self.assertTrue(png.exists() and png.stat().st_size > 20000)


if __name__ == "__main__":
    unittest.main()
