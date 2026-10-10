"""Tests of the physics, of the timeline, of the configuration and of the frames of the film "Oscillations of neutral kaons".

    python -m pytest test_kaon_oscillations.py
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

import kaon_oscillations as film
import kaon_physics as kp

HERE = Path(__file__).resolve().parent
PH = kp.PH


def evolve(h: np.ndarray, psi0: np.ndarray, t: float) -> np.ndarray:
    return kp.integrate(h, psi0, np.array([0.0, t]))[-1]


# ================================================================================================ the effective Hamiltonian

class HamiltonianTest(unittest.TestCase):
    def test_parameters_are_the_numbers_of_the_book(self) -> None:
        self.assertAlmostEqual(PH.tau_s_s, 8.954e-11)
        self.assertAlmostEqual(PH.tau_l_s, 5.116e-8)
        self.assertAlmostEqual(PH.delta_m_ev, 3.5e-6)                      # 3.5e-15 GeV
        self.assertAlmostEqual(kp.GAMMA_L, 8.954e-11 / 5.116e-8)
        self.assertAlmostEqual(kp.DM, 3.5e-6 / 6.582119569e-16 * 8.954e-11, delta=1e-12)
        self.assertAlmostEqual(PH.delta_m_ev / PH.hbar_ev_s / 1e10, 0.53, delta=0.01)             # 0.53e10 hbar/s

    def test_ratio_of_lifetimes_and_dm_tau(self) -> None:
        self.assertAlmostEqual(kp.TAU_L, 571.36, delta=0.05)
        self.assertAlmostEqual(kp.DM, 0.476, delta=0.002)
        self.assertAlmostEqual(kp.PERIOD, 2 * math.pi / kp.DM)

    def test_ks_and_kl_are_the_eigenvectors_with_the_right_eigenvalues(self) -> None:
        h = kp.hamiltonian()
        for vec, m, g in ((kp.KS, -0.5 * kp.DM, kp.GAMMA_S), (kp.KL, +0.5 * kp.DM, kp.GAMMA_L)):
            self.assertTrue(np.allclose(h @ vec, (m - 0.5j * g) * vec, atol=1e-14))

    def test_the_mass_difference_is_m_L_minus_m_S(self) -> None:
        w = np.linalg.eigvals(kp.hamiltonian())
        mL = max(w.real)
        mS = min(w.real)
        self.assertAlmostEqual(mL - mS, kp.DM, places=12)
        self.assertAlmostEqual(-2 * w[np.argmax(w.real)].imag, kp.GAMMA_L, places=12)
        self.assertAlmostEqual(-2 * w[np.argmin(w.real)].imag, kp.GAMMA_S, places=12)

    def test_k0_is_the_sum_of_ks_and_kl_and_the_basis_is_orthonormal(self) -> None:
        self.assertTrue(np.allclose(kp.K0, (kp.KS + kp.KL) / math.sqrt(2.0)))
        self.assertTrue(np.allclose(kp.K0BAR, (kp.KL - kp.KS) / math.sqrt(2.0)))
        self.assertAlmostEqual(abs(np.vdot(kp.KS, kp.KL)), 0.0, places=15)
        self.assertAlmostEqual(np.linalg.norm(kp.KS), 1.0)

    def test_cp_is_conserved_h_commutes_with_cp(self) -> None:
        cp = -kp.SX                                                      # CP K0 = -K0bar, CP K0bar = -K0 (the book)
        h = kp.hamiltonian()
        self.assertTrue(np.allclose(h @ cp, cp @ h, atol=1e-15))
        self.assertTrue(np.allclose(cp @ kp.KS, +kp.KS))                  # K_S = K_1 has CP = +1
        self.assertTrue(np.allclose(cp @ kp.KL, -kp.KL))                  # K_L = K_2 has CP = -1


# ================================================================================================ the exact evolution

class EvolutionTest(unittest.TestCase):
    def test_unitarity_without_decays(self) -> None:
        h = kp.hamiltonian(0.0, 0.0)
        for t in (1.0, 7.0, 40.0, 300.0):
            self.assertAlmostEqual(float(np.linalg.norm(evolve(h, kp.K0, t))), 1.0, places=9)
        run = kp.stable_run()
        self.assertTrue(np.allclose(np.sum(np.abs(run.psi) ** 2, axis=1), 1.0, atol=1e-9))

    def test_full_oscillation_without_decays_with_period_2pi_over_dm(self) -> None:
        run = kp.stable_run()
        a0, ab = run.at(np.array([np.pi / kp.DM, kp.PERIOD]))
        self.assertAlmostEqual(abs(a0[0]) ** 2, 0.0, places=5)            # all K0 became K0bar after t = pi / dm (linear interpolation between grid points)
        self.assertAlmostEqual(abs(ab[0]) ** 2, 1.0, places=5)
        self.assertAlmostEqual(abs(a0[1]) ** 2, 1.0, places=5)            # and returned after the period
        t = np.linspace(0.0, 14.0, 281)
        a0, ab = run.at(t)
        self.assertTrue(np.allclose(np.abs(a0) ** 2, np.cos(kp.DM * t / 2) ** 2, atol=1e-6))
        self.assertTrue(np.allclose(np.abs(ab) ** 2, np.sin(kp.DM * t / 2) ** 2, atol=1e-6))

    def test_numerical_integration_equals_the_formulas_of_the_book(self) -> None:
        run = kp.beam_run()
        i0 = np.abs(run.psi[:, 0]) ** 2
        ib = np.abs(run.psi[:, 1]) ** 2
        self.assertLess(float(np.max(np.abs(i0 - kp.p_survive(run.t)))), 1e-9)
        self.assertLess(float(np.max(np.abs(ib - kp.p_appear(run.t)))), 1e-9)

    def test_the_formulas_agree_with_an_independent_matrix_exponential(self) -> None:
        from scipy.linalg import expm
        h = kp.hamiltonian()
        for t in (0.3, 2.0, 9.0, 55.0, 700.0):
            psi = expm(-1j * h * t) @ kp.K0
            self.assertAlmostEqual(abs(psi[0]) ** 2, float(kp.p_survive(t)), places=10)
            self.assertAlmostEqual(abs(psi[1]) ** 2, float(kp.p_appear(t)), places=10)

    def test_sum_of_the_probabilities(self) -> None:
        run = kp.beam_run()
        tot = np.sum(np.abs(run.psi) ** 2, axis=1)
        self.assertLess(float(np.max(np.abs(tot - 0.5 * (np.exp(-kp.GAMMA_S * run.t) + np.exp(-kp.GAMMA_L * run.t))))), 1e-9)

    def test_trace_and_determinant_of_the_evolution_matrix(self) -> None:
        h = kp.hamiltonian()
        for t in (0.5, 3.0, 20.0, 400.0):
            u = np.column_stack([evolve(h, kp.K0, t), evolve(h, kp.K0BAR, t)])
            tr = np.exp(-(1j * (-0.5 * kp.DM) + 0.5 * kp.GAMMA_S) * t) + np.exp(-(1j * (0.5 * kp.DM) + 0.5 * kp.GAMMA_L) * t)
            self.assertAlmostEqual(abs(np.trace(u) - tr), 0.0, places=8)
            self.assertAlmostEqual(abs(np.linalg.det(u) - np.exp(-0.5 * (kp.GAMMA_S + kp.GAMMA_L) * t)), 0.0, places=8)

    def test_the_total_intensity_decays_with_the_rate_psi_gamma_psi(self) -> None:
        h = kp.hamiltonian()
        gamma = -2.0 * (0.5 * (h - h.conj().T) / 1j)                      # Gamma = i (H - H^dagger)
        gamma = 1j * (h - h.conj().T)
        for t in (0.4, 2.0, 15.0):
            psi = evolve(h, kp.K0, t)
            eps = 1e-5
            n1 = np.linalg.norm(evolve(h, kp.K0, t + eps)) ** 2
            n0 = np.linalg.norm(evolve(h, kp.K0, t - eps)) ** 2
            self.assertAlmostEqual((n1 - n0) / (2 * eps), -float(np.real(np.vdot(psi, gamma @ psi))), places=6)

    def test_the_oscillation_frequency_is_delta_m(self) -> None:
        """P(K0 -> K0) minus the incoherent part oscillates as cos(dm t): its zeros are at dm t = pi/2 + n pi."""
        t = np.linspace(0.0, 6.0, 6001)
        run = kp.beam_run()
        a0, ab = run.at(t)
        inter = np.abs(a0) ** 2 - np.abs(ab) ** 2                         # = e^{-(Gs+Gl)t/2} cos(dm t)
        ratio = inter / np.exp(-0.5 * (kp.GAMMA_S + kp.GAMMA_L) * t)
        self.assertTrue(np.allclose(ratio, np.cos(kp.DM * t), atol=1e-5))        # linear interpolation between the grid points of the run
        zero = t[np.argmax(ratio < 0)]
        self.assertAlmostEqual(zero, math.pi / (2 * kp.DM), delta=2e-3)

    def test_late_time_composition_is_fifty_fifty(self) -> None:
        run = kp.beam_run()
        for t in (100.0, 570.0, 1200.0, 2200.0):
            a0, ab = run.at(np.array([t]))
            self.assertAlmostEqual(abs(a0[0]) ** 2 / abs(ab[0]) ** 2, 1.0, places=6)
            self.assertAlmostEqual(abs(a0[0]) ** 2, 0.25 * math.exp(-kp.GAMMA_L * t), delta=1e-8)
            s, l = kp.s_l(a0, ab)
            self.assertLess(abs(s[0]) / abs(l[0]), 1e-9 + 1e-2 * math.exp(-0.5 * (t - 100)))

    def test_ks_is_gone_after_a_few_lifetimes_and_kl_survives(self) -> None:
        ps, pl = kp.project(kp.beam_run().psi)
        t = kp.beam_run().t
        self.assertTrue(np.allclose(np.abs(ps) ** 2, 0.5 * np.exp(-kp.GAMMA_S * t), atol=1e-9))     # the book: 1/2 e^{-Gamma_1 t}
        self.assertTrue(np.allclose(np.abs(pl) ** 2, 0.5 * np.exp(-kp.GAMMA_L * t), atol=1e-9))     # 1/2 e^{-Gamma_2 t}

    def test_the_time_grid_resolves_the_oscillation(self) -> None:
        self.assertLess(kp.DM * kp.CFG.integrator.coarse_dt, 0.05)                      # the phase per step of the grid is much smaller than 1

    def test_phasor_decomposition_a0_equals_s_plus_l(self) -> None:
        a0, ab = kp.beam_run().at(np.array([0.0, 1.0, 4.0, 30.0]))
        s, l = kp.s_l(a0, ab)
        self.assertTrue(np.allclose(s + l, a0))
        self.assertTrue(np.allclose(l - s, ab))
        t = np.array([0.0, 1.0, 4.0, 30.0])
        self.assertTrue(np.allclose(np.abs(s), 0.5 * np.exp(-0.5 * kp.GAMMA_S * t), atol=1e-9))
        self.assertTrue(np.allclose(np.abs(l), 0.5 * np.exp(-0.5 * kp.GAMMA_L * t), atol=1e-9))
        # the phases rotate in opposite directions with the angular velocity dm / 2 (the frame of the mean mass)
        self.assertTrue(np.allclose(np.angle(s * np.exp(-0.5j * kp.DM * t)), 0.0, atol=1e-8))
        self.assertTrue(np.allclose(np.angle(l * np.exp(+0.5j * kp.DM * t)), 0.0, atol=1e-8))


# ================================================================================================ what the detector sees

class DetectorTest(unittest.TestCase):
    def test_lepton_asymmetry_is_cos_over_cosh(self) -> None:
        t = np.linspace(0.0, 8.0, 801)
        a0, ab = kp.beam_run().at(t)
        self.assertTrue(np.allclose(kp.lepton_asymmetry(a0, ab), kp.asymmetry(t), atol=1e-8))
        self.assertTrue(np.allclose(kp.asymmetry(t), np.cos(kp.DM * t) / np.cosh(0.5 * (kp.GAMMA_L - kp.GAMMA_S) * t)))

    def test_asymmetry_without_width_difference_is_a_pure_cosine(self) -> None:
        t = np.linspace(0.0, 20.0, 400)
        self.assertTrue(np.allclose(kp.asymmetry(t, 1.0, 1.0), np.cos(kp.DM * t)))

    def test_the_hadronic_decays_do_not_oscillate(self) -> None:
        """Projection on K_S and K_L: 1/2 e^{-Gamma t} without any cosine (the book, eq. 19)."""
        ps, pl = kp.project(kp.beam_run().psi)
        t = kp.beam_run().t[:400]
        self.assertTrue(np.allclose(np.abs(ps[:400]) ** 2, 0.5 * np.exp(-t), atol=1e-10))

    def test_hyperon_yield_is_proportional_to_the_k0bar_intensity_and_has_a_maximum(self) -> None:
        t = np.linspace(0.0, 12.0, 2401)
        pb = kp.p_appear(t)
        self.assertGreater(float(pb.max()), 0.27)                          # the maximum above the plateau 1/4
        self.assertAlmostEqual(float(t[np.argmax(pb)]), 4.9, delta=0.4)
        self.assertAlmostEqual(float(kp.p_appear(11.0)), 0.25 * math.exp(-kp.GAMMA_L * 11.0), delta=0.002)


# ================================================================================================ regeneration

class RegenerationTest(unittest.TestCase):
    def setUp(self) -> None:
        self.R = kp.regeneration()
        self.f, self.fb = PH.f_k0, PH.f_k0bar

    def test_the_incident_beam_is_kl(self) -> None:
        ps, pl = kp.project(self.R["psi_in"])
        self.assertLess(abs(ps) ** 2, 1e-3)
        self.assertAlmostEqual(float(np.linalg.norm(self.R["psi_in"])), 1.0)
        self.assertAlmostEqual(abs(pl) ** 2, 1.0, delta=1e-3)

    def test_an_absorber_regenerates_ks(self) -> None:
        ps, pl = kp.project(self.R["psi_out"])
        self.assertAlmostEqual(abs(ps), abs(0.5 * (self.f - self.fb)), delta=2e-3)          # (f - fbar) / 2
        self.assertAlmostEqual(abs(pl), abs(0.5 * (self.f + self.fb)), delta=2e-3)          # (f + fbar) / 2

    def test_no_regeneration_if_k0_and_k0bar_are_absorbed_equally(self) -> None:
        for f in (1.0, 0.7, 0.2):
            psi = kp.slab_matrix(1.0, f, f) @ self.R["psi_in"]
            self.assertLess(abs(kp.project(psi)[0]), 1e-2 * f)                # only the residual K_S amplitude e^{-5} of the incident beam

    def test_regenerated_intensity_decays_with_the_ks_width_and_kl_stays(self) -> None:
        ta, after = self.R["ta"], self.R["after"]
        ps, pl = kp.project(after)
        self.assertTrue(np.allclose(np.abs(ps) ** 2, np.abs(ps[0]) ** 2 * np.exp(-kp.GAMMA_S * ta), rtol=1e-8, atol=1e-12))
        self.assertTrue(np.allclose(np.abs(pl) ** 2, np.abs(pl[0]) ** 2 * np.exp(-kp.GAMMA_L * ta), rtol=1e-8))

    def test_behind_the_absorber_the_beam_is_again_fifty_fifty_after_ks_is_gone(self) -> None:
        a0, ab = self.R["after"][-1]
        self.assertGreater(abs(ab) / abs(a0), 0.99)
        self.assertLess(abs(ab) / abs(a0), 1.01)

    def test_the_absorber_removes_more_k0bar_than_k0(self) -> None:
        self.assertLess(PH.f_k0bar, PH.f_k0)
        psi = kp.slab_matrix() @ self.R["psi_in"]
        self.assertGreater(abs(psi[0]), abs(psi[1]))

    def test_the_intensities_before_the_absorber_are_those_of_the_beam(self) -> None:
        ps, pl = kp.project(self.R["before"])
        tb = self.R["tb"] + PH.slab_time
        norm2 = self.R["norm"] ** 2
        self.assertTrue(np.allclose(np.abs(ps) ** 2 * norm2, 0.5 * np.exp(-kp.GAMMA_S * tb), rtol=1e-6, atol=1e-12))
        self.assertTrue(np.allclose(np.abs(pl) ** 2 * norm2, 0.5 * np.exp(-kp.GAMMA_L * tb), rtol=1e-8))

    def test_the_slab_matrix_goes_through_the_slab_smoothly(self) -> None:
        self.assertTrue(np.allclose(kp.slab_matrix(0.0), np.eye(2)))
        self.assertTrue(np.allclose(kp.slab_matrix(1.0), np.diag([self.f, self.fb])))
        self.assertTrue(np.allclose(kp.slab_matrix(0.5) @ kp.slab_matrix(0.5), kp.slab_matrix(1.0)))


# ================================================================================================ timeline and frames

class FilmTest(unittest.TestCase):
    def test_parts_follow_each_other(self) -> None:
        pb = film.PB
        self.assertEqual(pb[0], 0.0)
        self.assertEqual(pb[-1], film.CONTENT_TOTAL)
        self.assertEqual(len(pb) - 1, len(film.PART_KEYS))
        for ca in film.CARD_AT:
            self.assertIn(ca, pb)

    def test_the_length_of_the_film_is_in_the_asked_range(self) -> None:
        self.assertGreaterEqual(film.TOTAL, 60.0)
        self.assertLessEqual(film.TOTAL, 110.0)

    def test_timeline_inserts_the_cards(self) -> None:
        self.assertAlmostEqual(film.TOTAL, film.CONTENT_TOTAL + sum(film.CARD_S))
        prev = -1.0
        for tf in np.linspace(0.0, film.TOTAL - 0.01, 3000):
            tc, card, prog = film.timeline(float(tf))
            self.assertGreaterEqual(tc, prev - 1e-9)
            prev = tc
            if card is not None:
                self.assertTrue(0.0 <= prog < 1.0)
        self.assertEqual(film.timeline(0.5)[1], 0)
        self.assertIsNone(film.timeline(film.CARD_S[0] + 0.5)[1])
        self.assertAlmostEqual(film.timeline(film.CARD_S[0] + 0.5)[0], 0.5)

    def test_film_time_is_the_inverse_of_the_timeline(self) -> None:
        for tc in (0.5, 17.0, 25.0, 50.0, 60.0, 80.0, 93.0):
            tc2, card, _ = film.timeline(film.film_time(tc))
            self.assertIsNone(card)
            self.assertAlmostEqual(tc2, tc)

    def test_every_card_has_a_title_in_both_languages(self) -> None:
        for lang in ("en", "ru"):
            for key in film.CARD_KEYS + film.PART_KEYS:
                self.assertIn(key, film.TEXT[lang])
                if key in film.CARD_KEYS:
                    self.assertIn(key + "s", film.TEXT[lang])

    def test_sub_stages_are_inside_their_parts(self) -> None:
        c = film.CFG
        self.assertEqual(c.part2.stable[0], film.PB[1])
        self.assertEqual(c.part2.stable[1], c.part2.real[0])
        self.assertEqual(c.part2.real[1], c.part2.zoom[0])
        self.assertEqual(c.part2.zoom[1], film.PB[2])
        self.assertEqual(c.part3.lepton[1], film.PB[3])
        self.assertEqual(c.part4.sweep[1], film.PB[4])
        for fl in c.part3.foils:
            self.assertLess(fl, c.part3.x_hi)

    def test_the_cursor_of_the_stable_stage_covers_one_period(self) -> None:
        self.assertLessEqual(kp.PERIOD, film.CFG.part2.x_hi_early)
        self.assertLessEqual(kp.PERIOD, kp.PH.stable_t_end)

    def test_the_regeneration_axis_contains_the_run(self) -> None:
        c = film.CFG.part4
        self.assertGreaterEqual(-c.x_lo, kp.PH.regen_before)
        self.assertGreaterEqual(c.x_hi, c.sweep_end)
        self.assertLessEqual(c.sweep_end, kp.PH.regen_after)

    def test_numbers_inserted_into_the_texts(self) -> None:
        for lang in ("en", "ru"):
            v = film.values(lang)
            self.assertEqual(v["ratio"], "571")
            self.assertTrue(v["tau_s"].startswith("8" + ("{,}" if lang == "ru" else ".") + "95"))
            self.assertTrue(v["dm_ev"].startswith("3" + ("{,}" if lang == "ru" else ".") + "5"))
            tx = film.texts(lang)
            for key, text in tx.items():
                self.assertNotIn("@", text, key)

    def test_russian_numbers_have_a_decimal_comma(self) -> None:
        tx = film.texts("ru")
        for key, text in tx.items():
            self.assertIsNone(re.search(r"\d\.\d", text), key)

    def test_state_renders_in_both_languages_at_many_times(self) -> None:
        for lang in ("en", "ru"):
            for tc in np.linspace(0.5, film.CONTENT_TOTAL - 0.5, 9):
                out = Path(tempfile.mkdtemp()) / "f.png"
                film.render(out, (640, 360), 15, film.TOTAL, lang, snap=film.film_time(float(tc)))
                self.assertTrue(out.exists())


class FrameLayoutTest(unittest.TestCase):
    """No overlapping texts and no text outside the frame at the checked times (content seconds) of both languages."""

    TIMES = [3.0, 8.0, 12.5, 17.0, 19.5, 22.0, 27.0, 33.0, 38.0, 42.0, 45.0, 48.0, 51.0, 54.0, 58.0, 64.0, 69.0, 72.0, 75.5, 79.0, 84.0, 91.0, 93.5]

    def check(self, lang: str) -> None:
        for tc in self.TIMES:
            problems: list[str] = []
            out = Path(tempfile.mkdtemp()) / "f.png"
            film.render(out, (1280, 720), 30, film.TOTAL, lang, snap=film.film_time(tc), hook=lambda fig, t, card: problems.extend(film.frame_problems(fig)))
            self.assertEqual(problems, [], f"{lang} {tc}")

    def test_english(self) -> None:
        self.check("en")

    def test_russian(self) -> None:
        self.check("ru")

    def test_cards_have_no_overlaps(self) -> None:
        for lang in ("en", "ru"):
            for tf in (1.5, film.film_time(20.0) - 1.0, film.film_time(52.0) - 1.0, film.film_time(74.0) - 1.0):
                problems: list[str] = []
                out = Path(tempfile.mkdtemp()) / "f.png"
                film.render(out, (1280, 720), 30, film.TOTAL, lang, snap=tf, hook=lambda fig, t, card: problems.extend(film.frame_problems(fig)))
                self.assertEqual(problems, [], f"{lang} {tf}")


# ================================================================================================ the configuration

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

    def test_every_placeholder_has_a_value(self) -> None:
        names = set(film.values("en"))
        for lang in ("en", "ru"):
            for key, text in self.texts[lang].items():
                for name in re.findall(r"@([a-z0-9_]+)@", text):
                    self.assertIn(name, names, key)

    def test_no_russian_words_inside_formulas_and_no_english_in_russian(self) -> None:
        for key, text in self.texts["ru"].items():
            for formula in re.findall(r"\$([^$]*)\$", text):
                with self.subTest(key=key):
                    self.assertIsNone(re.search(r"[А-Яа-яЁё]", formula))
        words = re.compile(r"\b(the|and|of|with|beam|target|absorber)\b")
        for key, text in self.texts["ru"].items():
            with self.subTest(key=key):
                self.assertIsNone(words.search(re.sub(r"\$[^$]*\$", "", text)))

    def test_no_traces_of_discussions_in_the_texts(self) -> None:
        bad = re.compile(r"(author|audit|please check|honest|borrow|lend)", re.IGNORECASE)
        for lang, table in self.texts.items():
            for key, text in table.items():
                self.assertIsNone(bad.search(text), f"{lang}.{key}")

    def test_video_section_is_complete_and_16_to_9(self) -> None:
        v = self.cfg["video"]
        keys = {"width", "height", "fps", "dpi", "crf", "preset", "preview_width", "preview_height", "preview_fps", "fade_s", "reference_height"}
        self.assertTrue(keys <= set(v), keys - set(v))
        self.assertEqual(v["width"] * 9, v["height"] * 16)
        self.assertGreater(v["width"], v["preview_width"])
        self.assertIn("style", self.cfg)

    def test_script_reads_the_config(self) -> None:
        for name in ("kaon_oscillations.py", "kaon_physics.py"):
            src = (HERE / name).read_text(encoding="utf-8")
            self.assertIn("load_config", src)
            for forbidden in ('"libx264", "-preset", "slow"', "(1280, 720)", "dpi = 100\n"):
                self.assertNotIn(forbidden, src)

    def test_set_overrides_a_value(self) -> None:
        out = subprocess.run([sys.executable, "-c", "import kaon_oscillations as m; print(m.CFG.video.crf, m.kp.PH.delta_m_ev)", "--set", "video.crf=23", "--set", "physics.delta_m_ev=7e-6"],
                             cwd=HERE, capture_output=True, text=True, check=True).stdout.split()
        self.assertEqual(out, ["23", "7e-06"])

    def test_every_text_key_used_by_the_script_exists(self) -> None:
        src = (HERE / "kaon_oscillations.py").read_text(encoding="utf-8")
        used = set(re.findall(r'tx\["([a-zA-Z0-9_]+)"\]', src)) | set(re.findall(r'"((?:c[1-4][a-e]|f[1-4][a-c][0-9]?|lg_[a-z0-9]+|rx2?_[a-z0-9_]+|pl_[a-z0-9_]+|ph_[a-z0-9_]+|tb_[a-z0-9_]+|cn_[a-z]+|lp_[a-z]+|xl_[a-z]+|b_[slm](?:_[vn])?))"', src))
        used |= {"c1a", "c1b", "c1c", "c1d", "c1e", "c2a", "c2b", "c2c", "c2d", "c2e", "c3a", "c3b", "c3c", "c3d", "c3e", "c4a", "c4b", "c4c", "c4d", "c4e"}
        missing = used - set(self.texts["en"])
        self.assertFalse(missing, missing)

    def test_the_script_has_no_numeric_literals_for_the_picture(self) -> None:
        """Every number of the picture lives in config.toml: the drawing code only has halves, unit steps and tolerances."""
        src = (HERE / "kaon_oscillations.py").read_text(encoding="utf-8")
        body = src[src.index("def render("):]
        body = re.sub(r"\"[^\"\n]*\"", "", body)
        body = re.sub(r"#.*", "", body)
        allowed = {"0.0", "0.5", "1.0", "2.0", "3.0", "5.0", "10.0"}               # halves, units and the 1-2-5 sequence of the tick steps
        suspicious = [x for x in re.findall(r"(?<![\w.])\d+\.\d+(?![\w.])", body) if x not in allowed]
        self.assertEqual(suspicious, [])

    def test_snapshots_render_in_both_languages(self) -> None:
        for lang in ("en", "ru"):
            out = Path(tempfile.mkdtemp()) / f"s_{lang}.mp4"
            subprocess.run([sys.executable, "kaon_oscillations.py", "--lang", lang, "--content", "30", "--out", str(out)], cwd=HERE, check=True, capture_output=True)
            self.assertTrue(out.with_suffix(".png").exists())


if __name__ == "__main__":
    unittest.main()
