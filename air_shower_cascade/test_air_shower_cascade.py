"""Tests of the air-shower film: the toy model against its analytic limits and the conservation laws, and the film files.

    python -m pytest test_air_shower_cascade.py        (or python -m unittest test_air_shower_cascade)
"""

from __future__ import annotations

import math
import sys
import tempfile
import tomllib
import unittest
from fractions import Fraction
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.argv = sys.argv[:1]                    # the script reads --config / --set from argv when imported

import air_shower_cascade as asc  # noqa: E402
import shower_model as sm  # noqa: E402

P0 = asc.make_params()


def photon_peak(p: sm.Params, e0: float, seed: int, dx: float = 5.0) -> tuple[float, float]:
    sh = sm.simulate_photon(p.with_(x_ground=5000.0), seed, dx, e0_ev=e0, thin=False)
    return sh.peak()


class HeitlerLimitTest(unittest.TestCase):
    """The fixed-step cascade reproduces Heitler's formulas exactly."""

    def test_nmax_and_xmax_of_a_photon_cascade(self) -> None:
        pf = P0.with_(path="fixed", x_ground=1.0e5)
        for n in (4, 8, 12):
            e0 = P0.e_c_ev * 2 ** n
            sh = sm.simulate_photon(pf, 0, 1.0, e0_ev=e0, thin=False)
            xm, nm = sh.peak(smooth_bins=1)
            self.assertEqual(nm, e0 / P0.e_c_ev)
            self.assertAlmostEqual(xm, sm.heitler_x_max(P0.x0, e0, P0.e_c_ev), delta=1.0)

    def test_the_number_of_particles_doubles_every_step(self) -> None:
        pf = P0.with_(path="fixed", x_ground=1.0e5)
        e0 = P0.e_c_ev * 2 ** 6
        sh = sm.simulate_photon(pf, 0, 0.5, e0_ev=e0, thin=False)
        d = P0.x0 * math.log(2.0)
        n_all = sh.count((sm.GAMMA, sm.ELECTRON))
        for g in range(6):
            i = int(round((g + 0.5) * d / sh.dx))
            self.assertEqual(n_all[i], 2.0 ** g)

    def test_the_two_thirds_of_a_pure_cascade_are_charged(self) -> None:
        pf = P0.with_(path="fixed", x_ground=1.0e5)
        sh = sm.simulate_photon(pf, 0, 0.5, e0_ev=P0.e_c_ev * 2 ** 10, thin=False)
        d = P0.x0 * math.log(2.0)
        i = int(round(9.5 * d / sh.dx))
        ne, ng = sh.count((sm.ELECTRON,))[i], sh.count((sm.GAMMA,))[i]
        self.assertAlmostEqual(ne / (ne + ng), 2.0 / 3.0, delta=0.01)

    def test_analytic_profile_equals_the_fixed_step_simulation(self) -> None:
        pf = P0.with_(path="fixed", e0_ev=1.0e12, x_ground=1.0e5, e_thin_ev=float("inf"))
        sh = sm.simulate(pf, 1, 1.0, thin=False)
        g = np.arange(0.0, 1500.0, 1.0)
        an = sm.heitler_matthews_profile(pf, g)
        self.assertTrue(np.array_equal(sh.count((sm.GAMMA, sm.ELECTRON), g), an["em"]))
        self.assertTrue(np.array_equal(sh.count((sm.MUON,), g), an["mu"]))
        self.assertTrue(np.array_equal(sh.count((sm.PROTON, sm.PION), g), an["had"]))

    def test_the_random_depth_keeps_the_xmax_of_heitler(self) -> None:
        """Mean step X_0 ln 2: X_max follows X_0 ln(E_0/E_c) (within a few tens of g/cm^2) and N_max is a quarter of E_0/E_c."""
        for e0 in (1.0e12, 1.0e14):
            res = [photon_peak(P0, e0, s) for s in range(6)]
            xm = np.mean([r[0] for r in res])
            nm = np.mean([r[1] for r in res])
            self.assertAlmostEqual(xm, sm.heitler_x_max(P0.x0, e0, P0.e_c_ev), delta=30.0)
            self.assertTrue(0.12 < nm / (e0 / P0.e_c_ev) < 0.45, nm / (e0 / P0.e_c_ev))

    def test_xmax_grows_as_ln_e0(self) -> None:
        x = {e0: np.mean([photon_peak(P0, e0, s)[0] for s in range(6)]) for e0 in (1.0e12, 1.0e14)}
        self.assertAlmostEqual((x[1.0e14] - x[1.0e12]) / math.log(100.0), P0.x0, delta=0.25 * P0.x0)

    def test_track_length_is_proportional_to_the_energy(self) -> None:
        """The integral of N(X) is the fluorescence-light integral: proportional to E_0 (exactly so for energies that differ by
        a power of two, within the discreteness of the last generation, +-15 %, for others)."""
        def per_energy(e0: float) -> float:
            vals = []
            for s in range(3):
                sh = sm.simulate_photon(P0.with_(x_ground=1.0e5), s, 5.0, e0_ev=e0, thin=False)
                vals.append(float(sh.count((sm.GAMMA, sm.ELECTRON)).sum() * sh.dx) / e0)
            return float(np.mean(vals))
        k0 = per_energy(1.0e12)
        self.assertAlmostEqual(per_energy(1.0e12 * 2 ** 10) / k0, 1.0, delta=0.03)
        for e0 in (1.0e11, 1.0e13, 1.0e14):
            self.assertAlmostEqual(per_energy(e0) / k0, 1.0, delta=0.15)

    def test_moliere_radius_of_the_model(self) -> None:
        rm = sm.moliere_radius_m(P0.with_(x_ground=1030.0), 1030.0)
        self.assertAlmostEqual(rm, 71.7, delta=1.0)           # E_s X_0 / (E_c rho), rho = X / H


class ConservationTest(unittest.TestCase):
    def test_a_splitting_conserves_the_energy(self) -> None:
        sh = sm.simulate_photon(P0.with_(x_ground=1.0e5), 3, 5.0, e0_ev=1.0e11, thin=False)
        tr = sh.tr
        parent = tr["parent"]
        has = parent >= 0
        # the energy of the children of every splitting particle sums to the energy of the parent
        sums = np.bincount(parent[has], weights=tr["e"][has], minlength=len(parent))
        split = tr["e"] > P0.e_c_ev
        self.assertTrue(np.allclose(sums[split], tr["e"][split], rtol=1e-12))

    def test_the_total_energy_is_conserved_without_thinning(self) -> None:
        p = P0.with_(e0_ev=1.0e13, e_thin_ev=float("inf"))
        sh = sm.simulate(p, 2, 5.0, thin=False)
        total = (sh.deposited_energy() + sh.em_ground_energy() + sh.e_ground((sm.MUON, sm.NEUTRINO, sm.PION, sm.PROTON)))
        self.assertAlmostEqual(total / p.e0_ev, 1.0, places=9)

    def test_the_total_energy_is_conserved_on_average_with_thinning(self) -> None:
        tot = []
        for s in range(3):
            sh = sm.simulate(P0, s, 5.0)
            tot.append((sh.deposited_energy() + sh.em_ground_energy() + sh.e_ground((sm.MUON, sm.NEUTRINO, sm.PION, sm.PROTON))) / P0.e0_ev)
        self.assertAlmostEqual(float(np.mean(tot)), 1.0, delta=0.03)

    def test_a_hadronic_interaction_shares_the_energy_equally_and_conserves_it(self) -> None:
        sh = sm.simulate(P0.with_(e0_ev=1.0e13, e_thin_ev=float("inf")), 5, 5.0, thin=False)
        tr = sh.tr
        pr = np.flatnonzero(tr["kind"] == sm.PROTON)[0]
        kids = tr["parent"] == pr
        pions = kids & (tr["kind"] == sm.PION)
        self.assertEqual(int(pions.sum()), P0.n_ch)
        e_each = 1.0e13 / P0.n_sec
        self.assertTrue(np.allclose(tr["e"][pions], e_each))
        photons = sh.seeds["parent"] == pr
        self.assertEqual(int(photons.sum()), 2 * P0.n_pi0)
        self.assertAlmostEqual((tr["e"][pions].sum() + sh.seeds["e"][photons].sum()) / 1.0e13, 1.0, places=12)

    def test_the_transverse_momenta_of_an_interaction_cancel(self) -> None:
        sh = sm.simulate(P0.with_(e0_ev=1.0e13, e_thin_ev=float("inf")), 5, 5.0, thin=False)
        tr = sh.tr
        pr = np.flatnonzero(tr["kind"] == sm.PROTON)[0]
        # the primary is vertical: the energy-weighted mean direction of the secondaries is vertical too (equal energies)
        photons = sh.seeds["parent"] == pr
        u = sh.seeds["u"][photons]
        e_sec = 1.0e13 / P0.n_sec
        self.assertLess(float(np.abs(u.mean(axis=0)).max()), 3.0 * P0.pt_ev / e_sec)

    def test_a_pion_decay_conserves_the_energy(self) -> None:
        sh = sm.simulate(P0.with_(e0_ev=1.0e13, e_thin_ev=float("inf")), 5, 5.0, thin=False)
        tr = sh.tr
        mu = np.flatnonzero((tr["kind"] == sm.MUON))
        nu = np.flatnonzero((tr["kind"] == sm.NEUTRINO))
        self.assertEqual(len(mu), len(nu))
        parent_pion = tr["parent"][mu]
        self.assertTrue(np.allclose(tr["e"][mu] + tr["e"][nu], tr["e"][parent_pion], rtol=1e-12))
        f = tr["e"][mu] / tr["e"][parent_pion]
        self.assertGreaterEqual(float(f.min()), P0.muon_fraction_min - 1e-12)
        self.assertAlmostEqual(float(f.mean()), 0.5 * (1.0 + P0.muon_fraction_min), delta=0.02)


class HadronicTest(unittest.TestCase):
    def test_one_third_of_the_pions_are_neutral(self) -> None:
        sh = sm.simulate(P0.with_(e0_ev=1.0e14), 4, 5.0)
        n_pi0 = len(sh.seeds["x"]) / 2
        n_ch = int((sh.tr["kind"] == sm.PION).sum())
        self.assertAlmostEqual(n_pi0 / (n_pi0 + n_ch), 1.0 / 3.0, places=12)

    def test_muon_number_follows_the_model(self) -> None:
        """E_0 = (1.5 n_ch)^m E_dec: N_mu = n_ch^m = (E_0 / E_dec)^beta (every pion decays before the ground)."""
        beta = sm.matthews_beta(P0.n_ch)
        self.assertAlmostEqual(beta, math.log(10) / math.log(15), places=12)
        for m in (3, 4):
            e0 = P0.e_dec_ev * P0.n_sec ** m
            p = P0.with_(e0_ev=e0, x_ground=1.0e6, e_thin_ev=float("inf") if m < 4 else P0.e_thin_ev)
            sh = sm.simulate(p, 1, 20.0, thin=m >= 4)
            self.assertEqual(sh.n_ground((sm.MUON,)), P0.n_ch ** m)
            self.assertAlmostEqual(sm.matthews_n_mu(e0, P0.e_dec_ev, P0.n_ch) / P0.n_ch ** m, 1.0, places=9)
            self.assertEqual(sm.hadron_levels(e0, P0.e_dec_ev, P0.n_ch), m)

    def test_film_muon_number(self) -> None:
        M = asc.get_model()
        self.assertEqual(M.m_levels, 4)
        self.assertAlmostEqual(float(M.n_mu[-1]) / sm.matthews_n_mu(P0.e0_ev, P0.e_dec_ev, P0.n_ch), 1.0, delta=0.15)

    def test_the_electromagnetic_share_of_the_energy(self) -> None:
        M = asc.get_model()
        self.assertAlmostEqual(sm.electromagnetic_fraction(P0), 1.0 - (2.0 / 3.0) ** 4, places=12)
        self.assertAlmostEqual(M.sh.deposited_energy() / P0.e0_ev, sm.electromagnetic_fraction(P0) - M.sh.em_ground_energy() / P0.e0_ev, delta=0.03)

    def test_the_shower_maximum_of_the_film_agrees_with_the_estimate(self) -> None:
        M = asc.get_model()
        self.assertAlmostEqual(M.x_max, M.x_max_h, delta=0.2 * M.x_max_h)
        self.assertTrue(0.1 < M.n_max / M.n_max_h < 0.4, M.n_max / M.n_max_h)

    def test_the_first_interaction_is_typical(self) -> None:
        M = asc.get_model()
        self.assertTrue(0.3 * P0.lam_i < M.x_first < 1.5 * P0.lam_i, M.x_first)

    def test_lateral_distribution_has_the_scale_of_the_moliere_radius(self) -> None:
        M = asc.get_model()
        tr = M.sh.tr
        m = (tr["kind"] == sm.ELECTRON) & tr["ground"].astype(bool)
        r = np.hypot(tr["px1"][m], tr["py1"][m])
        o = np.argsort(r)
        cw = np.cumsum(tr["w"][m][o]) / tr["w"][m].sum()
        r90 = r[o][np.searchsorted(cw, 0.9)]
        rm = sm.moliere_radius_m(P0, P0.x_ground)
        self.assertTrue(0.7 * rm < r90 < 3.0 * rm, (r90, rm))

    def test_nkg_is_normalised(self) -> None:
        r = np.geomspace(1e-3, 1e5, 200000)
        rho = sm.nkg_density(r, 1.0, 1.3, 80.0)
        self.assertAlmostEqual(float(np.trapezoid(rho * 2 * math.pi * r, r)), 1.0, delta=0.02)

    def test_cherenkov_numbers_of_the_film(self) -> None:
        nm1 = asc.CFG.light.n_minus_one_sea
        e = sm.cherenkov_threshold_ev(np.array([nm1]), asc.CFG.light.electron_mass_ev)[0] / 1e6
        th = math.degrees(sm.cherenkov_angle(np.array([nm1]))[0])
        self.assertAlmostEqual(e, 21.1, delta=0.3)
        self.assertAlmostEqual(th, 1.39, delta=0.02)

    def test_altitude_and_depth_are_inverse(self) -> None:
        for h in (0.0, 3.0, 12.0):
            self.assertAlmostEqual(float(sm.altitude_km(P0, sm.depth_of_altitude(P0, h))), h, places=9)


class DrawnSampleTest(unittest.TestCase):
    def test_the_sample_is_a_consistent_forest(self) -> None:
        M = asc.get_model()
        s = M.s
        n = len(s["x0"])
        self.assertTrue(1000 < n < 20000, n)
        self.assertTrue(np.all(s["x1"] >= s["x0"]))
        self.assertTrue(np.all(np.diff(s["x0"]) >= 0))                     # sorted by the start depth
        for k in (sm.PROTON, sm.PION, sm.GAMMA, sm.ELECTRON, sm.MUON, sm.NEUTRINO):
            self.assertGreater(int((s["kind"] == k).sum()), 0)
        self.assertEqual(int((s["kind"] == sm.PROTON).sum()), 1)

    def test_the_tracks_of_the_sample_are_straight_in_the_drift_variable(self) -> None:
        M = asc.get_model()
        s = M.s
        m = np.flatnonzero((s["kind"] == sm.MUON))[:5]
        for i in m:
            x = np.array([0.5 * (s["x0"][i] + s["x1"][i])])
            y = asc.lateral_at(s["x0"][i:i + 1], s["x1"][i:i + 1], s["px0"][i:i + 1], s["px1"][i:i + 1], x)
            frac = math.log(x[0] / s["x0"][i]) / math.log(s["x1"][i] / s["x0"][i])
            self.assertAlmostEqual(float(y[0]), s["px0"][i] + frac * (s["px1"][i] - s["px0"][i]), places=9)


class FilmTest(unittest.TestCase):
    def test_parts_follow_each_other(self) -> None:
        self.assertEqual(asc.T_TREE[1], asc.T_SHOWER[0])
        self.assertEqual(asc.T_SHOWER[1], asc.T_GROUND[0])
        self.assertEqual(asc.T_GROUND[1], asc.T_LIGHT[0])
        self.assertAlmostEqual(asc.T_SHOWER[1] - asc.T_SHOWER[0], sum(asc.CFG.scene2.stage_s), places=6)
        self.assertTrue(60.0 <= asc.TOTAL <= 110.0, asc.TOTAL)

    def test_timeline_inserts_the_cards(self) -> None:
        prev = -1.0
        for tf in np.linspace(0.0, asc.TOTAL - 0.01, 3000):
            tc, card, prog = asc.timeline(float(tf))
            self.assertGreaterEqual(tc, prev - 1e-9)
            prev = tc
            if card is not None:
                self.assertTrue(0.0 <= prog < 1.0)
        for tc in (0.5, 20.0, 70.0, 90.0):
            tc2, card, _ = asc.timeline(asc.film_time(tc))
            self.assertIsNone(card)
            self.assertAlmostEqual(tc2, tc)

    def test_the_front_of_the_shower_is_monotonic_and_visits_the_key_depths(self) -> None:
        M = asc.get_model()
        ts = np.linspace(0.0, asc.T_SHOWER[1] - asc.T_SHOWER[0], 400)
        xs = np.array([asc.front_depth(M, float(t)) for t in ts])
        self.assertTrue(np.all(np.diff(xs) >= -1e-9))
        st = asc.stage_times()
        self.assertAlmostEqual(asc.front_depth(M, float(st[1])), M.x_first)
        self.assertAlmostEqual(asc.front_depth(M, float(st[3])), M.x_max)
        self.assertAlmostEqual(asc.front_depth(M, float(st[5])), P0.x_ground)

    def test_the_tree_of_part_one_is_heitlers(self) -> None:
        T = asc.tree_data()
        n = T["n"]
        self.assertEqual(len([e for e in T["edges"] if e["final"]]), 2 ** n)
        self.assertAlmostEqual(T["x_end"], sm.heitler_x_max(P0.x0, T["e0"], P0.e_c_ev) + P0.x0, places=9)
        n_gamma = sum(1 for e in T["edges"] if e["kind"] == sm.GAMMA)
        n_charged = sum(1 for e in T["edges"] if e["kind"] == sm.ELECTRON)
        self.assertEqual(n_gamma + n_charged, 2 ** (n + 1) - 1)
        self.assertAlmostEqual(n_charged / (n_gamma + n_charged), 2.0 / 3.0, delta=0.03)

    def test_the_ground_view_is_consistent(self) -> None:
        D = asc.ground_data()
        self.assertGreater(int((D["hits"] > 0).sum()), 50)
        self.assertGreater(int((D["mu_hits"] > 0).sum()), 10)
        self.assertGreater(D["rho_e"][0], D["rho_e"][-3])
        self.assertAlmostEqual(D["r_m"], sm.moliere_radius_m(P0, P0.x_ground))
        self.assertLess(D["ratio"][0], D["ratio"][1])                      # the muons fall off more slowly

    def test_every_caption_and_card_has_a_text_in_both_languages(self) -> None:
        for lang in ("en", "ru"):
            tx = asc.TEXT[lang]
            for table in (asc.CFG.tree.captions, asc.CFG.scene2.captions, asc.CFG.ground.captions, asc.CFG.light.captions):
                for _, key in table:
                    self.assertIn(key, tx)
            for key in asc.CARD_KEYS:
                self.assertIn(key, tx)
                self.assertIn(key + "s", tx)
            for pre, _, _ in list(asc.CFG.scene2.rules) + list(asc.CFG.tree.rules):
                for k in ("t", "1", "2", "3", "4"):
                    self.assertIn(f"{pre}_{k}", tx)

    def test_no_text_leaves_the_frame_or_overlaps(self) -> None:
        """Measured text boxes at times covering every part, in both languages."""
        times = [3.0, 9.0, 16.0, 20.5, 25.0, 31.0, 40.0, 48.0, 55.0, 61.0, 66.0, 72.0, 79.0, 84.5, 89.0, 92.5]
        for lang in ("en", "ru"):
            cv = asc.Canvas(1280, 720, lang)
            for tc in times:
                asc.frame_at(cv, tc, lang)
                self.assertEqual(asc.layout_problems(cv), [], (lang, tc))
            cv.plt.close(cv.fig)


class ConfigTest(unittest.TestCase):
    def test_both_languages_have_the_same_keys_and_balanced_dollars(self) -> None:
        texts = tomllib.loads((HERE / "texts.toml").read_text(encoding="utf-8"))
        self.assertEqual(set(texts), {"en", "ru"})
        self.assertEqual(set(texts["en"]), set(texts["ru"]))
        for lang in ("en", "ru"):
            for k, v in texts[lang].items():
                self.assertEqual(v.count("$") % 2, 0, (lang, k))
        for k, v in texts["en"].items():
            self.assertEqual(sorted(__import__("re").findall(r"<(\w+)>", v)), sorted(__import__("re").findall(r"<(\w+)>", texts["ru"][k])), k)

    def test_the_video_and_style_sections_are_complete(self) -> None:
        cfg = tomllib.loads((HERE / "config.toml").read_text(encoding="utf-8"))
        keys = {"width", "height", "fps", "dpi", "crf", "preset", "preview_width", "preview_height", "preview_fps", "fade_s", "reference_height"}
        self.assertTrue(keys <= set(cfg["video"]))
        self.assertEqual(cfg["video"]["width"] * 9, cfg["video"]["height"] * 16)
        self.assertIn("style", cfg)

    def test_the_script_reads_its_numbers_from_the_config(self) -> None:
        src = (HERE / "air_shower_cascade.py").read_text(encoding="utf-8")
        self.assertIn("load_config", src)
        for forbidden in ('"libx264", "-preset", "slow"', "(1280, 720)", "dpi = 100\n"):
            self.assertNotIn(forbidden, src)

    def test_set_overrides_a_value(self) -> None:
        import dvconfig
        old = sys.argv
        sys.argv = ["x", "--set", "physics.n_charged=12"]
        try:
            self.assertEqual(dvconfig.load_config(HERE).physics.n_charged, 12)
        finally:
            sys.argv = old

    def test_snapshots_render_in_both_languages(self) -> None:
        for lang in ("en", "ru"):
            for tc in (8.0, 40.0, 75.0, 90.0):
                with tempfile.TemporaryDirectory() as d:
                    out = Path(d) / "s.png"
                    asc.render(out, (640, 360), 30, asc.TOTAL, lang, snap_content=tc)
                    self.assertGreater(out.stat().st_size, 5000)


if __name__ == "__main__":
    unittest.main()
