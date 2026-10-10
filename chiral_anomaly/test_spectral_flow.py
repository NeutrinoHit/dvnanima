"""Physics tests of the spectral flow: the coefficients of the anomaly are obtained by counting levels, never typed in.

    python -m pytest test_spectral_flow.py
"""

from __future__ import annotations

import math
import unittest

import numpy as np

import spectral_flow as sf

RINGS = [sf.Ring(length=2.0 * math.pi, theta=0.5, e=1.0, field=0.3),
         sf.Ring(length=7.3, theta=0.5, e=0.8, field=0.11),
         sf.Ring(length=3.0, theta=0.27, e=2.0, field=0.5)]


class RingCountingTest(unittest.TestCase):
    def test_number_of_crossed_levels_is_eEL_t_over_2pi_in_each_branch(self) -> None:
        for ring in RINGS:
            for t in np.linspace(0.3, 40.0, 61):
                c = sf.charges(ring, float(t))
                expect = ring.e * ring.field * ring.length * t / (2.0 * math.pi)
                self.assertLessEqual(abs(c["NR"] - expect), 1.0 + 1e-9)
                self.assertLessEqual(abs(c["NL"] + expect), 1.0 + 1e-9)

    def test_the_average_rate_is_exact_for_a_long_time(self) -> None:
        for ring in RINGS:
            t = 4000.0 / ring.level_rate
            c = sf.charges(ring, t, nmax=6000)
            self.assertAlmostEqual(c["NR"] / t / ring.level_rate, 1.0, delta=2e-3)

    def test_crossings_are_equally_spaced_by_the_period(self) -> None:
        for ring in RINGS:
            tc = sf.crossing_times(ring, 60.0)
            self.assertTrue(np.allclose(np.diff(tc), 1.0 / ring.level_rate, rtol=1e-9))

    def test_counter_steps_at_the_crossing_times(self) -> None:
        ring = RINGS[0]
        tc = sf.crossing_times(ring, 20.0)
        for j, t in enumerate(tc):
            self.assertEqual(sf.charges(ring, float(t) - 1e-6)["NR"], j)
            self.assertEqual(sf.charges(ring, float(t) + 1e-6)["NR"], j + 1)

    def test_vector_charge_is_conserved_and_axial_charge_grows_at_eEL_over_pi(self) -> None:
        for ring in RINGS:
            ts = np.linspace(0.0, 50.0, 40)
            q = np.array([sf.charges(ring, float(t))["Q"] for t in ts])
            q5 = np.array([sf.charges(ring, float(t))["Q5"] for t in ts])
            self.assertTrue(np.all(q == 0))
            slope = np.polyfit(ts, q5, 1)[0]
            self.assertAlmostEqual(slope / (ring.e * ring.field * ring.length / math.pi), 1.0, delta=0.04)

    def test_density_form_of_the_1plus1_anomaly(self) -> None:
        """d(Q5/L)/dt = e E / pi: the length drops out."""
        for ring in RINGS:
            t = 3000.0 / ring.level_rate
            c = sf.charges(ring, t, nmax=5000)
            self.assertAlmostEqual(c["Q5"] / ring.length / t / (ring.e * ring.field / math.pi), 1.0, delta=2e-3)

    def test_one_period_returns_the_spectrum_but_not_the_state(self) -> None:
        ring = RINGS[0]
        t = 1.0 / ring.level_rate
        n = sf.labels(60)
        k0, k1 = np.sort(ring.k0(n)), np.sort(ring.k(n, t))
        inner = np.abs(k0) < 20.0
        self.assertTrue(np.allclose(k0[inner], np.sort(ring.k(n, t))[(np.abs(np.sort(ring.k(n, t))) < 20.0)], atol=1e-9))
        c = sf.charges(ring, t)
        self.assertEqual((c["NR"], c["NL"]), (1, -1))            # the same spectrum, one particle and one hole more


class RegularisationTest(unittest.TestCase):
    def test_energy_window_gives_the_same_flow_for_any_cutoff(self) -> None:
        ring = RINGS[0]
        d = ring.spacing
        for t in (3.1, 9.7, 16.2):
            exact = sf.charges(ring, t)
            for lam_lo, lam_hi in [(30 * d, 30 * d), (7 * d, 7 * d), (50 * d, 12 * d), (12 * d, 50 * d), (20 * d, None)]:
                w = sf.window_counts(ring, t, "energy", lam_lo, lam_hi)
                self.assertEqual(w["Q"], 0)
                self.assertEqual(w["R"], exact["NR"])                      # whole numbers of spacings: no rounding difference
                self.assertEqual(w["L"], exact["NL"])
                self.assertEqual(w["Q5"], exact["Q5"])

    def test_energy_window_with_arbitrary_cutoffs_agrees_within_one_level(self) -> None:
        ring = RINGS[0]
        for t in (3.1, 9.7, 16.2):
            exact = sf.charges(ring, t)
            for lam in (23.37, 41.9, 77.1):
                w = sf.window_counts(ring, t, "energy", lam)
                self.assertLessEqual(abs(w["Q"]), 1)                       # discreteness of the window edge: at most one level
                self.assertLessEqual(abs(w["R"] - exact["NR"]), 1)
                self.assertLessEqual(abs(w["L"] - exact["NL"]), 1)

    def test_smooth_regulator_converges_to_the_same_answer(self) -> None:
        ring = RINGS[0]
        t = 11.4
        exact = sf.charges(ring, t)["NR"]
        errs = []
        for lam in (20.0, 40.0, 80.0, 160.0):
            s = sf.smooth_charges(ring, t, lam)
            self.assertAlmostEqual(s["Q"], 0.0, places=9)
            errs.append(abs(s["R"] - ring.level_rate * t))
        self.assertTrue(all(a > b for a, b in zip(errs[:-1], errs[1:])))
        self.assertLess(errs[-1], 5e-3)
        self.assertLess(abs(sf.smooth_charges(ring, t, 160.0)["R"] - exact), 0.55)

    def test_a_cut_in_fixed_labels_keeps_the_axial_charge_but_the_cut_moves(self) -> None:
        ring = RINGS[0]
        for t in (3.1, 9.7, 16.2):
            w = sf.window_counts(ring, t, "label", 30 * ring.spacing)
            self.assertEqual((w["R"], w["L"], w["Q5"]), (0, 0, 0))
            self.assertGreater(sf.charges(ring, t)["NR"], 0)                    # while the levels did cross E = 0

    def test_ground_state_current_depends_on_the_gauge_only_for_the_label_cut(self) -> None:
        ring = RINGS[0]
        d = ring.spacing
        shifts = np.linspace(0.0, 12.0 * d, 49)
        j_energy = np.array([sf.vacuum_current(ring, float(s), "energy", 30.5 * d) for s in shifts])
        j_label = np.array([sf.vacuum_current(ring, float(s), "label", 30.5 * d) for s in shifts])
        self.assertLessEqual(int(np.abs(j_energy).max()), 1)
        self.assertGreater(abs(int(j_label[-1])), 20)
        self.assertAlmostEqual(np.polyfit(shifts / d, j_label, 1)[0], -2.0, delta=0.05)


class LatticeTest(unittest.TestCase):
    def test_lattice_conserves_the_number_of_electrons_and_shows_the_same_flow(self) -> None:
        for n_sites in (24, 40, 64):
            for s in np.linspace(0.05, 6.0, 40):
                c = sf.lattice_counts(n_sites, 0.5, float(s))
                self.assertEqual(c["filled"], n_sites // 2)
                self.assertEqual(c["Q"], 0)
                self.assertEqual(c["NR"], int(math.floor(s + 0.5)))
                self.assertEqual(c["NL"], -c["NR"])

    def test_the_two_fermi_points_are_the_two_branches(self) -> None:
        k = sf.lattice_levels(24, 0.5, 0.0)
        e = np.sin(k)
        slope = np.cos(k)
        self.assertEqual(int(np.sum((e < 0) & (slope > 0))), 6)       # the right branch below E = 0
        self.assertEqual(int(np.sum((e < 0) & (slope < 0))), 6)       # the left branch below E = 0


class LandauTest(unittest.TestCase):
    EB, LX, LY, NX, PZ = 1.0, 24.0, 4.0 * math.pi, 193, 0.37

    @classmethod
    def setUpClass(cls) -> None:
        cls.spec = sf.landau_spectrum(cls.EB, cls.LX, cls.LY, cls.NX, cls.PZ, range(-24, 25))
        cls.spec_l = sf.landau_spectrum(cls.EB, cls.LX, cls.LY, cls.NX, cls.PZ, range(-24, 25), chirality=-1)

    def test_lowest_level_is_one_chiral_branch(self) -> None:
        for s in self.spec:
            inside = np.abs(s["x"]) < 5.9                                                   # states of the interior of the box
            self.assertEqual(int(np.sum((np.abs(s["E"] - self.PZ) < 1e-6) & inside)), 1 if abs(s["centre_expected"]) < 5.9 else 0)
            self.assertEqual(int(np.sum((np.abs(s["E"] + self.PZ) < 1e-3) & inside)), 0)    # nothing at -p_z: the branch is chiral

    def test_left_handed_fermion_has_the_opposite_branch(self) -> None:
        for s in self.spec_l:
            inside = np.abs(s["x"]) < 5.9
            self.assertEqual(int(np.sum((np.abs(s["E"] + self.PZ) < 1e-6) & inside)), 1 if abs(s["centre_expected"]) < 5.9 else 0)
            self.assertEqual(int(np.sum((np.abs(s["E"] - self.PZ) < 1e-3) & inside)), 0)

    def test_higher_levels_come_in_pairs_with_the_known_energies(self) -> None:
        for s in self.spec:
            if abs(s["centre_expected"]) < 5.9:
                known = np.array(sf.landau_energies(self.EB, self.PZ, 4))
                inner = np.sort(s["E"][(np.abs(s["E"]) < 3.0) & (np.abs(s["x"]) < 5.9)])
                self.assertEqual(len(inner), len(known))
                self.assertTrue(np.allclose(inner, known, atol=1e-4))
                for n in (1, 2, 3, 4):
                    en = math.sqrt(self.PZ ** 2 + 2.0 * n * self.EB)
                    self.assertEqual(int(np.sum(np.abs(s["E"] - en) < 1e-4)), 1)
                    self.assertEqual(int(np.sum(np.abs(s["E"] + en) < 1e-4)), 1)

    def test_degeneracy_of_the_lowest_level_per_area_is_eB_over_2pi(self) -> None:
        for x_lo, x_hi in [(-4.0, 3.0), (-5.0, 5.0), (-2.5, 4.5)]:
            n = sf.landau_branch_count(self.spec, self.PZ, x_lo, x_hi)
            area = (x_hi - x_lo) * self.LY
            self.assertEqual(n, int(round(self.EB * area / (2.0 * math.pi))))
            self.assertAlmostEqual(n / area, self.EB / (2.0 * math.pi), delta=1e-9)

    def test_centres_are_p_y_over_eB(self) -> None:
        for s in self.spec:
            if abs(s["centre_expected"]) < 5.9:
                i = int(np.argmin(np.abs(s["E"] - self.PZ)))
                self.assertAlmostEqual(float(s["x"][i]), s["centre_expected"], places=3)

    def test_degeneracy_scales_with_the_field(self) -> None:
        eb = 0.5
        spec = sf.landau_spectrum(eb, 30.0, 8.0 * math.pi, 241, 0.2, range(-40, 41))
        n = sf.landau_branch_count(spec, 0.2, -5.0, 5.0)
        self.assertEqual(n, int(round(eb * 10.0 * 8.0 * math.pi / (2.0 * math.pi))))


class ThreePlusOneTest(unittest.TestCase):
    def test_3plus1_coefficient_by_counting(self) -> None:
        """dn_5/dt = 2 (levels per area) (levels per length and time) = e^2 E.B / (2 pi^2) = (2 alpha / pi) E.B."""
        eb, lx, ly, nx, pz = 1.0, 24.0, 4.0 * math.pi, 193, 0.37
        spec = sf.landau_spectrum(eb, lx, ly, nx, pz, range(-24, 25))
        x_lo, x_hi = -5.0, 5.0
        n_phi = sf.landau_branch_count(spec, pz, x_lo, x_hi)                       # counted orbits in the area (x_hi - x_lo) L_y
        area = (x_hi - x_lo) * ly
        ring = sf.Ring(length=40.0, theta=0.5, e=1.3, field=0.21)
        t = 2500.0 / ring.level_rate
        c = sf.charges(ring, t, nmax=5000)
        crossed = c["NR"]                                                          # right-handed levels of ONE ladder
        n5_rate = (n_phi * crossed - n_phi * c["NL"]) / (area * ring.length) / t   # (R particles + L holes) per volume and time
        e_field, e_charge = ring.field, ring.e
        b_field = eb / e_charge
        expect = e_charge ** 2 * e_field * b_field / (2.0 * math.pi ** 2)
        self.assertAlmostEqual(n5_rate / expect, 1.0, delta=3e-3)
        alpha = e_charge ** 2 / (4.0 * math.pi)
        self.assertAlmostEqual(n5_rate / ((2.0 * alpha / math.pi) * e_field * b_field), 1.0, delta=3e-3)

    def test_the_1plus1_coefficient_times_the_degeneracy_gives_3plus1(self) -> None:
        e, eb = 1.0, 0.7
        self.assertAlmostEqual((e / math.pi) * (eb / (2.0 * math.pi)), e * eb * e / (2.0 * math.pi ** 2), places=12)

    def test_isospin_current_weight_is_one_half(self) -> None:
        w = sf.isospin_anomaly_weight([2.0 / 3.0, -1.0 / 3.0], [0.5, -0.5], 3)
        self.assertAlmostEqual(w, 0.5, places=12)
        self.assertAlmostEqual(w * 2.0 / math.pi, 1.0 / math.pi, places=12)         # (alpha / pi) E.B for pi0 after the factor alpha


class PionTest(unittest.TestCase):
    ALPHA, MPI, FPI, HBAR = 1.0 / 137.035999, 134.9768e6, 92.2e6, 6.582119569e-16          # eV, eV, eV, eV s

    def test_width_matches_the_book(self) -> None:
        self.assertAlmostEqual(sf.pi0_width(self.ALPHA, self.MPI, self.FPI), 7.76, delta=0.02)

    def test_width_follows_from_the_amplitude(self) -> None:
        a = sf.pi0_amplitude(self.ALPHA, self.FPI)
        pk = self.MPI ** 2 / 2.0
        gamma = (1.0 / (32.0 * math.pi * self.MPI)) * a * a * 2.0 * pk * pk
        self.assertAlmostEqual(gamma / sf.pi0_width(self.ALPHA, self.MPI, self.FPI), 1.0, places=12)

    def test_measured_width_from_lifetime_and_branching(self) -> None:
        g = 0.98823 * self.HBAR / 8.43e-17
        self.assertAlmostEqual(g, 7.72, delta=0.05)

    def test_without_the_anomaly_the_width_is_about_a_thousand_times_smaller(self) -> None:
        g_meas = self.HBAR / 8.43e-17
        g_old = self.HBAR / 1e-13
        self.assertGreater(g_meas / g_old, 800.0)
        self.assertLess(g_meas / g_old, 1500.0)


if __name__ == "__main__":
    unittest.main()
