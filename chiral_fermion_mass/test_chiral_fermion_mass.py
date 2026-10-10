"""Tests of the physics (dirac1d.py) and of the film (chiral_fermion_mass.py).

    python -m unittest test_chiral_fermion_mass
"""

from __future__ import annotations

import math
import re
import sys
import tomllib
import unittest
from pathlib import Path

import numpy as np

import chiral_fermion_mass as cf
import dirac1d as D

HERE = Path(__file__).resolve().parent
G = D.Grid(64.0, 1024)


def mass_const(m: float):
    return lambda t: m


def mean_velocity(p0: float, sigma: float, m: float) -> float:
    """<p/E> over the momentum distribution of the Gaussian packet: the exact velocity of its centre (v(p0) plus a small curvature term)."""
    g = np.abs(np.fft.fft(D.gaussian(G, 0.0, sigma, p0))) ** 2
    return float(np.sum(g * D.velocity_ratio(G.p, m)) / np.sum(g))


def run_packet(psi0: np.ndarray, mass, times: np.ndarray) -> list[np.ndarray]:
    return [D.to_real(k) for k in D.evolve(D.to_fourier(psi0), G.p, mass, times, 0.004)]


class HamiltonianTest(unittest.TestCase):
    def test_eigenvalues_are_plus_minus_sqrt_p2_m2(self) -> None:
        p = np.linspace(-6, 6, 25)
        for m in (0.0, 1.0, 3.0):
            ev = np.linalg.eigvalsh(D.hamiltonian(p, m))
            e = np.sqrt(p ** 2 + m ** 2)
            self.assertTrue(np.allclose(ev[:, 0], -e, atol=1e-12))
            self.assertTrue(np.allclose(ev[:, 1], e, atol=1e-12))

    def test_h_squared_is_e_squared(self) -> None:
        p = np.linspace(-5, 5, 11)
        h = D.hamiltonian(p, 2.0)
        for k in range(len(p)):
            self.assertTrue(np.allclose(h[k] @ h[k], (p[k] ** 2 + 4.0) * np.eye(2), atol=1e-12))

    def test_eigenvectors_satisfy_the_eigenvalue_equation_and_are_orthonormal(self) -> None:
        p = np.linspace(-6, 6, 41)
        for m in (0.5, 3.0):
            h = D.hamiltonian(p, m)
            e = np.sqrt(p ** 2 + m ** 2)
            up, um = D.eigenvector(p, m, +1), D.eigenvector(p, m, -1)
            for k in range(len(p)):
                self.assertTrue(np.allclose(h[k] @ up[:, k], e[k] * up[:, k], atol=1e-12))
                self.assertTrue(np.allclose(h[k] @ um[:, k], -e[k] * um[:, k], atol=1e-12))
                self.assertAlmostEqual(float(up[:, k] @ um[:, k]), 0.0, places=12)
                self.assertAlmostEqual(float(up[:, k] @ up[:, k]), 1.0, places=12)

    def test_the_chirality_of_an_eigenvector_is_p_over_e(self) -> None:
        """<gamma_5> = |u_R|^2 - |u_L|^2 = +-p/E on the upper/lower branch: chirality is the velocity in 1+1."""
        p = np.linspace(-6, 6, 41)
        m = 2.0
        e = np.sqrt(p ** 2 + m ** 2)
        for sign in (+1, -1):
            u = D.eigenvector(p, m, sign)
            self.assertTrue(np.allclose(u[1] ** 2 - u[0] ** 2, sign * p / e, atol=1e-12))

    def test_the_massless_eigenvectors_are_the_chiral_waves(self) -> None:
        u = D.eigenvector(np.array([-2.0, 2.0]), 0.0, +1)
        self.assertTrue(np.allclose(u[:, 0], [1, 0]))              # p < 0, E > 0: left chirality
        self.assertTrue(np.allclose(u[:, 1], [0, 1]))              # p > 0, E > 0: right chirality

    def test_the_ratio_of_the_components_of_the_massive_packet(self) -> None:
        p, m = 3.0, 3.0
        u = D.eigenvector(np.array([p]), m, +1)[:, 0]
        e = math.hypot(p, m)
        self.assertAlmostEqual(float(u[0] / u[1]), m / (e + p), places=12)       # psi_L / psi_R = m / (E + p)


class EvolutionTest(unittest.TestCase):
    def test_norm_is_conserved_for_a_constant_mass(self) -> None:
        psi0 = D.chiral_packet(G, -5.0, 1.6, 3.0, "R")
        for psi in run_packet(psi0, mass_const(3.0), np.array([0.0, 5.0, 20.0])):
            self.assertAlmostEqual(D.norm(psi, G.dx), 1.0, places=11)

    def test_norm_is_conserved_for_a_time_dependent_mass(self) -> None:
        psi0 = D.chiral_packet(G, -5.0, 1.6, 3.0, "R")
        mass = lambda t: 3.0 * D.smoothstep(t, 1.0, 1.3)
        for psi in run_packet(psi0, mass, np.array([0.5, 1.15, 1.5, 6.0])):
            self.assertAlmostEqual(D.norm(psi, G.dx), 1.0, places=11)

    def test_a_massless_right_wave_moves_with_c_and_keeps_its_shape_and_chirality(self) -> None:
        psi0 = D.chiral_packet(G, -5.0, 1.6, 3.0, "R")
        for t, psi in zip((3.0, 9.0), run_packet(psi0, mass_const(0.0), np.array([3.0, 9.0]))):
            self.assertAlmostEqual(D.centroid(psi, G.x, G.dx), -5.0 + t, places=6)
            pr, pl = D.chirality_weights(psi, G.dx)
            self.assertAlmostEqual(pr, 1.0, places=12)
            self.assertAlmostEqual(pl, 0.0, places=12)
            shifted = D.chiral_packet(G, -5.0 + t, 1.6, 3.0, "R")
            self.assertTrue(np.allclose(np.abs(psi[1]), np.abs(shifted[1]), atol=1e-8))      # the envelope is unchanged

    def test_a_massless_left_wave_moves_to_the_left(self) -> None:
        psi0 = D.chiral_packet(G, 6.0, 1.6, -3.0, "L")
        psi = run_packet(psi0, mass_const(0.0), np.array([8.0]))[0]
        self.assertAlmostEqual(D.centroid(psi, G.x, G.dx), 6.0 - 8.0, places=6)
        pr, pl = D.chirality_weights(psi, G.dx)
        self.assertAlmostEqual(pl, 1.0, places=12)

    def test_massless_limit_of_the_chirality(self) -> None:
        """<gamma_5> of a pure psi_R packet stays 1 for m = 0 and falls with m as m^2/E^2 (the amplitude of the oscillation)."""
        psi0 = D.chiral_packet(G, 0.0, 1.6, 3.0, "R")
        times = np.linspace(0.0, 2.0, 41)
        g_small = [D.chirality_weights(p, G.dx) for p in run_packet(psi0, mass_const(0.1), times)]
        lowest = min(a - b for a, b in g_small)
        self.assertGreater(lowest, 1.0 - 2.0 * (0.1 ** 2 / (9.0 + 0.01)) - 1e-3)
        self.assertLess(lowest, 1.0)

    def test_the_group_velocity_is_p_over_e(self) -> None:
        for m in (1.0, 3.0, 6.0):
            psi0 = D.eigen_packet(G, -5.0, 1.6, 3.0, m, +1)
            x0 = D.centroid(psi0, G.x, G.dx)
            psi = run_packet(psi0, mass_const(m), np.array([10.0]))[0]
            v = (D.centroid(psi, G.x, G.dx) - x0) / 10.0
            self.assertAlmostEqual(v, mean_velocity(3.0, 1.6, m), delta=3e-4)            # exactly the mean of p/E over the packet
            self.assertAlmostEqual(v, D.group_velocity(3.0, m), delta=6e-3)             # = p0/E0 up to the curvature of v(p) (about 0.3 %)

    def test_an_eigen_packet_has_constant_chirality_equal_to_the_velocity(self) -> None:
        m = 3.0
        psi0 = D.eigen_packet(G, -5.0, 1.6, 3.0, m, +1)
        for psi in run_packet(psi0, mass_const(m), np.array([0.0, 4.0, 11.0])):
            pr, pl = D.chirality_weights(psi, G.dx)
            self.assertAlmostEqual(pr - pl, mean_velocity(3.0, 1.6, m), delta=3e-4)
            self.assertAlmostEqual(pr - pl, D.group_velocity(3.0, m), delta=6e-3)

    def test_the_chirality_is_the_velocity_operator(self) -> None:
        """d<x>/dt = <gamma_5> at every time, also while the mass is switched on."""
        psi0 = D.chiral_packet(G, -5.0, 1.6, 3.0, "R")
        mass = lambda t: 3.0 * D.smoothstep(t, 1.0, 1.3)
        h = 0.01
        times = np.arange(0.0, 5.0, h)
        states = run_packet(psi0, mass, times)
        xc = np.array([D.centroid(s, G.x, G.dx) for s in states])
        g = np.array([np.subtract(*D.chirality_weights(s, G.dx)) for s in states])
        v = (xc[2:] - xc[:-2]) / (2 * h)
        self.assertTrue(np.allclose(v, g[1:-1], atol=2e-3))

    def test_a_plane_wave_flips_with_the_frequency_two_e(self) -> None:
        """A single Fourier mode that is pure psi_R at t = 0: <gamma_5>(t) = p^2/E^2 + (m^2/E^2) cos(2 E t) and its spectrum peaks at 2E."""
        p, m = np.array([3.0]), 3.0
        e = math.hypot(3.0, m)
        t = np.arange(0, 400) * 0.01
        psi0 = np.array([[0.0], [1.0 + 0j]])
        out = D.evolve(psi0, p, mass_const(m), t, 0.005)
        g = np.abs(out[:, 1, 0]) ** 2 - np.abs(out[:, 0, 0]) ** 2
        self.assertTrue(np.allclose(g, D.mean_chirality_plane_wave(3.0, m, t), atol=1e-12))
        spec = np.abs(np.fft.rfft(g - g.mean()))
        freq = 2 * math.pi * np.fft.rfftfreq(len(t), 0.01)
        self.assertAlmostEqual(float(freq[int(np.argmax(spec))]), 2 * e, delta=2 * math.pi / (len(t) * 0.01))

    def test_at_rest_the_two_chiralities_flip_completely(self) -> None:
        """p = 0: P_R = cos^2(m t), P_L = sin^2(m t), the flip frequency is 2 m."""
        m = 2.5
        t = np.linspace(0, 5, 51)
        out = D.evolve(np.array([[0.0], [1.0 + 0j]]), np.array([0.0]), mass_const(m), t, 0.005)
        self.assertTrue(np.allclose(np.abs(out[:, 1, 0]) ** 2, np.cos(m * t) ** 2, atol=1e-12))
        self.assertTrue(np.allclose(np.abs(out[:, 0, 0]) ** 2, np.sin(m * t) ** 2, atol=1e-12))

    def test_a_plane_wave_eigenstate_only_gains_the_phase_exp_minus_i_e_t(self) -> None:
        """The dispersion relation E^2 = p^2 + m^2: the phase of an eigenvector grows as E t."""
        p, m = np.array([2.0]), 1.5
        e = math.hypot(2.0, m)
        for sign in (+1, -1):
            u = D.eigenvector(p, m, sign)
            out = D.evolve(u.astype(complex), p, mass_const(m), np.array([0.7]), 0.005)[0]
            self.assertTrue(np.allclose(out, np.exp(-1j * sign * e * 0.7) * u, atol=1e-12))

    def test_the_branch_weights_of_a_suddenly_switched_on_mass(self) -> None:
        """A pure psi_R plane wave with a sudden mass: the weights of the upper and lower branches are (1 +- p/E) / 2."""
        p, m = 3.0, 3.0
        w_up, w_dn = D.branch_weights(np.array([[0.0], [1.0 + 0j]]), np.array([p]), m, 1.0)
        e = math.hypot(p, m)
        self.assertAlmostEqual(float(w_up[0]), 0.5 * (1 + p / e), places=12)
        self.assertAlmostEqual(float(w_dn[0]), 0.5 * (1 - p / e), places=12)

    def test_the_two_parts_of_a_suddenly_massive_wave_run_in_opposite_directions(self) -> None:
        """The upper-branch part moves with +p/E and the lower-branch part with -p/E (the zig-zag that averages to p^2/E^2)."""
        m, p0 = 3.0, 3.0
        psi0 = D.chiral_packet(G, 0.0, 1.6, p0, "R")
        t = 8.0
        out = D.evolve(D.to_fourier(psi0), G.p, mass_const(m), np.array([t]), 0.004)[0]
        xp, wp = D.branch_centroid(out, G.p, m, G.x, G.dx, +1)
        xm, wm = D.branch_centroid(out, G.p, m, G.x, G.dx, -1)
        v = p0 / math.hypot(p0, m)
        self.assertAlmostEqual(xp / t, v, delta=0.02)
        self.assertAlmostEqual(xm / t, -v, delta=0.02)
        self.assertAlmostEqual(wp + wm, 1.0, places=9)
        self.assertAlmostEqual(wp, 0.5 * (1 + v), delta=0.01)

    def test_the_centre_of_a_suddenly_massive_wave_moves_with_p_squared_over_e_squared(self) -> None:
        m, p0 = 3.0, 3.0
        psi0 = D.chiral_packet(G, 0.0, 1.6, p0, "R")
        psi = run_packet(psi0, mass_const(m), np.array([12.0]))[0]
        self.assertAlmostEqual(D.centroid(psi, G.x, G.dx) / 12.0, p0 ** 2 / (p0 ** 2 + m ** 2), delta=0.01)


class FilmTest(unittest.TestCase):
    def test_parts_follow_each_other(self) -> None:
        self.assertEqual(cf.PARTS[0][0], 0.0)
        for a, b in zip(cf.PARTS[:-1], cf.PARTS[1:]):
            self.assertEqual(a[1], b[0])
        self.assertEqual(cf.PARTS[-1][1], cf.CONTENT_TOTAL)
        self.assertEqual(len(cf.PARTS), 5)

    def test_timeline_inserts_the_cards(self) -> None:
        self.assertAlmostEqual(cf.TOTAL, cf.CONTENT_TOTAL + cf.CARD_S * len(cf.CARD_AT))
        prev = -1.0
        for tf in np.linspace(0.0, cf.TOTAL - 0.01, 3000):
            tc, card, prog = cf.timeline(float(tf))
            self.assertGreaterEqual(tc, prev - 1e-9)
            prev = tc
            if card is not None:
                self.assertTrue(0.0 <= prog < 1.0)
        self.assertEqual(cf.timeline(0.5)[1], 0)
        self.assertEqual(cf.timeline(cf.CARD_S + 0.5)[1], 1)
        self.assertIsNone(cf.timeline(2 * cf.CARD_S + 0.5)[1])

    def test_film_time_is_the_inverse_of_the_timeline(self) -> None:
        for tc in (0.5, 14.0, 31.0, 50.0, 70.0, 79.0):
            tc2, card, _ = cf.timeline(cf.film_time(tc))
            self.assertIsNone(card)
            self.assertAlmostEqual(tc2, tc)

    def test_the_slow_motion_is_exactly_the_ramp_of_the_mass(self) -> None:
        P = cf.CFG.part2
        a, b = P.slow_label
        t0, t1 = cf.phys_time(P.segments, a), cf.phys_time(P.segments, b)
        self.assertAlmostEqual(t0, P.ramp_start, places=9)
        self.assertAlmostEqual(t1 - t0, P.ramp_duration, places=9)

    def test_the_condensate_dial_follows_the_mass(self) -> None:
        P = cf.CFG.part2
        self.assertAlmostEqual(cf.state2(0.5)["dial"]["m"], 0.0, places=12)
        end = cf.state2(P.captions[-1][0] + 1.0)["dial"]
        self.assertAlmostEqual(end["m"], cf.mass_of(P.f), places=9)
        self.assertAlmostEqual(end["v"], cf.HG.v_max, places=9)
        mids = [cf.state2(u)["dial"]["m"] for u in np.linspace(P.slow_label[0], P.slow_label[1], 9)]
        self.assertTrue(all(b >= a - 1e-12 for a, b in zip(mids[:-1], mids[1:])))             # the dial turns monotonically

    def test_m_is_f_v_over_root_two(self) -> None:
        self.assertAlmostEqual(cf.mass_of(1.0), 3.0, places=9)
        self.assertAlmostEqual(cf.mass_of(2.0), 2 * cf.mass_of(1.0), places=12)

    def test_the_runs_of_part_3_fill_the_part_and_get_heavier(self) -> None:
        runs = cf.CFG.part3.runs
        self.assertEqual(runs[0]["from"], 0.0)
        self.assertEqual(runs[-1]["to"], cf.PARTS[2][1] - cf.PARTS[2][0])
        for a, b in zip(runs[:-1], runs[1:]):
            self.assertEqual(a["to"], b["from"])
            self.assertLess(a["f"], b["f"])

    def test_the_flip_frequency_in_part_3_is_two_m(self) -> None:
        for k, run in enumerate(cf.CFG.part3.runs):
            S = cf.sim3(k)
            m = cf.mass_of(run["f"])
            w = np.abs(np.fft.fft(D.gaussian(G, 0.0, cf.PK.rest_sigma, 0.0))) ** 2
            w = w / w.sum()
            exact = sum(wk * D.mean_chirality_plane_wave(pk, m, S["t"]) for wk, pk in zip(w, G.p) if wk > 1e-14)      # the plane-wave result averaged over the packet
            self.assertTrue(np.allclose(S["g"], exact, atol=1e-6))
            self.assertTrue(np.allclose(S["g"], np.cos(2 * m * S["t"]), atol=0.12))        # = cos(2 m t), the momentum spread only dephases it a little

    def test_the_tracks_of_part_4_run_with_p_over_e(self) -> None:
        P = cf.CFG.part4
        for k, f in enumerate(P.f_values):
            S = cf.sim4(k)
            m = cf.mass_of(f)
            v = (S["xc"][-1] - S["xc"][0]) / (S["t"][-1] - S["t"][0])
            exact = mean_velocity(cf.PK.carrier_p, cf.PK.sigma, m)
            self.assertAlmostEqual(v, exact, delta=3e-4)
            self.assertAlmostEqual(v, D.group_velocity(cf.PK.carrier_p, m), delta=1e-2)
            self.assertAlmostEqual(float(S["g"][-1]), exact, delta=3e-4)                                   # <gamma_5> = v
        slower = [(cf.sim4(k)["xc"][-1]) for k in range(len(P.f_values))]
        self.assertTrue(all(a > b for a, b in zip(slower[:-1], slower[1:])))                  # heavier, slower

    def test_part_2_oscillation_decays_to_its_mean(self) -> None:
        S = cf.sim2()
        end = np.arange(len(S["t"]))[S["t"] > S["t"][-1] - 2.0]
        self.assertAlmostEqual(float(S["g"][end].mean()), S["mean_g"], delta=0.02)
        i0 = int(np.argmax(S["t"] > cf.CFG.part2.ramp_start + cf.CFG.part2.ramp_duration))
        self.assertGreater(float(S["g"][i0:i0 + 40].max() - S["g"][i0:i0 + 40].min()), 0.3)       # oscillates right after the switch-on
        self.assertLess(S["mean_g"], D.group_velocity(cf.PK.carrier_p, S["m_end"]))             # the centre is slower than the main packet

    def test_part_1_chirality_is_constant(self) -> None:
        S = cf.sim1()
        self.assertTrue(np.allclose(S["gR"], 1.0, atol=1e-9))
        self.assertTrue(np.allclose(S["gL"], -1.0, atol=1e-9))

    def test_state_is_defined_everywhere(self) -> None:
        for t in np.linspace(0.0, cf.CONTENT_TOTAL - 0.01, 61):
            st = cf.state(float(t))
            self.assertIn(st["cap"], cf.TEXT["en"])
            for key, *_ in st["formulas"]:
                self.assertIn(key, cf.TEXT["en"])
            if st.get("lanes"):
                self.assertTrue(np.all(np.isfinite(st["lanes"]["psiR"])))
                self.assertEqual(st["lanes"]["psiR"].shape, cf.XD.shape)

    def test_every_card_has_a_title_in_both_languages(self) -> None:
        for lang in ("en", "ru"):
            for key in cf.CARD_KEYS:
                self.assertIn(key, cf.TEXT[lang])
                self.assertIn(key + "s", cf.TEXT[lang]) if key != "h0" else self.assertIn("h0s", cf.TEXT[lang])


class ConfigTest(unittest.TestCase):
    def test_texts_have_the_same_keys_and_placeholders_in_both_languages_and_balanced_formulas(self) -> None:
        texts = tomllib.loads((HERE / "texts.toml").read_text(encoding="utf-8"))
        self.assertEqual(set(texts), {"en", "ru"})
        self.assertEqual(set(texts["en"]), set(texts["ru"]))
        for key in texts["en"]:
            self.assertEqual(sorted(re.findall(r"@(\w+)@", texts["en"][key])), sorted(re.findall(r"@(\w+)@", texts["ru"][key])), key)
        for lang, table in texts.items():
            for key, value in table.items():
                self.assertEqual(value.count("$") % 2, 0, f"{lang}.{key}: unbalanced $")

    def test_no_russian_inside_math(self) -> None:
        texts = tomllib.loads((HERE / "texts.toml").read_text(encoding="utf-8"))
        for key, value in texts["ru"].items():
            for part in value.split("$")[1::2]:
                self.assertIsNone(re.search(r"[а-яА-ЯёЁ]", part), f"ru.{key}: a Russian word inside a formula")

    def test_video_section_is_complete(self) -> None:
        v = tomllib.loads((HERE / "config.toml").read_text(encoding="utf-8"))["video"]
        keys = {"width", "height", "fps", "dpi", "crf", "preset", "preview_width", "preview_height", "preview_fps", "fade_s", "reference_height"}
        self.assertTrue(keys <= set(v))
        self.assertEqual(v["width"] * 9, v["height"] * 16)
        self.assertGreater(v["width"], v["preview_width"])
        self.assertIn("style", tomllib.loads((HERE / "config.toml").read_text(encoding="utf-8")))

    def test_the_script_has_no_hard_coded_video_settings(self) -> None:
        src = (HERE / "chiral_fermion_mass.py").read_text(encoding="utf-8")
        self.assertIn("load_config", src)
        for forbidden in ('"libx264", "-preset", "slow"', "(1280, 720)", "dpi = 100\n"):
            self.assertNotIn(forbidden, src)

    def test_set_overrides_a_value(self) -> None:
        sys.path.insert(0, str(HERE.parent))
        import dvconfig
        old = sys.argv
        try:
            sys.argv = ["film.py", "--set", "stage.view_half=11.5", "--set", "style.background='#000000'"]
            c = dvconfig.load_config(HERE)
            self.assertEqual(c.stage.view_half, 11.5)
            self.assertEqual(c.style.background, "#000000")
            sys.argv = ["film.py", "--set", "stage.view_hlaf=3"]
            with self.assertRaises(KeyError):
                dvconfig.load_config(HERE)
        finally:
            sys.argv = old


def boxes_overlap(a, b, tol: float = 2.0) -> bool:
    return min(a[2], b[2]) - max(a[0], b[0]) > tol and min(a[3], b[3]) - max(a[1], b[1]) > tol


class PictureTest(unittest.TestCase):
    TIMES = [0.5, 3.0, 6.0, 12.0, 14.0, 16.5, 19.0, 22.0, 26.0, 29.0, 32.0, 35.0, 40.0, 46.0, 50.0, 56.0, 62.0, 66.0, 70.0, 73.0, 76.0, 79.0]

    def painters(self):
        for lang in ("en", "ru"):
            yield lang, cf.Painter(lang, (cf.CFG.video.preview_width * 2, cf.CFG.video.preview_height * 2))

    def test_snapshots_render_in_both_languages_and_nothing_overlaps_or_leaves_the_frame(self) -> None:
        for lang, pt in self.painters():
            W, H = pt.W, pt.H
            for tc in self.TIMES:
                frame = pt.draw(cf.film_time(tc), cf.TOTAL)
                self.assertEqual(frame.shape, (H, W, 4))
                self.assertGreater(int(frame[..., :3].max()), 100)                    # something is drawn
                boxes = pt.text_boxes()
                for s, b in boxes:
                    self.assertGreaterEqual(b[0], -1, f"{lang} t={tc}: '{s}' leaves the frame on the left")
                    self.assertGreaterEqual(b[1], -1, f"{lang} t={tc}: '{s}' leaves the frame at the bottom")
                    self.assertLessEqual(b[2], W + 1, f"{lang} t={tc}: '{s}' leaves the frame on the right ({b[2]:.0f} > {W})")
                    self.assertLessEqual(b[3], H + 1, f"{lang} t={tc}: '{s}' leaves the frame at the top")
                for i in range(len(boxes)):
                    for j in range(i + 1, len(boxes)):
                        self.assertFalse(boxes_overlap(boxes[i][1], boxes[j][1]), f"{lang} t={tc}: '{boxes[i][0]}' overlaps '{boxes[j][0]}'")

    def test_russian_numbers_have_a_decimal_comma_and_english_a_point(self) -> None:
        for lang, pt in self.painters():
            for tc in (20.0, 36.0, 56.0, 77.0):
                pt.draw(cf.film_time(tc), cf.TOTAL)
                for s, _ in pt.text_boxes():
                    if lang == "ru":
                        self.assertIsNone(re.search(r"\d\.\d", s), f"ru: '{s}' has a decimal point")
                    else:
                        self.assertNotIn("{,}", s)

    def test_every_caption_fits_the_frame(self) -> None:
        for lang, pt in self.painters():
            r = pt.fig.canvas.get_renderer()
            for part in range(1, 6):
                P = cf.PART_CFG[part]
                for _, key in P.captions:
                    t = pt.fig.text(*cf.CFG.layout.caption_pos, pt.T(key), fontsize=cf.CFG.fonts.caption * pt.sc)
                    bb = t.get_window_extent(r)
                    t.remove()
                    self.assertLess(bb.x1, 0.97 * pt.W, f"{lang} caption {key} is too wide")


if __name__ == "__main__":
    unittest.main()
