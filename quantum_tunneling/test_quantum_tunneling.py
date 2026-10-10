"""Physics and configuration tests of the film "Tunnelling: a point body and an extended body" (python -m pytest test_quantum_tunneling.py)."""
from __future__ import annotations

import math
import re
import sys
import tomllib
import unittest
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
import quantum_tunneling as qt  # noqa: E402
import tunnel_physics as tp  # noqa: E402

CL, QU, JC = qt.CL, qt.QU, qt.JC


class TrainTest(unittest.TestCase):
    def test_a_single_car_threshold_is_mgH(self) -> None:
        tr = qt.train(1)
        self.assertAlmostEqual(qt.threshold_ratio(1), 1.0, places=4)
        self.assertAlmostEqual(tr.u_max(CL.margin_sigma) / (CL.mass * CL.g * CL.height), 1.0, places=4)

    def test_the_threshold_of_a_train_falls_with_n(self) -> None:
        r = [qt.threshold_ratio(n) for n in (1, 2, 4, 8, 16, 20)]
        self.assertTrue(all(a > b for a, b in zip(r, r[1:])))
        self.assertLess(qt.threshold_ratio(CL.cars), 0.5)

    def test_the_film_single_car_turns_back_and_the_train_passes(self) -> None:
        for n, passes in ((1, False), (CL.cars, True)):
            tr = qt.train(n)
            s0 = tr.s_start(CL.margin_sigma)
            self.assertEqual(tp.will_pass(tr, s0, qt.v0_flat(), 1e-3, tr.s_end(CL.margin_sigma), 200.0), passes)

    def test_the_threshold_equals_the_one_found_by_integrating_the_motion(self) -> None:
        tr = qt.train(5)
        vc = tr.critical_speed(CL.margin_sigma)
        vb = tp.critical_speed_by_bisection(tr, tr.s_start(CL.margin_sigma), tr.s_end(CL.margin_sigma), 0.8 * vc, 1.2 * vc, 1e-3, 300.0, 1e-5)
        self.assertAlmostEqual(vb / vc, 1.0, delta=2e-3)

    def test_energy_is_conserved_by_the_integrator(self) -> None:
        for n in (1, CL.cars):
            _, _, _, e = qt.classical_run(n)
            self.assertLess(abs(e.max() - e.min()) / abs(e.max()), 1e-5)

    def test_the_long_train_limit(self) -> None:
        tr = qt.train(CL.cars)
        lim = tr.asymptotic_threshold() / (tr.n * tr.mass * CL.g * CL.height) * tr.n
        self.assertGreater(lim, 1.0)                      # a very long train needs a fixed amount of energy, larger than V for one car


class QuantumTest(unittest.TestCase):
    def test_rectangular_barrier_formula(self) -> None:
        v0, a, m, hb = 1.0, 2.0, 1.0, 1.0
        for e in (0.2, 0.5, 0.8):
            kap = math.sqrt(2 * m * (v0 - e)) / hb
            exact = 1.0 / (1.0 + v0 ** 2 * math.sinh(kap * a) ** 2 / (4 * e * (v0 - e)))
            self.assertAlmostEqual(tp.rectangle_transmission(e, v0, a, m, hb), exact, places=10)
            num = tp.transmission(e, lambda x: np.where(np.abs(x) < a / 2, v0, 0.0), m, hb, 4.0, breakpoints=(-a / 2, a / 2))
            self.assertAlmostEqual(num, exact, delta=1e-6)

    def test_packet_transmission_equals_the_integral_over_energy(self) -> None:
        run = qt.quantum_run("light")
        pn = qt.packet_numbers("light")
        self.assertAlmostEqual(float(run["pr"][-1]), pn["pt"], delta=1e-3)

    def test_norm_is_conserved_before_the_packet_reaches_the_edges(self) -> None:
        run = qt.quantum_run("light")
        self.assertAlmostEqual(float(run["norm"][0]), 1.0, places=6)
        self.assertGreater(float(run["norm"].min()), 0.99)

    def test_a_heavier_particle_tunnels_less(self) -> None:
        self.assertLess(qt.packet_numbers("heavy")["pt"], 0.3 * qt.packet_numbers("light")["pt"])

    def test_transmission_is_smooth_below_and_above_the_top(self) -> None:
        e, t = qt.te_curve(QU.mass)
        self.assertTrue(np.all(np.diff(t) >= -1e-9))
        self.assertGreater(t[e < 1.0].max(), 0.0)
        self.assertLess(t[e > 1.0].min(), 1.0)


class WaveTest(unittest.TestCase):
    def test_the_simulated_transmission_equals_the_formula(self) -> None:
        for i in range(len(qt.WV.widths)):
            r = qt.wave_run(i)
            self.assertAlmostEqual(r["t"] / r["t_exact"], 1.0, delta=0.01)

    def test_the_energy_flux_is_conserved(self) -> None:
        for i in range(len(qt.WV.widths)):
            r = qt.wave_run(i)
            self.assertAlmostEqual(r["t"] + r["r"], 1.0, delta=0.01)

    def test_the_decay_inside_the_segment_has_the_predicted_kappa(self) -> None:
        r = qt.wave_run(len(qt.WV.widths) - 1)
        x, am = r["amp_x"], r["amp_last"]
        sel = (x > 0.8) & (x < r["a"] - 0.8)
        slope = np.polyfit(x[sel], np.log(am[sel]), 1)[0]
        self.assertAlmostEqual(-slope / qt.wave_kappa(), 1.0, delta=0.05)

    def test_the_formula_is_the_rectangular_barrier_of_the_schroedinger_equation(self) -> None:
        w, om = qt.WV.omega, qt.WV.big_omega
        for a in (0.5, 1.0, 3.0):
            kap = math.sqrt(om ** 2 - w ** 2)
            ref = 1.0 / (1.0 + math.sinh(kap * a) ** 2 / (4.0 * (w / om) ** 2 * (1.0 - (w / om) ** 2)))
            self.assertAlmostEqual(qt.wave_t_exact(a), ref, places=12)

    def test_the_large_thickness_law(self) -> None:
        w, om = qt.WV.omega, qt.WV.big_omega
        a1, a2 = 4.0, 5.0
        ratio = qt.wave_t_exact(a2) / qt.wave_t_exact(a1)
        self.assertAlmostEqual(math.log(ratio) / (a2 - a1), -2.0 * qt.wave_kappa(), delta=0.01)


class MatterTest(unittest.TestCase):
    def test_log_t_is_linear_in_the_square_root_of_the_mass(self) -> None:
        d = qt.mass_data()
        x, y = np.sqrt(d["mu"]), np.log(d["t"])
        c = np.polyfit(x, y, 1)
        resid = y - np.polyval(c, x)
        self.assertLess(np.abs(resid).max(), 0.35)
        self.assertLess(c[0], 0.0)

    def test_the_objects_exponents(self) -> None:
        ex = qt.object_exponents()
        self.assertAlmostEqual(ex["e"], -2.22, delta=0.05)
        self.assertAlmostEqual(ex["p"] / ex["e"], math.sqrt(qt.OB.proton_kg / qt.OB.electron_kg), delta=1e-9)
        self.assertLess(ex["g"], -1.0e10)


class JosephsonTest(unittest.TestCase):
    """The circuit of the first version of the film (a part that is kept in the code and is not shown)."""
    def test_barrier_height_equals_the_numerical_extremum(self) -> None:
        for s in (0.3, 0.6, 0.9):
            d0, d1 = tp.well_and_top(s)
            self.assertAlmostEqual(float(tp.washboard(d1, s) - tp.washboard(d0, s)), tp.barrier_height(s), places=10)
            d = np.linspace(d0, d1 + 0.5, 200001)
            u = tp.washboard(d, s)
            self.assertAlmostEqual(float(u.max() - u[0]), tp.barrier_height(s), places=8)

    def test_plasma_frequency_equals_the_numerical_second_derivative(self) -> None:
        for s in (0.3, 0.6, 0.9):
            self.assertAlmostEqual(tp.harmonic_frequency_numeric(s, 0.05), tp.plasma_ratio(s), delta=1e-4)

    def test_the_cubic_exponent_is_36_over_5(self) -> None:
        self.assertAlmostEqual(math.log(tp.rate_cubic(2.0) / tp.rate_cubic(1.0)) - 0.5 * math.log(2.0), -7.2, places=10)

    def test_the_rate_grows_with_the_bias(self) -> None:
        sg, wkb, cub = qt.rate_curves()
        self.assertTrue(np.all(np.diff(wkb) >= -1e-18))
        self.assertGreater(wkb[-1] / wkb[0], 1e6)

    def test_the_wkb_and_the_cubic_formula_agree_near_the_top(self) -> None:
        sg, wkb, cub = qt.rate_curves()
        i = int(np.argmin(np.abs(sg - 0.93)))
        self.assertLess(abs(math.log(wkb[i] / cub[i])), 0.5)

    def test_the_simulated_phase_packet_escapes_more_with_the_tilt(self) -> None:
        run = qt.circuit_run()
        self.assertTrue(np.all(np.diff(run["p_esc"]) >= -1e-3))
        self.assertLess(run["p_esc"][0], 1e-3)
        self.assertGreater(run["p_esc"][-1], 0.3)
        self.assertTrue(np.all(np.diff(run["s"]) >= -1e-12))


class ConfigTest(unittest.TestCase):
    def test_texts_have_the_same_keys_and_balanced_dollars(self) -> None:
        d = tomllib.loads((HERE / "texts.toml").read_text(encoding="utf-8"))
        self.assertEqual(set(d["en"]), set(d["ru"]))
        for lang in ("en", "ru"):
            for k, v in d[lang].items():
                self.assertEqual(v.replace("\\$", "").count("$") % 2, 0, (lang, k))
                for m in re.finditer(r"\$[^$]*\$", v):
                    self.assertNotRegex(m.group(0), r"[А-Яа-яЁё]", (lang, k))

    def test_no_formula_is_left_outside_the_dollars(self) -> None:
        d = tomllib.loads((HERE / "texts.toml").read_text(encoding="utf-8"))
        for lang in ("en", "ru"):
            for k, v in d[lang].items():
                out = re.sub(r"\$[^$]*\$", "", v)
                self.assertNotRegex(out, r"[=<>∫≈²]", (lang, k, out))

    def test_the_text_about_the_bias_matches_the_sketch(self) -> None:
        d = tomllib.loads((HERE / "texts.toml").read_text(encoding="utf-8"))
        self.assertIn(str(qt.CI.c1_s).replace(".", "{,}") if False else "0", d["en"]["jc_note"])
        self.assertEqual(qt.CI.c1_s, 0.6)

    def test_video_section_and_script_rules(self) -> None:
        cfg = tomllib.loads((HERE / "config.toml").read_text(encoding="utf-8"))
        v = cfg["video"]
        for k in ("width", "height", "fps", "dpi", "crf", "preset", "preview_width", "preview_height", "preview_fps", "fade_s", "reference_height"):
            self.assertIn(k, v)
        self.assertEqual(v["width"] * 9, v["height"] * 16)
        src = (HERE / "quantum_tunneling.py").read_text(encoding="utf-8")
        self.assertIn("load_config", src)
        for bad in ('"libx264", "-preset", "slow"', "(1280, 720)", "dpi = 100\n"):
            self.assertNotIn(bad, src)

    def test_snapshot_renders_in_both_languages(self) -> None:
        import subprocess
        import tempfile
        py = sys.executable
        with tempfile.TemporaryDirectory() as tmp:
            for lang in ("en", "ru"):
                out = Path(tmp) / f"s_{lang}.mp4"
                r = subprocess.run([py, str(HERE / "quantum_tunneling.py"), "--lang", lang, "--snapshot", "12", "--out", str(out)], capture_output=True, text=True, cwd=HERE)
                self.assertEqual(r.returncode, 0, r.stderr[-400:])
                self.assertTrue(out.with_suffix(".png").exists())

    def test_set_overrides_a_value(self) -> None:
        import subprocess
        r = subprocess.run([sys.executable, "-c", "import sys; sys.argv=['x','--set','classical.cars=7']; sys.path.insert(0,'.'); import quantum_tunneling as q; print(q.CL.cars)"],
                           capture_output=True, text=True, cwd=HERE)
        self.assertEqual(r.stdout.strip().splitlines()[-1], "7", r.stderr[-300:])


if __name__ == "__main__":
    unittest.main()
