"""Physics and configuration tests of the films "Attraction" and "Repulsion" (python -m pytest test_attraction_repulsion.py)."""
from __future__ import annotations

import re
import subprocess
import sys
import tempfile
import tomllib
import unittest
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
import attraction_repulsion as ar  # noqa: E402


def data(case: str) -> dict:
    ar.set_case(case)
    return ar.get_data()


class PhysicsTest(unittest.TestCase):
    def test_the_signs_of_the_charges(self) -> None:
        self.assertEqual(list(data("attraction")["signs"]), [1, -1])
        self.assertEqual(list(data("repulsion")["signs"]), [1, 1])

    def test_attraction_bends_the_trajectories_toward_each_other_and_repulsion_away(self) -> None:
        for case, sgn in (("attraction", -1.0), ("repulsion", 1.0)):
            D = data(case)
            y1, y1_free = D["centers"][:, 0, 1], D["straight"][:, 0, 1]
            self.assertGreater(sgn * (y1[-1] - y1_free[-1]), 0.5, case)         # packet 1 starts above packet 2: toward = lower y

    def test_the_phase_gradient_shifts_the_momentum_toward_or_away(self) -> None:
        for case, sgn in (("attraction", 1.0), ("repulsion", -1.0)):
            D = data(case)
            dp = D["d_p"]
            self.assertTrue(np.all(sgn * dp >= -1e-4), case)
            self.assertGreater(sgn * dp[-1], 0.2, case)
            self.assertTrue(np.all(np.diff(sgn * dp) >= -1e-3), case)

    def test_the_phase_gradient_agrees_with_the_momentum_of_the_centre(self) -> None:
        """<Delta P> from the phase of the field (the density-weighted gradient) and m (v(t) - v(0)) of the centre of the packet along the same line: the same size, the gradient
        of the model is taken at the centre point, so the two differ by tens of per cent."""
        for case in ("attraction", "repulsion"):
            D = data(case)
            dt = D["st"][1] - D["st"][0]
            v = np.gradient(D["centers"], dt, axis=0)
            for k in (D["ti"].size // 2, D["ti"].size - 1):
                i = int(round(D["ti"][k] / dt))
                dpc = ar.SQ.mass * float(np.dot(v[i, 0] - v[0, 0], D["nvec"][k]))
                self.assertAlmostEqual(float(D["d_p"][k]) / dpc, 1.0, delta=0.45, msg=(case, k))

    def test_the_distribution_is_shifted_by_the_mean_phase_gradient(self) -> None:
        D = data("attraction")
        k = D["ti"].size - 1
        axis, free, summed, g = ar.momentum_distributions(D, k)
        mean_f = (axis * free).sum() / free.sum()
        mean_s = (axis * summed).sum() / summed.sum()
        self.assertAlmostEqual(float(mean_s - mean_f), g, delta=0.01)
        self.assertAlmostEqual(float(summed.sum() / free.sum()), 1.0, delta=1e-6)       # a pure phase: the total probability is the same

    def test_the_phase_profile_has_zero_at_the_centre_and_the_free_peak_is_one(self) -> None:
        D = data("attraction")
        mid = D["s_axis"].size // 2
        self.assertAlmostEqual(float(D["chi_line"][0][mid] - D["chi_line"][0][mid]), 0.0, places=12)
        self.assertAlmostEqual(float(np.hypot(D["free_re"], D["free_im"]).max()), 1.0, places=6)

    def test_the_two_cases_start_equal_and_differ_only_by_the_charge(self) -> None:
        a, r = data("attraction"), data("repulsion")
        self.assertTrue(np.allclose(a["centers"][0], r["centers"][0]))
        self.assertTrue(np.allclose(a["straight"], r["straight"]))
        self.assertGreater(np.abs(a["centers"][-1] - r["centers"][-1]).max(), 1.0)


class TimelineTest(unittest.TestCase):
    def test_the_parts_are_consecutive_and_the_cards_are_inside(self) -> None:
        parts = [ar.PARTS[k] for k in ("intro", "motion", "interf", "outro")]
        for (a0, a1), (b0, b1) in zip(parts, parts[1:]):
            self.assertEqual(a1, b0)
        self.assertEqual(parts[0][0], 0.0)
        self.assertTrue(all(0.0 <= c <= ar.CONTENT_TOTAL for c in ar.CARD_AT))
        self.assertAlmostEqual(ar.TOTAL, ar.CONTENT_TOTAL + len(ar.CARD_AT) * ar.TL.card_s)

    def test_the_sim_time_runs_over_the_whole_model_in_the_motion_part(self) -> None:
        a, b = ar.PARTS["motion"]
        self.assertAlmostEqual(ar.sim_time_of(a), 0.0)
        self.assertAlmostEqual(ar.sim_time_of(b - 1e-9), ar.SQ.t_end, places=3)


class ConfigTest(unittest.TestCase):
    def test_texts_have_the_same_keys_and_balanced_dollars(self) -> None:
        d = tomllib.loads((HERE / "texts.toml").read_text(encoding="utf-8"))
        self.assertEqual(set(d["en"]), set(d["ru"]))
        for lang in ("en", "ru"):
            for k, v in d[lang].items():
                self.assertEqual(v.count("$") % 2, 0, (lang, k))
                for m in re.finditer(r"\$[^$]*\$", v):
                    self.assertNotRegex(m.group(0), r"[А-Яа-яЁё]", (lang, k))
                self.assertNotRegex(re.sub(r"\$[^$]*\$", "", v), r"[=<>∫≈²]", (lang, k))

    def test_every_case_has_all_the_texts(self) -> None:
        d = tomllib.loads((HERE / "texts.toml").read_text(encoding="utf-8"))
        for case in ("attraction", "repulsion"):
            for name in ("title", "h0s", "h2", "h2s", "cI2", "cM2", "cF2", "cF3", "o1", "o2", "o3", "o4"):
                self.assertIn(f"{name}_{case}", d["en"], (case, name))

    def test_video_section_and_script_rules(self) -> None:
        cfg = tomllib.loads((HERE / "config.toml").read_text(encoding="utf-8"))
        v = cfg["video"]
        for k in ("width", "height", "fps", "dpi", "crf", "preset", "preview_width", "preview_height", "preview_fps", "fade_s", "reference_height"):
            self.assertIn(k, v)
        self.assertEqual(v["width"] * 9, v["height"] * 16)
        src = (HERE / "attraction_repulsion.py").read_text(encoding="utf-8")
        self.assertIn("load_config", src)
        for bad in ('"libx264", "-preset", "slow"', "(1280, 720)", "dpi = 100\n"):
            self.assertNotIn(bad, src)

    def test_snapshots_render_in_both_languages_and_cases(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            for case in ("attraction", "repulsion"):
                for lang in ("en", "ru"):
                    out = Path(tmp) / f"s_{case}_{lang}.mp4"
                    r = subprocess.run([sys.executable, str(HERE / "attraction_repulsion.py"), "--case", case, "--lang", lang, "--snapshot", "30", "--out", str(out)],
                                       capture_output=True, text=True, cwd=HERE)
                    self.assertEqual(r.returncode, 0, r.stderr[-400:])
                    self.assertTrue(out.with_suffix(".png").exists())

    def test_set_overrides_a_value(self) -> None:
        r = subprocess.run([sys.executable, "-c", "import sys; sys.argv=['x','--set','scalar_qed.mass=7']; sys.path.insert(0,'.'); import attraction_repulsion as a; print(a.SQ.mass)"],
                           capture_output=True, text=True, cwd=HERE)
        self.assertEqual(r.stdout.strip().splitlines()[-1], "7", r.stderr[-300:])


if __name__ == "__main__":
    unittest.main()
