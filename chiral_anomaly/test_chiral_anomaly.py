"""Tests of the film script: configuration, texts, timeline, the numbers drawn on the screen and the layout of the frames.

    python -m pytest test_chiral_anomaly.py
"""

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
import chiral_anomaly as ca  # noqa: E402
import spectral_flow as sf  # noqa: E402


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

    def test_no_russian_inside_math_and_numbers_only_inside_formulas(self) -> None:
        texts = tomllib.loads((HERE / "texts.toml").read_text(encoding="utf-8"))
        for key, value in texts["ru"].items():
            for part in value.split("$")[1::2]:
                self.assertIsNone(re.search(r"[а-яА-ЯёЁ]", part), f"ru.{key}: a Russian word inside a formula")
        for lang, table in texts.items():
            for key, value in table.items():
                outside = " ".join(value.split("$")[0::2])
                for name in ("G", "Gm", "dG", "Gold"):
                    self.assertNotIn(f"@{name}@", outside, f"{lang}.{key}: a number outside a formula (the decimal comma needs a formula)")

    def test_every_key_used_by_the_script_exists_in_both_languages(self) -> None:
        src = (HERE / "chiral_anomaly.py").read_text(encoding="utf-8") + (HERE / "config.toml").read_text(encoding="utf-8")
        for lang in ("en", "ru"):
            for key in re.findall(r'self\.(?:tx|T)\(?\[?"(\w+)"', src):
                if key.endswith("_"):                                     # a prefix: cnt_ + NR, NL, Q, Q5
                    for suffix in ("NR", "NL", "Q", "Q5"):
                        self.assertIn(key + suffix, ca.TEXT[lang])
                else:
                    self.assertIn(key, ca.TEXT[lang], key)
            for part in ca.PARTCFG:
                for f in part.formulas:
                    self.assertIn(f["key"], ca.TEXT[lang])
                for _, key in part.captions:
                    if key:
                        self.assertIn(key, ca.TEXT[lang])
            for r in ca.CFG.part5.rows:
                self.assertIn(r["key"], ca.TEXT[lang])
            for key in ca.CARD_KEYS:
                self.assertIn(key, ca.TEXT[lang])
                self.assertIn(key + "s", ca.TEXT[lang])

    def test_video_section_is_complete(self) -> None:
        cfg = tomllib.loads((HERE / "config.toml").read_text(encoding="utf-8"))
        keys = {"width", "height", "fps", "dpi", "crf", "preset", "preview_width", "preview_height", "preview_fps", "fade_s", "reference_height"}
        self.assertTrue(keys <= set(cfg["video"]))
        self.assertEqual(cfg["video"]["width"] * 9, cfg["video"]["height"] * 16)
        self.assertGreater(cfg["video"]["width"], cfg["video"]["preview_width"])
        self.assertIn("style", cfg)

    def test_the_script_has_no_hard_coded_video_settings(self) -> None:
        src = (HERE / "chiral_anomaly.py").read_text(encoding="utf-8")
        self.assertIn("load_config", src)
        for forbidden in ('"libx264", "-preset", "slow"', "(1280, 720)", "dpi = 100\n"):
            self.assertNotIn(forbidden, src)

    def test_set_overrides_a_value_and_a_misspelt_key_is_an_error(self) -> None:
        sys.path.insert(0, str(HERE.parent))
        import dvconfig
        old = sys.argv
        try:
            sys.argv = ["film.py", "--set", "part2.rate=0.3", "--set", "style.background='#000000'"]
            c = dvconfig.load_config(HERE)
            self.assertEqual(c.part2.rate, 0.3)
            self.assertEqual(c.style.background, "#000000")
            sys.argv = ["film.py", "--set", "part2.rtae=3"]
            with self.assertRaises(KeyError):
                dvconfig.load_config(HERE)
        finally:
            sys.argv = old

    def test_length_of_the_film_is_in_the_range(self) -> None:
        self.assertGreaterEqual(ca.TOTAL, 60.0)
        self.assertLessEqual(ca.TOTAL, 110.0)


class TimelineTest(unittest.TestCase):
    def test_parts_follow_each_other_and_cards_come_first(self) -> None:
        self.assertEqual(ca.PARTS[0][0], 0.0)
        for a, b in zip(ca.PARTS[:-1], ca.PARTS[1:]):
            self.assertEqual(a[1], b[0])
        self.assertEqual(len(ca.CARD_AT), len(ca.PARTS) + 1)
        self.assertEqual(ca.timeline(0.5)[1], 0)
        self.assertEqual(ca.timeline(ca.CARD_S + 0.5)[1], 1)

    def test_timeline_never_runs_backwards(self) -> None:
        prev = -1.0
        for tf in np.linspace(0.0, ca.TOTAL - 0.01, 3000):
            tc, card, prog = ca.timeline(float(tf))
            self.assertGreaterEqual(tc, prev - 1e-9)
            prev = tc
            if card is not None:
                self.assertTrue(0.0 <= prog < 1.0)

    def test_film_time_is_the_inverse_of_the_timeline(self) -> None:
        for tc in (0.5, 16.0, 45.0, 70.0, 95.0):
            tc2, card, _ = ca.timeline(ca.film_time(tc))
            self.assertIsNone(card)
            self.assertAlmostEqual(tc2, tc)

    def test_captions_and_formulas_are_inside_their_parts(self) -> None:
        for (a, b), P in zip(ca.PARTS, ca.PARTCFG):
            dur = b - a
            starts = [t for t, _ in P.captions]
            self.assertEqual(starts, sorted(starts))
            self.assertEqual(starts[0], 0.0)
            self.assertLess(starts[-1], dur)
            for f in P.formulas:
                self.assertLessEqual(f["appear"][1], dur)
            for k in ("field_on",):
                if k in P:
                    self.assertLess(P[k], dur)


class NumbersOnTheScreenTest(unittest.TestCase):
    """What the counters show is counted from the levels, with the rates of the config."""

    def test_part2_counters_equal_the_formula_up_to_one_level(self) -> None:
        P = ca.CFG.part2
        ring = ca.ring_of(P.rate)
        self.assertAlmostEqual(ring.level_rate, P.rate, places=12)
        for u in np.linspace(P.field_on + 0.2, ca.PARTS[1][1] - ca.PARTS[1][0] - 0.2, 40):
            t = ca.flow_time(float(u), P)
            c = ca.counters_at(ring, t)
            self.assertLessEqual(abs(c["NR"] - P.rate * t), 1.0 + 1e-9)
            self.assertEqual(c["NL"], -c["NR"])
            self.assertEqual(c["Q"], 0)
            self.assertEqual(c["Q5"], 2 * c["NR"])

    def test_staircase_follows_the_line_t_over_T(self) -> None:
        d, nr, nlm = ca.staircase(ca.CFG.part2.rate)
        self.assertTrue(np.all(np.abs(nr - d) <= 1.0 + 1e-9))
        self.assertTrue(np.array_equal(nr, nlm))
        self.assertTrue(np.all(np.diff(nr) >= 0))

    def test_part3_window_counts_agree_with_the_level_counting(self) -> None:
        P = ca.CFG.part3
        ring = ca.ring_of(P.rate)
        for t in np.linspace(0.3, P.t_max_a, 25):
            w = sf.window_counts(ring, float(t), "energy", P.cut)
            c = ca.counters_at(ring, float(t))
            self.assertEqual((w["R"], w["L"]), (c["NR"], c["NL"]))
            f = sf.window_filled(ring, float(t), "energy", P.cut)
            s0 = sf.window_filled(ring, 0.0, "energy", P.cut)
            self.assertEqual(f["R"] + f["L"], s0["R"] + s0["L"])

    def test_part3_label_cut_keeps_the_axial_charge(self) -> None:
        P = ca.CFG.part3
        ring = ca.ring_of(P.rate)
        for t in np.linspace(0.3, 4.0, 15):
            self.assertEqual(sf.window_counts(ring, float(t), "label", P.cut)["Q5"], 0)

    def test_lattice_numbers_on_the_screen(self) -> None:
        LT = ca.CFG.lattice
        for flow in np.linspace(0.0, LT.flow_max, 37):
            c = sf.lattice_counts(LT.n_sites, LT.theta, float(flow))
            self.assertEqual(c["filled"], LT.n_sites // 2)
            self.assertEqual(c["NR"], -c["NL"])
            if abs(flow + LT.theta - round(flow + LT.theta)) > 1e-9:         # exactly at a crossing the level is neither a particle nor a hole
                self.assertEqual(c["NR"], int(math.floor(flow + LT.theta)))

    def test_part4_orbits_equal_the_counted_degeneracy(self) -> None:
        """The plate of the film shows rows * cols states: the states that a strip with eB S / 2 pi = 12 holds (counted by diagonalising H)."""
        LD = ca.CFG.landau
        eb, ly = 1.0, 4.0 * math.pi
        spec = sf.landau_spectrum(eb, 24.0, ly, 193, 0.37, range(-24, 25))
        n = sf.landau_branch_count(spec, 0.37, -3.0, 3.0)                      # a strip of the width 6: S = 6 L_y
        self.assertEqual(n, int(round(eb * 6.0 * ly / (2.0 * math.pi))))
        self.assertEqual(LD.rows * LD.cols, n)

    def test_pion_numbers(self) -> None:
        P = ca.PN
        g = sf.pi0_width(P.alpha, P.m_pi_ev, P.f_pi_ev)
        self.assertAlmostEqual(g, 7.76, delta=0.02)
        meas = P.branching * P.hbar_ev_s / P.tau_s
        self.assertAlmostEqual(meas, 7.72, delta=0.03)
        self.assertLess(abs(g - meas), 2.0 * P.gamma_meas_err)
        self.assertGreater(meas / (P.hbar_ev_s / P.tau_no_anomaly_s), 800.0)
        self.assertAlmostEqual(sf.isospin_anomaly_weight([2 / 3, -1 / 3], [0.5, -0.5], 3), 0.5, places=12)


def boxes_overlap(a, b, tol: float = 2.0) -> bool:
    return min(a[2], b[2]) - max(a[0], b[0]) > tol and min(a[3], b[3]) - max(a[1], b[1]) > tol


class PictureTest(unittest.TestCase):
    TIMES = [0.6, 3.0, 6.0, 9.0, 12.0, 14.0, 16.0, 19.0, 24.0, 30.0, 36.0, 39.0, 44.0, 47.0, 50.0, 54.0, 58.0, 61.0, 66.0, 70.0, 74.0, 78.0, 82.0, 87.0, 91.0, 94.0, 96.5]

    def painters(self):
        for lang in ("en", "ru"):
            yield lang, ca.Painter(lang, (ca.CFG.video.preview_width * 2, ca.CFG.video.preview_height * 2))

    def test_snapshots_render_in_both_languages_and_nothing_overlaps_or_leaves_the_frame(self) -> None:
        for lang, pt in self.painters():
            W, H = pt.W, pt.H
            for tc in self.TIMES:
                frame = pt.draw(ca.film_time(tc), ca.TOTAL)
                self.assertEqual(frame.shape, (H, W, 4))
                self.assertGreater(int(frame[..., :3].max()), 100)
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
            for tc in (30.0, 50.0, 90.0):
                pt.draw(ca.film_time(tc), ca.TOTAL)
                for s, _ in pt.text_boxes():
                    if lang == "ru":
                        self.assertIsNone(re.search(r"\d\.\d", s), f"ru: '{s}' has a decimal point")
                    if lang == "en":
                        self.assertNotIn("{,}", s)
                    self.assertNotIn("@", s)

    def test_every_caption_fits_the_frame(self) -> None:
        for lang, pt in self.painters():
            r = pt.fig.canvas.get_renderer()
            for P in ca.PARTCFG:
                for _, key in P.captions:
                    if not key:
                        continue
                    t = pt.fig.text(*ca.LY.caption_pos, pt.T(key, **pt.cap_numbers()), fontsize=ca.FN.caption * pt.sc, va="bottom", linespacing=1.25)
                    bb = t.get_window_extent(r)
                    t.remove()
                    self.assertLess(bb.x1, 0.985 * pt.W, f"{lang} caption {key} is too wide")
                    self.assertLess(bb.y1, 0.12 * pt.H + bb.height, f"{lang} caption {key} is too tall")

    def test_formulas_fit_their_rows(self) -> None:
        for lang, pt in self.painters():
            r = pt.fig.canvas.get_renderer()
            for P in ca.PARTCFG:
                for f in P.formulas:
                    t = pt.fig.text(f["x"], 0.5, pt.formula_text(f["key"]), fontsize=ca.FN.formula * pt.sc)
                    bb = t.get_window_extent(r)
                    t.remove()
                    self.assertLess(bb.x1, 0.99 * pt.W, f"{lang} formula {f['key']} is too wide")

    def test_the_answer_fits_the_frame(self) -> None:
        for lang, pt in self.painters():
            r = pt.fig.canvas.get_renderer()
            for key, pos, size in (("answer", ca.LY.answer_pos, ca.FN.answer), ("answer_q", ca.LY.answer_q_pos, ca.FN.answer_q)):
                t = pt.fig.text(*pos, pt.tx[key], fontsize=size * pt.sc, ha="center", va="center", linespacing=ca.CFG.part5.answer_spacing)
                bb = t.get_window_extent(r)
                t.remove()
                self.assertGreater(bb.x0, 0.03 * pt.W, f"{lang} {key}")
                self.assertLess(bb.x1, 0.97 * pt.W, f"{lang} {key}")


if __name__ == "__main__":
    unittest.main()
