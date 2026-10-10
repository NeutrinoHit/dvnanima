"""Tests of the film: the timeline, the renderer, the configuration and the layout of the frames (physics: test_ffbar.py).

    python -m unittest test_higgs_ff_spin test_ffbar
"""

from __future__ import annotations

import ast
import math
import subprocess
import sys
import tomllib
import unittest
from pathlib import Path

import numpy as np

import ffbar as fb
import higgs_ff_spin as hf
import isorender as ir

HERE = Path(__file__).resolve().parent
SAMPLE_TIMES = [1.0, 4.0, 9.0, 14.0, 17.0, 21.0, 26.0, 33.0, 40.0, 46.0, 50.0, 55.0, 60.0, 64.0, 68.0, 76.0, 82.0, 86.0, 90.0, 93.0]


class RendererTest(unittest.TestCase):
    @staticmethod
    def small_state():
        st = fb.common_axis_state([0, 0, 1], 1, 1)
        return lambda P: fb.psi_s(P, st, 1.0)

    def render(self, **kw):
        sh = ir.Shading(0.3, 0.6, 0.3, 24.0, (-0.45, 0.55, 0.7))
        args = dict(levels=[0.05, 0.10], alphas=[0.3, 1.0], shading=sh, r_max=5.0, view=4.6, size=48, samples=64)
        args.update(kw)
        return ir.render(self.small_state(), ir.camera(32.0, -55.0), **args)

    def test_hue_shift_equals_a_shift_of_the_phase(self) -> None:
        """The colour at the time t is the colour of the state with the phase shifted: S0, S1 give the same image as psi -> psi exp(i delta)."""
        st = fb.common_axis_state([0, 0, 1], 1, 1)
        delta = 1.3
        img = self.render()
        img2 = ir.render(lambda P: np.exp(1j * delta) * fb.psi_s(P, st, 1.0), ir.camera(32.0, -55.0), [0.05, 0.10], [0.3, 1.0],
                         ir.Shading(0.3, 0.6, 0.3, 24.0, (-0.45, 0.55, 0.7)), 5.0, 4.6, 48, 64)
        bg = np.zeros(3)
        a = img.rgb(delta, 0.58, 0.42, bg)
        b = img2.rgb(0.0, 0.58, 0.42, bg)
        self.assertLess(float(np.abs(a - b).max()), 1e-4)

    def test_the_donut_has_a_hole_and_the_dumbbell_two_lobes(self) -> None:
        donut = self.render(size=64)
        c = donut.cover
        self.assertGreater(float(c[32, 20]), 0.3)                          # the body of the ring on the left
        self.assertLess(float(c[0, 0]), 1e-9)                              # nothing in the corner
        st = fb.common_axis_state([0, 0, 1], 1, -1)
        sh = ir.Shading(0.3, 0.6, 0.3, 24.0, (-0.45, 0.55, 0.7))
        dumb = ir.render(lambda P: fb.psi_s(P, st, 1.0), ir.camera(0.0, -55.0), [0.05], [1.0], sh, 5.0, 4.6, 64, 64)
        col = dumb.cover[:, 32]
        self.assertGreater(float(col[20]), 0.5)                            # upper lobe
        self.assertGreater(float(col[44]), 0.5)                            # lower lobe
        self.assertLess(float(col[32]), 0.2)                               # the nodal plane between them

    def test_the_surface_is_where_the_density_equals_the_level(self) -> None:
        st = fb.common_axis_state([0, 0, 1], 1, 1)
        fn = lambda P: fb.psi_s(P, st, 1.0)
        sh = ir.Shading(0.3, 0.6, 0.3, 24.0, (-0.45, 0.55, 0.7))
        img = ir.render(fn, ir.camera(90.0, 0.0), [0.10], [1.0], sh, 5.0, 4.6, 200, 400)          # seen from above: rings of the donut
        n = 200
        a = (np.arange(n) + 0.5) / n * 9.2 - 4.6
        row = img.cover[n // 2]                                           # y = 0 line: the covered x are the points of the ring
        covered = a[row > 0.5]
        inner, outer = abs(covered).min(), abs(covered).max()
        for rr in (inner, outer):                                         # at z = 0 the density is 0.1 at the edge of the ring
            self.assertAlmostEqual(float(abs(fn(np.array([rr, 0.0, 0.0]))) ** 2), 0.10, delta=0.01)


class TimelineTest(unittest.TestCase):
    def test_parts_follow_each_other(self) -> None:
        parts = [hf.T_P1, hf.T_P2, hf.T_P3, hf.T_P4, hf.T_P5]
        for a, b in zip(parts[:-1], parts[1:]):
            self.assertEqual(a[1], b[0])
        self.assertEqual(parts[-1][1], hf.CONTENT_TOTAL)
        self.assertEqual(parts[0][0], 0.0)

    def test_the_length_is_in_the_asked_range(self) -> None:
        self.assertTrue(60.0 <= hf.TOTAL <= 110.0, hf.TOTAL)

    def test_timeline_inserts_the_chapter_cards(self) -> None:
        self.assertAlmostEqual(hf.TOTAL, hf.CONTENT_TOTAL + hf.CARD_S * len(hf.CARD_AT))
        prev = -1.0
        for tf in np.linspace(0.0, hf.TOTAL - 0.01, 3000):
            tc, card, prog = hf.timeline(float(tf))
            self.assertGreaterEqual(tc, prev - 1e-9)
            prev = tc
            if card is not None:
                self.assertTrue(0.0 <= prog < 1.0)
        for tc in (0.5, 17.0, 50.0, 70.0, 90.0):
            tc2, card, _ = hf.timeline(hf.film_time(tc))
            self.assertIsNone(card)
            self.assertAlmostEqual(tc2, tc)

    def test_scheduled_items_lie_inside_their_part(self) -> None:
        for name, (a, b) in zip(("p1", "p2", "p3", "p4", "p5"), hf.BOUNDS):
            P = getattr(hf.CFG, name)
            length = b - a
            for lst in ("formulas", "captions"):
                for it in P[lst]:
                    self.assertTrue(0.0 <= it["from"] < it["to"] <= length + 1e-9, (name, it))
            caps = sorted(P.captions, key=lambda i: i["from"])
            for x, y in zip(caps[:-1], caps[1:]):
                self.assertLessEqual(x["to"], y["from"] + 1e-9, (name, x, y))                 # one caption at a time
            self.assertAlmostEqual(caps[-1]["to"], length, places=6)

    def test_sweep_is_continuous_and_covers_zero_to_ninety_degrees(self) -> None:
        for P, rows in ((hf.CFG.p2, hf.CFG.p2.sweep), (hf.CFG.p3, hf.CFG.p3.sweep)):
            for (a0, a1, v0, v1), (b0, b1, w0, w1) in zip(rows[:-1], rows[1:]):
                self.assertEqual(a1, b0)
                self.assertEqual(v1, w0)
            vals = [hf.segments(t, rows) for t in np.linspace(0, rows[-1][1], 200)]
            self.assertAlmostEqual(min(vals), 0.0)
            self.assertAlmostEqual(max(vals), 90.0)

    def test_the_number_of_plane_waves_grows_and_saturates(self) -> None:
        ns = [hf.n_directions(t) for t in np.linspace(0.0, 16.0, 100)]
        self.assertEqual(ns[0], 1)
        self.assertTrue(all(b >= a for a, b in zip(ns[:-1], ns[1:])))
        self.assertEqual(ns[-1], hf.CFG.partial_sums.n_directions)


class ConfigTest(unittest.TestCase):
    texts = tomllib.loads((HERE / "texts.toml").read_text(encoding="utf-8"))
    cfg = tomllib.loads((HERE / "config.toml").read_text(encoding="utf-8"))

    def test_texts_have_the_same_keys_in_both_languages_and_balanced_formulas(self) -> None:
        self.assertEqual(set(self.texts), {"en", "ru"})
        self.assertEqual(set(self.texts["en"]), set(self.texts["ru"]))
        for lang, table in self.texts.items():
            for key, value in table.items():
                self.assertEqual(value.count("$") % 2, 0, f"{lang}.{key}: unbalanced $")
                for line in value.split("\n"):
                    self.assertEqual(line.count("$") % 2, 0, f"{lang}.{key}: a formula is split over two lines")

    def test_no_russian_words_inside_formulas_and_no_decimal_point_in_russian(self) -> None:
        import re
        for key, value in self.texts["ru"].items():
            for formula in re.findall(r"\$(.*?)\$", value, flags=re.S):
                self.assertIsNone(re.search(r"[А-Яа-яЁё]", formula), f"ru.{key}: Cyrillic inside $...$")
                self.assertIsNone(re.search(r"\d\.\d", formula), f"ru.{key}: decimal point in a formula")

    def test_every_text_key_used_by_the_config_exists(self) -> None:
        keys = set(self.texts["en"])
        def walk(o):
            if isinstance(o, dict):
                if isinstance(o.get("key"), str):
                    yield o["key"]
                for k in ("header", "header_p"):
                    if isinstance(o.get(k), str):
                        yield o[k]
                for v in o.values():
                    yield from walk(v)
            elif isinstance(o, list):
                for v in o:
                    yield from walk(v)
        for k in walk(self.cfg):
            self.assertIn(k, keys)
        for k in hf.CARD_KEYS:
            self.assertIn(k, keys)
            self.assertIn(k + "s", keys)

    def test_video_section_is_complete_and_16_to_9(self) -> None:
        v = self.cfg["video"]
        need = {"width", "height", "fps", "dpi", "crf", "preset", "preview_width", "preview_height", "preview_fps", "fade_s", "reference_height"}
        self.assertTrue(need <= set(v))
        self.assertEqual(v["width"] * 9, v["height"] * 16)
        self.assertGreater(v["width"], v["preview_width"])
        self.assertIn("style", self.cfg)

    def test_the_script_has_no_hard_coded_video_settings_and_uses_load_config(self) -> None:
        for name in ("higgs_ff_spin.py",):
            src = (HERE / name).read_text(encoding="utf-8")
            self.assertIn("load_config", src)
            for forbidden in ('"libx264", "-preset", "slow"', "(1280, 720)", "dpi = 100\n"):
                self.assertNotIn(forbidden, src)

    def test_set_override_works(self) -> None:
        r = subprocess.run([sys.executable, "-c", "import sys; sys.argv=['x','--set','iso.elevation_deg=40.0']; import higgs_ff_spin as h; print(h.CFG.iso.elevation_deg)"],
                           cwd=HERE, capture_output=True, text=True)
        self.assertEqual(r.stdout.strip(), "40.0")

    def test_no_stray_numeric_literals_in_the_drawing_script(self) -> None:
        """Only structural constants (0, 1, 2, 3, 0.5, 255, 16, 12 ... below) may appear as numbers in the script."""
        allowed = {0.0, 1.0, 2.0, 3.0, 4.0, 0.5, 0.7, 255.0, 1e-6, 1e-9, 1e-12, 0.02, 0.05, 0.01, 0.99}      # integers up to 20 are z-orders, indices, counts
        tree = ast.parse((HERE / "higgs_ff_spin.py").read_text(encoding="utf-8"))
        bad = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
                if node.value not in allowed and not (isinstance(node.value, int) and node.value <= 20):
                    bad.append((node.lineno, node.value))
        self.assertEqual(bad, [], f"numbers outside the configuration: {bad}")

    def test_the_cells_fit_in_the_frame(self) -> None:
        for name in ("p2", "p4"):
            P = self.cfg[name]
            half_w = 0.5 * P["cell_side"] * 9 / 16
            xs = P["columns_x"] if name == "p2" else [P["left_x"], P["right_x"]]
            for x in xs:
                self.assertTrue(half_w <= x <= 1.0 - half_w, (name, x))
            self.assertTrue(0.5 * P["cell_side"] <= P["cell_y"] <= 1.0 - 0.5 * P["cell_side"])


class FrameLayoutTest(unittest.TestCase):
    """Frames at sample times in both languages: nothing leaves the frame, texts do not overlap, no text is clipped by a shrink below the limit."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.cv = {lang: hf.Canvas((hf.CFG.video.width, hf.CFG.video.height), lang) for lang in ("en", "ru")}

    @staticmethod
    def boxes(cv):
        rend = cv.fig.canvas.get_renderer()
        out = []
        for t in cv.texts:
            if not t.get_text() or t.get_alpha() == 0:
                continue
            col = t.get_color()
            if isinstance(col, tuple) and len(col) == 4 and col[3] < 0.35:
                continue
            out.append((t, t.get_window_extent(rend)))
        return out

    def check(self, lang: str, tc: float) -> None:
        cv = self.cv[lang]
        hf.draw_frame(cv, hf.film_time(tc) + 1e-6)
        cv.fig.canvas.draw()
        bs = self.boxes(cv)
        for t, b in bs:
            self.assertGreaterEqual(b.x0, -1.0, (lang, tc, t.get_text()))
            self.assertLessEqual(b.x1, cv.W + 1.0, (lang, tc, t.get_text()))
            self.assertGreaterEqual(b.y0, -1.0, (lang, tc, t.get_text()))
            self.assertLessEqual(b.y1, cv.H + 1.0, (lang, tc, t.get_text()))
        for i in range(len(bs)):
            for j in range(i + 1, len(bs)):
                a, b = bs[i][1], bs[j][1]
                ox = min(a.x1, b.x1) - max(a.x0, b.x0)
                oy = min(a.y1, b.y1) - max(a.y0, b.y0)
                self.assertFalse(ox > 2.0 and oy > 2.0, f"{lang} t={tc}: '{bs[i][0].get_text()}' overlaps '{bs[j][0].get_text()}'")

    def test_no_overlaps_english(self) -> None:
        for tc in SAMPLE_TIMES:
            self.check("en", tc)

    def test_no_overlaps_russian(self) -> None:
        for tc in SAMPLE_TIMES:
            self.check("ru", tc)

    def test_cards_fit(self) -> None:
        for lang in ("en", "ru"):
            for tf in (0.8, 2.5, hf.film_time(16.0) - 1.0, hf.film_time(48.0) - 1.0, hf.film_time(61.0) - 1.0, hf.film_time(83.0) - 1.0):
                cv = self.cv[lang]
                hf.draw_frame(cv, tf + 0.85 * 0.0)
                cv.fig.canvas.draw()
                for t, b in self.boxes(cv):
                    self.assertTrue(b.x0 >= 0 and b.x1 <= cv.W, (lang, tf, t.get_text()))

    def test_snapshots_render_in_both_languages(self) -> None:
        for lang in ("en", "ru"):
            cv = self.cv[lang]
            hf.draw_frame(cv, hf.film_time(40.0))
            cv.fig.canvas.draw()
            img = np.asarray(cv.fig.canvas.buffer_rgba())
            self.assertEqual(img.shape[:2], (hf.CFG.video.height, hf.CFG.video.width))
            self.assertGreater(float(img[..., :3].mean()), 5.0)


class PlotTest(unittest.TestCase):
    def test_angular_factors_of_the_plots_are_the_physics_ones(self) -> None:
        th = np.linspace(0, math.pi / 2, 7)
        for r1, r2 in ((1, -1), (1, 1), (-1, -1)):
            st = fb.common_axis_state([0, 0, 1], r1, r2)
            for t in th:
                rh = np.array([math.sin(t), 0.0, math.cos(t)])
                self.assertAlmostEqual(float(fb.angular_scalar(rh, st.xi1, st.xi2)), 0.5 * (1 + r1 * r2 * (1 - 2 * math.cos(t) ** 2)), places=12)

    def test_pr_where_the_axis_density_cancels(self) -> None:
        e1, e2 = hf.eps_of(hf.CFG.p4.ramp[-1][3])
        x = np.linspace(0.05, 6.0, 4000)
        a, b = e1 * hf.BETA * fb.j1(x), e2 * fb.j0(x)
        sgn = np.sign(b - a)
        eq = x[int(np.argmax(sgn[1:] != sgn[0])) + 1]
        self.assertAlmostEqual(float(eq), 2.04, delta=0.05)
        self.assertAlmostEqual(float(fb.density_on_axis(float(eq), e1, e2, hf.BETA, -1)), 0.0, delta=1e-3)

    def test_decimal_comma_in_russian_numbers(self) -> None:
        self.assertEqual(hf.num(0.3, ".2f", "ru"), "0{,}30")
        self.assertEqual(hf.num(0.3, ".2f", "en"), "0.30")

    def test_the_palette_is_a_closed_wheel_of_distinct_colours(self) -> None:
        ph = np.linspace(0, 2 * math.pi, 7)
        c = hf.palette(ph)
        self.assertTrue(np.allclose(c[0], c[-1]))
        self.assertGreater(float(np.abs(c[0] - c[3]).max()), 0.3)


if __name__ == "__main__":
    unittest.main()
