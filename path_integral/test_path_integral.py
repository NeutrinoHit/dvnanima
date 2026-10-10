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

import path_integral as pi

HERE = Path(__file__).resolve().parent


class SlitsTest(unittest.TestCase):
    def test_number_of_paths(self) -> None:
        for n_scr, n_paths in ((1, 3), (2, 21), (3, 105)):
            verts, ph = pi.paths_to_detector(n_scr, 0.1)
            self.assertEqual(len(verts), n_paths)
            self.assertEqual(len(ph), n_paths)

    def test_symmetric_slits_give_a_symmetric_profile(self) -> None:
        for y in (0.05, 0.2, 0.33):
            self.assertAlmostEqual(abs(pi.amplitude(3, y)), abs(pi.amplitude(3, -y)), places=6)

    def test_all_paths_are_equal_length_at_the_axis_for_one_slit_screen(self) -> None:
        # the straight path through the central slits is the shortest one
        verts, ph = pi.paths_to_detector(3, 0.0)
        lengths = ph * pi.WAVELENGTH / (2 * math.pi)
        self.assertAlmostEqual(float(lengths.min()), 1.0, places=9)


class CornuTest(unittest.TestCase):
    def test_action_is_quadratic(self) -> None:
        self.assertAlmostEqual(float(pi.action_phase(2.0) / pi.action_phase(1.0)), 4.0)

    def test_sum_converges_to_the_stationary_phase_value(self) -> None:
        for c in (9.0, 40.0, 90.0):
            exact = pi.stationary_value(c)
            got = pi.cornu_sum(6.0, c, n=400001)
            self.assertAlmostEqual(abs(got), abs(exact), delta=0.03 * abs(exact))
            self.assertAlmostEqual(float(np.angle(got)), math.pi / 4, delta=0.1)

    def test_amplitude_scales_as_sqrt_of_hbar(self) -> None:
        self.assertAlmostEqual(abs(pi.stationary_value(90.0)) / abs(pi.stationary_value(9.0)), 1 / math.sqrt(10), places=12)


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
                    self.assertNotIn("\\bm", text)
                    self.assertNotIn("\\Box", text)

    def test_the_same_placeholders_in_both_languages(self) -> None:
        for key in self.texts["en"]:
            with self.subTest(key=key):
                self.assertEqual(sorted(re.findall(r"\{[a-z0-9_]+\}", self.texts["en"][key])),
                                 sorted(re.findall(r"\{[a-z0-9_]+\}", self.texts["ru"][key])))

    def test_video_section_is_complete_and_16_to_9(self) -> None:
        v = self.cfg["video"]
        keys = {"width", "height", "fps", "dpi", "crf", "preset", "preview_width", "preview_height", "preview_fps", "fade_s", "reference_height"}
        self.assertTrue(keys <= set(v), keys - set(v))
        self.assertEqual(v["width"] * 9, v["height"] * 16)
        self.assertGreater(v["width"], v["preview_width"])

    def test_script_reads_the_config(self) -> None:
        src = (HERE / "path_integral.py").read_text(encoding="utf-8")
        self.assertIn("load_config", src)
        for forbidden in ('"libx264", "-preset", "slow"', "(1280, 720)", "dpi = 100\n"):
            self.assertNotIn(forbidden, src)

    def test_set_overrides_a_value(self) -> None:
        out = subprocess.run([sys.executable, "-c", "import path_integral as m; print(m.CFG.video.crf)", "--set", "video.crf=23"],
                             cwd=HERE, capture_output=True, text=True, check=True).stdout.strip()
        self.assertEqual(out, "23")


class FilmSmokeTest(unittest.TestCase):
    def test_a_snapshot_in_each_language(self) -> None:
        from PIL import Image
        with tempfile.TemporaryDirectory() as d:
            sizes = {}
            for lang in ("en", "ru"):
                out = Path(d) / f"s_{lang}.mp4"
                subprocess.run([sys.executable, "path_integral.py", "--lang", lang, "--snapshot", "12", "--out", str(out)], cwd=HERE, check=True,
                               capture_output=True)
                sizes[lang] = Image.open(out.with_suffix(".png")).size
            self.assertEqual(sizes["en"], (pi.CFG.video.width, pi.CFG.video.height))
            self.assertEqual(sizes["en"], sizes["ru"])


if __name__ == "__main__":
    unittest.main()
