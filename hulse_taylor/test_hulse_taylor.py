from __future__ import annotations

import math
import subprocess
import sys
import tempfile
import tomllib
import unittest
from pathlib import Path

import numpy as np

import hulse_taylor as ht

HERE = Path(__file__).resolve().parent


class OrbitTest(unittest.TestCase):
    def test_semi_major_axis_from_keplers_third_law(self) -> None:
        # about 1.95e9 m (2.8 solar radii) for PSR B1913+16
        self.assertAlmostEqual(ht.semi_major_axis() / 1e9, 1.949, delta=0.005)

    def test_kepler_equation(self) -> None:
        m = np.linspace(0, 2 * math.pi, 50, endpoint=False)
        E = ht.kepler(m)
        np.testing.assert_allclose(E - ht.ECC * np.sin(E), m, atol=1e-10)

    def test_periastron_and_apastron_separations(self) -> None:
        _, _, r = ht.relative_orbit(np.array([0.0, math.pi]), 1.0)
        self.assertAlmostEqual(float(r[0]), 1 - ht.ECC, places=10)
        self.assertAlmostEqual(float(r[1]), 1 + ht.ECC, places=10)

    def test_orbit_is_an_ellipse(self) -> None:
        x, y, _ = ht.relative_orbit(np.linspace(0, 2 * math.pi, 200), 1.0)
        b = math.sqrt(1 - ht.ECC ** 2)
        np.testing.assert_allclose(((x + ht.ECC) / 1.0) ** 2 + (y / b) ** 2, 1.0, atol=1e-9)


class ShiftTest(unittest.TestCase):
    def test_periastron_shift_after_thirty_years(self) -> None:
        # about -38.6 s, as in the figure of the book (about -40 s in 2005)
        self.assertAlmostEqual(float(ht.periastron_shift(30.0)), -38.6, delta=0.1)

    def test_shift_is_quadratic_and_negative(self) -> None:
        self.assertAlmostEqual(float(ht.periastron_shift(20.0) / ht.periastron_shift(10.0)), 4.0, places=9)
        self.assertLess(float(ht.periastron_shift(5.0)), 0.0)

    def test_measured_derivative_agrees_with_gr(self) -> None:
        self.assertAlmostEqual(ht.PDOT_INTR / ht.PDOT_GR, 0.9983, delta=0.0005)

    def test_orbit_shrink_in_thirty_years(self) -> None:
        self.assertAlmostEqual(ht.fractional_shrink(30.0), -5.4e-8, delta=0.1e-8)

    def test_period_change_in_microseconds_per_year(self) -> None:
        self.assertAlmostEqual(ht.PDOT_GR * ht.YEAR * 1e6, -75.82, delta=0.02)


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
        import re
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
        src = (HERE / "hulse_taylor.py").read_text(encoding="utf-8")
        self.assertIn("load_config", src)
        for forbidden in ('"libx264", "-preset", "slow"', "(1280, 720)", "dpi = 100\n"):
            self.assertNotIn(forbidden, src)

    def test_model_values_come_from_the_config(self) -> None:
        self.assertEqual(ht.ECC, self.cfg["model"]["eccentricity"])
        self.assertEqual(ht.PDOT_GR, self.cfg["model"]["pdot_gr"])
        self.assertAlmostEqual(self.cfg["model"]["pdot_ratio"], ht.PDOT_INTR / ht.PDOT_GR, delta=0.0005)

    def test_set_overrides_a_value(self) -> None:
        out = subprocess.run([sys.executable, "-c", "import hulse_taylor as h; print(h.ECC)", "--set", "model.eccentricity=0.5"],
                             cwd=HERE, capture_output=True, text=True, check=True).stdout.strip()
        self.assertEqual(out, "0.5")


class FilmSmokeTest(unittest.TestCase):
    def test_a_snapshot_in_each_language(self) -> None:
        from PIL import Image
        with tempfile.TemporaryDirectory() as d:
            sizes = {}
            for lang in ("en", "ru"):
                out = Path(d) / f"s_{lang}.mp4"
                subprocess.run([sys.executable, "hulse_taylor.py", "--lang", lang, "--snapshot", "21", "--out", str(out)], cwd=HERE, check=True,
                               capture_output=True)
                sizes[lang] = Image.open(out.with_suffix(".png")).size
            self.assertEqual(sizes["en"], (ht.CFG.video.width, ht.CFG.video.height))
            self.assertEqual(sizes["en"], sizes["ru"])


if __name__ == "__main__":
    unittest.main()
