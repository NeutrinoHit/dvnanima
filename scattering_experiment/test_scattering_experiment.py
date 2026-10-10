from __future__ import annotations

import re
import subprocess
import sys
import tempfile
import tomllib
import unittest
from pathlib import Path

import numpy as np

import scattering_experiment as se

HERE = Path(__file__).resolve().parent


class AngularDistributionTest(unittest.TestCase):
    def test_density_is_normalized(self) -> None:
        c = np.linspace(-1, 1, 20001)
        self.assertAlmostEqual(float(np.trapezoid(se.angular_density(c), c)), 1.0, places=6)

    def test_sampling_follows_one_plus_cos_squared(self) -> None:
        c = se.sample_cos_theta(np.random.default_rng(0), 200_000)
        self.assertAlmostEqual(float(np.mean(c * c)), 0.4, delta=0.005)      # <cos^2> for (3/8)(1+c^2)
        self.assertAlmostEqual(float(np.mean(c)), 0.0, delta=0.005)          # no forward-backward asymmetry
        hist, edges = np.histogram(c, bins=10, range=(-1, 1), density=True)
        mid = 0.5 * (edges[1:] + edges[:-1])
        np.testing.assert_allclose(hist, se.angular_density(mid), atol=0.02)


class EventsTest(unittest.TestCase):
    def test_particles_meet_where_the_event_happens(self) -> None:
        crossings = se.make_events(1, 28.0)
        v = 560.0
        for cr in crossings[1:]:
            for ev in cr["events"]:
                t, xm, _ = se.event_time_and_point(ev, cr, v)
                xe = se.xc_of(t, cr, v) + cr["dx_e"][ev["ie"]]
                xp = -se.xc_of(t, cr, v) + cr["dx_p"][ev["ip"]]
                self.assertAlmostEqual(float(xe), float(xp), places=6)
                self.assertAlmostEqual(float(xe), float(xm), places=6)

    def test_first_event_is_unique_and_readable(self) -> None:
        cr = se.make_events(1, 28.0)[0]
        self.assertTrue(cr["first"])
        self.assertEqual(len(cr["events"]), 1)
        self.assertAlmostEqual(cr["events"][0]["c"], 0.62)

    def test_distinct_particles_per_crossing(self) -> None:
        for cr in se.make_events(3, 28.0):
            es = [ev["ie"] for ev in cr["events"]]
            ps = [ev["ip"] for ev in cr["events"]]
            self.assertEqual(len(set(es)), len(es))
            self.assertEqual(len(set(ps)), len(ps))


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
        src = (HERE / "scattering_experiment.py").read_text(encoding="utf-8")
        self.assertIn("load_config", src)
        for forbidden in ('"libx264", "-preset", "slow"', "(1280, 720)", "dpi = 100\n"):
            self.assertNotIn(forbidden, src)

    def test_set_overrides_a_value(self) -> None:
        out = subprocess.run([sys.executable, "-c", "import scattering_experiment as m; print(m.CFG.video.crf)", "--set", "video.crf=23"],
                             cwd=HERE, capture_output=True, text=True, check=True).stdout.strip()
        self.assertEqual(out, "23")


class FilmSmokeTest(unittest.TestCase):
    def test_a_snapshot_in_each_language(self) -> None:
        from PIL import Image
        with tempfile.TemporaryDirectory() as d:
            sizes = {}
            for lang in ("en", "ru"):
                out = Path(d) / f"s_{lang}.mp4"
                subprocess.run([sys.executable, "scattering_experiment.py", "--lang", lang, "--snapshot", "15", "--out", str(out)], cwd=HERE, check=True,
                               capture_output=True)
                sizes[lang] = Image.open(out.with_suffix(".png")).size
            self.assertEqual(sizes["en"], (se.CFG.video.width, se.CFG.video.height))
            self.assertEqual(sizes["en"], sizes["ru"])


if __name__ == "__main__":
    unittest.main()
