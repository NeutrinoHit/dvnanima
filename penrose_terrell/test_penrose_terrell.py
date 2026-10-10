from __future__ import annotations

import math
import tomllib
import unittest
from pathlib import Path

import numpy as np

import penrose_terrell as pt

HERE = Path(__file__).resolve().parent


class GeometryTest(unittest.TestCase):
    def test_football_is_a_truncated_icosahedron(self) -> None:
        self.assertEqual(len(pt.NORMALS), 32)
        self.assertEqual(int(pt.IS_PENTAGON.sum()), 12)
        np.testing.assert_allclose(np.linalg.norm(pt.NORMALS, axis=1), 1.0, atol=1e-9)

    def test_rest_frame_ball_is_an_ordinary_disc(self) -> None:
        xs = np.linspace(-1.2, 1.2, 241)
        X, Y = np.meshgrid(xs, xs)
        _, hit = pt.camera_image(X, Y, 0.0, 0.0)
        self.assertLess(np.mean(hit != (X ** 2 + Y ** 2 <= 1.0)), 1e-4)

    def test_camera_sees_a_round_ball_at_any_beta(self) -> None:
        # Penrose-Terrell: the outline stays a unit circle centred at x = beta * tau
        beta, tau = 0.99, 0.7
        xs = np.linspace(-3, 3, 1201)
        X, Y = np.meshgrid(xs, xs)
        _, hit = pt.camera_image(X, Y, tau, beta)
        r2 = (X - beta * tau) ** 2 + Y ** 2
        mismatch = np.mean(hit != (r2 <= 1.0))
        self.assertLess(mismatch, 2e-4)

    def test_snapshot_is_contracted_by_gamma(self) -> None:
        beta = 0.99
        gamma = 1 / math.sqrt(1 - beta ** 2)
        xs = np.linspace(-1.2, 1.2, 4801)
        _, hit = pt.camera_image(xs, np.zeros_like(xs), 0.0, beta, snapshot=True)
        self.assertAlmostEqual(xs[hit].max(), 1 / gamma, places=3)

    def test_apparent_rotation_angle_is_arcsin_beta(self) -> None:
        # body-frame direction seen at the image centre makes angle psi with the
        # line of sight, tan(psi) = gamma * beta, i.e. sin(psi) = beta
        beta = 0.99
        gamma = 1 / math.sqrt(1 - beta ** 2)
        w, y = 0.0, 0.0                      # x = beta * tau
        a, b, c = gamma ** 2, 0.0, gamma ** 2 * w * w + y * y - 1.0
        z = math.sqrt(-4 * a * c) / (2 * a)
        xb = gamma * (w - beta * z)
        psi = math.atan2(abs(xb), z)
        self.assertAlmostEqual(math.sin(psi), beta, places=12)


class ConfigTest(unittest.TestCase):
    def test_texts_have_the_same_keys_in_both_languages_and_balanced_formulas(self) -> None:
        texts = tomllib.loads((HERE / "texts.toml").read_text(encoding="utf-8"))
        self.assertEqual(set(texts), {"en", "ru"})
        self.assertEqual(set(texts["en"]), set(texts["ru"]))
        for lang, table in texts.items():
            for key, value in table.items():
                self.assertEqual(value.count("$") % 2, 0, f"{lang}.{key}: unbalanced $")
            self.assertIn("{beta}", table["sub_camera"])

    def test_decimal_comma_in_russian_formulas(self) -> None:
        self.assertEqual(pt.num(0.99, ".2f", "ru"), "0{,}99")
        self.assertEqual(pt.num(0.99, ".2f", "en"), "0.99")

    def test_frame_has_three_panels_and_a_header_in_both_languages(self) -> None:
        size = 60
        for lang in ("en", "ru"):
            img = pt.render_frame(0.0, 0.99, size, pt.CFG.scene.span, lang)
            self.assertEqual(img.size, (3 * size + 2 * pt.CFG.layout.gap, size + int(pt.CFG.layout.header * size)))
        en = np.asarray(pt.render_header("en", 0.99, size)).astype(int)
        ru = np.asarray(pt.render_header("ru", 0.99, size)).astype(int)
        self.assertGreater(np.abs(en - ru).max(), 0)         # the titles differ

    def test_configured_video_settings(self) -> None:
        v = pt.CFG.video
        self.assertGreater(v.panel_size, v.preview_panel_size)
        self.assertGreater(v.fps, 0)
        self.assertTrue(0.0 < pt.CFG.scene.beta < 1.0)

    def test_the_script_has_no_hard_coded_video_settings(self) -> None:
        src = (HERE / "penrose_terrell.py").read_text(encoding="utf-8")
        self.assertIn("load_config", src)
        for forbidden in ('"libx264", "-preset", "slow"', "(1280, 720)", "dpi = 100\n"):
            self.assertNotIn(forbidden, src)


if __name__ == "__main__":
    unittest.main()
