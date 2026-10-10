"""Tests of the shared configuration loader and of the configuration files of the film projects.

    python -m unittest test_dvconfig
"""

from __future__ import annotations

import sys
import tempfile
import tomllib
import unittest
from pathlib import Path

import dvconfig

HERE = Path(__file__).resolve().parent
FILMS = ["coupled_chain", "coupled_lattice_2d", "coupled_lattice_3d", "sm_fields", "pi0_double_dalitz", "free_wavepacket",
         "cherenkov_radiation", "pmt_multiplication", "thomson_tube", "simultaneity", "oam_beams", "circular_polarization",
         "electron_kick", "spinor_mobius", "hulse_taylor", "path_integral", "scattering_experiment", "running_charge", "rutherford",
         "meissner_photon_mass", "chiral_fermion_mass", "z_decay_asymmetry", "air_shower_cascade", "higgs_ff_spin",
         "quantum_tunneling", "kaon_oscillations", "chiral_anomaly", "gauge_principle", "attraction_repulsion", "wormhole_camera", "earth_shield"]
# a frame made of panels, not 16:9: only the text and the script rules apply (its own video section is tested in its own file)
PANEL_FILMS = ["penrose_terrell"]


def with_argv(args: list[str]):
    class Ctx:
        def __enter__(self):
            self.old = sys.argv
            sys.argv = ["film.py", *args]

        def __exit__(self, *exc):
            sys.argv = self.old
    return Ctx()


class LoaderTest(unittest.TestCase):
    def setUp(self) -> None:
        self.dir = Path(tempfile.mkdtemp())
        (self.dir / "config.toml").write_text("[a]\nx = 1\ny = [1, 2]\n[a.b]\nz = 'w'\n", encoding="utf-8")
        (self.dir / "other.toml").write_text("[a]\nx = 7\n", encoding="utf-8")
        (self.dir / "texts.toml").write_text("[en]\nk = 'hello'\n[ru]\nk = 'привет'\n", encoding="utf-8")

    def test_nested_attribute_access(self) -> None:
        with with_argv([]):
            c = dvconfig.load_config(self.dir)
        self.assertEqual((c.a.x, c.a.y, c.a.b.z), (1, [1, 2], "w"))

    def test_a_missing_key_is_an_error_with_its_path(self) -> None:
        with with_argv([]):
            c = dvconfig.load_config(self.dir)
        with self.assertRaises(KeyError) as cm:
            c.a.b.missing
        self.assertIn("a.b.missing", str(cm.exception))

    def test_set_overrides_a_value(self) -> None:
        with with_argv(["--set", "a.x=5", "--set", "a.y=[3,4]", "--set", "a.b.z='q'"]):
            c = dvconfig.load_config(self.dir)
        self.assertEqual((c.a.x, c.a.y, c.a.b.z), (5, [3, 4], "q"))

    def test_set_of_a_misspelt_key_is_an_error(self) -> None:
        with with_argv(["--set", "a.xx=5"]):
            with self.assertRaises(KeyError):
                dvconfig.load_config(self.dir)

    def test_config_flag_loads_another_file(self) -> None:
        with with_argv(["--config", str(self.dir / "other.toml")]):
            c = dvconfig.load_config(self.dir)
        self.assertEqual(c.a.x, 7)

    def test_the_configuration_is_read_only(self) -> None:
        with with_argv([]):
            c = dvconfig.load_config(self.dir)
        with self.assertRaises(AttributeError):
            c.a = 3

    def test_texts_by_language(self) -> None:
        self.assertEqual(dvconfig.load_texts(self.dir, "ru")["k"], "привет")


class FilmConfigTest(unittest.TestCase):
    def test_every_film_has_a_config_and_texts_in_both_languages(self) -> None:
        for film in FILMS + PANEL_FILMS:
            with self.subTest(film=film):
                d = HERE / film
                cfg = tomllib.loads((d / "config.toml").read_text(encoding="utf-8"))
                texts = tomllib.loads((d / "texts.toml").read_text(encoding="utf-8"))
                self.assertEqual(set(texts), {"en", "ru"})
                self.assertEqual(set(texts["en"]), set(texts["ru"]), "the two languages must have the same keys")
                for section in ("video", "style"):
                    self.assertIn(section, cfg)

    def test_the_video_section_is_complete_and_sane(self) -> None:
        keys = {"width", "height", "fps", "dpi", "crf", "preset", "preview_width", "preview_height", "preview_fps", "fade_s", "reference_height"}
        for film in FILMS:
            with self.subTest(film=film):
                v = tomllib.loads((HERE / film / "config.toml").read_text(encoding="utf-8"))["video"]
                self.assertTrue(keys <= set(v), keys - set(v))
                self.assertGreater(v["width"], v["preview_width"])
                self.assertEqual(v["width"] * 9, v["height"] * 16)          # 16:9
                self.assertGreater(v["fps"], 0)

    def test_a_film_script_reads_its_numbers_from_the_config(self) -> None:
        """No film script defines its own frame size, frame rate or encoder quality any more."""
        for film in FILMS + PANEL_FILMS:
            with self.subTest(film=film):
                src = (HERE / film / f"{film}.py").read_text(encoding="utf-8")
                self.assertIn("load_config", src)
                for forbidden in ('"libx264", "-preset", "slow"', "(1280, 720)", "dpi = 100\n"):
                    self.assertNotIn(forbidden, src)


if __name__ == "__main__":
    unittest.main()
