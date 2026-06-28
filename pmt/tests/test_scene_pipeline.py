from __future__ import annotations

from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from pmt.config.loader import merge_base_and_scene
from pmt.io.hdf5_io import load_scene_hdf5
from pmt.io.paths import resolve_single_output_paths


PMT_DIR = Path(__file__).resolve().parents[1]
MAIN_PATH = PMT_DIR / "main.py"


def _write_base_config(path: Path) -> None:
    path.write_text(
        """
[scene]
name = "base_scene"
title = "Base"
kind = "surface_to_plate"
physics_mode = "numerical"
launch_mode = "surface"

[grid]
nx = 80
ny = 64
lx_cm = 40.0
ly_cm = 28.0

[time]
dt = 2.0e-11
steps = 220
integration_substeps = 1
bz_t = 0.0

[electrons]
count = 28
initial_energy_ev = 0.35
initial_energy_spread_ev = 0.05
angle_spread_deg = 4.0
seed = 11

[electrodes]
background_voltage = 0.0
cathode_voltage = 0.0
anode_voltage = 600.0
focus_voltage = 300.0

[geometry]
cathode_shape = "line"
line_cathode_x_cm = -14.0
line_cathode_height_cm = 16.0
line_cathode_thickness_cm = 0.5
receiver_kind = "point"
receiver_point_x_cm = 10.0
receiver_point_y_cm = 0.0
receiver_radius_cm = 0.7
launch_point_x_cm = -10.0
launch_point_y_cm = 0.0

[central]
center_x_cm = 10.0
center_y_cm = 0.0
kappa = 25.0
softening_cm = 0.3

[render]
figure_width = 6.5
figure_height = 4.2
dpi = 90
show_field_lines = false
max_trajectories = 30

[solver]
sor_omega = 1.85
sor_tolerance = 1.0e-6
sor_max_iterations = 3500
""".strip()
        + "\n",
        encoding="utf-8",
    )


def _write_scene_config(path: Path) -> None:
    path.write_text(
        """
[scene]
name = "cli_single"
title = "CLI Single"
kind = "point_field"
physics_mode = "central"
launch_mode = "point"

[electrons]
count = 18
launch_angle_deg = 12.0

[electrodes]
anode_voltage = 900.0

[geometry]
launch_point_x_cm = -12.0
launch_point_y_cm = -1.0
""".strip()
        + "\n",
        encoding="utf-8",
    )


def _run_cli(args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(MAIN_PATH), *args],
        cwd=str(PMT_DIR),
        text=True,
        capture_output=True,
        check=False,
    )


class ScenePipelineTests(unittest.TestCase):
    def test_merge_base_and_scene(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            base = tmp_path / "base.toml"
            scene = tmp_path / "scene.toml"
            _write_base_config(base)
            _write_scene_config(scene)

            merged = merge_base_and_scene(base, scene)

            self.assertEqual(merged["scene"]["name"], "cli_single")
            self.assertEqual(merged["scene"]["physics_mode"], "central")
            self.assertEqual(merged["electrodes"]["anode_voltage"], 900.0)
            self.assertEqual(merged["electrodes"]["cathode_voltage"], 0.0)

    def test_output_filename_generation(self) -> None:
        png_path, h5_path = resolve_single_output_paths(
            scene_name="surface_to_plate",
            output=None,
            output_dir=Path("out"),
            hdf5_output=None,
            hdf5_dir=None,
            save_hdf5=True,
        )
        self.assertEqual(png_path, Path("out") / "surface_to_plate.png")
        self.assertEqual(h5_path, Path("out") / "surface_to_plate.h5")

        png_explicit, h5_explicit = resolve_single_output_paths(
            scene_name="surface_to_plate",
            output=Path("custom.png"),
            output_dir=Path("out"),
            hdf5_output=Path("custom.h5"),
            hdf5_dir=None,
            save_hdf5=False,
        )
        self.assertEqual(png_explicit, Path("custom.png"))
        self.assertEqual(h5_explicit, Path("custom.h5"))

    def test_cli_single_scene_and_png_save(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            base = tmp_path / "base.toml"
            scene = tmp_path / "scene.toml"
            out_png = tmp_path / "scene.png"
            _write_base_config(base)
            _write_scene_config(scene)

            proc = _run_cli(
                [
                    "--base",
                    str(base),
                    "--scene",
                    str(scene),
                    "--output",
                    str(out_png),
                ]
            )

            self.assertEqual(proc.returncode, 0, msg=proc.stderr)
            self.assertTrue(out_png.exists())
            self.assertIn("[cli_single]", proc.stdout)

    def test_cli_hdf5_save(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            base = tmp_path / "base.toml"
            scene = tmp_path / "scene.toml"
            out_png = tmp_path / "scene.png"
            out_h5 = tmp_path / "scene.h5"
            _write_base_config(base)
            _write_scene_config(scene)

            proc = _run_cli(
                [
                    "--base",
                    str(base),
                    "--scene",
                    str(scene),
                    "--output",
                    str(out_png),
                    "--hdf5",
                    str(out_h5),
                ]
            )

            self.assertEqual(proc.returncode, 0, msg=proc.stderr)
            self.assertTrue(out_png.exists())
            self.assertTrue(out_h5.exists())

            data = load_scene_hdf5(out_h5)
            self.assertIn("metadata", data)
            self.assertIn("positions_m", data)
            self.assertEqual(data["positions_m"].ndim, 3)

    def test_cli_dry_run(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            base = tmp_path / "base.toml"
            scene = tmp_path / "scene.toml"
            out_png = tmp_path / "scene.png"
            _write_base_config(base)
            _write_scene_config(scene)

            proc = _run_cli(
                [
                    "--base",
                    str(base),
                    "--scene",
                    str(scene),
                    "--output",
                    str(out_png),
                    "--dry-run",
                ]
            )

            self.assertEqual(proc.returncode, 0, msg=proc.stderr)
            self.assertIn("DRY RUN", proc.stdout)
            self.assertIn('"scene"', proc.stdout)
            self.assertIn("cli_single", proc.stdout)
            self.assertFalse(out_png.exists())


if __name__ == "__main__":
    unittest.main()
