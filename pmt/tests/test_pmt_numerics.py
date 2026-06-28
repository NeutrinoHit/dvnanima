from __future__ import annotations

import tempfile
import unittest

import numpy as np

from pmt.data_io import build_dataset, load_dataset, save_dataset
from pmt.numerics import (
    GridConfig,
    ParticleConfig,
    SimulationConfig,
    bilinear_interpolate_scalar,
    boris_push_2d,
    build_cathode_dynode_geometry,
    compute_electric_field,
    make_grid,
    solve_laplace_red_black_sor,
    trace_particles,
)


def _test_config() -> SimulationConfig:
    return SimulationConfig(
        grid=GridConfig(nx=64, ny=48, lx=0.024, ly=0.016),
        particles=ParticleConfig(
            count=32,
            steps=220,
            dt=6.0e-12,
            mass=9.1093837015e-31,
            charge=-1.602176634e-19,
            initial_energy_ev=0.30,
            initial_energy_spread_ev=0.08,
            angle_spread_deg=8.0,
            emission_jitter_y=6.0e-5,
            integration_substeps=1,
        ),
        cathode_voltage=0.0,
        dynode_voltage=280.0,
        background_voltage=0.0,
        cathode_x=-0.009,
        cathode_height=0.008,
        cathode_thickness=8.0e-4,
        dynode_center_x=0.006,
        dynode_center_y=0.001,
        dynode_length=0.006,
        dynode_thickness=9.0e-4,
        dynode_angle_deg=-20.0,
        bz_t=0.0,
        sor_omega=1.90,
        sor_tolerance=1.0e-8,
        sor_max_iterations=9000,
        seed=7,
    )


class PMTNumericsTests(unittest.TestCase):
    def test_bilinear_interpolation_exact_for_plane(self) -> None:
        x_axis = np.linspace(-2.0, 2.0, 41)
        y_axis = np.linspace(-1.5, 1.5, 33)
        x_grid, y_grid = np.meshgrid(x_axis, y_axis, indexing="ij")

        a, b, c = 2.3, -1.7, 0.9
        field = a * x_grid + b * y_grid + c

        rng = np.random.default_rng(42)
        x = rng.uniform(float(x_axis[0]), float(x_axis[-1]), size=300)
        y = rng.uniform(float(y_axis[0]), float(y_axis[-1]), size=300)

        interp = bilinear_interpolate_scalar(field, x, y, x_axis, y_axis)
        ref = a * x + b * y + c
        self.assertLess(float(np.max(np.abs(interp - ref))), 1e-12)

    def test_compute_electric_field_from_linear_potential(self) -> None:
        x_axis = np.linspace(-1.0, 1.0, 47)
        y_axis = np.linspace(-0.7, 0.7, 39)
        x_grid, y_grid = np.meshgrid(x_axis, y_axis, indexing="ij")
        dx = float(x_axis[1] - x_axis[0])
        dy = float(y_axis[1] - y_axis[0])

        phi = 3.2 * x_grid - 4.1 * y_grid + 0.4
        ex, ey = compute_electric_field(phi, dx=dx, dy=dy)

        self.assertLess(float(np.max(np.abs(ex + 3.2))), 1e-12)
        self.assertLess(float(np.max(np.abs(ey - 4.1))), 1e-12)

    def test_laplace_solver_respects_fixed_values(self) -> None:
        cfg = _test_config()
        _, _, x_grid, y_grid, dx, dy = make_grid(cfg)
        geometry = build_cathode_dynode_geometry(cfg, x_grid, y_grid, dx)

        phi, iterations, residual = solve_laplace_red_black_sor(
            fixed_mask=geometry.fixed_mask,
            fixed_values=geometry.fixed_values,
            dx=dx,
            dy=dy,
            omega=cfg.sor_omega,
            tolerance=cfg.sor_tolerance,
            max_iterations=cfg.sor_max_iterations,
        )

        self.assertGreater(iterations, 0)
        self.assertLess(residual, 1e-6)
        self.assertLess(float(np.max(np.abs(phi[geometry.fixed_mask] - geometry.fixed_values[geometry.fixed_mask]))), 1e-13)

    def test_boris_bz_rotation_preserves_speed_without_electric_field(self) -> None:
        rng = np.random.default_rng(123)
        velocity = rng.normal(0.0, 2.0, size=(128, 2))
        electric = np.zeros_like(velocity)

        speed0 = np.linalg.norm(velocity, axis=1)
        v = velocity
        for _ in range(200):
            v = boris_push_2d(v, electric, bz_t=0.75, q_over_m=-1.3, dt=0.04)
        speed1 = np.linalg.norm(v, axis=1)

        rel_err = np.max(np.abs(speed1 - speed0) / np.maximum(speed0, 1e-12))
        self.assertLess(float(rel_err), 2e-12)

    def test_trace_particles_straight_line_for_zero_fields(self) -> None:
        x_axis = np.linspace(-10.0, 10.0, 201)
        y_axis = np.linspace(-10.0, 10.0, 161)
        ex = np.zeros((x_axis.size, y_axis.size), dtype=float)
        ey = np.zeros_like(ex)

        positions0 = np.array([[-1.0, -0.5], [0.3, 0.9]], dtype=float)
        velocities0 = np.array([[0.7, -0.2], [-0.4, 0.1]], dtype=float)

        tracks = trace_particles(
            x_axis=x_axis,
            y_axis=y_axis,
            ex=ex,
            ey=ey,
            dynode_mask=np.zeros_like(ex, dtype=bool),
            cathode_mask=np.zeros_like(ex, dtype=bool),
            positions0=positions0,
            velocities0=velocities0,
            steps=40,
            dt=0.05,
            charge=-1.0,
            mass=1.0,
            bz_t=0.0,
        )

        t = tracks["time_axis"][:, None, None]
        ref = positions0[None, :, :] + t * velocities0[None, :, :]

        self.assertEqual(tracks["positions"].shape, ref.shape)
        self.assertLess(float(np.max(np.abs(tracks["positions"] - ref))), 1e-12)
        self.assertTrue(bool(np.all(tracks["alive"])))

    def test_hdf5_round_trip(self) -> None:
        cfg = _test_config()
        cfg = SimulationConfig(
            grid=GridConfig(nx=44, ny=34, lx=cfg.grid.lx, ly=cfg.grid.ly),
            particles=ParticleConfig(
                count=10,
                steps=90,
                dt=cfg.particles.dt,
                mass=cfg.particles.mass,
                charge=cfg.particles.charge,
                initial_energy_ev=cfg.particles.initial_energy_ev,
                initial_energy_spread_ev=cfg.particles.initial_energy_spread_ev,
                angle_spread_deg=cfg.particles.angle_spread_deg,
                emission_jitter_y=cfg.particles.emission_jitter_y,
                integration_substeps=cfg.particles.integration_substeps,
            ),
            cathode_voltage=cfg.cathode_voltage,
            dynode_voltage=cfg.dynode_voltage,
            background_voltage=cfg.background_voltage,
            cathode_x=cfg.cathode_x,
            cathode_height=cfg.cathode_height,
            cathode_thickness=cfg.cathode_thickness,
            dynode_center_x=cfg.dynode_center_x,
            dynode_center_y=cfg.dynode_center_y,
            dynode_length=cfg.dynode_length,
            dynode_thickness=cfg.dynode_thickness,
            dynode_angle_deg=cfg.dynode_angle_deg,
            bz_t=cfg.bz_t,
            sor_omega=cfg.sor_omega,
            sor_tolerance=cfg.sor_tolerance,
            sor_max_iterations=6000,
            seed=cfg.seed,
        )

        dataset = build_dataset(cfg, storage_dtype=np.float32)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = save_dataset(f"{tmpdir}/pmt_roundtrip.h5", dataset)
            loaded = load_dataset(path)

        self.assertEqual(loaded["metadata"]["dataset_version"], dataset["metadata"]["dataset_version"])
        self.assertEqual(tuple(loaded["particle_positions"].shape), tuple(dataset["particle_positions"].shape))
        np.testing.assert_allclose(loaded["potential"], dataset["potential"], rtol=0.0, atol=0.0)
        np.testing.assert_array_equal(loaded["electrode_id"], dataset["electrode_id"])


if __name__ == "__main__":
    unittest.main()
