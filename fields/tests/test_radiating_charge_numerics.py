from __future__ import annotations

import unittest

import numpy as np

from fields.radiating_charge.data_io import build_dataset
from fields.radiating_charge.numerics import (
    CoulombTrajectoryConfig,
    GridConfig,
    ObservableConfig,
    OscillatorTrajectoryConfig,
    RadiatingChargeConfig,
    RenderConfig,
    SourceConfig,
    TimeConfig,
    TrajectoryConfig,
    TrajectorySolution,
    electric_field_components,
    solve_retarded_times,
    solve_trajectory,
    evaluate_lw_fields,
)


def _uniform_trajectory(vx: float = 0.3) -> TrajectorySolution:
    t_nodes = np.linspace(-8.0, 8.0, 2001)
    x = vx * t_nodes
    y = np.zeros_like(t_nodes)
    z = np.zeros_like(t_nodes)
    r_nodes = np.column_stack([x, y, z])
    v_nodes = np.column_stack([np.full_like(t_nodes, vx), np.zeros_like(t_nodes), np.zeros_like(t_nodes)])
    a_nodes = np.zeros_like(v_nodes)
    return TrajectorySolution(t_nodes=t_nodes, r_nodes=r_nodes, v_nodes=v_nodes, a_nodes=a_nodes)


def _base_cfg(model: str) -> RadiatingChargeConfig:
    return RadiatingChargeConfig(
        grid=GridConfig(nx=36, ny=28, lx=12.0, ly=9.0, z_obs=0.25, mask_radius=0.06),
        time=TimeConfig(t_start=-2.5, t_end=2.5, trajectory_dt=0.004, animation_num_samples=31),
        source=SourceConfig(charge=1.0, c=6.0, eps0=1.0),
        observables=ObservableConfig(lower_surface="e_mag", upper_surface="s_mag", keys=("e_mag", "bz", "s_mag")),
        render=RenderConfig(
            lower_height_target=1.0,
            upper_height_target=0.9,
            lower_visual_gain=1.0,
            upper_visual_gain=1.0,
            upper_plane_shift=2.0,
            display_transform="signed_log",
            playback_time_scale=1.0,
        ),
        trajectory=TrajectoryConfig(model=model),
        coulomb=CoulombTrajectoryConfig(
            mass=1.0,
            center_charge=2.6,
            softening=0.2,
            center_x=0.0,
            center_y=0.0,
            center_z=0.0,
            x0=-4.0,
            y0=1.1,
            z0=0.0,
            vx0=1.15,
            vy0=0.0,
            vz0=0.0,
        ),
        oscillator=OscillatorTrajectoryConfig(
            mass=1.0,
            omega0=1.3,
            gamma=0.0,
            center_x=0.0,
            center_y=0.0,
            center_z=0.0,
            x0=1.7,
            y0=0.0,
            z0=0.0,
            vx0=0.15,
            vy0=0.0,
            vz0=0.0,
        ),
    )


class RadiatingChargeNumericsTests(unittest.TestCase):
    def test_retarded_time_residual(self) -> None:
        traj = _uniform_trajectory(vx=0.25)
        x_axis = np.linspace(-2.0, 2.0, 9)
        y_axis = np.linspace(-1.5, 1.5, 7)
        x_grid, y_grid = np.meshgrid(x_axis, y_axis, indexing="ij")

        t_obs = 1.2
        t_ret, residual = solve_retarded_times(
            traj,
            x_grid,
            y_grid,
            z_obs=0.2,
            t_obs=t_obs,
            c=1.0,
        )
        self.assertLess(float(np.max(np.abs(residual))), 1e-6)

        r_ret, _, _ = traj.eval(t_ret)
        distance = np.sqrt((x_grid - r_ret[..., 0]) ** 2 + (y_grid - r_ret[..., 1]) ** 2 + (0.2 - r_ret[..., 2]) ** 2)
        eq_residual = t_obs - t_ret - distance
        self.assertLess(float(np.max(np.abs(eq_residual))), 1e-6)

    def test_b_equals_n_cross_e_over_c(self) -> None:
        traj = _uniform_trajectory(vx=0.2)
        x_axis = np.linspace(-2.0, 2.0, 11)
        y_axis = np.linspace(-1.5, 1.5, 9)
        x_grid, y_grid = np.meshgrid(x_axis, y_axis, indexing="ij")

        t_ret, _ = solve_retarded_times(traj, x_grid, y_grid, z_obs=0.35, t_obs=1.1, c=1.0)
        fields = evaluate_lw_fields(
            traj,
            t_ret,
            x_grid,
            y_grid,
            z_obs=0.35,
            charge=1.0,
            c=1.0,
            eps0=1.0,
            mask_radius=0.0,
        )

        r_ret, _, _ = traj.eval(t_ret)
        R = np.stack((x_grid - r_ret[..., 0], y_grid - r_ret[..., 1], 0.35 - r_ret[..., 2]), axis=-1)
        n = R / np.maximum(np.linalg.norm(R, axis=-1, keepdims=True), 1e-12)
        e = np.stack((fields["ex"], fields["ey"], fields["ez"]), axis=-1)
        b_ref = np.cross(n, e) / 1.0
        b = np.stack((fields["bx"], fields["by"], fields["bz"]), axis=-1)

        self.assertLess(float(np.max(np.abs(b - b_ref))), 2e-6)

    def test_uniform_motion_has_zero_acceleration_component(self) -> None:
        traj = _uniform_trajectory(vx=0.33)
        x_axis = np.linspace(-1.8, 1.8, 10)
        y_axis = np.linspace(-1.3, 1.3, 8)
        x_grid, y_grid = np.meshgrid(x_axis, y_axis, indexing="ij")

        t_ret, _ = solve_retarded_times(traj, x_grid, y_grid, z_obs=0.15, t_obs=0.8, c=1.0)
        _, e_acc = electric_field_components(
            traj,
            t_ret,
            x_grid,
            y_grid,
            z_obs=0.15,
            charge=1.0,
            c=1.0,
            eps0=1.0,
        )
        self.assertLess(float(np.max(np.abs(e_acc))), 1e-8)

    def test_coulomb_trajectory_conserves_energy_and_angular_momentum(self) -> None:
        cfg = _base_cfg(model="coulomb")
        traj = solve_trajectory(cfg, t_min=-3.0, t_max=3.0)

        rel = traj.r_nodes - np.array([cfg.coulomb.center_x, cfg.coulomb.center_y, cfg.coulomb.center_z], dtype=float)
        r_soft = np.sqrt(np.sum(rel * rel, axis=1) + cfg.coulomb.softening**2)
        speed2 = np.sum(traj.v_nodes * traj.v_nodes, axis=1)

        k_e = 1.0 / (4.0 * np.pi * cfg.source.eps0)
        kappa = k_e * cfg.source.charge * cfg.coulomb.center_charge

        energy = 0.5 * cfg.coulomb.mass * speed2 + kappa / r_soft
        lz = cfg.coulomb.mass * (rel[:, 0] * traj.v_nodes[:, 1] - rel[:, 1] * traj.v_nodes[:, 0])

        energy_spread = (float(np.max(energy)) - float(np.min(energy))) / max(float(np.mean(np.abs(energy))), 1e-12)
        lz_spread = (float(np.max(lz)) - float(np.min(lz))) / max(float(np.mean(np.abs(lz))), 1e-12)

        self.assertLess(energy_spread, 5e-3)
        self.assertLess(lz_spread, 5e-3)

    def test_oscillator_matches_analytic_solution(self) -> None:
        cfg = _base_cfg(model="oscillator")
        traj = solve_trajectory(cfg, t_min=cfg.time.t_start, t_max=cfg.time.t_end)

        t = traj.t_nodes
        tau = t - cfg.time.t_start
        omega = cfg.oscillator.omega0
        x0 = cfg.oscillator.x0
        vx0 = cfg.oscillator.vx0
        x_ref = x0 * np.cos(omega * tau) + (vx0 / omega) * np.sin(omega * tau)

        max_abs_error = float(np.max(np.abs(traj.r_nodes[:, 0] - x_ref)))
        self.assertLess(max_abs_error, 8e-3)

    def test_trajectory_anchor_at_t_start(self) -> None:
        cfg = _base_cfg(model="coulomb")
        traj = solve_trajectory(cfg, t_min=cfg.time.t_start - 2.0, t_max=cfg.time.t_end)
        r0, v0, _ = traj.eval(cfg.time.t_start)
        self.assertLess(float(np.max(np.abs(r0 - np.array([cfg.coulomb.x0, cfg.coulomb.y0, cfg.coulomb.z0])))), 1e-8)
        self.assertLess(float(np.max(np.abs(v0 - np.array([cfg.coulomb.vx0, cfg.coulomb.vy0, cfg.coulomb.vz0])))), 1e-8)

    def test_radiation_channel_nonzero_for_accelerated_motion(self) -> None:
        cfg = _base_cfg(model="coulomb")
        cfg = RadiatingChargeConfig(
            grid=GridConfig(nx=18, ny=14, lx=7.0, ly=5.0, z_obs=0.2, mask_radius=0.05),
            time=TimeConfig(t_start=-1.3, t_end=1.3, trajectory_dt=0.005, animation_num_samples=11),
            source=cfg.source,
            observables=ObservableConfig(lower_surface="e_mag", upper_surface="s_rad_mag", keys=("e_rad_mag", "s_rad_mag")),
            render=cfg.render,
            trajectory=cfg.trajectory,
            coulomb=cfg.coulomb,
            oscillator=cfg.oscillator,
        )
        dataset = build_dataset(cfg)
        self.assertGreater(float(np.max(dataset["e_rad_mag"])), 0.0)
        self.assertGreater(float(np.percentile(dataset["e_rad_mag"], 99)), 0.0)
        self.assertGreater(float(np.max(dataset["s_rad_mag"])), 0.0)

    def test_dataset_smoke(self) -> None:
        cfg = _base_cfg(model="oscillator")
        cfg = RadiatingChargeConfig(
            grid=GridConfig(nx=20, ny=16, lx=8.0, ly=6.0, z_obs=0.2, mask_radius=0.05),
            time=TimeConfig(t_start=-1.5, t_end=1.5, trajectory_dt=0.006, animation_num_samples=12),
            source=cfg.source,
            observables=ObservableConfig(lower_surface="e_mag", upper_surface="s_rad_mag", keys=("e_mag", "bz", "s_mag", "e_rad_mag", "s_rad_mag")),
            render=cfg.render,
            trajectory=cfg.trajectory,
            coulomb=cfg.coulomb,
            oscillator=cfg.oscillator,
        )

        dataset = build_dataset(cfg)
        self.assertEqual(dataset["e_mag"].shape, (12, 20, 16))
        self.assertEqual(dataset["bz"].shape, (12, 20, 16))
        self.assertEqual(dataset["e_rad_mag"].shape, (12, 20, 16))
        self.assertGreater(float(np.max(dataset["e_rad_mag"])), 0.0)
        self.assertEqual(dataset["source_centers"].shape, (12, 1, 2))
        self.assertTrue(np.isfinite(dataset["retarded_residual_max"]).all())


if __name__ == "__main__":
    unittest.main()
