from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mac_e_filter.dynamics import (  # noqa: E402
    ELECTRON,
    IntegrationSettings,
    adiabatic_electrostatic_transport_along_path,
    adiabatic_pitch_angle_deg,
    adiabatic_transport_along_path,
    decompose_velocity,
    integrate_relativistic_magnetic_trajectory,
    integrate_relativistic_electromagnetic_trajectory,
    normalized_momentum_from_angles,
    uniform_field_exact_solution,
)
from mac_e_filter.electrostatics import (  # noqa: E402
    FiniteCylindricalElectrodePotential,
)
from mac_e_filter.fields import UniformAxialField  # noqa: E402


class RelativisticDynamicsTests(unittest.TestCase):
    def test_full_lorentz_static_e_and_b_conserves_total_energy(
        self,
    ) -> None:
        class UniformAxialPotential:
            field_v_m = 100.0

            def potential_cylindrical(self, rho_m, z_m):
                del rho_m
                return -self.field_v_m * np.asarray(z_m, dtype=float)

            def electric_field_cylindrical(self, rho_m, z_m):
                shape = np.broadcast(
                    np.asarray(rho_m),
                    np.asarray(z_m),
                ).shape
                return np.zeros(shape), np.full(shape, self.field_v_m)

        magnetic_field = UniformAxialField("uniform test field", 1.0e-3)
        electric_potential = UniformAxialPotential()
        solution = integrate_relativistic_electromagnetic_trajectory(
            magnetic_field,
            electric_potential,
            [0.0, 0.0, 0.0],
            normalized_momentum_from_angles(18_600.0, 20.0),
            lower_stop_z_m=-0.01,
            upper_stop_z_m=1.0,
            maximum_time_s=1.0e-7,
            settings=IntegrationSettings(
                field_bound_t=1.0e-3,
                output_samples=501,
            ),
        )
        self.assertEqual(
            solution.solver_statistics["exit_kind"],
            "transmitted",
        )
        potential = electric_potential.potential_cylindrical(
            np.hypot(
                solution.position_m[:, 0],
                solution.position_m[:, 1],
            ),
            solution.position_m[:, 2],
        )
        total_energy_ev = (
            solution.diagnostics.gamma * ELECTRON.mass_kg * 299_792_458.0**2
            + ELECTRON.charge_c * potential
        ) / 1.602_176_634e-19
        self.assertLess(np.ptp(total_energy_ev), 5.0e-8)

    def test_cylindrical_potential_and_field_are_consistent(self) -> None:
        model = FiniteCylindricalElectrodePotential(
            radius_m=4.9,
            length_m=23.28,
            boundary_potential_v=-18_700.0,
            quadrature_order=256,
            k_max_inv_m=8.0,
        )
        center = float(model.potential_cylindrical(0.0, 0.0))
        left = float(model.potential_cylindrical(0.0, -4.0))
        right = float(model.potential_cylindrical(0.0, 4.0))
        self.assertLess(center, -18_000.0)
        self.assertAlmostEqual(left, right, places=9)

        rho = 1.2
        z = -5.0
        step = 1.0e-4
        e_rho, e_z = model.electric_field_cylindrical(rho, z)
        numerical_e_rho = -(
            model.potential_cylindrical(rho + step, z)
            - model.potential_cylindrical(rho - step, z)
        ) / (2.0 * step)
        numerical_e_z = -(
            model.potential_cylindrical(rho, z + step)
            - model.potential_cylindrical(rho, z - step)
        ) / (2.0 * step)
        self.assertAlmostEqual(
            float(e_rho),
            float(numerical_e_rho),
            delta=2.0e-5,
        )
        self.assertAlmostEqual(
            float(e_z),
            float(numerical_e_z),
            delta=2.0e-5,
        )

    def test_electrostatic_adiabatic_transport_transmits_and_reflects(
        self,
    ) -> None:
        z = np.linspace(0.0, 2.0, 4001)
        path = np.column_stack(
            (np.zeros_like(z), np.zeros_like(z), z)
        )
        field = np.column_stack(
            (np.zeros_like(z), np.zeros_like(z), np.ones_like(z))
        )

        transmitted = adiabatic_electrostatic_transport_along_path(
            path,
            field,
            -9_000.0 * np.sin(0.5 * np.pi * z) ** 2,
            kinetic_energy_ev=10_000.0,
            initial_pitch_deg=10.0,
        )
        self.assertEqual(transmitted.outcome, "transmitted")
        self.assertIsNone(transmitted.turning_position_m)
        self.assertLess(np.ptp(transmitted.total_energy_ev), 1.0e-9)
        self.assertLess(
            np.ptp(transmitted.magnetic_moment_j_per_t)
            / transmitted.magnetic_moment_j_per_t[0],
            1.0e-13,
        )

        reflected = adiabatic_electrostatic_transport_along_path(
            path,
            field,
            -12_000.0 * np.sin(0.5 * np.pi * z) ** 2,
            kinetic_energy_ev=10_000.0,
            initial_pitch_deg=10.0,
        )
        self.assertEqual(reflected.outcome, "reflected")
        self.assertIsNotNone(reflected.turning_position_m)
        self.assertAlmostEqual(
            float(reflected.path_position_m[-1, 2]),
            0.0,
            places=12,
        )
        self.assertLess(
            np.ptp(reflected.total_energy_ev),
            1.0e-9,
        )

    def test_adiabatic_transport_conserves_invariant(self) -> None:
        z = np.linspace(0.0, 2.0, 1001)
        field_magnitude = 3.6 * np.exp(-4.0 * z)
        field = np.column_stack(
            (
                np.zeros_like(z),
                np.zeros_like(z),
                field_magnitude,
            )
        )
        path = np.column_stack(
            (np.zeros_like(z), np.zeros_like(z), z)
        )
        solution = adiabatic_transport_along_path(
            path,
            field,
            kinetic_energy_ev=18_600.0,
            initial_pitch_deg=50.0,
        )

        invariant = (
            solution.perpendicular_speed_m_s**2
            / solution.magnetic_field_magnitude_t
        )
        np.testing.assert_allclose(
            invariant,
            invariant[0],
            rtol=5e-15,
        )
        self.assertLess(solution.pitch_angle_deg[-1], 1.0)
        self.assertGreater(
            solution.parallel_speed_m_s[-1],
            solution.parallel_speed_m_s[0],
        )
        self.assertGreater(solution.time_s[-1], 0.0)
        self.assertGreater(solution.cumulative_turns[-1], 0.0)

    def test_velocity_decomposition(self) -> None:
        velocity = np.array([3.0, 4.0, 12.0])
        parallel, perpendicular = decompose_velocity(
            velocity,
            np.array([0.0, 0.0, -2.0]),
        )
        np.testing.assert_allclose(parallel, [0.0, 0.0, 12.0])
        np.testing.assert_allclose(perpendicular, [3.0, 4.0, 0.0])
        self.assertAlmostEqual(float(np.dot(parallel, perpendicular)), 0.0)
        self.assertAlmostEqual(
            float(np.linalg.norm(velocity) ** 2),
            float(
                np.linalg.norm(parallel) ** 2
                + np.linalg.norm(perpendicular) ** 2
            ),
        )
        with self.assertRaises(ValueError):
            decompose_velocity(velocity, np.zeros(3))

    def test_adiabatic_pitch_mapping_and_mirror_condition(self) -> None:
        mapped = adiabatic_pitch_angle_deg(
            np.array([15.0, 45.0, 90.0]),
            initial_field_t=3.6,
            final_field_t=0.35e-3,
        )
        np.testing.assert_allclose(
            np.sin(np.deg2rad(mapped)) ** 2 / 0.35e-3,
            np.sin(np.deg2rad([15.0, 45.0, 90.0])) ** 2 / 3.6,
            rtol=2e-15,
        )
        self.assertLess(float(mapped[-1]), 0.57)
        with self.assertRaises(ValueError):
            adiabatic_pitch_angle_deg(60.0, 1.0, 2.0)

    def test_uniform_field_matches_exact_relativistic_helix(self) -> None:
        field_z_t = -1.0e-3
        field = UniformAxialField("uniform test field", field_z_t)
        initial_position = np.array([0.1, -0.2, 0.0])
        initial_u = normalized_momentum_from_angles(18_600.0, 40.0, 23.0)
        settings = IntegrationSettings(
            relative_tolerance=1.0e-11,
            position_atol_m=1.0e-12,
            normalized_momentum_atol=1.0e-13,
            max_gyro_phase_rad=0.15,
            field_bound_t=abs(field_z_t),
            output_samples=1001,
        )

        numerical = integrate_relativistic_magnetic_trajectory(
            field,
            initial_position,
            initial_u,
            stop_z_m=2.0,
            maximum_time_s=1.0e-7,
            settings=settings,
        )
        exact_position, exact_u = uniform_field_exact_solution(
            initial_position,
            initial_u,
            numerical.time_s,
            field_z_t,
            ELECTRON,
        )

        position_error = np.max(
            np.linalg.norm(numerical.position_m - exact_position, axis=1)
        )
        momentum_error = np.max(
            np.linalg.norm(
                numerical.normalized_momentum - exact_u,
                axis=1,
            )
        )
        self.assertLess(position_error, 2.0e-11)
        self.assertLess(momentum_error, 2.0e-11)

    def test_magnetic_field_does_no_work(self) -> None:
        field = UniformAxialField("uniform test field", 0.8e-3)
        initial_u = normalized_momentum_from_angles(18_600.0, 35.0)
        solution = integrate_relativistic_magnetic_trajectory(
            field,
            [0.0, 0.0, 0.0],
            initial_u,
            stop_z_m=1.0,
            maximum_time_s=1.0e-7,
            settings=IntegrationSettings(
                field_bound_t=0.8e-3,
                output_samples=501,
            ),
        )

        self.assertLess(
            solution.summary()["kinetic_energy_relative_span"], 2.0e-11
        )


if __name__ == "__main__":
    unittest.main()
