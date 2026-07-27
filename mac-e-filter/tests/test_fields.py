from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
from scipy.constants import mu_0

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mac_e_filter.fields import (  # noqa: E402
    CircularLoop,
    UniformAxialField,
    integrate_axisymmetric_field_line_radius,
)


class CircularLoopTests(unittest.TestCase):
    def test_uniform_field_line_keeps_constant_radius(self) -> None:
        field = UniformAxialField("uniform", -2.0)
        z = np.linspace(-3.0, 4.0, 101)
        radius = integrate_axisymmetric_field_line_radius(
            field,
            z,
            0.037,
        )
        np.testing.assert_allclose(radius, 0.037, atol=1e-14)

    def test_axis_field_uses_analytic_limit(self) -> None:
        loop = CircularLoop(
            name="test loop",
            radius_m=1.7,
            center_z_m=-0.2,
            ampere_turns=-1234.0,
        )
        z = np.linspace(-1.0, 1.0, 11)
        b_rho, b_z = loop.field_cylindrical(np.zeros_like(z), z)

        np.testing.assert_array_equal(b_rho, 0.0)
        np.testing.assert_allclose(b_z, loop.axis_field(z), rtol=0.0, atol=0.0)

    def test_elliptic_integral_solution_against_direct_biot_savart(self) -> None:
        """Compare with a direct, independent line integral around the wire."""

        loop = CircularLoop(
            name="test loop",
            radius_m=1.3,
            center_z_m=-0.15,
            ampere_turns=731.0,
        )
        rho = 0.7
        z = 0.4
        b_rho, b_z = loop.field_cylindrical(rho, z)

        samples = 32768
        phi = (np.arange(samples) + 0.5) * (2.0 * np.pi / samples)
        d_phi = 2.0 * np.pi / samples
        source_x = loop.radius_m * np.cos(phi)
        source_y = loop.radius_m * np.sin(phi)
        displacement = np.column_stack(
            (
                rho - source_x,
                -source_y,
                np.full(samples, z - loop.center_z_m),
            )
        )
        dl = np.column_stack(
            (
                -loop.radius_m * np.sin(phi),
                loop.radius_m * np.cos(phi),
                np.zeros(samples),
            )
        )
        integrand = np.cross(dl, displacement) / (
            np.linalg.norm(displacement, axis=1)[:, None] ** 3
        )
        direct = (
            mu_0
            * loop.ampere_turns
            / (4.0 * np.pi)
            * d_phi
            * integrand.sum(axis=0)
        )

        self.assertAlmostEqual(float(b_rho), direct[0], delta=abs(direct[0]) * 2e-12)
        self.assertAlmostEqual(float(b_z), direct[2], delta=abs(direct[2]) * 2e-12)
        self.assertAlmostEqual(direct[1], 0.0, delta=np.linalg.norm(direct) * 2e-12)

    def test_vacuum_maxwell_residuals(self) -> None:
        """Finite-difference Maxwell checks away from the current source."""

        loop = CircularLoop(
            name="test loop",
            radius_m=1.2,
            center_z_m=-0.1,
            ampere_turns=900.0,
        )
        rho = 0.45
        z = 0.30
        step = 2.0e-5

        br_plus_r, bz_plus_r = loop.field_cylindrical(rho + step, z)
        br_minus_r, bz_minus_r = loop.field_cylindrical(rho - step, z)
        br_plus_z, bz_plus_z = loop.field_cylindrical(rho, z + step)
        br_minus_z, bz_minus_z = loop.field_cylindrical(rho, z - step)

        divergence = (
            ((rho + step) * br_plus_r - (rho - step) * br_minus_r)
            / (2.0 * step * rho)
            + (bz_plus_z - bz_minus_z) / (2.0 * step)
        )
        curl_phi = (
            (br_plus_z - br_minus_z) / (2.0 * step)
            - (bz_plus_r - bz_minus_r) / (2.0 * step)
        )
        local_field_scale = max(
            abs(float(bz_plus_z)),
            abs(float(bz_minus_z)),
            abs(float(br_plus_r)),
            abs(float(br_minus_r)),
        )
        derivative_scale = local_field_scale / loop.radius_m

        self.assertLess(abs(float(divergence)) / derivative_scale, 2e-8)
        self.assertLess(abs(float(curl_phi)) / derivative_scale, 2e-8)


if __name__ == "__main__":
    unittest.main()
