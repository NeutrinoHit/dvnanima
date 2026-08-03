from __future__ import annotations

import unittest

import numpy as np

from accelerator_optics.quadrupole import (
    AlternatingGradientLattice,
    QuadrupoleRayModel,
    lorentz_force_xy,
    quadrupole_field,
)


class QuadrupoleFieldTests(unittest.TestCase):
    def test_field_is_zero_on_axis(self) -> None:
        np.testing.assert_allclose(quadrupole_field(0.0, 0.0), [0.0, 0.0, 0.0])

    def test_field_magnitude_grows_linearly_with_radius(self) -> None:
        gradient = 1.7
        for x, y in ((1.0, 0.0), (0.0, 2.0), (3.0, 4.0)):
            magnitude = np.linalg.norm(quadrupole_field(x, y, gradient))
            self.assertAlmostEqual(magnitude, gradient * np.hypot(x, y), places=12)

    def test_lorentz_force_focuses_x_and_defocuses_y(self) -> None:
        force_x = lorentz_force_xy(1.0, 0.0)
        force_y = lorentz_force_xy(0.0, 1.0)
        self.assertLess(force_x[0], 0.0)
        self.assertGreater(force_y[1], 0.0)


class QuadrupoleRayTests(unittest.TestCase):
    def setUp(self) -> None:
        self.model = QuadrupoleRayModel()

    def test_ray_is_continuous_at_magnet_edges(self) -> None:
        epsilon = 1e-9
        for edge in (self.model.magnet_start, self.model.magnet_end):
            for plane in ("focusing", "defocusing"):
                left = self.model.offset_and_slope(edge - epsilon, 0.3, plane)[0]
                right = self.model.offset_and_slope(edge + epsilon, 0.3, plane)[0]
                self.assertAlmostEqual(left, right, places=8)

    def test_parallel_focusing_rays_meet_at_focal_position(self) -> None:
        for initial_offset in (-0.4, -0.2, 0.2, 0.4):
            offset, _ = self.model.offset_and_slope(
                self.model.focal_position,
                initial_offset,
                "focusing",
            )
            self.assertAlmostEqual(offset, 0.0, places=12)

    def test_defocusing_plane_expands(self) -> None:
        initial = 0.3
        final, _ = self.model.offset_and_slope(3.4, initial, "defocusing")
        self.assertGreater(abs(final), abs(initial))


class AlternatingGradientLatticeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.lattice = AlternatingGradientLattice()

    def test_separated_lenses_do_not_cancel(self) -> None:
        identity = np.eye(2)
        for plane in ("x", "y"):
            self.assertFalse(
                np.allclose(self.lattice.cell_matrix(plane), identity)
            )

    def test_both_planes_are_stable(self) -> None:
        for plane in ("x", "y"):
            trace = float(np.trace(self.lattice.cell_matrix(plane)))
            self.assertLess(abs(trace), 2.0)

    def test_matched_envelope_repeats_after_one_cell(self) -> None:
        for plane in ("x", "y"):
            covariance = self.lattice.matched_covariance(plane)
            matrix = self.lattice.cell_matrix(plane)
            transported = matrix @ covariance @ matrix.T
            np.testing.assert_allclose(transported, covariance, atol=1e-12)

    def test_envelopes_remain_bounded(self) -> None:
        for plane in ("x", "y"):
            sizes = [
                self.lattice.sigma(position, plane)
                for position in np.linspace(0.0, self.lattice.total_length, 101)
            ]
            self.assertGreater(min(sizes), 0.0)
            self.assertLess(max(sizes), 1.0)


if __name__ == "__main__":
    unittest.main()
