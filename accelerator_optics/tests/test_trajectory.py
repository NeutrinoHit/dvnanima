from __future__ import annotations

import unittest

import numpy as np

from accelerator_optics.trajectory import SectorTrajectory


class SectorTrajectoryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.trajectory = SectorTrajectory(
            center_x=0.45,
            center_y=-2.70,
            radius=2.40,
            entry_start_y=-7.10,
            exit_end_x=3.35,
        )

    def test_endpoints_are_exact(self) -> None:
        start, start_tangent = self.trajectory.point_and_tangent(0.0)
        end, end_tangent = self.trajectory.point_and_tangent(1.0)
        np.testing.assert_allclose(start, self.trajectory.entry_start)
        np.testing.assert_allclose(end, self.trajectory.exit_end)
        np.testing.assert_allclose(start_tangent, [0.0, 1.0, 0.0])
        np.testing.assert_allclose(end_tangent, [1.0, 0.0, 0.0])

    def test_tangent_has_unit_length(self) -> None:
        for progress in np.linspace(0.0, 1.0, 101):
            _, tangent = self.trajectory.point_and_tangent(float(progress))
            self.assertAlmostEqual(float(np.linalg.norm(tangent)), 1.0, places=12)

    def test_arc_has_constant_radius(self) -> None:
        start = self.trajectory.entry_length / self.trajectory.total_length
        stop = (
            self.trajectory.entry_length + self.trajectory.arc_length
        ) / self.trajectory.total_length
        for progress in np.linspace(start, stop, 31):
            point, _ = self.trajectory.point_and_tangent(float(progress))
            distance = np.linalg.norm(point - self.trajectory.center)
            self.assertAlmostEqual(float(distance), self.trajectory.radius, places=11)

    def test_segments_join_continuously(self) -> None:
        first_join = self.trajectory.entry_length / self.trajectory.total_length
        second_join = (
            self.trajectory.entry_length + self.trajectory.arc_length
        ) / self.trajectory.total_length
        point_first, tangent_first = self.trajectory.point_and_tangent(first_join)
        point_second, tangent_second = self.trajectory.point_and_tangent(second_join)
        np.testing.assert_allclose(point_first, self.trajectory.entrance, atol=1e-12)
        np.testing.assert_allclose(tangent_first, [0.0, 1.0, 0.0], atol=1e-12)
        np.testing.assert_allclose(point_second, self.trajectory.exit, atol=1e-12)
        np.testing.assert_allclose(tangent_second, [1.0, 0.0, 0.0], atol=1e-12)

    def test_lorentz_force_is_normal_and_points_inward(self) -> None:
        arc_start = self.trajectory.entry_length / self.trajectory.total_length
        arc_stop = (
            self.trajectory.entry_length + self.trajectory.arc_length
        ) / self.trajectory.total_length
        for progress in np.linspace(arc_start, arc_stop, 31):
            point, tangent = self.trajectory.point_and_tangent(float(progress))
            normal = self.trajectory.inward_normal(float(progress))
            toward_center = self.trajectory.center - point
            self.assertAlmostEqual(float(np.dot(tangent, normal)), 0.0, places=12)
            self.assertAlmostEqual(float(np.linalg.norm(normal)), 1.0, places=12)
            self.assertGreater(float(np.dot(normal, toward_center)), 0.0)

    def test_lorentz_force_is_absent_outside_magnet(self) -> None:
        self.assertFalse(self.trajectory.is_inside_field(0.0))
        self.assertFalse(self.trajectory.is_inside_field(1.0))
        with self.assertRaises(ValueError):
            self.trajectory.inward_normal(0.0)


if __name__ == "__main__":
    unittest.main()
