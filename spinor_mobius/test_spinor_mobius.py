from __future__ import annotations

import math
import unittest

import numpy as np

import spinor_mobius as sm


class SpinorTest(unittest.TestCase):
    def test_strip_vector_flips_after_one_turn_and_returns_after_two(self) -> None:
        for u in (0.0, 0.7, 2.1, 5.9):
            np.testing.assert_allclose(sm.across(u + 2 * math.pi), -sm.across(u), atol=1e-12)
            np.testing.assert_allclose(sm.across(u + 4 * math.pi), sm.across(u), atol=1e-12)

    def test_vector_lies_across_the_strip(self) -> None:
        # the vector is tangent to the strip and orthogonal to the core circle direction
        for u in (0.3, 1.9, 4.4):
            tangent_core = np.array([-math.sin(u), math.cos(u), 0.0])
            self.assertAlmostEqual(float(sm.across(u) @ tangent_core), 0.0, places=12)
            self.assertAlmostEqual(float(np.linalg.norm(sm.across(u))), 1.0, places=12)

    def test_vector_is_the_derivative_of_the_surface_in_v(self) -> None:
        u = 1.3
        h = 1e-6
        d = (sm.strip_point(np.array(u), np.array(h)) - sm.strip_point(np.array(u), np.array(-h))) / (2 * h)
        np.testing.assert_allclose(d, sm.across(u), atol=1e-8)

    def test_spinor_component(self) -> None:
        self.assertAlmostEqual(sm.spinor_component(0.0), 1.0)
        self.assertAlmostEqual(sm.spinor_component(2 * math.pi), -1.0)
        self.assertAlmostEqual(sm.spinor_component(4 * math.pi), 1.0)

    def test_angle_schedule(self) -> None:
        total = 20.0
        self.assertEqual(sm.angle_at(0.0, total), 0.0)
        self.assertAlmostEqual(sm.angle_at(total, total), 4 * math.pi)
        values = [sm.angle_at(t, total) for t in np.linspace(0, total, 400)]
        self.assertTrue(all(b >= a - 1e-12 for a, b in zip(values, values[1:])))
        self.assertTrue(any(abs(v - 2 * math.pi) < 1e-9 for v in values))


if __name__ == "__main__":
    unittest.main()
