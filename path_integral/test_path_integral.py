from __future__ import annotations

import math
import unittest

import numpy as np

import path_integral as pi


class SlitsTest(unittest.TestCase):
    def test_number_of_paths(self) -> None:
        for n_scr, n_paths in ((1, 3), (2, 21), (3, 105)):
            verts, ph = pi.paths_to_detector(n_scr, 0.1)
            self.assertEqual(len(verts), n_paths)
            self.assertEqual(len(ph), n_paths)

    def test_symmetric_slits_give_a_symmetric_profile(self) -> None:
        for y in (0.05, 0.2, 0.33):
            self.assertAlmostEqual(abs(pi.amplitude(3, y)), abs(pi.amplitude(3, -y)), places=6)

    def test_all_paths_are_equal_length_at_the_axis_for_one_slit_screen(self) -> None:
        # the straight path through the central slits is the shortest one
        verts, ph = pi.paths_to_detector(3, 0.0)
        lengths = ph * pi.WAVELENGTH / (2 * math.pi)
        self.assertAlmostEqual(float(lengths.min()), 1.0, places=9)


class CornuTest(unittest.TestCase):
    def test_action_is_quadratic(self) -> None:
        self.assertAlmostEqual(float(pi.action_phase(2.0) / pi.action_phase(1.0)), 4.0)

    def test_sum_converges_to_the_stationary_phase_value(self) -> None:
        for c in (9.0, 40.0, 90.0):
            exact = pi.stationary_value(c)
            got = pi.cornu_sum(6.0, c, n=400001)
            self.assertAlmostEqual(abs(got), abs(exact), delta=0.03 * abs(exact))
            self.assertAlmostEqual(float(np.angle(got)), math.pi / 4, delta=0.1)

    def test_amplitude_scales_as_sqrt_of_hbar(self) -> None:
        self.assertAlmostEqual(abs(pi.stationary_value(90.0)) / abs(pi.stationary_value(9.0)), 1 / math.sqrt(10), places=12)


if __name__ == "__main__":
    unittest.main()
