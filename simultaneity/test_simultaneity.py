from __future__ import annotations

import math
import unittest

import numpy as np

import simultaneity as sm


class LorentzTest(unittest.TestCase):
    def test_order_of_the_three_flashes(self) -> None:
        self.assertEqual(sm.order(sm.EVENTS, 0.3), "CBA")
        self.assertEqual(sm.order(sm.EVENTS, -0.5), "ABC")
        for v in (0.05, 0.3, 0.9):
            self.assertEqual(sm.order(sm.EVENTS, v), "CBA")
            self.assertEqual(sm.order(sm.EVENTS, -v), "ABC")

    def test_times_follow_the_formula(self) -> None:
        for v in (-0.5, 0.3, 0.8):
            for name, (x, t) in sm.EVENTS.items():
                self.assertAlmostEqual(sm.boost(x, t, v)[1], -sm.gamma(v) * v * x, places=12)

    def test_interval_is_invariant(self) -> None:
        for v in (-0.6, 0.3, 0.9):
            for a in sm.PAIRS.values():
                for b in sm.PAIRS.values():
                    xa, ta = sm.boost(*a, v)
                    xb, tb = sm.boost(*b, v)
                    self.assertAlmostEqual((tb - ta) ** 2 - (xb - xa) ** 2, sm.interval2(a, b), places=10)

    def test_timelike_pair_never_changes_its_order(self) -> None:
        self.assertGreater(sm.interval2(sm.PAIRS["P"], sm.PAIRS["Q"]), 0)
        for v in np.linspace(-0.99, 0.99, 199):
            self.assertGreater(sm.boost(*sm.PAIRS["Q"], v)[1] - sm.boost(*sm.PAIRS["P"], v)[1], 0)

    def test_spacelike_pair_changes_its_order(self) -> None:
        self.assertLess(sm.interval2(sm.PAIRS["P"], sm.PAIRS["R"]), 0)
        signs = {math.copysign(1, sm.boost(*sm.PAIRS["R"], v)[1] - sm.boost(*sm.PAIRS["P"], v)[1]) for v in np.linspace(-0.6, 0.6, 61)}
        self.assertEqual(signs, {1.0, -1.0})

    def test_axes_of_the_moving_frame(self) -> None:
        for v in (-0.5, 0.3):
            # the ct' axis is x = v ct: its points have x' = 0
            for ct in (-1.0, 0.7, 2.0):
                self.assertAlmostEqual(sm.boost(v * ct, ct, v)[0], 0.0, places=12)
            # the x' axis is ct = v x: its points have t' = 0
            for x in (-2.0, 0.5, 1.5):
                self.assertAlmostEqual(sm.boost(x, v * x, v)[1], 0.0, places=12)

    def test_line_of_simultaneity_is_a_line_of_constant_t_prime(self) -> None:
        v, tp = -0.5, 0.4
        g = sm.gamma(v)
        for x in (-1.0, 0.0, 2.0):
            ct = v * x + tp / g
            self.assertAlmostEqual(sm.boost(x, ct, v)[1], tp, places=12)

    def test_read_off_point_on_the_time_axis(self) -> None:
        v = 0.3
        g = sm.gamma(v)
        for name, (x, t) in sm.EVENTS.items():
            tp = sm.boost(x, t, v)[1]
            px, pt = g * v * tp, g * tp
            self.assertAlmostEqual(sm.boost(px, pt, v)[0], 0.0, places=12)       # on the ct' axis
            self.assertAlmostEqual(sm.boost(px, pt, v)[1], tp, places=12)        # with the reading t'


class ScheduleTest(unittest.TestCase):
    def test_velocity_in_the_passes(self) -> None:
        self.assertEqual(sm.velocity(8.0), 0.0)
        self.assertEqual(sm.velocity(20.0), 0.3)
        self.assertEqual(sm.velocity(30.0), -0.5)
        self.assertAlmostEqual(sm.velocity(38.0), -0.6)
        self.assertAlmostEqual(sm.velocity(51.0), 0.6)

    def test_clock(self) -> None:
        self.assertIsNone(sm.clock(2.0))
        self.assertAlmostEqual(sm.clock(9.0), 0.0, delta=0.15)
        self.assertAlmostEqual(sm.clock(15.0), -sm.T_SWEEP)


if __name__ == "__main__":
    unittest.main()
