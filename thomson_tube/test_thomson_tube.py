from __future__ import annotations

import math
import unittest

import numpy as np

import thomson_tube as tt


class PhysicsTest(unittest.TestCase):
    def test_speed_after_the_gun(self) -> None:
        state = np.array([[tt.X_CATH, 0.0, 0.0, 0.0]])
        while state[0, 0] < tt.X_P0 - 0.5:
            tt.step(state, 0.0, 0.0)
        self.assertAlmostEqual(float(state[0, 2]), tt.V0, delta=0.05)

    def test_no_fields_no_deflection(self) -> None:
        self.assertAlmostEqual(tt.screen_deflection(0.0, 0.0), 0.0, places=9)

    def test_electric_deflection_matches_the_formula(self) -> None:
        y = tt.screen_deflection(tt.E0, 0.0)
        self.assertGreater(y, 0.0)                                       # towards the positive plate
        self.assertAlmostEqual(y / tt.deflection_electric(tt.E0), 1.0, delta=0.04)

    def test_magnetic_deflection_is_opposite_and_matches(self) -> None:
        y = tt.screen_deflection(0.0, tt.B_BAL)
        self.assertLess(y, 0.0)
        self.assertAlmostEqual(y / tt.deflection_magnetic(tt.B_BAL), 1.0, delta=0.06)

    def test_crossed_fields_balance_at_v_equal_e_over_b(self) -> None:
        self.assertAlmostEqual(tt.E0 / tt.B_BAL, tt.V0, places=12)
        self.assertAlmostEqual(tt.screen_deflection(tt.E0, tt.B_BAL), 0.0, delta=0.02)
        self.assertGreater(tt.screen_deflection(tt.E0, 0.8 * tt.B_BAL), 0.0)
        self.assertLess(tt.screen_deflection(tt.E0, 1.2 * tt.B_BAL), 0.0)

    def test_charge_to_mass_is_recovered_from_the_measured_deflection(self) -> None:
        y = tt.screen_deflection(tt.E0, 0.0)
        self.assertAlmostEqual(tt.charge_to_mass(y, tt.E0, tt.B_BAL), 1.0, delta=0.05)

    def test_the_beam_stays_between_the_plates(self) -> None:
        state = np.array([[tt.X_CATH, 0.0, 0.0, 0.0]])
        ymax = 0.0
        while state[0, 0] < tt.X_P1:
            tt.step(state, tt.E0, 0.0)
            ymax = max(ymax, abs(state[0, 1]))
        self.assertLess(ymax, 1.1)

    def test_schedule(self) -> None:
        self.assertEqual(tt.fields(3.0), (0.0, 0.0))
        e, b = tt.fields(12.0)
        self.assertAlmostEqual(e, tt.E0)
        self.assertAlmostEqual(b, 0.0)
        e, b = tt.fields(24.0)
        self.assertAlmostEqual(e, 0.0)
        self.assertAlmostEqual(b, tt.B_BAL)
        e, b = tt.fields(41.0)
        self.assertAlmostEqual(e, tt.E0)
        self.assertAlmostEqual(b, tt.B_BAL)


if __name__ == "__main__":
    unittest.main()
