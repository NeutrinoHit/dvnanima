from __future__ import annotations

import unittest

import numpy as np

from accelerator_optics.sextupole import (
    ChromaticFocusModel,
    dispersive_offset,
    effective_gradient_correction,
    sextupole_field,
)


class SextupoleFieldTests(unittest.TestCase):
    def test_field_is_zero_on_axis(self) -> None:
        np.testing.assert_allclose(sextupole_field(0.0, 0.0), [0.0, 0.0, 0.0])

    def test_field_magnitude_is_quadratic_in_radius(self) -> None:
        strength = 1.8
        for x, y in ((1.0, 0.0), (0.0, 2.0), (3.0, 4.0)):
            magnitude = np.linalg.norm(sextupole_field(x, y, strength))
            expected = 0.5 * abs(strength) * (x**2 + y**2)
            self.assertAlmostEqual(float(magnitude), expected, places=12)

    def test_dispersion_makes_correction_linear_in_momentum_error(self) -> None:
        for deviation in (-0.2, 0.0, 0.2):
            correction = effective_gradient_correction(deviation, 1.7, 0.8)
            self.assertAlmostEqual(correction, 1.7 * 0.8 * deviation, places=12)
            self.assertAlmostEqual(dispersive_offset(deviation, 1.7), 1.7 * deviation)


class ChromaticFocusTests(unittest.TestCase):
    def setUp(self) -> None:
        self.model = ChromaticFocusModel()
        self.deviations = (-0.12, 0.0, 0.12)

    def test_quadrupole_focus_depends_on_momentum(self) -> None:
        positions = [
            self.model.focal_position(deviation, corrected=False)
            for deviation in self.deviations
        ]
        self.assertLess(positions[0], positions[1])
        self.assertLess(positions[1], positions[2])

    def test_sextupole_reduces_focal_spread(self) -> None:
        before = [
            self.model.focal_position(deviation, corrected=False)
            for deviation in self.deviations
        ]
        after = [
            self.model.focal_position(deviation, corrected=True)
            for deviation in self.deviations
        ]
        self.assertLess(max(after) - min(after), 0.2 * (max(before) - min(before)))

    def test_nominal_particle_is_unchanged(self) -> None:
        before = self.model.focal_position(0.0, corrected=False)
        after = self.model.focal_position(0.0, corrected=True)
        self.assertAlmostEqual(before, after, places=12)

    def test_trajectory_is_continuous_at_both_magnets(self) -> None:
        epsilon = 1e-9
        for position in (
            self.model.quadrupole_position,
            self.model.sextupole_position,
        ):
            left = self.model.ray_offset(
                position - epsilon, 0.6, 0.12, corrected=True
            )
            right = self.model.ray_offset(
                position + epsilon, 0.6, 0.12, corrected=True
            )
            self.assertAlmostEqual(left, right, places=8)


if __name__ == "__main__":
    unittest.main()
