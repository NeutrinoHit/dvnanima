from __future__ import annotations

import unittest

import numpy as np

from accelerator_optics.ring_model import (
    RingConfig,
    SymplecticRingTracker,
    generate_momentum_spectrum,
)


class RingModelTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = RingConfig()
        self.tracker = SymplecticRingTracker(self.config)

    def test_linear_cell_maps_are_symplectic(self) -> None:
        symplectic = np.array([[0.0, 1.0], [-1.0, 0.0]])
        for deviation in (-0.12, 0.0, 0.12):
            for plane in ("x", "y"):
                matrix = self.tracker.one_cell_linear_matrix(deviation, plane)
                np.testing.assert_allclose(
                    matrix.T @ symplectic @ matrix,
                    symplectic,
                    atol=1e-12,
                )
                self.assertAlmostEqual(float(np.linalg.det(matrix)), 1.0, places=12)

    def test_sextupole_kick_preserves_phase_space_volume(self) -> None:
        state = np.array([[0.04, 0.01, -0.03, 0.02, 0.08]])
        epsilon = 1e-7

        def mapped(vector: np.ndarray) -> np.ndarray:
            sample = vector.reshape(1, 5).copy()
            self.tracker.sextupole_kick(sample, "F")
            return sample[0, :4]

        jacobian = np.zeros((4, 4))
        for column in range(4):
            plus = state[0].copy()
            minus = state[0].copy()
            plus[column] += epsilon
            minus[column] -= epsilon
            jacobian[:, column] = (mapped(plus) - mapped(minus)) / (2.0 * epsilon)
        self.assertAlmostEqual(float(np.linalg.det(jacobian)), 1.0, places=8)

    def test_sextupoles_improve_survival_for_momentum_spectrum(self) -> None:
        initial = generate_momentum_spectrum()
        without = self.tracker.track(initial, turns=18, sextupoles_on=False)
        with_correction = self.tracker.track(initial, turns=18, sextupoles_on=True)
        survival_without = without.survival_by_sample[-1]
        survival_with = with_correction.survival_by_sample[-1]
        self.assertGreater(survival_with, survival_without + 0.25)
        self.assertGreater(survival_with, 0.80)

    def test_same_initial_ensemble_is_used_for_both_modes(self) -> None:
        first = generate_momentum_spectrum(seed=17)
        second = generate_momentum_spectrum(seed=17)
        np.testing.assert_allclose(first, second)


if __name__ == "__main__":
    unittest.main()
