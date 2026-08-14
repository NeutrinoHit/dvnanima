from __future__ import annotations

import unittest

import numpy as np

from solar_photon_diffusion.physics import (
    BCZ_RADIUS_FRACTION,
    C_LIGHT_CM_S,
    SECONDS_PER_YEAR,
    SolarProfile,
    cumulative_delay_seconds,
    load_profile,
    mean_exit_time_seconds,
    production_weighted_exit_time_seconds,
    radius_at_delay_fraction,
)


class PhotonDiffusionPhysicsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.profile = load_profile()

    def test_constant_medium_matches_spherical_first_passage_solution(self) -> None:
        radius = np.linspace(0.001, 1.0, 1001) * 2.0e10
        density = np.full_like(radius, 10.0)
        opacity = np.full_like(radius, 0.4)
        profile = SolarProfile(
            radius_fraction=radius / radius[-1],
            radius_cm=radius,
            temperature_k=np.ones_like(radius),
            density_g_cm3=density,
            hydrogen_fraction=np.full_like(radius, 0.7),
            metal_fraction=np.full_like(radius, 0.02),
            opacity_cm2_g=opacity,
            luminosity_erg_s=np.linspace(0.0, 1.0, radius.size),
        )
        expected = radius[-1] ** 2 * density[0] * opacity[0] / (2.0 * C_LIGHT_CM_S)
        self.assertAlmostEqual(mean_exit_time_seconds(profile), expected, delta=5e-6 * expected)

    def test_guenther_profile_reproduces_reference_times(self) -> None:
        centre_to_surface = mean_exit_time_seconds(self.profile) / SECONDS_PER_YEAR
        centre_to_bcz = mean_exit_time_seconds(
            self.profile, escape_radius_fraction=BCZ_RADIUS_FRACTION
        ) / SECONDS_PER_YEAR
        weighted_to_surface = (
            production_weighted_exit_time_seconds(self.profile) / SECONDS_PER_YEAR
        )
        self.assertAlmostEqual(centre_to_surface, 41399.0, delta=2.0)
        self.assertAlmostEqual(centre_to_bcz, 31572.2, delta=2.0)
        self.assertAlmostEqual(weighted_to_surface, 36111.6, delta=2.0)

    def test_cumulative_delay_and_inverse_are_consistent(self) -> None:
        radius, cumulative = cumulative_delay_seconds(self.profile)
        fractions = np.linspace(0.0, 1.0, 21)
        inverted = radius_at_delay_fraction(self.profile, fractions)
        recovered = np.interp(inverted, radius, cumulative / cumulative[-1])
        np.testing.assert_allclose(recovered, fractions, atol=2e-5)
        self.assertTrue(np.all(np.diff(cumulative) >= 0.0))

    def test_core_mean_free_path_is_about_fifty_micrometres(self) -> None:
        millimetres = self.profile.mean_free_path_cm[0] * 10.0
        self.assertAlmostEqual(millimetres, 0.05142, delta=0.0001)


if __name__ == "__main__":
    unittest.main()
