from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
DEFAULT_PROFILE_PATH = ROOT / "data" / "solar_model_2010_grey_profile.csv"

C_LIGHT_CM_S = 2.99792458e10
SECONDS_PER_YEAR = 365.25 * 24.0 * 3600.0
# First shell in the bundled model where the radiative and adiabatic
# temperature gradients coincide.
BCZ_RADIUS_FRACTION = 0.713073966


@dataclass(frozen=True)
class SolarProfile:
    """Radial quantities needed by the grey radiative-diffusion model.

    Opacity is the Rosseland mean in cm^2/g, density is in g/cm^3, and
    radius is in cm.  Luminosity is the enclosed luminosity in erg/s.
    """

    radius_fraction: np.ndarray
    radius_cm: np.ndarray
    temperature_k: np.ndarray
    density_g_cm3: np.ndarray
    hydrogen_fraction: np.ndarray
    metal_fraction: np.ndarray
    opacity_cm2_g: np.ndarray
    luminosity_erg_s: np.ndarray

    def __post_init__(self) -> None:
        arrays = (
            self.radius_fraction,
            self.radius_cm,
            self.temperature_k,
            self.density_g_cm3,
            self.hydrogen_fraction,
            self.metal_fraction,
            self.opacity_cm2_g,
            self.luminosity_erg_s,
        )
        sizes = {np.asarray(item).size for item in arrays}
        if len(sizes) != 1 or not sizes or next(iter(sizes)) < 2:
            raise ValueError("All profile arrays must have the same length >= 2.")
        if np.any(np.diff(self.radius_cm) <= 0):
            raise ValueError("Profile radii must be strictly increasing.")
        if np.any(self.density_g_cm3 <= 0) or np.any(self.opacity_cm2_g <= 0):
            raise ValueError("Density and opacity must be positive.")

    @property
    def solar_radius_cm(self) -> float:
        return float(np.interp(1.0, self.radius_fraction, self.radius_cm))

    @property
    def mean_free_path_cm(self) -> np.ndarray:
        return 1.0 / (self.opacity_cm2_g * self.density_g_cm3)

    @property
    def diffusion_coefficient_cm2_s(self) -> np.ndarray:
        return C_LIGHT_CM_S * self.mean_free_path_cm / 3.0


def load_profile(path: str | Path = DEFAULT_PROFILE_PATH) -> SolarProfile:
    table = np.genfromtxt(path, delimiter=",", names=True, dtype=float)
    if table.ndim == 0:
        table = table.reshape(1)
    return SolarProfile(
        radius_fraction=table["radius_fraction"],
        radius_cm=table["radius_cm"],
        temperature_k=table["temperature_k"],
        density_g_cm3=table["density_g_cm3"],
        hydrogen_fraction=table["hydrogen_fraction"],
        metal_fraction=table["metal_fraction"],
        opacity_cm2_g=table["opacity_cm2_g"],
        luminosity_erg_s=table["luminosity_erg_s"],
    )


def _sample_interval(
    profile: SolarProfile,
    start_radius_fraction: float,
    escape_radius_fraction: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    start = float(start_radius_fraction)
    stop = float(escape_radius_fraction)
    if not 0.0 <= start < stop <= 1.0:
        raise ValueError("Require 0 <= start_radius_fraction < escape_radius_fraction <= 1.")

    rf = profile.radius_fraction
    radius = profile.radius_cm
    density = profile.density_g_cm3
    opacity = profile.opacity_cm2_g
    inside = (rf > start) & (rf < stop)
    samples = np.r_[start, rf[inside], stop]

    # The tabulation begins just off-centre.  The first shell is an excellent
    # constant central extrapolation over that tiny omitted interval.
    radius_samples = np.interp(samples, np.r_[0.0, rf], np.r_[0.0, radius])
    density_samples = np.interp(samples, np.r_[0.0, rf], np.r_[density[0], density])
    opacity_samples = np.interp(samples, np.r_[0.0, rf], np.r_[opacity[0], opacity])
    return samples, radius_samples, density_samples, opacity_samples


def cumulative_delay_seconds(
    profile: SolarProfile,
    escape_radius_fraction: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Return radius fraction and cumulative contribution to the mean delay.

    For spherical diffusion with D(r)=c/[3 kappa_R(r) rho(r)], the mean
    first-passage time from the centre obeys

        -1 = r^-2 d/dr [r^2 D(r) dT/dr]

    with T(R_escape)=0 and regularity at r=0.  Integrating once gives

        T(0) = integral_0^R r kappa_R(r) rho(r) / c dr.
    """

    rf, radius, density, opacity = _sample_interval(
        profile, 0.0, escape_radius_fraction
    )
    integrand = radius * opacity * density / C_LIGHT_CM_S
    increments = 0.5 * (integrand[:-1] + integrand[1:]) * np.diff(radius)
    cumulative = np.r_[0.0, np.cumsum(increments)]
    return rf, cumulative


def mean_exit_time_seconds(
    profile: SolarProfile,
    start_radius_fraction: float = 0.0,
    escape_radius_fraction: float = 1.0,
) -> float:
    """Mean first-passage time for energy born at one radius."""

    _, radius, density, opacity = _sample_interval(
        profile, start_radius_fraction, escape_radius_fraction
    )
    integrand = radius * opacity * density / C_LIGHT_CM_S
    return float(np.trapezoid(integrand, radius))


def production_weighted_exit_time_seconds(
    profile: SolarProfile,
    escape_radius_fraction: float = 1.0,
) -> float:
    """Average the first-passage time over the model's luminosity production."""

    rf, cumulative = cumulative_delay_seconds(profile, escape_radius_fraction)
    total_from_centre = float(cumulative[-1])

    use = profile.radius_fraction <= escape_radius_fraction
    source_radius = np.r_[0.0, profile.radius_fraction[use]]
    enclosed_luminosity = np.r_[0.0, profile.luminosity_erg_s[use]]
    shell_luminosity = np.maximum(np.diff(enclosed_luminosity), 0.0)
    shell_radius = 0.5 * (source_radius[:-1] + source_radius[1:])
    time_from_shell = total_from_centre - np.interp(shell_radius, rf, cumulative)
    return float(np.average(time_from_shell, weights=shell_luminosity))


def radius_at_delay_fraction(
    profile: SolarProfile,
    delay_fraction: np.ndarray | float,
    escape_radius_fraction: float = 1.0,
) -> np.ndarray:
    """Invert the cumulative mean-delay integral for animation purposes."""

    fraction = np.asarray(delay_fraction, dtype=float)
    rf, cumulative = cumulative_delay_seconds(profile, escape_radius_fraction)
    normalized = cumulative / cumulative[-1]
    return np.interp(np.clip(fraction, 0.0, 1.0), normalized, rf)


def profile_summary(profile: SolarProfile) -> dict[str, float]:
    central_full = mean_exit_time_seconds(profile)
    central_radiative = mean_exit_time_seconds(
        profile, escape_radius_fraction=BCZ_RADIUS_FRACTION
    )
    weighted_full = production_weighted_exit_time_seconds(profile)
    weighted_radiative = production_weighted_exit_time_seconds(
        profile, escape_radius_fraction=BCZ_RADIUS_FRACTION
    )
    return {
        "central_to_surface_years": central_full / SECONDS_PER_YEAR,
        "central_to_bcz_years": central_radiative / SECONDS_PER_YEAR,
        "production_weighted_to_surface_years": weighted_full / SECONDS_PER_YEAR,
        "production_weighted_to_bcz_years": weighted_radiative / SECONDS_PER_YEAR,
        "outer_zone_diffusion_years": (central_full - central_radiative)
        / SECONDS_PER_YEAR,
        "central_mean_free_path_cm": float(profile.mean_free_path_cm[0]),
        "central_density_g_cm3": float(profile.density_g_cm3[0]),
        "central_opacity_cm2_g": float(profile.opacity_cm2_g[0]),
    }


def _print_report() -> None:
    summary = profile_summary(load_profile())
    print("Grey radiative-diffusion model (Guenther 2010 solar profile)")
    print(f"centre -> base of convection zone: {summary['central_to_bcz_years']:,.1f} yr")
    print(f"centre -> photosphere:             {summary['central_to_surface_years']:,.1f} yr")
    print(
        "production-weighted -> photosphere: "
        f"{summary['production_weighted_to_surface_years']:,.1f} yr"
    )
    print(
        "central mean free path:             "
        f"{summary['central_mean_free_path_cm'] * 10.0:.4f} mm"
    )


if __name__ == "__main__":
    _print_report()
