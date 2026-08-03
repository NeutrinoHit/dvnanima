from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def sextupole_field(x: float, y: float, strength: float = 1.0) -> np.ndarray:
    """Ideal normal sextupole field in the transverse plane."""

    return np.array(
        [
            strength * x * y,
            0.5 * strength * (x**2 - y**2),
            0.0,
        ],
        dtype=float,
    )


@dataclass(frozen=True)
class ChromaticFocusModel:
    """Thin quadrupole followed by a dispersive sextupole correction."""

    quadrupole_position: float = -2.0
    sextupole_position: float = -0.6
    nominal_focal_length: float = 3.0

    def __post_init__(self) -> None:
        if self.sextupole_position <= self.quadrupole_position:
            raise ValueError("sextupole must follow the quadrupole")
        if self.nominal_focal_length <= 0.0:
            raise ValueError("nominal_focal_length must be positive")
        if self.sextupole_position - self.quadrupole_position >= self.nominal_focal_length:
            raise ValueError("the sextupole must be placed before the nominal focus")

    @property
    def separation(self) -> float:
        return self.sextupole_position - self.quadrupole_position

    @property
    def correction_strength(self) -> float:
        """First-order strength that cancels the chromatic focal shift."""

        inverse_focal = 1.0 / self.nominal_focal_length
        remaining = 1.0 - self.separation * inverse_focal
        return inverse_focal / remaining**2

    def focal_length(self, momentum_deviation: float) -> float:
        if momentum_deviation <= -1.0:
            raise ValueError("momentum deviation must exceed -1")
        return self.nominal_focal_length * (1.0 + momentum_deviation)

    def state_after_quadrupole(
        self,
        initial_offset: float,
        momentum_deviation: float,
    ) -> tuple[float, float]:
        slope = -initial_offset / self.focal_length(momentum_deviation)
        return initial_offset, slope

    def state_at_sextupole(
        self,
        initial_offset: float,
        momentum_deviation: float,
    ) -> tuple[float, float]:
        _, slope = self.state_after_quadrupole(initial_offset, momentum_deviation)
        offset = initial_offset + slope * self.separation
        return offset, slope

    def corrected_state_after_sextupole(
        self,
        initial_offset: float,
        momentum_deviation: float,
    ) -> tuple[float, float]:
        offset, slope = self.state_at_sextupole(initial_offset, momentum_deviation)
        correction = self.correction_strength * momentum_deviation
        return offset, slope - correction * offset

    def focal_position(self, momentum_deviation: float, corrected: bool) -> float:
        if not corrected:
            return self.quadrupole_position + self.focal_length(momentum_deviation)
        offset, slope = self.corrected_state_after_sextupole(
            1.0, momentum_deviation
        )
        return self.sextupole_position - offset / slope

    def ray_offset(
        self,
        longitudinal: float,
        initial_offset: float,
        momentum_deviation: float,
        *,
        corrected: bool,
    ) -> float:
        if longitudinal <= self.quadrupole_position:
            return initial_offset

        _, quadrupole_slope = self.state_after_quadrupole(
            initial_offset, momentum_deviation
        )
        if not corrected or longitudinal <= self.sextupole_position:
            return initial_offset + quadrupole_slope * (
                longitudinal - self.quadrupole_position
            )

        sextupole_offset, corrected_slope = self.corrected_state_after_sextupole(
            initial_offset, momentum_deviation
        )
        return sextupole_offset + corrected_slope * (
            longitudinal - self.sextupole_position
        )

    def sample_ray(
        self,
        initial_offset: float,
        momentum_deviation: float,
        *,
        corrected: bool,
        start: float = -3.5,
        stop: float = 3.25,
        count: int = 220,
    ) -> list[np.ndarray]:
        if count < 2:
            raise ValueError("count must be at least two")
        return [
            np.array(
                [
                    longitudinal,
                    self.ray_offset(
                        float(longitudinal),
                        initial_offset,
                        momentum_deviation,
                        corrected=corrected,
                    ),
                    0.0,
                ]
            )
            for longitudinal in np.linspace(start, stop, count)
        ]


def dispersive_offset(momentum_deviation: float, dispersion: float) -> float:
    return dispersion * momentum_deviation


def effective_gradient_correction(
    momentum_deviation: float,
    dispersion: float,
    sextupole_strength: float,
) -> float:
    return sextupole_strength * dispersive_offset(
        momentum_deviation, dispersion
    )
