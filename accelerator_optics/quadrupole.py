from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def quadrupole_field(x: float, y: float, gradient: float = 1.0) -> np.ndarray:
    """Ideal normal quadrupole field in the transverse plane."""

    return np.array([gradient * y, gradient * x, 0.0], dtype=float)


def lorentz_force_xy(
    x: float,
    y: float,
    *,
    charge_speed: float = 1.0,
    gradient: float = 1.0,
) -> np.ndarray:
    """Transverse force for a particle moving along the positive z axis."""

    field = quadrupole_field(x, y, gradient)
    return charge_speed * np.array([-field[1], field[0], 0.0])


@dataclass(frozen=True)
class QuadrupoleRayModel:
    """Parallel rays through a hard-edge quadrupole and the following drift."""

    magnet_start: float = -0.8
    magnet_end: float = 0.8
    strength: float = 0.55

    def __post_init__(self) -> None:
        if self.magnet_end <= self.magnet_start:
            raise ValueError("magnet_end must exceed magnet_start")
        if self.strength <= 0.0:
            raise ValueError("strength must be positive")

    @property
    def length(self) -> float:
        return self.magnet_end - self.magnet_start

    @property
    def focal_position(self) -> float:
        phase = self.strength * self.length
        return self.magnet_end + 1.0 / (
            self.strength * np.tan(phase)
        )

    def offset_and_slope(
        self,
        longitudinal: float,
        initial_offset: float,
        plane: str,
    ) -> tuple[float, float]:
        if plane not in {"focusing", "defocusing"}:
            raise ValueError("plane must be 'focusing' or 'defocusing'")

        if longitudinal <= self.magnet_start:
            return initial_offset, 0.0

        inside_length = min(longitudinal, self.magnet_end) - self.magnet_start
        phase = self.strength * inside_length
        if plane == "focusing":
            offset = initial_offset * np.cos(phase)
            slope = -initial_offset * self.strength * np.sin(phase)
        else:
            offset = initial_offset * np.cosh(phase)
            slope = initial_offset * self.strength * np.sinh(phase)

        if longitudinal > self.magnet_end:
            offset += slope * (longitudinal - self.magnet_end)
        return float(offset), float(slope)

    def sample_points(
        self,
        initial_offset: float,
        plane: str,
        *,
        start: float = -3.4,
        stop: float = 3.4,
        count: int = 180,
    ) -> list[np.ndarray]:
        if count < 2:
            raise ValueError("count must be at least two")
        return [
            np.array(
                [
                    longitudinal,
                    self.offset_and_slope(
                        float(longitudinal), initial_offset, plane
                    )[0],
                    0.0,
                ]
            )
            for longitudinal in np.linspace(start, stop, count)
        ]


@dataclass(frozen=True)
class AlternatingGradientLattice:
    """Thin-lens F-D lattice with matched beam envelopes in both planes."""

    drift_length: float = 1.0
    focal_length: float = 1.0
    cells: int = 2
    emittance: float = 0.08

    def __post_init__(self) -> None:
        if self.drift_length <= 0.0:
            raise ValueError("drift_length must be positive")
        if self.focal_length <= 0.0:
            raise ValueError("focal_length must be positive")
        if self.cells < 1:
            raise ValueError("cells must be positive")
        if self.emittance <= 0.0:
            raise ValueError("emittance must be positive")
        for plane in ("x", "y"):
            if abs(np.trace(self.cell_matrix(plane)) / 2.0) >= 1.0:
                raise ValueError("lattice parameters are outside the stable region")

    @property
    def total_length(self) -> float:
        return 2.0 * self.cells * self.drift_length

    def drift_matrix(self, length: float | None = None) -> np.ndarray:
        distance = self.drift_length if length is None else length
        return np.array([[1.0, distance], [0.0, 1.0]])

    def lens_matrix(self, kind: str, plane: str) -> np.ndarray:
        if kind not in {"F", "D"}:
            raise ValueError("kind must be 'F' or 'D'")
        if plane not in {"x", "y"}:
            raise ValueError("plane must be 'x' or 'y'")
        focuses = (kind == "F" and plane == "x") or (
            kind == "D" and plane == "y"
        )
        kick = (-1.0 if focuses else 1.0) / self.focal_length
        return np.array([[1.0, 0.0], [kick, 1.0]])

    def cell_matrix(self, plane: str) -> np.ndarray:
        drift = self.drift_matrix()
        return (
            self.lens_matrix("F", plane)
            @ drift
            @ self.lens_matrix("D", plane)
            @ drift
        )

    def matched_covariance(self, plane: str) -> np.ndarray:
        matrix = self.cell_matrix(plane)
        cosine = np.trace(matrix) / 2.0
        phase_advance = np.arccos(cosine)
        sine = np.sin(phase_advance)
        beta = matrix[0, 1] / sine
        alpha = (matrix[0, 0] - matrix[1, 1]) / (2.0 * sine)
        gamma = (1.0 + alpha**2) / beta
        return self.emittance * np.array(
            [[beta, -alpha], [-alpha, gamma]],
            dtype=float,
        )

    def covariance_at(self, longitudinal: float, plane: str) -> np.ndarray:
        position = float(np.clip(longitudinal, 0.0, self.total_length))
        covariance = self.matched_covariance(plane)
        current = 0.0
        boundary_count = 2 * self.cells

        for boundary_index in range(1, boundary_count + 1):
            boundary = boundary_index * self.drift_length
            if position <= boundary:
                drift = self.drift_matrix(position - current)
                return drift @ covariance @ drift.T

            drift = self.drift_matrix(self.drift_length)
            covariance = drift @ covariance @ drift.T
            kind = "D" if boundary_index % 2 == 1 else "F"
            lens = self.lens_matrix(kind, plane)
            covariance = lens @ covariance @ lens.T
            current = boundary

        return covariance

    def sigma(self, longitudinal: float, plane: str) -> float:
        return float(np.sqrt(self.covariance_at(longitudinal, plane)[0, 0]))

    def sample_envelope(self, plane: str, count: int = 240) -> list[np.ndarray]:
        if count < 2:
            raise ValueError("count must be at least two")
        return [
            np.array([position, self.sigma(float(position), plane), 0.0])
            for position in np.linspace(0.0, self.total_length, count)
        ]
