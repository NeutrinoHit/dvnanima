from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SectorTrajectory:
    """Straight entry, circular sector, and straight exit at constant speed."""

    center_x: float
    center_y: float
    radius: float
    entry_start_y: float
    exit_end_x: float

    def __post_init__(self) -> None:
        if self.radius <= 0.0:
            raise ValueError("radius must be positive")
        if self.entry_start_y >= self.center_y:
            raise ValueError("entry_start_y must lie below the sector entrance")
        if self.exit_end_x <= self.center_x:
            raise ValueError("exit_end_x must lie to the right of the sector exit")

    @property
    def center(self) -> np.ndarray:
        return np.array([self.center_x, self.center_y, 0.0], dtype=float)

    @property
    def entrance(self) -> np.ndarray:
        return np.array(
            [self.center_x - self.radius, self.center_y, 0.0],
            dtype=float,
        )

    @property
    def exit(self) -> np.ndarray:
        return np.array(
            [self.center_x, self.center_y + self.radius, 0.0],
            dtype=float,
        )

    @property
    def entry_start(self) -> np.ndarray:
        return np.array(
            [self.center_x - self.radius, self.entry_start_y, 0.0],
            dtype=float,
        )

    @property
    def exit_end(self) -> np.ndarray:
        return np.array(
            [self.exit_end_x, self.center_y + self.radius, 0.0],
            dtype=float,
        )

    @property
    def entry_length(self) -> float:
        return self.center_y - self.entry_start_y

    @property
    def arc_length(self) -> float:
        return 0.5 * np.pi * self.radius

    @property
    def exit_length(self) -> float:
        return self.exit_end_x - self.center_x

    @property
    def total_length(self) -> float:
        return self.entry_length + self.arc_length + self.exit_length

    def point_and_tangent(self, progress: float) -> tuple[np.ndarray, np.ndarray]:
        """Return position and unit tangent for a constant-speed progress value."""

        distance = float(np.clip(progress, 0.0, 1.0)) * self.total_length

        if distance <= self.entry_length:
            point = self.entry_start + np.array([0.0, distance, 0.0])
            return point, np.array([0.0, 1.0, 0.0])

        distance -= self.entry_length
        if distance <= self.arc_length:
            theta = np.pi - distance / self.radius
            point = self.center + self.radius * np.array(
                [np.cos(theta), np.sin(theta), 0.0]
            )
            tangent = np.array([np.sin(theta), -np.cos(theta), 0.0])
            return point, tangent

        distance -= self.arc_length
        point = self.exit + np.array([distance, 0.0, 0.0])
        return point, np.array([1.0, 0.0, 0.0])

    def is_inside_field(self, progress: float) -> bool:
        distance = float(np.clip(progress, 0.0, 1.0)) * self.total_length
        return self.entry_length <= distance <= self.entry_length + self.arc_length

    def inward_normal(self, progress: float) -> np.ndarray:
        """Return the Lorentz-force direction inside the hard-edge dipole."""

        if not self.is_inside_field(progress):
            raise ValueError("the particle is outside the magnetic field")
        _, tangent = self.point_and_tangent(progress)
        return np.array([tangent[1], -tangent[0], 0.0])

    def sample_points(self, count: int = 180) -> list[np.ndarray]:
        if count < 2:
            raise ValueError("count must be at least two")
        return [
            self.point_and_tangent(progress)[0]
            for progress in np.linspace(0.0, 1.0, count)
        ]
