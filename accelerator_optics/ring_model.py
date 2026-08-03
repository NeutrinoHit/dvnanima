from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class RingConfig:
    cells: int = 3
    drift_length: float = 1.0
    quadrupole_strength: float = 1.90
    sextupole_dispersion: float = 3.0
    aperture: float = 0.72

    def __post_init__(self) -> None:
        if self.cells < 2:
            raise ValueError("cells must be at least two")
        if self.drift_length <= 0.0:
            raise ValueError("drift_length must be positive")
        if self.quadrupole_strength <= 0.0:
            raise ValueError("quadrupole_strength must be positive")
        if self.sextupole_dispersion == 0.0:
            raise ValueError("sextupole_dispersion must be non-zero")
        if self.aperture <= 0.0:
            raise ValueError("aperture must be positive")

    @property
    def samples_per_turn(self) -> int:
        return 2 * self.cells


@dataclass(frozen=True)
class TrackingResult:
    states: np.ndarray
    alive: np.ndarray
    loss_step: np.ndarray
    config: RingConfig

    @property
    def turns(self) -> int:
        return (self.states.shape[0] - 1) // self.config.samples_per_turn

    @property
    def survival_by_sample(self) -> np.ndarray:
        return np.mean(self.alive, axis=1)

    def rms_sizes(self) -> tuple[np.ndarray, np.ndarray]:
        sigma_x = np.zeros(self.states.shape[0])
        sigma_y = np.zeros(self.states.shape[0])
        for index, (state, alive) in enumerate(zip(self.states, self.alive)):
            if np.any(alive):
                sigma_x[index] = float(np.std(state[alive, 0]))
                sigma_y[index] = float(np.std(state[alive, 2]))
        return sigma_x, sigma_y


def generate_momentum_spectrum(
    count: int = 72,
    *,
    seed: int = 20260803,
    momentum_half_width: float = 0.18,
) -> np.ndarray:
    if count < 3:
        raise ValueError("count must be at least three")
    if not 0.0 < momentum_half_width < 1.0:
        raise ValueError("momentum_half_width must lie between zero and one")
    generator = np.random.default_rng(seed)
    states = np.zeros((count, 5), dtype=float)
    states[:, 0] = generator.normal(0.0, 0.0018, count)
    states[:, 1] = generator.normal(0.0, 0.0014, count)
    states[:, 2] = generator.normal(0.0, 0.0018, count)
    states[:, 3] = generator.normal(0.0, 0.0014, count)
    states[:, 4] = np.linspace(-momentum_half_width, momentum_half_width, count)
    return states


class SymplecticRingTracker:
    """Educational thin-element ring with chromatic quadrupoles and sextupoles."""

    def __init__(self, config: RingConfig | None = None) -> None:
        self.config = config or RingConfig()

    def drift(self, states: np.ndarray) -> None:
        length = self.config.drift_length
        states[:, 0] += length * states[:, 1]
        states[:, 2] += length * states[:, 3]

    def quadrupole_kick(self, states: np.ndarray, kind: str) -> None:
        if kind not in {"F", "D"}:
            raise ValueError("kind must be 'F' or 'D'")
        strength = self.config.quadrupole_strength / (1.0 + states[:, 4])
        sign = 1.0 if kind == "F" else -1.0
        states[:, 1] -= sign * strength * states[:, 0]
        states[:, 3] += sign * strength * states[:, 2]

    def sextupole_kick(self, states: np.ndarray, kind: str) -> None:
        if kind not in {"F", "D"}:
            raise ValueError("kind must be 'F' or 'D'")
        sign = 1.0 if kind == "F" else -1.0
        strength = (
            sign
            * self.config.quadrupole_strength
            / self.config.sextupole_dispersion
        )
        x = states[:, 0].copy()
        y = states[:, 2].copy()
        delta = states[:, 4]
        dispersive_x = self.config.sextupole_dispersion * delta

        states[:, 1] -= strength * dispersive_x * x
        states[:, 1] -= 0.5 * strength * (x**2 - y**2)
        states[:, 3] += strength * dispersive_x * y
        states[:, 3] += strength * x * y

    def half_cell(self, states: np.ndarray, kind: str, sextupoles_on: bool) -> None:
        self.drift(states)
        self.quadrupole_kick(states, kind)
        if sextupoles_on:
            self.sextupole_kick(states, kind)

    def physical_radius(self, states: np.ndarray) -> np.ndarray:
        physical_x = (
            states[:, 0]
            + self.config.sextupole_dispersion * states[:, 4]
        )
        return np.hypot(physical_x, states[:, 2])

    def track(
        self,
        initial_states: np.ndarray,
        *,
        turns: int,
        sextupoles_on: bool,
    ) -> TrackingResult:
        if turns < 1:
            raise ValueError("turns must be positive")
        if initial_states.ndim != 2 or initial_states.shape[1] != 5:
            raise ValueError("initial_states must have shape (particles, 5)")

        particle_count = initial_states.shape[0]
        sample_count = turns * self.config.samples_per_turn + 1
        history = np.zeros((sample_count, particle_count, 5), dtype=float)
        alive_history = np.zeros((sample_count, particle_count), dtype=bool)
        loss_step = np.full(particle_count, -1, dtype=int)
        states = initial_states.astype(float, copy=True)
        alive = self.physical_radius(states) <= self.config.aperture
        history[0] = states
        alive_history[0] = alive

        sample = 0
        for _ in range(turns):
            for _ in range(self.config.cells):
                for kind in ("D", "F"):
                    sample += 1
                    active_indices = np.flatnonzero(alive)
                    if active_indices.size:
                        active = states[active_indices].copy()
                        self.half_cell(active, kind, sextupoles_on)
                        states[active_indices] = active
                    finite = np.all(np.isfinite(states), axis=1)
                    within_aperture = self.physical_radius(states) <= self.config.aperture
                    newly_lost = alive & (~finite | ~within_aperture)
                    loss_step[newly_lost] = sample
                    alive &= finite & within_aperture
                    history[sample] = states
                    alive_history[sample] = alive

        return TrackingResult(
            states=history,
            alive=alive_history,
            loss_step=loss_step,
            config=self.config,
        )

    def one_cell_linear_matrix(self, momentum_deviation: float, plane: str) -> np.ndarray:
        if plane not in {"x", "y"}:
            raise ValueError("plane must be 'x' or 'y'")
        length = self.config.drift_length
        base = self.config.quadrupole_strength / (1.0 + momentum_deviation)
        drift = np.array([[1.0, length], [0.0, 1.0]])

        def lens(kind: str) -> np.ndarray:
            focuses_x = kind == "F"
            focuses = focuses_x if plane == "x" else not focuses_x
            kick = -base if focuses else base
            return np.array([[1.0, 0.0], [kick, 1.0]])

        return lens("F") @ drift @ lens("D") @ drift
