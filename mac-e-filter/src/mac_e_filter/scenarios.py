"""Configuration loading for reproducible trajectory scenarios."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import tomllib

import numpy as np
from numpy.typing import NDArray

from .dynamics import (
    ELECTRON,
    IntegrationSettings,
    normalized_momentum_from_angles,
)

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class KatrinTrajectoryScenario:
    name: str
    field_configuration: str
    kinetic_energy_ev: float
    pitch_to_positive_z_deg: float
    gyro_phase_deg: float
    initial_position_m: FloatArray
    stop_z_m: float
    maximum_time_s: float
    integration: IntegrationSettings

    def initial_normalized_momentum(self) -> FloatArray:
        return normalized_momentum_from_angles(
            self.kinetic_energy_ev,
            self.pitch_to_positive_z_deg,
            self.gyro_phase_deg,
            ELECTRON,
        )


def load_katrin_trajectory_scenario(
    path: str | Path,
) -> KatrinTrajectoryScenario:
    config_path = Path(path)
    with config_path.open("rb") as stream:
        data = tomllib.load(stream)

    particle = data["particle"]
    trajectory = data["trajectory"]
    integration = data["integration"]
    field = data["field"]
    return KatrinTrajectoryScenario(
        name=str(data["scenario"]["name"]),
        field_configuration=str(field["configuration"]),
        kinetic_energy_ev=float(particle["kinetic_energy_ev"]),
        pitch_to_positive_z_deg=float(
            particle["pitch_to_positive_z_deg"]
        ),
        gyro_phase_deg=float(particle.get("gyro_phase_deg", 0.0)),
        initial_position_m=np.array(
            [
                float(trajectory.get("start_x_m", 0.0)),
                float(trajectory.get("start_y_m", 0.0)),
                float(trajectory["start_z_m"]),
            ]
        ),
        stop_z_m=float(trajectory["stop_z_m"]),
        maximum_time_s=float(trajectory["maximum_time_ns"]) * 1.0e-9,
        integration=IntegrationSettings(
            relative_tolerance=float(integration["relative_tolerance"]),
            position_atol_m=float(integration["position_atol_m"]),
            normalized_momentum_atol=float(
                integration["normalized_momentum_atol"]
            ),
            max_gyro_phase_rad=float(integration["max_gyro_phase_rad"]),
            field_bound_t=float(integration["field_bound_t"]),
            output_samples=int(integration["output_samples"]),
        ),
    )

