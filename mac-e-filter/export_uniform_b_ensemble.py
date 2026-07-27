#!/usr/bin/env python3
"""Export three exact relativistic helices in a uniform magnetic field."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
import tomllib

import numpy as np
from scipy.constants import c, elementary_charge

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from mac_e_filter.dynamics import (  # noqa: E402
    ELECTRON,
    IntegrationSettings,
    integrate_relativistic_magnetic_trajectory,
    normalized_momentum_from_angles,
    uniform_field_exact_solution,
)
from mac_e_filter.fields import UniformAxialField  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "uniform_b_ensemble.toml",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=PROJECT_ROOT / "datasets" / "uniform_b_ensemble.npz",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with args.config.open("rb") as stream:
        config = tomllib.load(stream)

    field_t = float(config["field"]["magnetic_field_t"])
    if field_t == 0.0:
        raise ValueError("magnetic_field_t must be non-zero")
    kinetic_energy_ev = float(config["particle"]["kinetic_energy_ev"])
    physical_duration_s = (
        float(config["animation"]["physical_duration_ns"]) * 1.0e-9
    )
    sample_count = int(config["animation"]["samples"])
    start_z_m = float(config["animation"]["start_z_m"])
    time_s = np.linspace(0.0, physical_duration_s, sample_count)

    ensemble = config["ensemble"]
    pitch_deg = np.asarray(ensemble["pitch_angles_deg"], dtype=float)
    gyro_phase_deg = np.asarray(ensemble["gyro_phases_deg"], dtype=float)
    center_x_m = np.asarray(
        ensemble["guiding_center_x_m"],
        dtype=float,
    )
    center_y_m = np.asarray(
        ensemble["guiding_center_y_m"],
        dtype=float,
    )
    labels = [str(value) for value in ensemble["labels"]]
    colors = [str(value) for value in ensemble["colors"]]
    track_count = pitch_deg.size
    if not (
        gyro_phase_deg.size == track_count
        and center_x_m.size == track_count
        and center_y_m.size == track_count
        and len(labels) == track_count
        and len(colors) == track_count
    ):
        raise ValueError("All ensemble arrays must have equal length")

    validation_config = config["validation"]
    validation_settings = IntegrationSettings(
        relative_tolerance=float(
            validation_config["relative_tolerance"]
        ),
        position_atol_m=float(validation_config["position_atol_m"]),
        normalized_momentum_atol=float(
            validation_config["normalized_momentum_atol"]
        ),
        max_gyro_phase_rad=float(
            validation_config["max_gyro_phase_rad"]
        ),
        field_bound_t=abs(field_t),
        output_samples=int(validation_config["output_samples"]),
    )
    field_model = UniformAxialField("exact uniform axial field", field_t)

    positions = []
    momenta = []
    velocities = []
    guiding_centers = []
    tracks_metadata = []
    numerical_validation = []
    for index in range(track_count):
        initial_u = normalized_momentum_from_angles(
            kinetic_energy_ev,
            float(pitch_deg[index]),
            float(gyro_phase_deg[index]),
            ELECTRON,
        )
        gamma = float(np.sqrt(1.0 + np.dot(initial_u, initial_u)))
        signed_omega = (
            ELECTRON.charge_c * field_t / (ELECTRON.mass_kg * gamma)
        )
        velocity_factor = c / gamma
        initial_position = np.array(
            [
                center_x_m[index]
                - velocity_factor / signed_omega * initial_u[1],
                center_y_m[index]
                + velocity_factor / signed_omega * initial_u[0],
                start_z_m,
            ]
        )
        position, momentum = uniform_field_exact_solution(
            initial_position,
            initial_u,
            time_s,
            field_t,
            ELECTRON,
        )
        velocity = c * momentum / gamma
        positions.append(position)
        momenta.append(momentum)
        velocities.append(velocity)
        guiding_centers.append(
            np.column_stack(
                (
                    np.full(time_s.size, center_x_m[index]),
                    np.full(time_s.size, center_y_m[index]),
                    position[:, 2],
                )
            )
        )

        physical_momentum = ELECTRON.mass_kg * c * np.linalg.norm(initial_u)
        perpendicular_momentum = (
            physical_momentum * np.sin(np.deg2rad(pitch_deg[index]))
        )
        larmor_radius_m = perpendicular_momentum / (
            abs(ELECTRON.charge_c) * abs(field_t)
        )
        speed_m_s = float(np.linalg.norm(velocity[0]))
        turns = abs(signed_omega) * physical_duration_s / (2.0 * np.pi)

        target_z_m = float(position[-1, 2])
        numerical = integrate_relativistic_magnetic_trajectory(
            field_model,
            initial_position,
            initial_u,
            stop_z_m=target_z_m,
            maximum_time_s=1.05 * physical_duration_s,
            particle=ELECTRON,
            settings=validation_settings,
        )
        exact_validation_position, exact_validation_momentum = (
            uniform_field_exact_solution(
                initial_position,
                initial_u,
                numerical.time_s,
                field_t,
                ELECTRON,
            )
        )
        position_error_m = float(
            np.max(
                np.linalg.norm(
                    numerical.position_m - exact_validation_position,
                    axis=1,
                )
            )
        )
        momentum_error = float(
            np.max(
                np.linalg.norm(
                    numerical.normalized_momentum
                    - exact_validation_momentum,
                    axis=1,
                )
            )
        )
        numerical_validation.append(
            {
                "pitch_angle_deg": float(pitch_deg[index]),
                "maximum_position_error_m": position_error_m,
                "maximum_normalized_momentum_error": momentum_error,
                "function_evaluations": int(
                    numerical.solver_statistics["nfev"]
                ),
            }
        )
        tracks_metadata.append(
            {
                "label": labels[index],
                "color": colors[index],
                "pitch_angle_deg": float(pitch_deg[index]),
                "gyro_phase_deg": float(gyro_phase_deg[index]),
                "guiding_center_x_m": float(center_x_m[index]),
                "guiding_center_y_m": float(center_y_m[index]),
                "speed_m_s": speed_m_s,
                "parallel_speed_m_s": float(
                    speed_m_s * np.cos(np.deg2rad(pitch_deg[index]))
                ),
                "perpendicular_speed_m_s": float(
                    speed_m_s * np.sin(np.deg2rad(pitch_deg[index]))
                ),
                "larmor_radius_m": float(larmor_radius_m),
                "cyclotron_frequency_hz": float(
                    abs(signed_omega) / (2.0 * np.pi)
                ),
                "turns": float(turns),
                "final_z_m": target_z_m,
            }
        )

    position_array = np.asarray(positions)
    momentum_array = np.asarray(momenta)
    velocity_array = np.asarray(velocities)
    gamma_array = np.sqrt(
        1.0 + np.einsum("tni,tni->tn", momentum_array, momentum_array)
    )
    kinetic_energy_array = (
        (gamma_array - 1.0)
        * ELECTRON.mass_kg
        * c**2
        / elementary_charge
    )
    metadata = {
        "dataset_version": 1,
        "dataset_type": "uniform_b_exact_ensemble",
        "scenario_name": str(config["scenario"]["name"]),
        "particle": asdict(ELECTRON),
        "kinetic_energy_ev": kinetic_energy_ev,
        "magnetic_field_t": field_t,
        "electric_field": "zero",
        "solution": {
            "name": "exact relativistic uniform-B helix",
            "equations": [
                "dx/dt = c u/gamma",
                "du/dt = q (u cross B)/(m gamma)",
                "omega = q B/(gamma m)",
            ],
            "coordinate_scale": "physical; no helix-radius exaggeration",
        },
        "physical_duration_s": physical_duration_s,
        "tracks": tracks_metadata,
        "numerical_validation": {
            "solver": {
                "method": "DOP853",
                "settings": asdict(validation_settings),
            },
            "tracks": numerical_validation,
        },
        "kinetic_energy_span_ev": float(
            np.max(np.ptp(kinetic_energy_array, axis=1))
        ),
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out,
        metadata_json=np.array(
            json.dumps(metadata, ensure_ascii=False),
            dtype="<U65535",
        ),
        time_s=time_s,
        position_m=position_array,
        guiding_center_m=np.asarray(guiding_centers),
        normalized_momentum=momentum_array,
        velocity_m_s=velocity_array,
        kinetic_energy_ev=kinetic_energy_array,
        magnetic_field_t=np.broadcast_to(
            np.array([0.0, 0.0, field_t]),
            position_array.shape,
        ).copy(),
    )

    print(f"Saved exact uniform-B ensemble: {args.out}")
    print(
        f"B={field_t * 1e3:.3f} mT, "
        f"K={kinetic_energy_ev:.3f} eV, "
        f"duration={physical_duration_s * 1e9:.3f} ns"
    )
    for track, validation in zip(
        tracks_metadata,
        numerical_validation,
        strict=True,
    ):
        print(
            f"{track['label']}: rL={track['larmor_radius_m']:.6f} m, "
            f"z_final={track['final_z_m']:+.4f} m, "
            f"N={track['turns']:.4f}, "
            f"validation dx={validation['maximum_position_error_m']:.3g} m, "
            f"du={validation['maximum_normalized_momentum_error']:.3g}"
        )


if __name__ == "__main__":
    main()

