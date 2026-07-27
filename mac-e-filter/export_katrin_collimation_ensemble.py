#!/usr/bin/env python3
"""Export several exact central-spectrometer full-orbit trajectories."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
import tomllib

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from mac_e_filter.dynamics import (  # noqa: E402
    ELECTRON,
    IntegrationSettings,
    adiabatic_pitch_angle_deg,
    integrate_relativistic_magnetic_trajectory,
    normalized_momentum_from_angles,
)
from mac_e_filter.katrin_nominal import build_katrin_2013_field  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT
        / "configs"
        / "katrin_2013_collimation_ensemble.toml",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=PROJECT_ROOT
        / "datasets"
        / "katrin_2013_collimation_ensemble.npz",
    )
    return parser.parse_args()


def _interpolate_columns(
    target_time: np.ndarray,
    source_time: np.ndarray,
    values: np.ndarray,
) -> np.ndarray:
    return np.column_stack(
        [
            np.interp(target_time, source_time, values[:, component])
            for component in range(values.shape[1])
        ]
    )


def main() -> None:
    args = parse_args()
    with args.config.open("rb") as stream:
        config = tomllib.load(stream)

    field_configuration = str(config["field"]["configuration"])
    field_model = build_katrin_2013_field(field_configuration)
    kinetic_energy_ev = float(config["particle"]["kinetic_energy_ev"])
    trajectory_config = config["trajectory"]
    initial_position = np.array(
        [
            float(trajectory_config.get("start_x_m", 0.0)),
            float(trajectory_config.get("start_y_m", 0.0)),
            float(trajectory_config["start_z_m"]),
        ]
    )
    stop_z_m = float(trajectory_config["stop_z_m"])
    maximum_time_s = float(trajectory_config["maximum_time_ns"]) * 1.0e-9

    ensemble_config = config["ensemble"]
    pitch_angles = np.asarray(
        ensemble_config["pitch_angles_deg"], dtype=float
    )
    gyro_phases = np.asarray(
        ensemble_config["gyro_phases_deg"], dtype=float
    )
    labels = [str(item) for item in ensemble_config["labels"]]
    colors = [str(item) for item in ensemble_config["colors"]]
    track_count = pitch_angles.size
    if not (
        gyro_phases.size == track_count
        and len(labels) == track_count
        and len(colors) == track_count
    ):
        raise ValueError("All ensemble arrays must have equal length")

    integration_config = config["integration"]
    settings = IntegrationSettings(
        relative_tolerance=float(integration_config["relative_tolerance"]),
        position_atol_m=float(integration_config["position_atol_m"]),
        normalized_momentum_atol=float(
            integration_config["normalized_momentum_atol"]
        ),
        max_gyro_phase_rad=float(
            integration_config["max_gyro_phase_rad"]
        ),
        field_bound_t=float(integration_config["field_bound_t"]),
        output_samples=int(integration_config["samples_per_track"]),
    )

    solutions = []
    for index, (pitch, phase) in enumerate(
        zip(pitch_angles, gyro_phases, strict=True),
        start=1,
    ):
        print(
            f"Integrating track {index}/{track_count}: "
            f"local pitch={pitch:g}°",
            flush=True,
        )
        solutions.append(
            integrate_relativistic_magnetic_trajectory(
                field_model.total,
                initial_position,
                normalized_momentum_from_angles(
                    kinetic_energy_ev,
                    float(pitch),
                    float(phase),
                    ELECTRON,
                ),
                stop_z_m=stop_z_m,
                maximum_time_s=maximum_time_s,
                particle=ELECTRON,
                settings=settings,
            )
        )

    common_time = np.linspace(
        0.0,
        max(float(solution.time_s[-1]) for solution in solutions),
        int(integration_config["common_time_samples"]),
    )
    position = []
    normalized_momentum = []
    velocity = []
    magnetic_field = []
    field_magnitude = []
    pitch = []
    active = []
    tracks_metadata = []
    for index, solution in enumerate(solutions):
        diagnostics = solution.diagnostics
        position.append(
            _interpolate_columns(
                common_time, solution.time_s, solution.position_m
            )
        )
        normalized_momentum.append(
            _interpolate_columns(
                common_time,
                solution.time_s,
                solution.normalized_momentum,
            )
        )
        velocity.append(
            _interpolate_columns(
                common_time,
                solution.time_s,
                diagnostics.velocity_m_s,
            )
        )
        magnetic_field.append(
            _interpolate_columns(
                common_time,
                solution.time_s,
                diagnostics.magnetic_field_t,
            )
        )
        field_magnitude.append(
            np.interp(
                common_time,
                solution.time_s,
                diagnostics.magnetic_field_magnitude_t,
            )
        )
        pitch.append(
            np.interp(
                common_time,
                solution.time_s,
                diagnostics.pitch_angle_deg,
            )
        )
        active.append(common_time <= solution.time_s[-1])
        center_pitch = float(
            np.interp(
                0.0,
                solution.position_m[:, 2],
                diagnostics.pitch_angle_deg,
            )
        )
        tracks_metadata.append(
            {
                "label": labels[index],
                "color": colors[index],
                "initial_local_pitch_deg": float(pitch_angles[index]),
                "gyro_phase_deg": float(gyro_phases[index]),
                "center_instantaneous_pitch_deg": center_pitch,
                "final_instantaneous_pitch_deg": float(
                    diagnostics.pitch_angle_deg[-1]
                ),
                "flight_time_s": float(solution.time_s[-1]),
                "solver": solution.solver_statistics,
            }
        )

    source_mapping = config["source_mapping"]
    source_field_t = float(source_mapping["source_field_t"])
    source_pitch = np.asarray(
        source_mapping["source_pitch_angles_deg"], dtype=float
    )
    analyzing_field_t = abs(float(field_model.total.axis_field(0.0)))
    mapped_pitch = adiabatic_pitch_angle_deg(
        source_pitch,
        source_field_t,
        analyzing_field_t,
    )
    metadata = {
        "dataset_version": 1,
        "dataset_type": "katrin_collimation_ensemble",
        "scenario_name": str(config["scenario"]["name"]),
        "particle": asdict(ELECTRON),
        "kinetic_energy_ev": kinetic_energy_ev,
        "field_configuration": field_configuration,
        "field_reference": field_model.reference,
        "field_validity_note": field_model.validity_note,
        "electric_field": "zero",
        "initial_position_m": initial_position.tolist(),
        "stop_z_m": stop_z_m,
        "tracks": tracks_metadata,
        "interpretation": (
            "Track pitch angles are local at z=-6.5 m. They demonstrate the "
            "reversible magnetic-only redistribution between transverse and "
            "longitudinal momentum inside the documented central field."
        ),
        "source_to_analysis_adiabatic_mapping": {
            "status": "guiding-centre comparison, not integrated full orbit",
            "source_field_t": source_field_t,
            "analyzing_field_t": analyzing_field_t,
            "source_pitch_angles_deg": source_pitch.tolist(),
            "mapped_analysis_pitch_angles_deg": mapped_pitch.tolist(),
            "relation": "sin(theta_a)^2 / B_a = sin(theta_s)^2 / B_s",
        },
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out,
        metadata_json=np.array(
            json.dumps(metadata, ensure_ascii=False),
            dtype="<U65535",
        ),
        time_s=common_time,
        position_m=np.asarray(position),
        normalized_momentum=np.asarray(normalized_momentum),
        velocity_m_s=np.asarray(velocity),
        magnetic_field_t=np.asarray(magnetic_field),
        magnetic_field_magnitude_t=np.asarray(field_magnitude),
        pitch_angle_deg=np.asarray(pitch),
        active=np.asarray(active),
    )
    print(f"Saved ensemble dataset: {args.out}")
    print("Full-orbit central-field tracks:")
    for track in tracks_metadata:
        print(
            f"  {track['label']}: "
            f"pitch {track['initial_local_pitch_deg']:.3f}° -> "
            f"{track['center_instantaneous_pitch_deg']:.3f}° at z=0 -> "
            f"{track['final_instantaneous_pitch_deg']:.3f}°"
        )
    print("Adiabatic source-to-analysis comparison:")
    for source_angle, analysis_angle in zip(
        source_pitch, mapped_pitch, strict=True
    ):
        print(f"  {source_angle:.1f}° -> {analysis_angle:.6f}°")


if __name__ == "__main__":
    main()

