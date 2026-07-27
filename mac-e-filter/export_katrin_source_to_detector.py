#!/usr/bin/env python3
"""Export KATRIN source-to-detector adiabatic guiding-centre transport."""

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
    adiabatic_transport_along_path,
    integrate_relativistic_magnetic_trajectory,
    normalized_momentum_from_angles,
)
from mac_e_filter.fields import (  # noqa: E402
    integrate_axisymmetric_field_line_radius,
)
from mac_e_filter.katrin_nominal import (  # noqa: E402
    SUPERCONDUCTING_SOURCES_2013,
    build_katrin_2013_field,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT
        / "configs"
        / "katrin_2013_source_to_detector.toml",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=PROJECT_ROOT
        / "datasets"
        / "katrin_2013_source_to_detector.npz",
    )
    return parser.parse_args()


def _interp(
    common_time_s: np.ndarray,
    source_time_s: np.ndarray,
    values: np.ndarray,
) -> np.ndarray:
    return np.interp(common_time_s, source_time_s, values)


def main() -> None:
    args = parse_args()
    with args.config.open("rb") as stream:
        config = tomllib.load(stream)

    field_configuration = str(config["field"]["configuration"])
    field_model = build_katrin_2013_field(field_configuration)
    kinetic_energy_ev = float(config["particle"]["kinetic_energy_ev"])
    beamline = config["beamline"]
    start_z_m = float(beamline["start_z_m"])
    stop_z_m = float(beamline["stop_z_m"])
    path_samples = int(beamline["path_samples"])
    common_time_samples = int(beamline["common_time_samples"])
    source_radius_m = float(beamline["source_guiding_center_radius_m"])
    source_flux_radius_m = float(beamline["source_flux_tube_radius_m"])

    z_m = np.linspace(start_z_m, stop_z_m, path_samples)
    guiding_radius_m = integrate_axisymmetric_field_line_radius(
        field_model.total,
        z_m,
        source_radius_m,
    )
    flux_radius_m = integrate_axisymmetric_field_line_radius(
        field_model.total,
        z_m,
        source_flux_radius_m,
    )

    ensemble = config["ensemble"]
    source_pitch_deg = np.asarray(
        ensemble["source_pitch_angles_deg"],
        dtype=float,
    )
    azimuth_deg = np.asarray(ensemble["azimuth_angles_deg"], dtype=float)
    labels = [str(value) for value in ensemble["labels"]]
    colors = [str(value) for value in ensemble["colors"]]
    count = source_pitch_deg.size
    if not (
        azimuth_deg.size == count
        and len(labels) == count
        and len(colors) == count
    ):
        raise ValueError("All ensemble arrays must have equal length")

    raw_solutions = []
    raw_positions = []
    raw_fields = []
    for pitch, azimuth in zip(
        source_pitch_deg,
        azimuth_deg,
        strict=True,
    ):
        phi = np.deg2rad(float(azimuth))
        cosine = np.cos(phi)
        sine = np.sin(phi)
        position = np.column_stack(
            (
                guiding_radius_m * cosine,
                guiding_radius_m * sine,
                z_m,
            )
        )
        b_rho, b_z = field_model.total.field_cylindrical(
            guiding_radius_m,
            z_m,
        )
        magnetic_field = np.column_stack(
            (
                b_rho * cosine,
                b_rho * sine,
                b_z,
            )
        )
        raw_positions.append(position)
        raw_fields.append(magnetic_field)
        raw_solutions.append(
            adiabatic_transport_along_path(
                position,
                magnetic_field,
                kinetic_energy_ev=kinetic_energy_ev,
                initial_pitch_deg=float(pitch),
                particle=ELECTRON,
            )
        )

    maximum_time_s = max(
        float(solution.time_s[-1]) for solution in raw_solutions
    )
    common_time_s = np.linspace(
        0.0,
        maximum_time_s,
        common_time_samples,
    )

    guiding_center_m = []
    magnetic_field_t = []
    field_magnitude_t = []
    pitch_angle_deg = []
    parallel_speed_m_s = []
    perpendicular_speed_m_s = []
    larmor_radius_m = []
    gyro_phase_rad = []
    cumulative_turns = []
    adiabaticity_per_radian = []
    active = []
    tracks_metadata = []
    for index, solution in enumerate(raw_solutions):
        coordinates = np.column_stack(
            [
                _interp(common_time_s, solution.time_s, solution.path_position_m[:, axis])
                for axis in range(3)
            ]
        )
        fields = np.column_stack(
            [
                _interp(common_time_s, solution.time_s, solution.magnetic_field_t[:, axis])
                for axis in range(3)
            ]
        )
        guiding_center_m.append(coordinates)
        magnetic_field_t.append(fields)
        field_magnitude_t.append(
            _interp(
                common_time_s,
                solution.time_s,
                solution.magnetic_field_magnitude_t,
            )
        )
        pitch_angle_deg.append(
            _interp(
                common_time_s,
                solution.time_s,
                solution.pitch_angle_deg,
            )
        )
        parallel_speed_m_s.append(
            _interp(
                common_time_s,
                solution.time_s,
                solution.parallel_speed_m_s,
            )
        )
        perpendicular_speed_m_s.append(
            _interp(
                common_time_s,
                solution.time_s,
                solution.perpendicular_speed_m_s,
            )
        )
        larmor_radius_m.append(
            _interp(
                common_time_s,
                solution.time_s,
                solution.larmor_radius_m,
            )
        )
        gyro_phase_rad.append(
            _interp(
                common_time_s,
                solution.time_s,
                solution.gyro_phase_rad,
            )
        )
        cumulative_turns.append(
            _interp(
                common_time_s,
                solution.time_s,
                solution.cumulative_turns,
            )
        )
        adiabaticity_per_radian.append(
            _interp(
                common_time_s,
                solution.time_s,
                solution.adiabaticity_per_radian,
            )
        )
        active.append(common_time_s <= solution.time_s[-1])

        analyzing_index = int(
            np.argmin(np.abs(solution.path_position_m[:, 2]))
        )
        tracks_metadata.append(
            {
                "label": labels[index],
                "color": colors[index],
                "source_pitch_deg": float(source_pitch_deg[index]),
                "azimuth_deg": float(azimuth_deg[index]),
                "source_pitch_recovered_deg": float(
                    solution.pitch_angle_deg[0]
                ),
                "analysis_pitch_deg": float(
                    solution.pitch_angle_deg[analyzing_index]
                ),
                "detector_pitch_deg": float(
                    solution.pitch_angle_deg[-1]
                ),
                "analysis_larmor_radius_m": float(
                    solution.larmor_radius_m[analyzing_index]
                ),
                "flight_time_s": float(solution.time_s[-1]),
                "cyclotron_turns": float(solution.cumulative_turns[-1]),
                "max_adiabaticity_per_radian": float(
                    np.max(solution.adiabaticity_per_radian)
                ),
            }
        )

    source_field_t = float(
        raw_solutions[0].magnetic_field_magnitude_t[0]
    )
    axis_field_t = np.abs(field_model.total.axis_field(z_m))
    analysis_index = int(np.argmin(np.abs(z_m)))

    validation_config = config["central_full_lorentz_validation"]
    validation_start_z_m = float(validation_config["start_z_m"])
    validation_stop_z_m = float(validation_config["stop_z_m"])
    validation_start_field_t = abs(
        float(field_model.total.axis_field(validation_start_z_m))
    )
    validation_analysis_field_t = abs(
        float(field_model.total.axis_field(0.0))
    )
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
        field_bound_t=float(validation_config["field_bound_t"]),
        output_samples=int(validation_config["output_samples"]),
    )
    validation_adiabatic_pitch = []
    validation_full_lorentz_pitch = []
    validation_relative_difference = []
    validation_nfev = []
    for source_pitch in source_pitch_deg:
        local_pitch = float(
            adiabatic_pitch_angle_deg(
                source_pitch,
                source_field_t,
                validation_start_field_t,
            )
        )
        adiabatic_analysis_pitch = float(
            adiabatic_pitch_angle_deg(
                source_pitch,
                source_field_t,
                validation_analysis_field_t,
            )
        )
        exact_solution = integrate_relativistic_magnetic_trajectory(
            field_model.total,
            [0.0, 0.0, validation_start_z_m],
            normalized_momentum_from_angles(
                kinetic_energy_ev,
                local_pitch,
                0.0,
                ELECTRON,
            ),
            stop_z_m=validation_stop_z_m,
            maximum_time_s=(
                float(validation_config["maximum_time_ns"]) * 1.0e-9
            ),
            particle=ELECTRON,
            settings=validation_settings,
        )
        exact_analysis_pitch = float(
            np.interp(
                0.0,
                exact_solution.position_m[:, 2],
                exact_solution.diagnostics.pitch_angle_deg,
            )
        )
        validation_adiabatic_pitch.append(adiabatic_analysis_pitch)
        validation_full_lorentz_pitch.append(exact_analysis_pitch)
        validation_relative_difference.append(
            100.0
            * (
                exact_analysis_pitch / adiabatic_analysis_pitch
                - 1.0
            )
        )
        validation_nfev.append(
            int(exact_solution.solver_statistics["nfev"])
        )

    metadata = {
        "dataset_version": 2,
        "dataset_type": "katrin_source_to_detector_adiabatic",
        "scenario_name": str(config["scenario"]["name"]),
        "particle": asdict(ELECTRON),
        "kinetic_energy_ev": kinetic_energy_ev,
        "electric_field": "zero",
        "field_configuration": field_configuration,
        "field_reference": field_model.reference,
        "field_validity_note": field_model.validity_note,
        "transport_method": {
            "name": "relativistic first-adiabatic-invariant guiding centre",
            "conserved_quantities": [
                "total momentum magnitude",
                "p_perp^2 / |B|",
            ],
            "not_a_full_orbit": True,
            "reason": (
                "The published 2013 table does not provide an as-built field "
                "map for precision full-orbit tracking near superconducting "
                "magnets. The open Kassiopeia repository does not ship the "
                "KATRIN experiment configuration."
            ),
        },
        "central_full_lorentz_validation": {
            "interval_z_m": [
                validation_start_z_m,
                validation_stop_z_m,
            ],
            "source_pitch_deg": source_pitch_deg.tolist(),
            "adiabatic_analysis_pitch_deg": validation_adiabatic_pitch,
            "full_lorentz_analysis_pitch_deg": (
                validation_full_lorentz_pitch
            ),
            "relative_difference_percent": (
                validation_relative_difference
            ),
            "full_lorentz_equation": (
                "dx/dt = p/(gamma m), dp/dt = q v cross B"
            ),
            "solver": {
                "method": "DOP853",
                "settings": asdict(validation_settings),
                "function_evaluations": validation_nfev,
            },
        },
        "beamline": {
            "start_z_m": start_z_m,
            "stop_z_m": stop_z_m,
            "source_guiding_center_radius_m": source_radius_m,
            "source_flux_tube_radius_m": source_flux_radius_m,
            "source_field_t": source_field_t,
            "analysis_axis_field_t": float(axis_field_t[analysis_index]),
            "maximum_axis_field_t": float(np.max(axis_field_t)),
        },
        "tracks": tracks_metadata,
        "superconducting_sources": [
            {
                "name": source.name,
                "z_m": source.center_z_m,
                "published_typical_max_field_t": source.typical_max_field_t,
            }
            for source in SUPERCONDUCTING_SOURCES_2013
        ],
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out,
        metadata_json=np.array(
            json.dumps(metadata, ensure_ascii=False),
            dtype="<U65535",
        ),
        time_s=common_time_s,
        guiding_center_m=np.asarray(guiding_center_m),
        magnetic_field_t=np.asarray(magnetic_field_t),
        magnetic_field_magnitude_t=np.asarray(field_magnitude_t),
        pitch_angle_deg=np.asarray(pitch_angle_deg),
        parallel_speed_m_s=np.asarray(parallel_speed_m_s),
        perpendicular_speed_m_s=np.asarray(perpendicular_speed_m_s),
        larmor_radius_m=np.asarray(larmor_radius_m),
        gyro_phase_rad=np.asarray(gyro_phase_rad),
        cumulative_turns=np.asarray(cumulative_turns),
        adiabaticity_per_radian=np.asarray(adiabaticity_per_radian),
        active=np.asarray(active),
        field_line_z_m=z_m,
        guiding_radius_m=guiding_radius_m,
        flux_radius_m=flux_radius_m,
        axis_field_magnitude_t=axis_field_t,
    )

    print(f"Saved source-to-detector dataset: {args.out}")
    print(
        f"B: source={source_field_t:.9g} T, "
        f"analysis={axis_field_t[analysis_index]:.9g} T, "
        f"max={np.max(axis_field_t):.9g} T"
    )
    for track in tracks_metadata:
        print(
            f"{track['label']}: "
            f"{track['source_pitch_deg']:.3f}° -> "
            f"{track['analysis_pitch_deg']:.6f}° -> "
            f"{track['detector_pitch_deg']:.3f}°; "
            f"N={track['cyclotron_turns']:.1f}, "
            f"t={track['flight_time_s'] * 1e9:.3f} ns"
        )


if __name__ == "__main__":
    main()
