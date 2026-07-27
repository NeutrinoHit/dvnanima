#!/usr/bin/env python3
"""Export relativistic adiabatic KATRIN transport with electrostatic retardation."""

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
    adiabatic_electrostatic_transport_along_path,
    integrate_relativistic_electromagnetic_trajectory,
    kinetic_energy_ev_to_normalized_momentum,
)
from mac_e_filter.electrostatics import (  # noqa: E402
    FiniteCylindricalElectrodePotential,
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
        default=PROJECT_ROOT / "configs" / "katrin_2013_mac_e.toml",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=PROJECT_ROOT / "datasets" / "katrin_2013_mac_e.npz",
    )
    return parser.parse_args()


def _interpolate(
    common_time_s: np.ndarray,
    source_time_s: np.ndarray,
    values: np.ndarray,
) -> np.ndarray:
    return np.interp(common_time_s, source_time_s, values)


def _interpolate_vector(
    common_time_s: np.ndarray,
    source_time_s: np.ndarray,
    values: np.ndarray,
) -> np.ndarray:
    return np.column_stack(
        [
            _interpolate(common_time_s, source_time_s, values[:, axis])
            for axis in range(values.shape[1])
        ]
    )


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
    b_rho, b_z = field_model.total.field_cylindrical(
        guiding_radius_m,
        z_m,
    )
    field_magnitude_t = np.hypot(b_rho, b_z)

    electrostatic = config["electrostatic"]
    unit_potential_model = FiniteCylindricalElectrodePotential(
        radius_m=float(electrostatic["cylinder_radius_m"]),
        length_m=float(electrostatic["electrode_length_m"]),
        boundary_potential_v=1.0,
        quadrature_order=int(electrostatic["quadrature_order"]),
        k_max_inv_m=float(electrostatic["k_max_inv_m"]),
    )
    unit_path_potential = unit_potential_model.potential_cylindrical(
        guiding_radius_m,
        z_m,
    )
    analysis_index = int(np.argmin(np.abs(z_m)))
    unit_retarding_response = float(
        unit_path_potential[analysis_index] - unit_path_potential[0]
    )
    retarding_energy_ev = float(electrostatic["retarding_energy_ev"])
    boundary_potential_v = -retarding_energy_ev / unit_retarding_response
    potential_model = unit_potential_model.with_boundary_potential(
        boundary_potential_v
    )
    path_potential_v = boundary_potential_v * unit_path_potential
    path_e_rho_v_m, path_e_z_v_m = (
        potential_model.electric_field_cylindrical(
            guiding_radius_m,
            z_m,
        )
    )
    axis_potential_v = potential_model.potential_cylindrical(
        np.zeros_like(z_m),
        z_m,
    )
    _, axis_e_z_v_m = potential_model.electric_field_cylindrical(
        np.zeros_like(z_m),
        z_m,
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
        magnetic_field = np.column_stack(
            (
                b_rho * cosine,
                b_rho * sine,
                b_z,
            )
        )
        raw_solutions.append(
            adiabatic_electrostatic_transport_along_path(
                position,
                magnetic_field,
                path_potential_v,
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
    electric_potential_v = []
    kinetic_energy = []
    total_energy = []
    pitch_angle_deg = []
    parallel_speed_m_s = []
    perpendicular_speed_m_s = []
    parallel_normalized_momentum = []
    perpendicular_normalized_momentum = []
    larmor_radius_m = []
    cumulative_turns = []
    active = []
    tracks_metadata = []
    for index, solution in enumerate(raw_solutions):
        guiding_center_m.append(
            _interpolate_vector(
                common_time_s,
                solution.time_s,
                solution.path_position_m,
            )
        )
        magnetic_field_t.append(
            _interpolate_vector(
                common_time_s,
                solution.time_s,
                solution.magnetic_field_t,
            )
        )
        electric_potential_v.append(
            _interpolate(
                common_time_s,
                solution.time_s,
                solution.electric_potential_v,
            )
        )
        kinetic_energy.append(
            _interpolate(
                common_time_s,
                solution.time_s,
                solution.kinetic_energy_ev,
            )
        )
        total_energy.append(
            _interpolate(
                common_time_s,
                solution.time_s,
                solution.total_energy_ev,
            )
        )
        pitch_angle_deg.append(
            _interpolate(
                common_time_s,
                solution.time_s,
                solution.pitch_angle_deg,
            )
        )
        parallel_speed_m_s.append(
            _interpolate(
                common_time_s,
                solution.time_s,
                solution.parallel_speed_m_s,
            )
        )
        perpendicular_speed_m_s.append(
            _interpolate(
                common_time_s,
                solution.time_s,
                solution.perpendicular_speed_m_s,
            )
        )
        parallel_normalized_momentum.append(
            _interpolate(
                common_time_s,
                solution.time_s,
                solution.parallel_normalized_momentum,
            )
        )
        perpendicular_normalized_momentum.append(
            _interpolate(
                common_time_s,
                solution.time_s,
                solution.perpendicular_normalized_momentum,
            )
        )
        larmor_radius_m.append(
            _interpolate(
                common_time_s,
                solution.time_s,
                solution.larmor_radius_m,
            )
        )
        cumulative_turns.append(
            _interpolate(
                common_time_s,
                solution.time_s,
                solution.cumulative_turns,
            )
        )
        active.append(common_time_s <= solution.time_s[-1])

        total_energy_residual_ev = float(
            np.ptp(solution.total_energy_ev)
        )
        invariant = (
            solution.perpendicular_normalized_momentum**2
            / solution.magnetic_field_magnitude_t
        )
        invariant_relative_span = float(
            np.ptp(invariant) / invariant[0]
        )
        if solution.outcome == "transmitted":
            local_analysis_index = int(
                np.argmin(np.abs(solution.path_position_m[:, 2]))
            )
            analysis_pitch_deg = float(
                solution.pitch_angle_deg[local_analysis_index]
            )
            analysis_kinetic_energy_ev = float(
                solution.kinetic_energy_ev[local_analysis_index]
            )
            turning_z_m = None
        else:
            analysis_pitch_deg = None
            analysis_kinetic_energy_ev = None
            turning_z_m = float(solution.turning_position_m[2])
        tracks_metadata.append(
            {
                "label": labels[index],
                "color": colors[index],
                "source_pitch_deg": float(source_pitch_deg[index]),
                "azimuth_deg": float(azimuth_deg[index]),
                "outcome": solution.outcome,
                "turning_z_m": turning_z_m,
                "analysis_pitch_deg": analysis_pitch_deg,
                "analysis_kinetic_energy_ev": analysis_kinetic_energy_ev,
                "minimum_kinetic_energy_ev": float(
                    np.min(solution.kinetic_energy_ev)
                ),
                "flight_time_s": float(solution.time_s[-1]),
                "cyclotron_turns": float(solution.cumulative_turns[-1]),
                "total_energy_residual_ev": total_energy_residual_ev,
                "magnetic_invariant_relative_span": (
                    invariant_relative_span
                ),
            }
        )

    convergence = config["convergence"]
    check_indices = np.linspace(
        0,
        z_m.size - 1,
        int(convergence["check_samples"]),
    ).round().astype(int)
    refined_unit = FiniteCylindricalElectrodePotential(
        radius_m=potential_model.radius_m,
        length_m=potential_model.length_m,
        boundary_potential_v=boundary_potential_v,
        quadrature_order=int(convergence["refined_quadrature_order"]),
        k_max_inv_m=float(convergence["refined_k_max_inv_m"]),
    )
    refined_potential = refined_unit.potential_cylindrical(
        guiding_radius_m[check_indices],
        z_m[check_indices],
    )
    baseline_potential = potential_model.potential_cylindrical(
        guiding_radius_m[check_indices],
        z_m[check_indices],
    )
    convergence_max_abs_v = float(
        np.max(np.abs(refined_potential - baseline_potential))
    )

    source_potential_v = float(path_potential_v[0])
    analysis_potential_v = float(path_potential_v[analysis_index])
    validation_config = config["central_full_lorentz_validation"]
    validation_start_z_m = float(validation_config["start_z_m"])
    validation_lower_stop_z_m = float(
        validation_config["lower_stop_z_m"]
    )
    validation_upper_stop_z_m = float(
        validation_config["upper_stop_z_m"]
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
    validation_potential_model = FiniteCylindricalElectrodePotential(
        radius_m=potential_model.radius_m,
        length_m=potential_model.length_m,
        boundary_potential_v=boundary_potential_v,
        quadrature_order=int(
            validation_config["electric_quadrature_order"]
        ),
        k_max_inv_m=potential_model.k_max_inv_m,
    )
    validation_radius_m = float(
        np.interp(validation_start_z_m, z_m, guiding_radius_m)
    )
    validation_potential_v = float(
        potential_model.potential_cylindrical(
            validation_radius_m,
            validation_start_z_m,
        )
    )
    validation_b_rho, validation_b_z = (
        field_model.total.field_cylindrical(
            validation_radius_m,
            validation_start_z_m,
        )
    )
    validation_field_magnitude_t = float(
        np.hypot(validation_b_rho, validation_b_z)
    )
    source_u = kinetic_energy_ev_to_normalized_momentum(
        kinetic_energy_ev,
        ELECTRON,
    )
    source_gamma = float(np.sqrt(1.0 + source_u**2))
    rest_energy_j = ELECTRON.mass_kg * c**2
    normalized_total_energy = (
        source_gamma
        + ELECTRON.charge_c * source_potential_v / rest_energy_j
    )
    validation_gamma = (
        normalized_total_energy
        - ELECTRON.charge_c * validation_potential_v / rest_energy_j
    )
    validation_results = []
    for track_index, (source_pitch, azimuth) in enumerate(
        zip(source_pitch_deg, azimuth_deg, strict=True)
    ):
        perpendicular_u_squared = (
            source_u * np.sin(np.deg2rad(source_pitch))
        ) ** 2 * validation_field_magnitude_t / field_magnitude_t[0]
        total_u_squared = validation_gamma**2 - 1.0
        parallel_u_squared = total_u_squared - perpendicular_u_squared
        if parallel_u_squared <= 0.0:
            raise RuntimeError(
                "Validation interval starts beyond a turning point"
            )
        phi = np.deg2rad(float(azimuth))
        radial_unit = np.array([np.cos(phi), np.sin(phi), 0.0])
        azimuthal_unit = np.array([-np.sin(phi), np.cos(phi), 0.0])
        local_b = np.array(
            [
                float(validation_b_rho) * radial_unit[0],
                float(validation_b_rho) * radial_unit[1],
                float(validation_b_z),
            ]
        )
        forward_field_unit = (
            np.sign(float(validation_b_z))
            * local_b
            / np.linalg.norm(local_b)
        )
        initial_u = (
            np.sqrt(parallel_u_squared) * forward_field_unit
            + np.sqrt(perpendicular_u_squared) * azimuthal_unit
        )
        initial_position = np.array(
            [
                validation_radius_m * radial_unit[0],
                validation_radius_m * radial_unit[1],
                validation_start_z_m,
            ]
        )
        full_solution = integrate_relativistic_electromagnetic_trajectory(
            field_model.total,
            validation_potential_model,
            initial_position,
            initial_u,
            lower_stop_z_m=validation_lower_stop_z_m,
            upper_stop_z_m=validation_upper_stop_z_m,
            maximum_time_s=(
                float(validation_config["maximum_time_us"]) * 1.0e-6
            ),
            particle=ELECTRON,
            settings=validation_settings,
        )
        full_radius = np.hypot(
            full_solution.position_m[:, 0],
            full_solution.position_m[:, 1],
        )
        full_potential = validation_potential_model.potential_cylindrical(
            full_radius,
            full_solution.position_m[:, 2],
        )
        full_total_energy_ev = (
            full_solution.diagnostics.gamma * rest_energy_j
            + ELECTRON.charge_c * full_potential
        ) / elementary_charge
        full_outcome = str(
            full_solution.solver_statistics["exit_kind"]
        )
        adiabatic_outcome = str(
            tracks_metadata[track_index]["outcome"]
        )
        full_turning_z_m = float(
            np.max(full_solution.position_m[:, 2])
        )
        adiabatic_turning_z_m = tracks_metadata[track_index][
            "turning_z_m"
        ]
        validation_results.append(
            {
                "source_pitch_deg": float(source_pitch),
                "adiabatic_outcome": adiabatic_outcome,
                "full_lorentz_outcome": full_outcome,
                "full_lorentz_turning_z_m": full_turning_z_m,
                "adiabatic_turning_z_m": adiabatic_turning_z_m,
                "turning_z_difference_m": (
                    None
                    if adiabatic_turning_z_m is None
                    else full_turning_z_m - adiabatic_turning_z_m
                ),
                "total_energy_residual_ev": float(
                    np.ptp(full_total_energy_ev)
                ),
                "magnetic_moment_relative_span": float(
                    np.ptp(
                        full_solution.diagnostics.magnetic_moment_j_per_t
                    )
                    / full_solution.diagnostics.magnetic_moment_j_per_t[0]
                ),
                "function_evaluations": int(
                    full_solution.solver_statistics["nfev"]
                ),
            }
        )

    metadata = {
        "dataset_version": 1,
        "dataset_type": "katrin_mac_e_adiabatic_electrostatic",
        "scenario_name": str(config["scenario"]["name"]),
        "particle": asdict(ELECTRON),
        "kinetic_energy_ev": kinetic_energy_ev,
        "field_configuration": field_configuration,
        "field_reference": field_model.reference,
        "field_validity_note": field_model.validity_note,
        "electrostatic_model": {
            "name": str(electrostatic["model"]),
            "equation": (
                "Phi=(2 V0/pi) integral_0^inf sin(kL/2)/k "
                "cos(kz) I0(k rho)/I0(kR) dk"
            ),
            "vacuum_equation": "axisymmetric Laplace equation",
            "radius_m": potential_model.radius_m,
            "length_m": potential_model.length_m,
            "boundary_potential_v": boundary_potential_v,
            "source_potential_v": source_potential_v,
            "analysis_potential_v": analysis_potential_v,
            "retarding_difference_v": (
                analysis_potential_v - source_potential_v
            ),
            "quadrature_order": potential_model.quadrature_order,
            "k_max_inv_m": potential_model.k_max_inv_m,
            "convergence_max_abs_v": convergence_max_abs_v,
            "status": (
                "documented dimensions and retarding scale; explicitly "
                "idealized cylindrical boundary, not an as-built KATRIN map"
            ),
        },
        "transport_method": {
            "name": (
                "relativistic electrostatic first-adiabatic-invariant "
                "guiding centre"
            ),
            "conserved_quantities": [
                "gamma m c^2 + q Phi",
                "p_perp^2 / |B|",
            ],
            "reflection_condition": "p_parallel^2 = 0",
            "not_a_full_orbit": True,
        },
        "central_full_lorentz_validation": {
            "interval_z_m": [
                validation_lower_stop_z_m,
                validation_upper_stop_z_m,
            ],
            "equation": (
                "dx/dt=p/(gamma m); "
                "dp/dt=q(E+p/(gamma m) cross B)"
            ),
            "solver": {
                "method": "DOP853",
                "settings": asdict(validation_settings),
                "electric_quadrature_order": (
                    validation_potential_model.quadrature_order
                ),
            },
            "tracks": validation_results,
        },
        "beamline": {
            "start_z_m": start_z_m,
            "stop_z_m": stop_z_m,
            "source_guiding_center_radius_m": source_radius_m,
            "source_flux_tube_radius_m": source_flux_radius_m,
            "source_field_t": float(field_magnitude_t[0]),
            "analysis_axis_field_t": float(
                np.abs(field_model.total.axis_field(0.0))
            ),
            "maximum_axis_field_t": float(
                np.max(np.abs(field_model.total.axis_field(z_m)))
            ),
        },
        "tracks": tracks_metadata,
        "superconducting_sources": [
            {
                "name": source.name,
                "z_m": source.center_z_m,
                "published_typical_max_field_t": (
                    source.typical_max_field_t
                ),
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
        electric_potential_v=np.asarray(electric_potential_v),
        kinetic_energy_ev=np.asarray(kinetic_energy),
        total_energy_ev=np.asarray(total_energy),
        pitch_angle_deg=np.asarray(pitch_angle_deg),
        parallel_speed_m_s=np.asarray(parallel_speed_m_s),
        perpendicular_speed_m_s=np.asarray(perpendicular_speed_m_s),
        parallel_normalized_momentum=np.asarray(
            parallel_normalized_momentum
        ),
        perpendicular_normalized_momentum=np.asarray(
            perpendicular_normalized_momentum
        ),
        larmor_radius_m=np.asarray(larmor_radius_m),
        cumulative_turns=np.asarray(cumulative_turns),
        active=np.asarray(active),
        field_line_z_m=z_m,
        guiding_radius_m=guiding_radius_m,
        flux_radius_m=flux_radius_m,
        axis_field_magnitude_t=np.abs(field_model.total.axis_field(z_m)),
        axis_electric_potential_v=axis_potential_v,
        axis_electric_field_z_v_m=axis_e_z_v_m,
        guiding_path_electric_potential_v=path_potential_v,
        path_electric_field_rho_v_m=path_e_rho_v_m,
        path_electric_field_z_v_m=path_e_z_v_m,
    )

    print(f"Saved electrostatic MAC-E dataset: {args.out}")
    print(
        f"Phi: source={source_potential_v:.6f} V, "
        f"analysis={analysis_potential_v:.6f} V, "
        f"delta={analysis_potential_v - source_potential_v:.6f} V"
    )
    print(
        f"Laplace quadrature convergence: "
        f"max |delta Phi|={convergence_max_abs_v:.6g} V"
    )
    for track in tracks_metadata:
        terminal = (
            f"turn z={track['turning_z_m']:+.4f} m"
            if track["outcome"] == "reflected"
            else (
                f"K_analysis={track['analysis_kinetic_energy_ev']:.6f} eV"
            )
        )
        print(
            f"{track['label']}: {track['outcome']}; {terminal}; "
            f"K_min={track['minimum_kinetic_energy_ev']:.6f} eV, "
            f"t={track['flight_time_s'] * 1e6:.6f} us, "
            f"dE={track['total_energy_residual_ev']:.3g} eV"
        )
    for result in validation_results:
        comparison = (
            ""
            if result["turning_z_difference_m"] is None
            else (
                f", dz_turn={result['turning_z_difference_m']:+.4g} m"
            )
        )
        print(
            "full Lorentz "
            f"{result['source_pitch_deg']:.0f} deg: "
            f"{result['full_lorentz_outcome']}, "
            f"dE={result['total_energy_residual_ev']:.3g} eV"
            f"{comparison}, nfev={result['function_evaluations']}"
        )


if __name__ == "__main__":
    main()
