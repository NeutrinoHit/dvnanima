#!/usr/bin/env python3
"""Integrate and export an 18.6 keV electron in the KATRIN 2013 field."""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from mac_e_filter.data_io import save_trajectory_dataset  # noqa: E402
from mac_e_filter.dynamics import (  # noqa: E402
    ELECTRON,
    TrajectorySolution,
    integrate_relativistic_magnetic_trajectory,
)
from mac_e_filter.katrin_nominal import build_katrin_2013_field  # noqa: E402
from mac_e_filter.scenarios import (  # noqa: E402
    KatrinTrajectoryScenario,
    load_katrin_trajectory_scenario,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "katrin_2013_trajectory.toml",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=PROJECT_ROOT
        / "datasets"
        / "katrin_2013_electron_18p6kev.npz",
    )
    parser.add_argument(
        "--skip-convergence",
        action="store_true",
        help="Do not run the second, refined integration.",
    )
    return parser.parse_args()


def integrate(
    scenario: KatrinTrajectoryScenario,
    *,
    refined: bool = False,
) -> tuple[TrajectorySolution, object]:
    field_model = build_katrin_2013_field(scenario.field_configuration)
    settings = scenario.integration
    if refined:
        settings = replace(
            settings,
            relative_tolerance=settings.relative_tolerance / 4.0,
            position_atol_m=settings.position_atol_m / 4.0,
            normalized_momentum_atol=(
                settings.normalized_momentum_atol / 4.0
            ),
            max_gyro_phase_rad=settings.max_gyro_phase_rad / 2.0,
        )
    solution = integrate_relativistic_magnetic_trajectory(
        field_model.total,
        scenario.initial_position_m,
        scenario.initial_normalized_momentum(),
        stop_z_m=scenario.stop_z_m,
        maximum_time_s=scenario.maximum_time_s,
        particle=ELECTRON,
        settings=settings,
    )
    return solution, field_model


def convergence_report(
    baseline: TrajectorySolution,
    refined: TrajectorySolution,
) -> dict[str, float]:
    baseline_z = baseline.position_m[:, 2]
    refined_z = refined.position_m[:, 2]
    if np.any(np.diff(baseline_z) <= 0.0) or np.any(np.diff(refined_z) <= 0.0):
        raise RuntimeError(
            "The convergence comparison assumes a monotonic transmitted orbit"
        )
    common_z = np.linspace(
        max(baseline_z[0], refined_z[0]),
        min(baseline_z[-1], refined_z[-1]),
        2001,
    )

    def interpolate_by_z(values: np.ndarray, source_z: np.ndarray) -> np.ndarray:
        return np.column_stack(
            [
                np.interp(common_z, source_z, values[:, component])
                for component in range(values.shape[1])
            ]
        )

    baseline_position = interpolate_by_z(baseline.position_m, baseline_z)
    refined_position = interpolate_by_z(refined.position_m, refined_z)
    baseline_u = interpolate_by_z(
        baseline.normalized_momentum, baseline_z
    )
    refined_u = interpolate_by_z(refined.normalized_momentum, refined_z)
    return {
        "flight_time_difference_s": abs(
            float(baseline.time_s[-1] - refined.time_s[-1])
        ),
        "endpoint_position_difference_m": float(
            np.linalg.norm(baseline.position_m[-1] - refined.position_m[-1])
        ),
        "endpoint_normalized_momentum_difference": float(
            np.linalg.norm(
                baseline.normalized_momentum[-1]
                - refined.normalized_momentum[-1]
            )
        ),
        "path_position_max_difference_m": float(
            np.max(np.linalg.norm(baseline_position - refined_position, axis=1))
        ),
        "path_normalized_momentum_max_difference": float(
            np.max(np.linalg.norm(baseline_u - refined_u, axis=1))
        ),
    }


def main() -> None:
    args = parse_args()
    scenario = load_katrin_trajectory_scenario(args.config)

    print("Integrating baseline full Lorentz trajectory...")
    baseline, field_model = integrate(scenario)
    convergence = None
    if not args.skip_convergence:
        print("Integrating refined trajectory for convergence...")
        refined, _ = integrate(scenario, refined=True)
        convergence = convergence_report(baseline, refined)

    metadata = {
        "scenario": {
            **asdict(scenario),
            "initial_position_m": scenario.initial_position_m.tolist(),
            "integration": asdict(scenario.integration),
        },
        "particle": asdict(ELECTRON),
        "field_reference": field_model.reference,
        "field_validity_note": field_model.validity_note,
        "equations": {
            "momentum_variable": "u = p/(m c)",
            "dx_dt": "c u / sqrt(1 + |u|^2)",
            "du_dt": "q (u cross B) / (m sqrt(1 + |u|^2))",
            "electric_field": "zero in this scenario",
        },
        "convergence": convergence,
    }
    output = save_trajectory_dataset(args.out, baseline, metadata)

    print(f"Saved dataset: {output}")
    print("Trajectory summary:")
    for key, value in baseline.summary().items():
        print(f"  {key}: {value:.12g}")
    print("Solver:")
    for key, value in baseline.solver_statistics.items():
        print(f"  {key}: {value}")
    if convergence is not None:
        print("Baseline/refined convergence:")
        for key, value in convergence.items():
            print(f"  {key}: {value:.12g}")


if __name__ == "__main__":
    main()

