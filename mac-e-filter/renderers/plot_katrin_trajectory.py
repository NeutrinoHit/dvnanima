#!/usr/bin/env python3
"""Render diagnostics from a precomputed KATRIN trajectory dataset."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from mac_e_filter.data_io import load_trajectory_dataset  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=PROJECT_ROOT
        / "datasets"
        / "katrin_2013_electron_18p6kev.npz",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=PROJECT_ROOT
        / "media"
        / "katrin_2013_electron_trajectory.png",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    import matplotlib.pyplot as plt

    dataset = load_trajectory_dataset(args.dataset)
    metadata = dataset["metadata"]
    position = dataset["position_m"]
    z = position[:, 2]
    radius = np.hypot(position[:, 0], position[:, 1])
    field_microtesla = 1.0e6 * dataset["magnetic_field_magnitude_t"]
    pitch = dataset["pitch_angle_deg"]
    energy = dataset["kinetic_energy_ev"]
    magnetic_moment = dataset["magnetic_moment_j_per_t"]

    figure = plt.figure(figsize=(12.0, 8.2), constrained_layout=True)
    grid = figure.add_gridspec(2, 2)
    axis_longitudinal = figure.add_subplot(grid[0, 0])
    axis_transverse = figure.add_subplot(grid[0, 1])
    axis_field = figure.add_subplot(grid[1, 0])
    axis_invariants = figure.add_subplot(grid[1, 1])

    axis_longitudinal.plot(z, position[:, 0], label="x(z)", linewidth=1.3)
    axis_longitudinal.plot(z, position[:, 1], label="y(z)", linewidth=1.3)
    axis_longitudinal.plot(z, radius, label=r"$\rho(z)$", linewidth=1.8)
    axis_longitudinal.set_xlabel("z [m]")
    axis_longitudinal.set_ylabel("transverse coordinate [m]")
    axis_longitudinal.set_title("Full-orbit transverse motion")
    axis_longitudinal.grid(alpha=0.22)
    axis_longitudinal.legend(ncols=3)

    axis_transverse.plot(
        position[:, 0],
        position[:, 1],
        linewidth=1.4,
        color="#6A4C93",
    )
    axis_transverse.scatter(
        [position[0, 0]],
        [position[0, 1]],
        color="#2A9D8F",
        s=35,
        label="start",
        zorder=3,
    )
    axis_transverse.scatter(
        [position[-1, 0]],
        [position[-1, 1]],
        color="#D1495B",
        s=35,
        label="finish",
        zorder=3,
    )
    axis_transverse.set_aspect("equal", adjustable="box")
    axis_transverse.set_xlabel("x [m]")
    axis_transverse.set_ylabel("y [m]")
    axis_transverse.set_title("Transverse projection (true scale)")
    axis_transverse.grid(alpha=0.22)
    axis_transverse.legend()

    field_line = axis_field.plot(
        z,
        field_microtesla,
        color="#1768AC",
        label=r"$|\mathbf{B}|$",
    )[0]
    axis_field.set_xlabel("z [m]")
    axis_field.set_ylabel(r"$|\mathbf{B}|$ [µT]", color="#1768AC")
    axis_field.tick_params(axis="y", labelcolor="#1768AC")
    pitch_axis = axis_field.twinx()
    pitch_line = pitch_axis.plot(
        z,
        pitch,
        color="#D1495B",
        label="pitch angle",
    )[0]
    pitch_axis.set_ylabel("pitch angle [deg]", color="#D1495B")
    pitch_axis.tick_params(axis="y", labelcolor="#D1495B")
    axis_field.set_title("Field and instantaneous pitch")
    axis_field.grid(alpha=0.22)
    axis_field.legend(
        [field_line, pitch_line],
        [field_line.get_label(), pitch_line.get_label()],
        loc="best",
    )

    energy_relative = (energy / energy[0] - 1.0) * 1.0e9
    moment_relative = (magnetic_moment / magnetic_moment[0] - 1.0) * 100.0
    energy_line = axis_invariants.plot(
        z,
        energy_relative,
        color="#2A9D8F",
        label=r"$\Delta K/K_0$",
    )[0]
    axis_invariants.set_xlabel("z [m]")
    axis_invariants.set_ylabel(
        r"kinetic-energy deviation [$10^{-9}$]",
        color="#2A9D8F",
    )
    axis_invariants.tick_params(axis="y", labelcolor="#2A9D8F")
    moment_axis = axis_invariants.twinx()
    moment_line = moment_axis.plot(
        z,
        moment_relative,
        color="#F4A261",
        label=r"$\Delta\mu/\mu_0$",
    )[0]
    moment_axis.set_ylabel(
        "first adiabatic-invariant change [%]",
        color="#F4A261",
    )
    moment_axis.tick_params(axis="y", labelcolor="#F4A261")
    axis_invariants.set_title("Exact energy invariant vs adiabatic diagnostic")
    axis_invariants.grid(alpha=0.22)
    axis_invariants.legend(
        [energy_line, moment_line],
        [energy_line.get_label(), moment_line.get_label()],
        loc="best",
    )

    summary = metadata["summary"]
    title = (
        "18.6 keV electron in the published KATRIN 2013 nominal field\n"
        f"flight time {summary['flight_time_s'] * 1e9:.3f} ns; "
        f"max radius {summary['radius_max_m']:.3f} m; "
        "no electric field"
    )
    figure.suptitle(title)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.out, dpi=180)
    plt.close(figure)
    print(f"Saved trajectory diagnostic: {args.out}")


if __name__ == "__main__":
    main()
