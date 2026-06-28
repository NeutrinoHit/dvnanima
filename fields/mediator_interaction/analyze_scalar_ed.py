from __future__ import annotations

import argparse
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np

from fields.mediator_interaction.numerics import (
    ScalarEDSimulationConfig,
    signed_charge_centers,
    simulate_scalar_ed,
)


def total_charge_series(charge_frames: np.ndarray, dx: float, dy: float) -> np.ndarray:
    return charge_frames.sum(axis=(1, 2)) * dx * dy


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run scalar electrodynamics numerics without Manim.")
    parser.add_argument("--out", type=Path, default=None, help="Optional .npz output path.")
    parser.add_argument("--t-final", type=float, default=None)
    parser.add_argument("--dt", type=float, default=None)
    parser.add_argument("--output-dt", type=float, default=None)
    parser.add_argument("--nx", type=int, default=None)
    parser.add_argument("--ny", type=int, default=None)
    parser.add_argument("--lx", type=float, default=None)
    parser.add_argument("--ly", type=float, default=None)
    parser.add_argument("--charge-e", type=float, default=None)
    parser.add_argument("--carrier-k", type=float, default=None)
    parser.add_argument("--sigma", type=float, default=None)
    parser.add_argument("--initial-offset", type=float, default=None)
    parser.add_argument("--impact-parameter", type=float, default=None)
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    cfg = ScalarEDSimulationConfig()
    overrides = {
        "t_final": args.t_final,
        "dt": args.dt,
        "output_dt": args.output_dt,
        "nx": args.nx,
        "ny": args.ny,
        "lx": args.lx,
        "ly": args.ly,
        "charge_e": args.charge_e,
        "carrier_k": args.carrier_k,
        "sigma": args.sigma,
        "initial_offset": args.initial_offset,
        "impact_parameter": args.impact_parameter,
    }
    overrides = {key: value for key, value in overrides.items() if value is not None}
    if overrides:
        cfg = replace(cfg, **overrides)

    sim = simulate_scalar_ed(cfg)
    x_axis = sim["x_axis"]
    y_axis = sim["y_axis"]
    times = sim["times"]
    density_frames = sim["density_frames"]
    a0_frames = sim["a0_frames"]
    charge_frames = sim["charge_frames"]
    matter_energy = sim["matter_energy"]
    coulomb_energy = sim["coulomb_energy"]
    total_energy = sim["total_energy"]

    dx = float(x_axis[1] - x_axis[0])
    dy = float(y_axis[1] - y_axis[0])
    spatial_scale = min(dx, dy)
    q_series = total_charge_series(charge_frames, dx, dy)

    centers_initial = signed_charge_centers(charge_frames[0], x_axis, y_axis)
    centers_final = signed_charge_centers(charge_frames[-1], x_axis, y_axis)

    print("Config:")
    for key, value in asdict(cfg).items():
        print(f"  {key} = {value}")

    print("\nDiagnostics:")
    print(f"  t_end = {float(times[-1]):.6f}")
    print(f"  nframes = {len(times)}")
    print(f"  dx = {dx:.6f}, dy = {dy:.6f}")
    print(f"  sigma / min(dx,dy) = {cfg.sigma / spatial_scale:.3f}")
    print(f"  density max = {float(np.max(density_frames)):.6f}")
    print(f"  |A0| max = {float(np.max(np.abs(a0_frames))):.6f}")
    print(f"  total charge: min = {float(np.min(q_series)):.6e}, max = {float(np.max(q_series)):.6e}")
    print(f"  charge drift = {float(np.max(np.abs(q_series - q_series[0]))):.6e}")
    print(f"  matter energy drift = {float(np.max(np.abs(matter_energy - matter_energy[0]))):.6e}")
    print(f"  Coulomb energy drift = {float(np.max(np.abs(coulomb_energy - coulomb_energy[0]))):.6e}")
    print(f"  total energy drift = {float(np.max(np.abs(total_energy - total_energy[0]))):.6e}")
    print(f"  initial centers: positive = {centers_initial['positive']}, negative = {centers_initial['negative']}")
    print(f"  final centers:   positive = {centers_final['positive']}, negative = {centers_final['negative']}")
    print("\nNote: centers are computed from positive and negative parts of rho, not from |phi|^2.")
    if cfg.sigma < 2.0 * spatial_scale:
        print("Warning: sigma is close to or below the grid scale; expect aliasing and unreliable visualisation.")

    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            args.out,
            times=times,
            x_axis=x_axis,
            y_axis=y_axis,
            density_frames=density_frames,
            a0_frames=a0_frames,
            charge_frames=charge_frames,
            matter_energy=matter_energy,
            coulomb_energy=coulomb_energy,
            total_energy=total_energy,
        )
        print(f"\nSaved fields to {args.out}")


if __name__ == "__main__":
    main()
