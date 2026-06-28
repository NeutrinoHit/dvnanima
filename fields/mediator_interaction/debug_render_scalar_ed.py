from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter

from fields.mediator_interaction.numerics import (
    ScalarEDSimulationConfig,
    signed_charge_centers,
    simulate_scalar_ed,
)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fast matplotlib debug renderer for scalar electrodynamics.")
    parser.add_argument("--out", type=Path, default=None, help="Optional output GIF path.")
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
    parser.add_argument("--show-a0", action="store_true", help="Show a second panel with A0.")
    parser.add_argument("--show-charge", action="store_true", help="Show a second panel with rho.")
    parser.add_argument("--show-centers", action="store_true", help="Overlay centers of positive and negative charge.")
    parser.add_argument("--fps", type=int, default=24)
    return parser


def build_config(args: argparse.Namespace) -> ScalarEDSimulationConfig:
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
    return replace(cfg, **overrides) if overrides else cfg


def main() -> None:
    args = build_arg_parser().parse_args()
    cfg = build_config(args)
    sim = simulate_scalar_ed(cfg)

    x_axis = sim["x_axis"]
    y_axis = sim["y_axis"]
    times = sim["times"]
    density_frames = sim["density_frames"]
    a0_frames = sim["a0_frames"]
    charge_frames = sim["charge_frames"]

    panels = [("density", density_frames)]
    if args.show_charge:
        panels.append(("charge", charge_frames))
    elif args.show_a0:
        panels.append(("a0", a0_frames))

    fig, axes = plt.subplots(1, len(panels), figsize=(6.5 * len(panels), 5.6), constrained_layout=True)
    if len(panels) == 1:
        axes = [axes]

    extent = [float(x_axis[0]), float(x_axis[-1]), float(y_axis[0]), float(y_axis[-1])]
    artists: list[tuple] = []

    centers_positive = None
    centers_negative = None
    if args.show_centers:
        signed_centers = [signed_charge_centers(frame, x_axis, y_axis) for frame in charge_frames]
        centers_positive = np.array([item["positive"] for item in signed_centers], dtype=float)
        centers_negative = np.array([item["negative"] for item in signed_centers], dtype=float)

    for ax, (name, frames) in zip(axes, panels, strict=True):
        if name == "density":
            vmax = max(float(np.max(frames)), 1e-9)
            image = ax.imshow(
                frames[0].T,
                origin="lower",
                extent=extent,
                cmap="Blues",
                vmin=0.0,
                vmax=vmax,
                interpolation="nearest",
                aspect="auto",
            )
            ax.set_title(r"$|\varphi|^2$")
        elif name == "charge":
            vmax = max(float(np.max(np.abs(frames))), 1e-9)
            image = ax.imshow(
                frames[0].T,
                origin="lower",
                extent=extent,
                cmap="coolwarm",
                vmin=-vmax,
                vmax=vmax,
                interpolation="nearest",
                aspect="auto",
            )
            ax.set_title(r"$\rho$")
        else:
            vmax = max(float(np.max(np.abs(frames))), 1e-9)
            image = ax.imshow(
                frames[0].T,
                origin="lower",
                extent=extent,
                cmap="PuOr_r",
                vmin=-vmax,
                vmax=vmax,
                interpolation="nearest",
                aspect="auto",
            )
            ax.set_title(r"$A_0$")

        ax.set_xlabel("x")
        ax.set_ylabel("y")
        colorbar = fig.colorbar(image, ax=ax, shrink=0.88)
        colorbar.ax.tick_params(labelsize=8)

        pos_marker = neg_marker = pos_path = neg_path = None
        if args.show_centers:
            pos_marker, = ax.plot([], [], "o", color="tab:red", markersize=5)
            neg_marker, = ax.plot([], [], "o", color="gold", markersize=5)
            pos_path, = ax.plot([], [], "-", color="tab:red", linewidth=1.4, alpha=0.9)
            neg_path, = ax.plot([], [], "-", color="gold", linewidth=1.4, alpha=0.9)

        artists.append((image, pos_marker, neg_marker, pos_path, neg_path))

    time_text = fig.suptitle(f"t = {times[0]:.3f}")

    def update(frame_index: int):
        updated = []
        time_text.set_text(f"t = {times[frame_index]:.3f}")
        updated.append(time_text)
        for (name, frames), (image, pos_marker, neg_marker, pos_path, neg_path) in zip(panels, artists, strict=True):
            image.set_data(frames[frame_index].T)
            updated.append(image)
            if args.show_centers and centers_positive is not None and centers_negative is not None:
                px, py = centers_positive[frame_index]
                nx, ny = centers_negative[frame_index]
                pos_marker.set_data([px], [py])
                neg_marker.set_data([nx], [ny])
                pos_path.set_data(centers_positive[: frame_index + 1, 0], centers_positive[: frame_index + 1, 1])
                neg_path.set_data(centers_negative[: frame_index + 1, 0], centers_negative[: frame_index + 1, 1])
                updated.extend([pos_marker, neg_marker, pos_path, neg_path])
        return updated

    animation = FuncAnimation(fig, update, frames=len(times), interval=1000 / max(args.fps, 1), blit=False)

    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        animation.save(args.out, writer=PillowWriter(fps=args.fps))
        print(f"Saved debug animation to {args.out}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
