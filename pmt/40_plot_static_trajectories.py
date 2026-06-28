from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from pmt.data_io import load_dataset
from pmt.numerics import build_simulation_bundle, load_pmt_config, make_preview_config


def _require_matplotlib():
    try:
        import matplotlib.pyplot as plt
        from matplotlib.patches import Polygon, Rectangle, Wedge
    except Exception as exc:
        raise RuntimeError("matplotlib is required for static trajectory plotting") from exc
    return plt, Polygon, Rectangle, Wedge


def _select_indices(total: int, max_count: int) -> np.ndarray:
    if total <= max_count:
        return np.arange(total, dtype=np.int64)
    return np.unique(np.linspace(0, total - 1, max_count, dtype=np.int64))


def _rotated_rect_polygon(cx: float, cy: float, length: float, thickness: float, angle_deg: float) -> np.ndarray:
    theta = np.deg2rad(angle_deg)
    c = float(np.cos(theta))
    s = float(np.sin(theta))
    local = np.array(
        [
            [-0.5 * length, -0.5 * thickness],
            [0.5 * length, -0.5 * thickness],
            [0.5 * length, 0.5 * thickness],
            [-0.5 * length, 0.5 * thickness],
        ],
        dtype=float,
    )
    rot = np.array([[c, -s], [s, c]], dtype=float)
    pts = local @ rot.T
    pts[:, 0] += cx
    pts[:, 1] += cy
    return pts


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Static diagnostics plot: PMT geometry + trajectories + field lines.")
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("pmt.toml"))
    parser.add_argument("--data", type=Path, default=None, help="Use existing HDF5 dataset instead of re-running simulation.")
    parser.add_argument("--preview", action="store_true", help="Use preview-scaled simulation when --data is not set.")
    parser.add_argument("--grid-scale", type=float, default=0.40)
    parser.add_argument("--particle-scale", type=float, default=0.25)
    parser.add_argument("--step-scale", type=float, default=0.25)
    parser.add_argument("--max-particles", type=int, default=220)
    parser.add_argument("--stream-density", type=float, default=1.1)
    parser.add_argument("--field-mode", choices=("e", "electron_force"), default="electron_force")
    parser.add_argument("--out", type=Path, default=Path(__file__).with_name("pmt_static_trajectories.png"))
    parser.add_argument("--dpi", type=int, default=180)
    return parser


def main() -> None:
    args = _build_arg_parser().parse_args()
    plt, Polygon, Rectangle, Wedge = _require_matplotlib()

    if args.data is not None:
        dataset = load_dataset(args.data)
        cfg = dataset["metadata"]["config"]
        x_axis = np.asarray(dataset["x_axis"], dtype=float)
        y_axis = np.asarray(dataset["y_axis"], dtype=float)
        potential = np.asarray(dataset["potential"], dtype=float)
        ex = np.asarray(dataset["ex"], dtype=float)
        ey = np.asarray(dataset["ey"], dtype=float)
        positions = np.asarray(dataset["particle_positions"], dtype=float)
        alive = np.asarray(dataset["particle_alive"], dtype=bool)
        impact_mask = np.asarray(dataset["particle_impact_mask"], dtype=bool)
        impact_position = np.asarray(dataset["particle_impact_position"], dtype=float)
        total_particles = int(positions.shape[1])
    else:
        cfg_obj = load_pmt_config(args.config)
        if args.preview:
            cfg_obj = make_preview_config(
                cfg_obj,
                grid_scale=args.grid_scale,
                particle_scale=args.particle_scale,
                step_scale=args.step_scale,
            )
        sim = build_simulation_bundle(cfg_obj)
        x_axis = np.asarray(sim["x_axis"], dtype=float)
        y_axis = np.asarray(sim["y_axis"], dtype=float)
        potential = np.asarray(sim["potential"], dtype=float)
        ex = np.asarray(sim["ex"], dtype=float)
        ey = np.asarray(sim["ey"], dtype=float)
        positions = np.asarray(sim["particle_positions"], dtype=float)
        alive = np.asarray(sim["particle_alive"], dtype=bool)
        impact_mask = np.asarray(sim["particle_impact_mask"], dtype=bool)
        impact_position = np.asarray(sim["particle_impact_position"], dtype=float)
        total_particles = int(cfg_obj.particles.count)
        cfg = {
            "photocathode_shape": cfg_obj.photocathode_shape,
            "photocathode_diameter": cfg_obj.photocathode_diameter,
            "photocathode_thickness": cfg_obj.photocathode_thickness,
            "photocathode_center_x": cfg_obj.photocathode_center_x,
            "photocathode_center_y": cfg_obj.photocathode_center_y,
            "cathode_x": cfg_obj.cathode_x,
            "cathode_height": cfg_obj.cathode_height,
            "cathode_thickness": cfg_obj.cathode_thickness,
            "dynode_center_x": cfg_obj.dynode_center_x,
            "dynode_center_y": cfg_obj.dynode_center_y,
            "dynode_length": cfg_obj.dynode_length,
            "dynode_thickness": cfg_obj.dynode_thickness,
            "dynode_angle_deg": cfg_obj.dynode_angle_deg,
        }

    idx = _select_indices(total_particles, args.max_particles)
    shown = int(idx.shape[0])

    x_cm = x_axis * 100.0
    y_cm = y_axis * 100.0

    fig, ax = plt.subplots(figsize=(12.4, 8.6), dpi=args.dpi)
    fig.patch.set_facecolor("black")
    ax.set_facecolor("black")

    pot_q = np.percentile(potential, [1.0, 99.0])
    vmin = float(pot_q[0])
    vmax = float(pot_q[1]) if abs(float(pot_q[1] - pot_q[0])) > 1e-12 else float(np.max(potential) + 1e-9)

    ax.imshow(
        potential.T,
        extent=[float(x_cm[0]), float(x_cm[-1]), float(y_cm[0]), float(y_cm[-1])],
        origin="lower",
        cmap="magma",
        vmin=vmin,
        vmax=vmax,
        alpha=0.88,
        interpolation="nearest",
        aspect="auto",
    )

    fx = ex
    fy = ey
    if args.field_mode == "electron_force":
        fx = -ex
        fy = -ey

    ax.streamplot(
        x_cm,
        y_cm,
        fx.T,
        fy.T,
        color="#ff3b30",
        linewidth=0.75,
        density=args.stream_density,
        arrowsize=0.70,
        minlength=0.04,
    )

    for pidx in idx:
        path = positions[:, pidx, :]
        finite = np.isfinite(path[:, 0]) & np.isfinite(path[:, 1])
        if np.count_nonzero(finite) < 2:
            continue
        pts = path[finite]
        ax.plot(pts[:, 0] * 100.0, pts[:, 1] * 100.0, color="white", alpha=0.35, linewidth=0.9)

    hit_idx = idx[impact_mask[idx]]
    if hit_idx.size:
        hit_pts = impact_position[hit_idx]
        good = np.isfinite(hit_pts[:, 0]) & np.isfinite(hit_pts[:, 1])
        if np.any(good):
            ax.scatter(
                hit_pts[good, 0] * 100.0,
                hit_pts[good, 1] * 100.0,
                s=10,
                c="#fff7cc",
                alpha=0.88,
                linewidths=0.0,
            )

    shape = str(cfg.get("photocathode_shape", "line")).strip().lower()
    if shape == "hemisphere":
        cx = float(cfg["photocathode_center_x"]) * 100.0
        cy = float(cfg["photocathode_center_y"]) * 100.0
        r = 0.5 * float(cfg["photocathode_diameter"]) * 100.0
        t = float(cfg["photocathode_thickness"]) * 100.0
        ring = Wedge(
            center=(cx, cy),
            r=r,
            theta1=90.0,
            theta2=270.0,
            width=t,
            facecolor="#3f8cff",
            edgecolor="#8ec4ff",
            linewidth=1.6,
            alpha=0.28,
        )
        ax.add_patch(ring)
    else:
        cathode = Rectangle(
            (
                (float(cfg["cathode_x"]) - 0.5 * float(cfg["cathode_thickness"])) * 100.0,
                (-0.5 * float(cfg["cathode_height"])) * 100.0,
            ),
            width=float(cfg["cathode_thickness"]) * 100.0,
            height=float(cfg["cathode_height"]) * 100.0,
            facecolor="#3f8cff",
            edgecolor="#8ec4ff",
            linewidth=1.6,
            alpha=0.28,
        )
        ax.add_patch(cathode)

    dynode_pts = _rotated_rect_polygon(
        cx=float(cfg["dynode_center_x"]) * 100.0,
        cy=float(cfg["dynode_center_y"]) * 100.0,
        length=float(cfg["dynode_length"]) * 100.0,
        thickness=float(cfg["dynode_thickness"]) * 100.0,
        angle_deg=float(cfg["dynode_angle_deg"]),
    )
    dynode_patch = Polygon(
        dynode_pts,
        closed=True,
        facecolor="#ff8d5a",
        edgecolor="#ffd0b6",
        linewidth=1.6,
        alpha=0.70,
    )
    ax.add_patch(dynode_patch)

    border = Rectangle(
        (float(x_cm[0]), float(y_cm[0])),
        width=float(x_cm[-1] - x_cm[0]),
        height=float(y_cm[-1] - y_cm[0]),
        fill=False,
        edgecolor="#7d7d7d",
        linewidth=1.2,
        alpha=0.85,
    )
    ax.add_patch(border)

    collected = int(np.count_nonzero(impact_mask))
    eff = 100.0 * collected / max(total_particles, 1)
    mode_label = "electron force" if args.field_mode == "electron_force" else "E field"
    ax.set_title(
        f"PMT static check | shown {shown}/{total_particles} | collected {collected} ({eff:.2f}%) | lines: {mode_label}",
        color="white",
        fontsize=13,
        pad=12,
    )

    ax.set_xlabel("x [cm]", color="white")
    ax.set_ylabel("y [cm]", color="white")
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_color("#8a8a8a")
    ax.set_aspect("equal", adjustable="box")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(args.out, facecolor=fig.get_facecolor())
    print(f"Saved static trajectory diagnostics to {args.out}")


if __name__ == "__main__":
    main()
