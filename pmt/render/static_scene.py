from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from pmt.physics.simulate import SceneResult
from pmt.physics.types import M_TO_CM


def _require_matplotlib():
    try:
        if "MPLCONFIGDIR" not in os.environ:
            mpl_cache = Path(__file__).resolve().parents[1] / ".mplconfig"
            mpl_cache.mkdir(parents=True, exist_ok=True)
            os.environ["MPLCONFIGDIR"] = str(mpl_cache)

        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import Circle, Polygon, Rectangle, Wedge
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("matplotlib is required for PNG rendering") from exc
    return plt, Circle, Polygon, Rectangle, Wedge


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


def render_scene_png(result: SceneResult, output_path: str | Path) -> Path:
    plt, Circle, Polygon, Rectangle, Wedge = _require_matplotlib()

    cfg = result.config
    render_cfg = cfg.render
    geometry_cfg = cfg.geometry

    x_cm = result.x_axis_m * M_TO_CM
    y_cm = result.y_axis_m * M_TO_CM

    scalar_map = render_cfg.scalar_map
    if scalar_map == "field_magnitude":
        scalar = np.sqrt(result.field.ex * result.field.ex + result.field.ey * result.field.ey)
        scalar_label = "|E| [V/m]"
    else:
        scalar = result.field.potential
        scalar_label = "Potential [V]"

    finite_scalar = np.asarray(scalar[np.isfinite(scalar)], dtype=float)
    if finite_scalar.size == 0:
        vmin, vmax = -1.0, 1.0
    else:
        q = np.percentile(finite_scalar, [1.0, 99.0])
        vmin = float(q[0])
        vmax = float(q[1])
        if abs(vmax - vmin) < 1e-12:
            vmax = vmin + 1.0

    fig, ax = plt.subplots(
        figsize=(render_cfg.figure_width_in, render_cfg.figure_height_in),
        dpi=render_cfg.dpi,
    )
    fig.patch.set_facecolor(render_cfg.background)
    ax.set_facecolor(render_cfg.background)

    im = ax.imshow(
        scalar.T,
        extent=[float(x_cm[0]), float(x_cm[-1]), float(y_cm[0]), float(y_cm[-1])],
        origin="lower",
        cmap=render_cfg.colormap,
        vmin=vmin,
        vmax=vmax,
        interpolation="nearest",
        aspect="auto",
        alpha=0.90,
    )

    if render_cfg.show_equipotential_lines and render_cfg.equipotential_count > 0:
        finite_phi = np.asarray(result.field.potential[np.isfinite(result.field.potential)], dtype=float)
        if finite_phi.size:
            p_lo, p_hi = np.percentile(finite_phi, [20.0, 92.0])
            if p_hi > p_lo:
                levels = np.linspace(float(p_lo), float(p_hi), int(render_cfg.equipotential_count))
                ax.contour(
                    x_cm,
                    y_cm,
                    result.field.potential.T,
                    levels=levels,
                    colors=render_cfg.equipotential_color,
                    linewidths=render_cfg.equipotential_linewidth,
                    alpha=0.78,
                )

    if render_cfg.show_field_lines:
        mode = render_cfg.field_line_mode
        if mode == "quiver":
            base = max(1.0, render_cfg.field_line_density)
            step = max(1, int(round(9.0 / base)))
            u = np.array(result.field.ex[::step, ::step], copy=True)
            v = np.array(result.field.ey[::step, ::step], copy=True)
            if render_cfg.normalize_field_arrows:
                mag = np.hypot(u, v)
                norm = mag > 1e-20
                u = np.where(norm, u / np.maximum(mag, 1e-20), 0.0)
                v = np.where(norm, v / np.maximum(mag, 1e-20), 0.0)
                u *= render_cfg.field_arrow_length_cm
                v *= render_cfg.field_arrow_length_cm
            ax.quiver(
                x_cm[::step],
                y_cm[::step],
                u.T,
                v.T,
                color=render_cfg.field_line_color,
                alpha=0.75,
                width=0.002,
                angles="xy",
                scale_units="xy",
                scale=1.0 if render_cfg.normalize_field_arrows else 4.0e4,
                headwidth=3.5,
                headlength=4.5,
            )
        else:
            ax.streamplot(
                x_cm,
                y_cm,
                result.field.ex.T,
                result.field.ey.T,
                density=render_cfg.field_line_density,
                color=render_cfg.field_line_color,
                linewidth=0.75,
                arrowsize=0.70,
                minlength=0.06,
            )

    total_particles = int(result.tracks.positions_m.shape[1])
    idx = _select_indices(total_particles, render_cfg.max_trajectories)
    for pidx in idx:
        track = result.tracks.positions_m[:, pidx, :]
        finite = np.isfinite(track[:, 0]) & np.isfinite(track[:, 1])
        if np.count_nonzero(finite) < 2:
            continue
        pts = track[finite]
        ax.plot(
            pts[:, 0] * M_TO_CM,
            pts[:, 1] * M_TO_CM,
            color=render_cfg.trajectory_color,
            alpha=render_cfg.trajectory_alpha,
            linewidth=render_cfg.trajectory_linewidth,
        )

    impact = result.tracks.impact_position_m[result.tracks.status == 1]
    if impact.size:
        finite_impact = np.isfinite(impact[:, 0]) & np.isfinite(impact[:, 1])
        if np.any(finite_impact):
            ax.scatter(
                impact[finite_impact, 0] * M_TO_CM,
                impact[finite_impact, 1] * M_TO_CM,
                s=10,
                c="#fff7cc",
                alpha=0.90,
                linewidths=0.0,
            )

    if geometry_cfg.enable_cathode:
        shape = geometry_cfg.cathode_shape
        if shape == "hemisphere":
            patch = Wedge(
                center=(geometry_cfg.photocathode_center_x_m * M_TO_CM, geometry_cfg.photocathode_center_y_m * M_TO_CM),
                r=0.5 * geometry_cfg.photocathode_diameter_m * M_TO_CM,
                theta1=90.0,
                theta2=270.0,
                width=geometry_cfg.photocathode_thickness_m * M_TO_CM,
                facecolor=render_cfg.electrode_cathode_color,
                edgecolor=render_cfg.electrode_cathode_color,
                linewidth=1.6,
                alpha=0.28,
            )
            ax.add_patch(patch)
        else:
            patch = Rectangle(
                (
                    (geometry_cfg.line_cathode_x_m - 0.5 * geometry_cfg.line_cathode_thickness_m) * M_TO_CM,
                    (-0.5 * geometry_cfg.line_cathode_height_m) * M_TO_CM,
                ),
                width=geometry_cfg.line_cathode_thickness_m * M_TO_CM,
                height=geometry_cfg.line_cathode_height_m * M_TO_CM,
                facecolor=render_cfg.electrode_cathode_color,
                edgecolor=render_cfg.electrode_cathode_color,
                linewidth=1.6,
                alpha=0.28,
            )
            ax.add_patch(patch)

    if geometry_cfg.enable_receiver:
        receiver_kind = geometry_cfg.receiver_kind
        if receiver_kind == "point":
            ax.add_patch(
                Circle(
                    (geometry_cfg.receiver_point_x_m * M_TO_CM, geometry_cfg.receiver_point_y_m * M_TO_CM),
                    radius=geometry_cfg.receiver_radius_m * M_TO_CM,
                    facecolor=render_cfg.electrode_receiver_color,
                    edgecolor="#ffe2cf",
                    linewidth=1.2,
                    alpha=0.75,
                )
            )
        else:
            dynode_pts = _rotated_rect_polygon(
                cx=geometry_cfg.plate_center_x_m * M_TO_CM,
                cy=geometry_cfg.plate_center_y_m * M_TO_CM,
                length=geometry_cfg.plate_length_m * M_TO_CM,
                thickness=geometry_cfg.plate_thickness_m * M_TO_CM,
                angle_deg=geometry_cfg.plate_angle_deg,
            )
            ax.add_patch(
                Polygon(
                    dynode_pts,
                    closed=True,
                    facecolor=render_cfg.electrode_receiver_color,
                    edgecolor="#ffe2cf",
                    linewidth=1.4,
                    alpha=0.76,
                )
            )

    if geometry_cfg.focus_enabled:
        focus_pts = _rotated_rect_polygon(
            cx=geometry_cfg.focus_center_x_m * M_TO_CM,
            cy=geometry_cfg.focus_center_y_m * M_TO_CM,
            length=geometry_cfg.focus_length_m * M_TO_CM,
            thickness=geometry_cfg.focus_thickness_m * M_TO_CM,
            angle_deg=geometry_cfg.focus_angle_deg,
        )
        ax.add_patch(
            Polygon(
                focus_pts,
                closed=True,
                facecolor=render_cfg.electrode_focus_color,
                edgecolor="#eafff0",
                linewidth=1.2,
                alpha=0.55,
            )
        )

    ax.add_patch(
        Rectangle(
            (float(x_cm[0]), float(y_cm[0])),
            width=float(x_cm[-1] - x_cm[0]),
            height=float(y_cm[-1] - y_cm[0]),
            fill=False,
            edgecolor="#8c8c8c",
            linewidth=1.1,
            alpha=0.85,
        )
    )

    stats = result.stats
    scene_title = cfg.scene.title
    ax.set_title(scene_title, color="white", fontsize=17, pad=14)

    summary = (
        f"electrons: {stats.electron_count}\n"
        f"collected: {stats.collected_count}\n"
        f"efficiency: {100.0 * stats.collection_efficiency:.2f}%\n"
        f"misses: {stats.miss_count}\n"
        f"time: {stats.compute_seconds * 1e3:.1f} ms"
    )
    ax.text(
        0.985,
        0.985,
        summary,
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=11,
        color="white",
        bbox={"boxstyle": "round,pad=0.45", "facecolor": "black", "alpha": 0.35, "edgecolor": "#aaaaaa"},
    )

    params = (
        f"scene={cfg.scene.name} | kind={cfg.scene.kind} | physics={cfg.scene.physics_mode} | "
        f"Bz={cfg.time.bz_t:.4g} T | dt={cfg.time.dt_s:.2e} s | steps={cfg.time.steps}"
    )
    ax.text(
        0.01,
        -0.085,
        params,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        color="#dddddd",
    )

    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.018)
    cbar.set_label(scalar_label, color="white")
    cbar.ax.yaxis.set_tick_params(color="white")
    for tick in cbar.ax.get_yticklabels():
        tick.set_color("white")

    ax.set_xlabel("x [cm]", color="white")
    ax.set_ylabel("y [cm]", color="white")
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_color("#9a9a9a")

    ax.set_aspect("equal", adjustable="box")

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, facecolor=fig.get_facecolor())
    plt.close(fig)
    return out
