from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
from manim import *

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from pmt.data_io import load_dataset


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    return default if raw is None else int(raw)


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    return default if raw is None else float(raw)


def _env_flag(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _resolve_data_path() -> Path:
    env_path = os.environ.get("PMT_DATA_PATH")
    if env_path:
        path = Path(env_path).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"PMT_DATA_PATH does not exist: {path}")
        return path

    base = Path(__file__).resolve().parent
    candidates = (
        base / "pmt_dataset.h5",
        base / "pmt_dataset_preview.h5",
    )
    for path in candidates:
        if path.exists():
            return path

    raise FileNotFoundError(
        "No PMT dataset found. Export one first, e.g.: "
        "python pmt/export_pmt_dataset.py --config pmt/pmt.toml --out pmt/pmt_dataset.h5"
    )


def _select_particle_indices(total_count: int, max_display: int) -> np.ndarray:
    if total_count <= max_display:
        return np.arange(total_count, dtype=np.int64)
    return np.unique(np.linspace(0, total_count - 1, max_display, dtype=np.int64))


def _colormap_potential(norm_values: np.ndarray) -> np.ndarray:
    anchors = np.array([0.0, 0.25, 0.50, 0.75, 1.0], dtype=float)
    colors = np.array(
        [
            [0, 0, 0],
            [18, 22, 64],
            [74, 76, 112],
            [174, 70, 62],
            [255, 188, 148],
        ],
        dtype=float,
    )
    clipped = np.clip(norm_values, 0.0, 1.0)
    out = np.empty(clipped.shape + (3,), dtype=np.uint8)
    for channel in range(3):
        out[..., channel] = np.clip(np.interp(clipped, anchors, colors[:, channel]), 0.0, 255.0).astype(np.uint8)
    return out


def _point_mapper(
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    panel_center: np.ndarray,
    scale: float,
):
    x_mid = 0.5 * (x_min + x_max)
    y_mid = 0.5 * (y_min + y_max)

    def world_point(x: float, y: float) -> np.ndarray:
        return np.array(
            [
                float(panel_center[0] + scale * (x - x_mid)),
                float(panel_center[1] + scale * (y - y_mid)),
                0.0,
            ]
        )

    return world_point


def _build_field_arrows(
    x_axis: np.ndarray,
    y_axis: np.ndarray,
    ex: np.ndarray,
    ey: np.ndarray,
    electrode_id: np.ndarray,
    world_point,
    panel_width: float,
    panel_height: float,
) -> VGroup:
    nx = x_axis.shape[0]
    ny = y_axis.shape[0]

    stride_x = max(1, nx // 18)
    stride_y = max(1, ny // 12)

    mag = np.hypot(ex, ey)
    mask_free = electrode_id == 0
    if np.any(mask_free):
        ref = float(np.percentile(mag[mask_free], 90.0))
    else:
        ref = float(np.percentile(mag, 90.0))
    ref = max(ref, 1e-14)

    arrow_len_ref = 0.045 * min(panel_width, panel_height)
    arrows = VGroup()

    for i in range(stride_x // 2, nx, stride_x):
        for j in range(stride_y // 2, ny, stride_y):
            if electrode_id[i, j] != 0:
                continue
            v = np.array([ex[i, j], ey[i, j]], dtype=float)
            v_norm = float(np.linalg.norm(v))
            if v_norm < 0.25 * ref:
                continue

            direction = v / v_norm
            length = arrow_len_ref * min(v_norm / ref, 1.5)
            start = world_point(float(x_axis[i]), float(y_axis[j]))
            end = start + np.array([length * direction[0], length * direction[1], 0.0])
            arrows.add(
                Arrow(
                    start,
                    end,
                    buff=0.0,
                    stroke_width=1.4,
                    color="#ff3b30",
                    tip_length=0.07,
                    max_tip_length_to_length_ratio=0.35,
                    max_stroke_width_to_length_ratio=6.0,
                ).set_opacity(0.72)
            )

    return arrows


class PMTSchematicFromHDF5(Scene):
    def construct(self):
        data_path = _resolve_data_path()
        dataset = load_dataset(data_path)

        x_axis = np.asarray(dataset["x_axis"], dtype=float)
        y_axis = np.asarray(dataset["y_axis"], dtype=float)
        potential = np.asarray(dataset["potential"], dtype=float)
        ex = np.asarray(dataset["ex"], dtype=float)
        ey = np.asarray(dataset["ey"], dtype=float)
        electrode_id = np.asarray(dataset["electrode_id"], dtype=np.uint8)

        time_axis = np.asarray(dataset["time_axis"], dtype=float)
        particle_positions = np.asarray(dataset["particle_positions"], dtype=float)
        particle_alive = np.asarray(dataset["particle_alive"], dtype=bool)
        impact_mask = np.asarray(dataset["particle_impact_mask"], dtype=bool)
        impact_step = np.asarray(dataset["particle_impact_step"], dtype=np.int64)
        impact_position = np.asarray(dataset["particle_impact_position"], dtype=float)

        metadata_cfg = dataset["metadata"]["config"]

        x_min = float(x_axis[0])
        x_max = float(x_axis[-1])
        y_min = float(y_axis[0])
        y_max = float(y_axis[-1])
        lx = x_max - x_min
        ly = y_max - y_min

        panel_width = 0.82 * float(config.frame_width)
        panel_height = 0.72 * float(config.frame_height)
        scale = min(panel_width / max(lx, 1e-12), panel_height / max(ly, 1e-12))
        panel_center = np.array([0.0, -0.25, 0.0], dtype=float)
        world_point = _point_mapper(x_min, x_max, y_min, y_max, panel_center=panel_center, scale=scale)

        self.camera.background_color = "#000000"

        potential_lo = float(np.percentile(potential, 1.0))
        potential_hi = float(np.percentile(potential, 99.0))
        if abs(potential_hi - potential_lo) < 1e-12:
            potential_lo = float(np.min(potential))
            potential_hi = float(np.max(potential) + 1e-9)
        potential_norm = (potential - potential_lo) / (potential_hi - potential_lo)

        # Map grid (x, y) into image rows/cols: rows correspond to y from top to bottom.
        potential_image_data = _colormap_potential(np.flipud(potential_norm.T))
        potential_image = ImageMobject(potential_image_data)
        potential_image.set_resampling_algorithm(RESAMPLING_ALGORITHMS["nearest"])
        potential_image.width = scale * lx
        potential_image.height = scale * ly
        potential_image.move_to(panel_center)
        potential_image.set_opacity(0.88)

        boundary = Rectangle(width=scale * lx, height=scale * ly)
        boundary.move_to(panel_center)
        boundary.set_stroke(color="#8d8d8d", width=1.6, opacity=0.95)
        boundary.set_fill(opacity=0.0)

        photocathode_shape = str(metadata_cfg.get("photocathode_shape", "line")).strip().lower()
        if photocathode_shape == "hemisphere":
            radius = 0.5 * float(metadata_cfg["photocathode_diameter"])
            thickness = float(metadata_cfg["photocathode_thickness"])
            inner = max(0.0, radius - thickness)
            cathode = AnnularSector(
                inner_radius=scale * inner,
                outer_radius=scale * radius,
                start_angle=PI / 2.0,
                angle=PI,
            )
            cathode.move_to(
                world_point(
                    float(metadata_cfg["photocathode_center_x"]),
                    float(metadata_cfg["photocathode_center_y"]),
                )
            )
        else:
            cathode = Rectangle(
                width=scale * float(metadata_cfg["cathode_thickness"]),
                height=scale * float(metadata_cfg["cathode_height"]),
            )
            cathode.move_to(world_point(float(metadata_cfg["cathode_x"]), 0.0))
        cathode.set_fill(color="#2b83f6", opacity=0.26)
        cathode.set_stroke(color="#6db2ff", width=2.0, opacity=0.92)

        dynode = Rectangle(
            width=scale * float(metadata_cfg["dynode_length"]),
            height=scale * float(metadata_cfg["dynode_thickness"]),
        )
        dynode.move_to(
            world_point(
                float(metadata_cfg["dynode_center_x"]),
                float(metadata_cfg["dynode_center_y"]),
            )
        )
        dynode.rotate(float(metadata_cfg["dynode_angle_deg"]) * DEGREES)
        dynode.set_fill(color="#c24a2a", opacity=0.68)
        dynode.set_stroke(color="#ff9b7a", width=2.0, opacity=0.95)

        dynode_glow = dynode.copy()
        dynode_glow.set_stroke(width=0.0)
        dynode_glow.set_fill(color="#ffd8a6", opacity=0.0)

        show_field = _env_flag("PMT_SHOW_FIELD_ARROWS", True)
        field_arrows = (
            _build_field_arrows(
                x_axis=x_axis,
                y_axis=y_axis,
                ex=ex,
                ey=ey,
                electrode_id=electrode_id,
                world_point=world_point,
                panel_width=scale * lx,
                panel_height=scale * ly,
            )
            if show_field
            else VGroup()
        )

        title = Text("PMT Principle Scheme", color="#f4f4f4", font_size=38)
        title.to_edge(UP, buff=0.30)
        subtitle = Text(
            f"Data: {data_path.name}",
            color="#aaaaaa",
            font_size=19,
        )
        subtitle.next_to(title, DOWN, buff=0.13)

        cathode_label = Text("Photocathode", color="#7bc0ff", font_size=21)
        cathode_label.next_to(cathode, LEFT, buff=0.18)
        dynode_label = Text("Dynode", color="#ffb08a", font_size=21)
        dynode_label.next_to(dynode, UP, buff=0.12)

        total_particles = int(particle_positions.shape[1])
        max_display_particles = max(8, _env_int("PMT_MAX_PARTICLES", 260))
        particle_indices = _select_particle_indices(total_particles, max_display_particles)
        shown_particles = int(particle_indices.shape[0])

        frame_count = int(time_axis.shape[0])
        hits_per_frame = np.zeros(frame_count, dtype=np.int32)
        valid_hits = impact_step[(impact_step >= 0) & (impact_step < frame_count)]
        if valid_hits.size:
            binc = np.bincount(valid_hits, minlength=frame_count)
            hits_per_frame[: binc.shape[0]] = binc[:frame_count]
        cumulative_hits = np.cumsum(hits_per_frame)

        stat_box = RoundedRectangle(corner_radius=0.12, width=4.7, height=1.3)
        stat_box.to_corner(UR, buff=0.24)
        stat_box.set_fill(color="#0f0f0f", opacity=0.82)
        stat_box.set_stroke(color="#737373", width=1.0, opacity=0.75)

        emitted_text = Text(f"emitted: {total_particles}", color="#eaeaea", font_size=22)
        emitted_text.move_to(stat_box.get_center() + np.array([0.0, 0.35, 0.0]))

        collected_prefix = Text("collected:", color="#eaeaea", font_size=22)
        collected_prefix.move_to(stat_box.get_center() + np.array([-0.85, 0.02, 0.0]))
        collected_value = DecimalNumber(0, num_decimal_places=0, color="#ffe8ad", font_size=24)
        collected_value.next_to(collected_prefix, RIGHT, buff=0.10)

        efficiency_prefix = Text("efficiency:", color="#eaeaea", font_size=22)
        efficiency_prefix.move_to(stat_box.get_center() + np.array([-0.83, -0.30, 0.0]))
        efficiency_value = DecimalNumber(0.0, num_decimal_places=1, color="#8de389", font_size=24)
        efficiency_value.next_to(efficiency_prefix, RIGHT, buff=0.10)
        efficiency_unit = Text("%", color="#8de389", font_size=24)
        efficiency_unit.next_to(efficiency_value, RIGHT, buff=0.06)

        time_prefix = Text("t:", color="#eaeaea", font_size=22)
        time_prefix.to_corner(UL, buff=0.25)
        time_value = DecimalNumber(0.0, num_decimal_places=3, color="#f7f7f7", font_size=24)
        time_value.next_to(time_prefix, RIGHT, buff=0.08)
        time_unit = Text("ns", color="#eaeaea", font_size=22)
        time_unit.next_to(time_value, RIGHT, buff=0.08)

        shown_text = Text(
            f"shown particles: {shown_particles}/{total_particles}",
            color="#e3e3e3",
            font_size=18,
        )
        shown_text.to_corner(DR, buff=0.20)

        trail_group = VGroup()
        show_trails = _env_flag("PMT_SHOW_TRAILS", True)
        if show_trails:
            trail_count = min(shown_particles, max(8, _env_int("PMT_TRAIL_PARTICLES", 60)))
            trail_stride = max(1, _env_int("PMT_TRAIL_STRIDE", 4))
            for idx in particle_indices[:trail_count]:
                path = particle_positions[:, idx, :]
                finite = np.isfinite(path[:, 0]) & np.isfinite(path[:, 1])
                if np.count_nonzero(finite) < 3:
                    continue
                pts = path[finite][::trail_stride]
                if pts.shape[0] < 2:
                    continue
                scene_pts = [world_point(float(px), float(py)) for px, py in pts]
                trail = VMobject()
                trail.set_points_smoothly(scene_pts)
                trail.set_stroke(color="#ffffff", width=1.15, opacity=0.22)
                trail_group.add(trail)

        electron_color = "#ffffff"
        impact_color = "#ffffff"
        dot_radius = 0.018

        electron_dots = VGroup(
            *[
                Dot(radius=dot_radius, color=electron_color, fill_opacity=0.0, stroke_width=0.0)
                for _ in range(shown_particles)
            ]
        )

        tracker = ValueTracker(float(time_axis[0]))

        def frame_index() -> int:
            idx = int(np.searchsorted(time_axis, float(tracker.get_value()), side="right") - 1)
            return int(np.clip(idx, 0, frame_count - 1))

        sampled_impact_mask = impact_mask[particle_indices]
        sampled_impact_step = impact_step[particle_indices]
        sampled_impact_position = impact_position[particle_indices]

        def update_dots(dots: VGroup) -> None:
            idx = frame_index()
            pos = particle_positions[idx, particle_indices]
            alive = particle_alive[idx, particle_indices]

            for k, dot in enumerate(dots):
                if alive[k] and np.isfinite(pos[k]).all():
                    dot.move_to(world_point(float(pos[k, 0]), float(pos[k, 1])))
                    dot.set_color(electron_color)
                    dot.set_opacity(1.0)
                    continue

                if sampled_impact_mask[k] and sampled_impact_step[k] >= 0 and sampled_impact_step[k] <= idx:
                    hit_pos = sampled_impact_position[k]
                    if np.isfinite(hit_pos).all():
                        dot.move_to(world_point(float(hit_pos[0]), float(hit_pos[1])))
                        dot.set_color(impact_color)
                        dot.set_opacity(0.88)
                        continue

                dot.set_opacity(0.0)

        electron_dots.add_updater(update_dots)

        def update_stats(_: Mobject) -> None:
            idx = frame_index()
            hits = int(cumulative_hits[idx])
            collected_value.set_value(hits)
            efficiency_value.set_value(100.0 * hits / max(total_particles, 1))
            time_value.set_value(float(time_axis[idx]) * 1e9)
            dynode_glow.set_fill(color="#ffe3c4", opacity=0.03 + 0.30 * (hits / max(total_particles, 1)))

        stat_driver = VMobject()
        stat_driver.add_updater(update_stats)

        self.add(stat_driver)

        self.play(
            FadeIn(title, shift=0.15 * DOWN),
            FadeIn(subtitle, shift=0.10 * DOWN),
            run_time=0.8,
        )
        self.play(
            FadeIn(potential_image),
            Create(boundary),
            run_time=0.9,
        )
        self.play(
            FadeIn(cathode),
            FadeIn(dynode),
            FadeIn(cathode_label, shift=0.08 * RIGHT),
            FadeIn(dynode_label, shift=0.08 * DOWN),
            run_time=0.8,
        )

        if len(field_arrows) > 0:
            self.play(LaggedStart(*[FadeIn(a) for a in field_arrows], lag_ratio=0.02), run_time=1.0)

        if len(trail_group) > 0:
            self.play(FadeIn(trail_group), run_time=0.5)

        self.play(
            FadeIn(stat_box),
            FadeIn(emitted_text),
            FadeIn(collected_prefix),
            FadeIn(collected_value),
            FadeIn(efficiency_prefix),
            FadeIn(efficiency_value),
            FadeIn(efficiency_unit),
            FadeIn(time_prefix),
            FadeIn(time_value),
            FadeIn(time_unit),
            FadeIn(shown_text),
            run_time=0.7,
        )

        self.add(dynode_glow)
        self.play(FadeIn(electron_dots), run_time=0.45)

        render_seconds = max(1.0, _env_float("PMT_RENDER_SECONDS", 16.0))
        self.play(
            tracker.animate.set_value(float(time_axis[-1])),
            run_time=render_seconds,
            rate_func=linear,
        )

        electron_dots.clear_updaters()
        stat_driver.clear_updaters()
        self.wait(0.5)
