from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from manim import *

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from fields.common.config import apply_manim_profile, scene_bundle
from fields.common.math_text import make_panel
from fields.common.style import COLORS, reference_plane, style_surface
from fields.mediator_interaction.numerics import (
    ScalarEDSimulationConfig,
    bilinear_sample,
    sample_frames,
    signed_charge_centers,
    simulate_scalar_ed,
)

PROFILE = apply_manim_profile()
SCENE_CFG = scene_bundle("mediator", PROFILE)
config.background_color = COLORS["background"]
SIM_CFG = ScalarEDSimulationConfig()


class MediatorInteractionScene(ThreeDScene):
    def construct(self):
        layout = SCENE_CFG["layout"]
        show_a0_surface = layout.get("show_a0_surface", True)
        show_charge_centers = layout.get("show_charge_centers", False)
        colors = COLORS
        self.camera.background_color = colors["background"]
        self.set_camera_orientation(
            phi=layout["camera_phi"] * DEGREES,
            theta=layout["camera_theta"] * DEGREES,
            zoom=layout["camera_zoom"],
            frame_center=np.array(layout["frame_center"]),
        )

        info = make_panel(
            "Scalar electrodynamics with A0",
            [
                ("math", r"(\partial_t+i e A_0)^2\varphi-\nabla^2\varphi+m^2\varphi=0", colors["text"]),
                ("math", r"-\nabla^2 A_0=-i e\!\left(\varphi^*\pi-\pi^*\varphi\right)", colors["accent"]),
                (
                    "text",
                    "Lower surface: |varphi|^2. Upper surface: A_0 + z_0."
                    if show_a0_surface
                    else "Debug view: only |varphi|^2 is rendered.",
                    colors["muted_text"],
                ),
            ],
            title_scale=layout["title_scale"],
            line_scale=layout["formula_scale"],
            top_buff=layout["top_buff"],
        )
        legend_items = [Text("lower surface: |varphi|^2", color=colors["phi_fill"]).scale(layout["note_scale"])]
        if show_a0_surface:
            legend_items.append(Text("upper surface: A_0 + z0", color=colors["chi_fill"]).scale(layout["note_scale"]))
        legend = VGroup(*legend_items).arrange(DOWN, aligned_edge=LEFT, buff=0.08)
        legend.to_edge(LEFT, buff=0.35 if PROFILE == "widescreen" else 0.24)
        legend.shift(1.4 * DOWN if PROFILE == "widescreen" else 2.3 * DOWN)
        self.add_fixed_in_frame_mobjects(info, legend)

        simulation = simulate_scalar_ed(SIM_CFG)
        x_axis = simulation["x_axis"]
        y_axis = simulation["y_axis"]
        times = simulation["times"]
        density_frames = simulation["density_frames"]
        a0_frames = simulation["a0_frames"]
        charge_frames = simulation["charge_frames"]

        x_range = (float(x_axis[0]), float(x_axis[-1]))
        y_range = (float(y_axis[0]), float(y_axis[-1]))
        density_peak = max(float(np.max(density_frames)), 1e-9)
        a0_peak = max(float(np.max(np.abs(a0_frames))), 1e-9)
        base_z_scale = layout["field_z_scale"]
        phi_target_height = layout.get("phi_target_height", base_z_scale)
        chi_target_height = layout.get("chi_target_height", base_z_scale)
        phi_visual_gain = layout.get("phi_visual_gain", 1.0)
        chi_visual_gain = layout.get("chi_visual_gain", 1.0)
        phi_z_scale = phi_visual_gain * phi_target_height / density_peak
        chi_z_scale = chi_visual_gain * chi_target_height / a0_peak
        chi_shift = layout["mediator_shift"]
        playback_time_scale = layout.get("playback_time_scale", 1.0)

        lower_plane = reference_plane(x_range, y_range, layout["plane_resolution"], z_level=0.0)
        lower_plane.set_stroke(colors["phi_stroke"], width=0.52, opacity=0.18)
        upper_plane = None
        if show_a0_surface:
            upper_plane = reference_plane(x_range, y_range, layout["plane_resolution"], z_level=chi_shift)
            upper_plane.set_stroke(colors["chi_stroke"], width=0.46, opacity=0.14)

        tracker = ValueTracker(0.0)

        def current_density() -> np.ndarray:
            return sample_frames(times, density_frames, tracker.get_value())

        def current_a0() -> np.ndarray:
            return sample_frames(times, a0_frames, tracker.get_value())

        def density_height(x_val: float, y_val: float) -> float:
            return phi_z_scale * bilinear_sample(x_axis, y_axis, current_density(), x_val, y_val)

        def a0_height(x_val: float, y_val: float) -> float:
            return chi_shift + chi_z_scale * bilinear_sample(x_axis, y_axis, current_a0(), x_val, y_val)

        def make_density_surface() -> Surface:
            surface = Surface(
                lambda u, v: np.array([u, v, density_height(u, v)]),
                u_range=x_range,
                v_range=y_range,
                resolution=layout["surface_resolution"],
            )
            return style_surface(surface, colors["phi_fill"], colors["phi_stroke"], fill_opacity=0.82, stroke_width=1.10, stroke_opacity=0.92)

        def make_a0_surface() -> Surface:
            surface = Surface(
                lambda u, v: np.array([u, v, a0_height(u, v)]),
                u_range=x_range,
                v_range=y_range,
                resolution=layout["surface_resolution"],
            )
            return style_surface(surface, colors["chi_fill"], colors["chi_stroke"], fill_opacity=0.44, stroke_width=1.05, stroke_opacity=0.84)

        density_surface = always_redraw(make_density_surface)
        a0_surface = always_redraw(make_a0_surface) if show_a0_surface else None

        center_series_positive = None
        center_series_negative = None
        positive_marker = None
        negative_marker = None
        positive_path = None
        negative_path = None

        if show_charge_centers:
            signed_centers = [signed_charge_centers(frame, x_axis, y_axis) for frame in charge_frames]
            center_series_positive = np.array([item["positive"] for item in signed_centers], dtype=float)
            center_series_negative = np.array([item["negative"] for item in signed_centers], dtype=float)

            def sample_center(series: np.ndarray) -> np.ndarray:
                t_value = tracker.get_value()
                if t_value <= times[0]:
                    return series[0]
                if t_value >= times[-1]:
                    return series[-1]
                idx = int(np.searchsorted(times, t_value, side="right") - 1)
                idx = max(0, min(idx, len(times) - 2))
                frame_dt = max(float(times[idx + 1] - times[idx]), 1e-9)
                tau = (t_value - times[idx]) / frame_dt
                return (1.0 - tau) * series[idx] + tau * series[idx + 1]

            def marker_point(series: np.ndarray) -> np.ndarray:
                cx, cy = sample_center(series)
                return np.array([cx, cy, density_height(float(cx), float(cy)) + 0.09])

            positive_marker = always_redraw(
                lambda: Dot3D(
                    point=marker_point(center_series_positive),
                    radius=0.055,
                    color=colors["accent"],
                    resolution=(8, 8),
                )
            )
            negative_marker = always_redraw(
                lambda: Dot3D(
                    point=marker_point(center_series_negative),
                    radius=0.055,
                    color=colors["chi_fill"],
                    resolution=(8, 8),
                )
            )
            positive_path = TracedPath(
                positive_marker.get_center,
                stroke_color=colors["accent"],
                stroke_width=3.0,
                dissipating_time=None,
            )
            negative_path = TracedPath(
                negative_marker.get_center,
                stroke_color=colors["chi_fill"],
                stroke_width=3.0,
                dissipating_time=None,
            )

        self.play(FadeIn(info, shift=0.12 * DOWN), FadeIn(legend), run_time=0.9)
        intro_anims = [Create(lower_plane), FadeIn(density_surface)]
        if show_a0_surface and upper_plane is not None and a0_surface is not None:
            intro_anims.extend([Create(upper_plane), FadeIn(a0_surface)])
        self.play(*intro_anims, run_time=1.2)
        if show_charge_centers and positive_marker is not None and negative_marker is not None:
            self.add(positive_path, negative_path, positive_marker, negative_marker)
        self.play(
            tracker.animate.set_value(float(times[-1])),
            run_time=float(times[-1]) * playback_time_scale,
            rate_func=linear,
        )
        self.wait(0.7)
