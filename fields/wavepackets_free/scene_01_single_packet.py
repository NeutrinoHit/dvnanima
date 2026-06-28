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
from fields.wavepackets_free.numerics import PacketParameters, packet_center, packet_diagnostics, packet_field

PROFILE = apply_manim_profile()
SCENE_CFG = scene_bundle("single_packet", PROFILE)
config.background_color = COLORS["background"]

PACKET = PacketParameters(
    amplitude=1.15,
    sigma=0.62,
    mass=1.25,
    kx=3.5,
    ky=1.2,
    x0=-3.6,
    y0=-1.4,
)
DURATION = 7.4


class SingleGaussianWavePacket(ThreeDScene):
    def construct(self):
        layout = SCENE_CFG["layout"]
        colors = COLORS
        self.camera.background_color = colors["background"]
        self.set_camera_orientation(
            phi=layout["camera_phi"] * DEGREES,
            theta=layout["camera_theta"] * DEGREES,
            zoom=layout["camera_zoom"],
            frame_center=np.array(layout["frame_center"]),
        )

        info = make_panel(
            "Gaussian wave packet",
            [
                ("math", r"\phi(t,\mathbf r)\approx \mathcal A(t)e^{-\Delta_\parallel^2/(2\sigma_\parallel^2)-\Delta_\perp^2/(2\sigma_\perp^2)}\cos\Phi", colors["text"]),
                ("math", r"\mathbf v_g=\frac{\mathbf k_0}{\omega_0}", colors["accent"]),
                ("text", "Center drifts, envelope broadens, amplitude decays.", colors["muted_text"]),
            ],
            title_scale=layout["title_scale"],
            line_scale=layout["formula_scale"],
            top_buff=layout["top_buff"],
        )
        self.add_fixed_in_frame_mobjects(info)

        x_range = layout["x_range"]
        y_range = layout["y_range"]
        plane = reference_plane(x_range, y_range, layout["plane_resolution"])

        diagnostics = packet_diagnostics(PACKET)
        tracker = ValueTracker(0.0)
        z_scale = layout["field_z_scale"]

        def field_value(x_val: float, y_val: float) -> float:
            return z_scale * float(packet_field(x_val, y_val, tracker.get_value(), PACKET))

        def make_surface() -> Surface:
            surface = Surface(
                lambda u, v: np.array([u, v, field_value(u, v)]),
                u_range=x_range,
                v_range=y_range,
                resolution=layout["surface_resolution"],
            )
            return style_surface(surface, colors["phi_fill"], colors["phi_stroke"], fill_opacity=0.64)

        packet_surface = always_redraw(make_surface)

        center_arrow = Arrow(
            start=np.array([PACKET.x0, PACKET.y0, 0.02]),
            end=np.array([
                PACKET.x0 + 1.15 * diagnostics["v_group"][0],
                PACKET.y0 + 1.15 * diagnostics["v_group"][1],
                0.02,
            ]),
            buff=0.0,
            color=colors["accent"],
            stroke_width=4.0,
            tip_length=0.18,
            max_stroke_width_to_length_ratio=1000.0,
        )
        center_label = MathTex(r"\mathbf v_g", color=colors["accent"]).scale(layout["label_scale"])
        center_label.move_to(center_arrow.get_end() + np.array([0.32, 0.18, 0.0]))

        moving_marker = always_redraw(
            lambda: Dot3D(
                point=np.array([*packet_center(PACKET, tracker.get_value()), 0.02]),
                radius=0.045,
                color=colors["accent"],
                resolution=(5, 5),
            )
        )

        self.play(FadeIn(info, shift=0.12 * DOWN), run_time=0.9)
        self.play(Create(plane), FadeIn(packet_surface), run_time=1.1)
        self.wait(0.35)
        self.play(GrowArrow(center_arrow), FadeIn(center_label), FadeIn(moving_marker), run_time=0.7)
        self.play(tracker.animate.set_value(DURATION), run_time=DURATION, rate_func=linear)
        self.wait(0.6)
