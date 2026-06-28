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
from fields.wavepackets_free.numerics import PacketParameters, superposed_field

PROFILE = apply_manim_profile()
SCENE_CFG = scene_bundle("two_packets", PROFILE)
config.background_color = COLORS["background"]

LEFT_PACKET = PacketParameters(
    amplitude=0.82,
    sigma=0.58,
    mass=1.2,
    kx=3.1,
    ky=0.0,
    x0=-3.35,
    y0=0.0,
)
RIGHT_PACKET = PacketParameters(
    amplitude=0.82,
    sigma=0.58,
    mass=1.2,
    kx=-3.1,
    ky=0.0,
    x0=3.35,
    y0=0.0,
)
DURATION = 7.2


class TwoFreeWavePackets(ThreeDScene):
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
            "Two free wave packets",
            [
                ("math", r"\phi(t,\mathbf r)=\phi_1(t,\mathbf r)+\phi_2(t,\mathbf r)", colors["text"]),
                ("text", "No interaction: linear superposition only.", colors["accent"]),
                ("text", "Overlap creates interference; the packets then recover and continue.", colors["muted_text"]),
            ],
            title_scale=layout["title_scale"],
            line_scale=layout["formula_scale"],
            top_buff=layout["top_buff"],
        )
        self.add_fixed_in_frame_mobjects(info)

        x_range = layout["x_range"]
        y_range = layout["y_range"]
        plane = reference_plane(x_range, y_range, layout["plane_resolution"])
        tracker = ValueTracker(0.0)
        z_scale = layout["field_z_scale"]

        def field_value(x_val: float, y_val: float) -> float:
            return z_scale * float(superposed_field(x_val, y_val, tracker.get_value(), [LEFT_PACKET, RIGHT_PACKET]))

        def make_surface() -> Surface:
            surface = Surface(
                lambda u, v: np.array([u, v, field_value(u, v)]),
                u_range=x_range,
                v_range=y_range,
                resolution=layout["surface_resolution"],
            )
            return style_surface(surface, colors["phi_fill"], colors["phi_stroke"], fill_opacity=0.64)

        packet_surface = always_redraw(make_surface)
        superposition_note = Text("Linear superposition", color=colors["accent"]).scale(layout["note_scale"])
        superposition_note.to_edge(DOWN, buff=0.35 if PROFILE == "widescreen" else 1.25)
        self.add_fixed_in_frame_mobjects(superposition_note)

        self.play(FadeIn(info, shift=0.12 * DOWN), run_time=0.9)
        self.play(Create(plane), FadeIn(packet_surface), FadeIn(superposition_note), run_time=1.1)
        self.play(tracker.animate.set_value(DURATION), run_time=DURATION, rate_func=linear)
        self.wait(0.6)
