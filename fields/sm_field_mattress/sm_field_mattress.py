from __future__ import annotations

from pathlib import Path
import configparser
import math
import os
import sys

import numpy as np
from manim import *


ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


FIELD_SPECS = [
    ("u", "quark", "#FF6B6B"),
    ("d", "quark", "#FF8E72"),
    ("c", "quark", "#FFB347"),
    ("s", "quark", "#FFD166"),
    ("t", "quark", "#A8E063"),
    ("b", "quark", "#55D6BE"),
    ("e", "lepton", "#7CC7FF"),
    ("mu", "lepton", "#5EA8FF"),
    ("tau", "lepton", "#A3A1FF"),
    ("nu_e", "neutrino", "#C4B5FD"),
    ("nu_mu", "neutrino", "#B388FF"),
    ("nu_tau", "neutrino", "#E0A8FF"),
    ("gamma", "gauge", "#F6F7EB"),
    ("g", "gauge", "#76E39B"),
    ("W", "gauge", "#FFD166"),
    ("Z", "gauge", "#DADADA"),
    ("H", "scalar", "#FF7AC8"),
]


def load_cfg(paths: list[Path]) -> dict:
    cfg = configparser.ConfigParser(inline_comment_prefixes=(";",))
    cfg.read([str(path) for path in paths])

    def get(section: str, key: str, cast=str, fallback=None):
        if fallback is None:
            return cast(cfg[section][key])
        return cast(cfg.get(section, key, fallback=str(fallback)))

    def get_bool(section: str, key: str, fallback=False):
        return cfg.getboolean(section, key, fallback=fallback)

    profile = get("manim", "profile", str, "widescreen").strip().lower()
    if profile not in {"shorts", "widescreen"}:
        profile = "widescreen"

    manim_defaults = {
        "shorts": {
            "pixel_width": 1080,
            "pixel_height": 1920,
            "frame_width": 9.0,
            "frame_height": 16.0,
        },
        "widescreen": {
            "pixel_width": 1920,
            "pixel_height": 1080,
            "frame_width": 16.0,
            "frame_height": 9.0,
        },
    }[profile]

    return {
        "manim": {
            "profile": profile,
            "pixel_width": manim_defaults["pixel_width"],
            "pixel_height": manim_defaults["pixel_height"],
            "frame_width": manim_defaults["frame_width"],
            "frame_height": manim_defaults["frame_height"],
            "frame_rate": get("manim", "frame_rate", int, 30),
            "background_color": get("manim", "background_color", str, "#000000"),
        },
        "scene": {
            "show_labels": get_bool("scene", "show_labels", True),
            "show_legend": get_bool("scene", "show_legend", True),
            "show_field_tags": get_bool("scene", "show_field_tags", True),
        },
        "stack": {
            "width": get("stack", "width", float, 7.6),
            "depth": get("stack", "depth", float, 4.6),
            "layer_spacing": get("stack", "layer_spacing", float, 0.105),
            "surface_resolution_x": get("stack", "surface_resolution_x", int, 13),
            "surface_resolution_y": get("stack", "surface_resolution_y", int, 8),
            "line_samples": get("stack", "line_samples", int, 26),
            "vacuum_amplitude": get("stack", "vacuum_amplitude", float, 0.028),
            "vacuum_speed": get("stack", "vacuum_speed", float, 1.35),
            "wave_amplitude": get("stack", "wave_amplitude", float, 0.075),
            "wave_packet_amplitude": get("stack", "wave_packet_amplitude", float, 0.18),
            "wave_packet_sigma": get("stack", "wave_packet_sigma", float, 0.82),
            "wave_packet_speed": get("stack", "wave_packet_speed", float, 1.12),
            "wave_packet_wavenumber": get("stack", "wave_packet_wavenumber", float, 3.2),
            "fill_opacity": get("stack", "fill_opacity", float, 0.07),
            "stroke_opacity": get("stack", "stroke_opacity", float, 0.62),
            "stroke_width": get("stack", "stroke_width", float, 0.82),
            "cutout_center_x": get("stack", "cutout_center_x", float, -0.85),
            "cutout_center_y": get("stack", "cutout_center_y", float, -0.70),
            "cutout_width": get("stack", "cutout_width", float, 2.7),
            "cutout_depth": get("stack", "cutout_depth", float, 2.0),
            "cutout_edge_samples": get("stack", "cutout_edge_samples", int, 24),
            "cutout_edge_opacity": get("stack", "cutout_edge_opacity", float, 0.92),
            "cutout_edge_width": get("stack", "cutout_edge_width", float, 1.18),
        },
        "camera": {
            "phi": get("camera", "phi", float, 62.0),
            "theta": get("camera", "theta", float, -48.0),
            "zoom": get("camera", "zoom", float, 1.0),
            "frame_center_x": get("camera", "frame_center_x", float, 0.0),
            "frame_center_y": get("camera", "frame_center_y", float, -0.10),
            "frame_center_z": get("camera", "frame_center_z", float, 0.05),
        },
        "timing": {
            "build_time": get("timing", "build_time", float, 2.1),
            "vacuum_time": get("timing", "vacuum_time", float, 2.0),
            "wave_time": get("timing", "wave_time", float, get("timing", "excitation_time", float, 7.4)),
            "cutaway_reveal_time": get("timing", "cutaway_reveal_time", float, 2.0),
            "cutaway_time": get("timing", "cutaway_time", float, 6.0),
            "hold_time": get("timing", "hold_time", float, 1.2),
        },
        "colors": {
            "text": get("colors", "text", str, "#FFFFFF"),
            "muted": get("colors", "muted", str, "#C9D4E5"),
            "accent": get("colors", "accent", str, "#FFD166"),
            "panel": get("colors", "panel", str, "#05080C"),
            "panel_stroke": get("colors", "panel_stroke", str, "#7CC7FF"),
        },
    }


BASE_DIR = Path(__file__).resolve().parent
BASE_CFG_PATH = BASE_DIR / "run.cfg"
CFG_ENV_VAR = "SM_FIELD_MATTRESS_CONFIG"


def resolve_cfg_paths() -> list[Path]:
    override = os.environ.get(CFG_ENV_VAR)
    if not override:
        return [BASE_CFG_PATH]

    override_path = Path(override).expanduser()
    if not override_path.is_absolute():
        override_path = BASE_DIR / override_path

    if override_path == BASE_CFG_PATH:
        return [BASE_CFG_PATH]
    return [BASE_CFG_PATH, override_path]


CFG = load_cfg(resolve_cfg_paths())


def apply_render_geometry(manim_params: dict) -> None:
    aspect = manim_params["frame_width"] / manim_params["frame_height"]
    long_side = max(int(config.pixel_width), int(config.pixel_height))
    if aspect >= 1.0:
        config.pixel_width = long_side
        config.pixel_height = max(1, int(round(long_side / aspect)))
    else:
        config.pixel_height = long_side
        config.pixel_width = max(1, int(round(long_side * aspect)))
    config.frame_width = manim_params["frame_width"]
    config.frame_height = manim_params["frame_height"]
    config.frame_rate = manim_params["frame_rate"]


apply_render_geometry(CFG["manim"])


def field_z(index: int, layer_spacing: float) -> float:
    return (index - 0.5 * (len(FIELD_SPECS) - 1)) * layer_spacing


class StandardModelFieldMattress(ThreeDScene):
    def construct(self):
        p = CFG["stack"]
        c = CFG["colors"]
        scene_cfg = CFG["scene"]
        timing = CFG["timing"]
        camera_cfg = CFG["camera"]

        self.camera.background_color = CFG["manim"]["background_color"]
        self.set_camera_orientation(
            phi=camera_cfg["phi"] * DEGREES,
            theta=camera_cfg["theta"] * DEGREES,
            zoom=camera_cfg["zoom"],
            frame_center=np.array(
                [
                    camera_cfg["frame_center_x"],
                    camera_cfg["frame_center_y"],
                    camera_cfg["frame_center_z"],
                ]
            ),
        )

        clock = ValueTracker(0.0)
        visibility = ValueTracker(0.0)
        cutout = ValueTracker(0.0)
        x_min, x_max = -0.5 * p["width"], 0.5 * p["width"]
        y_min, y_max = -0.5 * p["depth"], 0.5 * p["depth"]

        def height_for(field_name: str, field_idx: int, u: float, v: float) -> float:
            t_val = clock.get_value()
            z0 = field_z(field_idx, p["layer_spacing"])
            phase = 0.72 * field_idx
            group_offset = {"quark": 0.0, "lepton": 0.7, "neutrino": 1.3, "gauge": 2.1, "scalar": 2.7}
            group_phase = group_offset.get(FIELD_SPECS[field_idx][1], 0.0)

            vacuum = (
                math.sin(1.10 * u + 0.72 * v + p["vacuum_speed"] * t_val + phase)
                + 0.52 * math.sin(0.62 * u - 1.08 * v + 0.78 * p["vacuum_speed"] * t_val + 0.37 * phase)
                + 0.28 * math.sin(1.55 * v - 0.46 * u - 0.55 * p["vacuum_speed"] * t_val + group_phase)
            )

            direction = 0.38 * field_idx + group_phase
            cx = 0.34 * p["width"] * math.sin(0.23 * t_val + 0.63 * field_idx)
            cy = 0.34 * p["depth"] * math.cos(0.19 * t_val + 0.47 * field_idx)
            dx = u - cx
            dy = v - cy
            local = math.exp(-(dx * dx + dy * dy) / (2.0 * p["wave_packet_sigma"] ** 2))
            travel_coord = math.cos(direction) * dx + math.sin(direction) * dy
            packet = local * math.cos(p["wave_packet_wavenumber"] * travel_coord - p["wave_packet_speed"] * t_val + phase)

            return z0 + visibility.get_value() * (p["vacuum_amplitude"] * vacuum + p["wave_amplitude"] * math.sin(0.35 * t_val + phase) + p["wave_packet_amplitude"] * packet)

        def cutout_bounds() -> tuple[float, float, float, float]:
            amount = max(0.0, min(1.0, cutout.get_value()))
            width = p["cutout_width"] * amount
            depth = p["cutout_depth"] * amount
            cx = p["cutout_center_x"]
            cy = p["cutout_center_y"]
            return (
                max(x_min, cx - 0.5 * width),
                min(x_max, cx + 0.5 * width),
                max(y_min, cy - 0.5 * depth),
                min(y_max, cy + 0.5 * depth),
            )

        def add_polyline(mesh: VMobject, points: list[np.ndarray]) -> None:
            if len(points) < 2:
                return
            if not mesh.has_points():
                mesh.set_points_as_corners(points)
                return
            mesh.start_new_path(points[0])
            mesh.add_points_as_corners(points[1:])

        def x_segments_for(y: float) -> list[tuple[float, float]]:
            if cutout.get_value() < 0.015:
                return [(x_min, x_max)]
            x_left, x_right, y_low, y_high = cutout_bounds()
            if y_low < y < y_high:
                return [(x_min, x_left), (x_right, x_max)]
            return [(x_min, x_max)]

        def y_segments_for(x: float) -> list[tuple[float, float]]:
            if cutout.get_value() < 0.015:
                return [(y_min, y_max)]
            x_left, x_right, y_low, y_high = cutout_bounds()
            if x_left < x < x_right:
                return [(y_min, y_low), (y_high, y_max)]
            return [(y_min, y_max)]

        def segment_samples(length: float, full_length: float) -> int:
            return max(3, int(round(p["line_samples"] * length / full_length)))

        def make_field_mesh(field_idx: int, field_name: str, color: str) -> VMobject:
            mesh = VMobject()
            x_values = np.linspace(x_min, x_max, p["surface_resolution_x"] + 1)
            y_values = np.linspace(y_min, y_max, p["surface_resolution_y"] + 1)

            for y in y_values:
                for x0, x1 in x_segments_for(float(y)):
                    if x1 - x0 < 0.04:
                        continue
                    xs = np.linspace(x0, x1, segment_samples(x1 - x0, p["width"]))
                    add_polyline(
                        mesh,
                        [np.array([x, y, height_for(field_name, field_idx, x, y)]) for x in xs],
                    )

            for x in x_values:
                for y0, y1 in y_segments_for(float(x)):
                    if y1 - y0 < 0.04:
                        continue
                    ys = np.linspace(y0, y1, segment_samples(y1 - y0, p["depth"]))
                    add_polyline(
                        mesh,
                        [np.array([x, y, height_for(field_name, field_idx, x, y)]) for y in ys],
                    )

            alpha = visibility.get_value()
            mesh.set_stroke(color, width=p["stroke_width"], opacity=p["stroke_opacity"] * alpha)
            return mesh

        def make_curve(points: list[np.ndarray], color: str, width: float, opacity: float) -> VMobject:
            curve = VMobject()
            curve.set_points_as_corners(points)
            curve.set_stroke(color, width=width, opacity=opacity)
            return curve

        def make_cutout_edges() -> VGroup:
            amount = max(0.0, min(1.0, cutout.get_value()))
            if amount < 0.02:
                return VGroup()

            x_left, x_right, y_low, y_high = cutout_bounds()
            alpha = visibility.get_value() * amount
            samples = max(4, p["cutout_edge_samples"])
            edges = VGroup()
            for idx, (field_name, _group, color) in enumerate(FIELD_SPECS):
                opacity = p["cutout_edge_opacity"] * alpha
                width = p["cutout_edge_width"]
                front = [
                    np.array([x, y_low, height_for(field_name, idx, x, y_low)])
                    for x in np.linspace(x_left, x_right, samples)
                ]
                back = [
                    np.array([x, y_high, height_for(field_name, idx, x, y_high)])
                    for x in np.linspace(x_left, x_right, samples)
                ]
                left = [
                    np.array([x_left, y, height_for(field_name, idx, x_left, y)])
                    for y in np.linspace(y_low, y_high, samples)
                ]
                right = [
                    np.array([x_right, y, height_for(field_name, idx, x_right, y)])
                    for y in np.linspace(y_low, y_high, samples)
                ]
                edges.add(
                    make_curve(front, color, width, opacity),
                    make_curve(back, color, 0.65 * width, 0.70 * opacity),
                    make_curve(left, color, width, opacity),
                    make_curve(right, color, 0.65 * width, 0.70 * opacity),
                )

            corner_opacity = 0.42 * alpha
            for x in (x_left, x_right):
                for y in (y_low, y_high):
                    vertical = [
                        np.array([x, y, height_for(field_name, idx, x, y)])
                        for idx, (field_name, _group, _color) in enumerate(FIELD_SPECS)
                    ]
                    edges.add(make_curve(vertical, CFG["colors"]["panel_stroke"], 1.4, corner_opacity))
            return edges

        field_surfaces = VGroup()
        for idx, (field_name, _group, color) in enumerate(FIELD_SPECS):
            field_surfaces.add(always_redraw(lambda idx=idx, field_name=field_name, color=color: make_field_mesh(idx, field_name, color)))
        cutout_edges = always_redraw(make_cutout_edges)

        if scene_cfg["show_labels"]:
            title = Text("The Standard Model as Quantum Fields", color=c["text"], weight=BOLD, font="Arial").scale(0.52)
            subtitle = Text("Each particle type corresponds to a field filling space.", color=c["muted"], font="Arial").scale(0.30)
            title_group = VGroup(title, subtitle).arrange(DOWN, aligned_edge=LEFT, buff=0.10)
            title_group.to_corner(UL, buff=0.38)
            self.add_fixed_in_frame_mobjects(title_group)

            caption = Text("Vacuum: the fields are present everywhere.", color=c["accent"], weight=BOLD, font="Arial").scale(0.38)
            caption.to_edge(DOWN, buff=0.40)
            self.add_fixed_in_frame_mobjects(caption)
        else:
            title_group = VGroup()
            caption = VGroup()

        if scene_cfg["show_legend"]:
            legend = self.make_legend(c)
            self.add_fixed_in_frame_mobjects(legend)
        else:
            legend = VGroup()

        if scene_cfg["show_field_tags"]:
            tag_group = self.make_field_tags(p, c)
            self.add_fixed_in_frame_mobjects(tag_group)
        else:
            tag_group = VGroup()

        self.add(field_surfaces, cutout_edges)
        fade_ins = []
        for obj in (title_group, legend, tag_group):
            if len(obj) > 0:
                obj.set_opacity(0.0)
                fade_ins.append(obj.animate.set_opacity(1.0))

        self.play(
            visibility.animate.set_value(1.0),
            *fade_ins,
            run_time=timing["build_time"],
            rate_func=smooth,
        )
        self.play(clock.animate.set_value(timing["vacuum_time"]), run_time=timing["vacuum_time"], rate_func=linear)

        if scene_cfg["show_labels"]:
            next_caption = Text("Different fields carry different wave patterns.", color=c["accent"], weight=BOLD, font="Arial").scale(0.38)
            next_caption.to_edge(DOWN, buff=0.40)
            self.add_fixed_in_frame_mobjects(next_caption)
            self.play(FadeOut(caption), FadeIn(next_caption), run_time=0.35)
            self.remove(caption)
            caption = next_caption

        wave_end = timing["vacuum_time"] + timing["wave_time"]
        self.play(clock.animate.set_value(wave_end), run_time=timing["wave_time"], rate_func=linear)

        if scene_cfg["show_labels"]:
            cutaway_caption = Text("Cutaway: waves run through the whole field stack.", color=c["accent"], weight=BOLD, font="Arial").scale(0.38)
            cutaway_caption.to_edge(DOWN, buff=0.40)
            self.add_fixed_in_frame_mobjects(cutaway_caption)
            self.play(FadeOut(caption), FadeIn(cutaway_caption), run_time=0.35)
            self.remove(caption)
            caption = cutaway_caption

        cutaway_reveal_end = wave_end + timing["cutaway_reveal_time"]
        self.play(
            clock.animate.set_value(cutaway_reveal_end),
            cutout.animate.set_value(1.0),
            run_time=timing["cutaway_reveal_time"],
            rate_func=smooth,
        )

        cutaway_end = cutaway_reveal_end + timing["cutaway_time"]
        self.play(clock.animate.set_value(cutaway_end), run_time=timing["cutaway_time"], rate_func=linear)

        if scene_cfg["show_labels"]:
            final_caption = Text("Particles are quanta of these field excitations.", color=c["accent"], weight=BOLD, font="Arial").scale(0.38)
            final_caption.to_edge(DOWN, buff=0.40)
            self.add_fixed_in_frame_mobjects(final_caption)
            self.play(FadeOut(caption), FadeIn(final_caption), run_time=0.35)
            self.remove(caption)

        self.play(clock.animate.set_value(cutaway_end + timing["hold_time"]), run_time=timing["hold_time"], rate_func=linear)

    def make_legend(self, c: dict) -> VGroup:
        panel = RoundedRectangle(
            width=4.85,
            height=2.36,
            corner_radius=0.10,
            color=c["panel_stroke"],
            stroke_width=1.3,
            fill_color=c["panel"],
            fill_opacity=0.66,
        )
        heading = Text("SM field content", color=c["text"], weight=BOLD, font="Arial").scale(0.26)
        rows = VGroup(
            MathTex(r"\text{quarks: } u,d,c,s,t,b\quad(\times 3)", color="#FFD166").scale(0.41),
            MathTex(r"\text{charged leptons: } e,\mu,\tau", color="#7CC7FF").scale(0.41),
            MathTex(r"\text{neutrinos: } \nu_e,\nu_\mu,\nu_\tau", color="#C4B5FD").scale(0.41),
            MathTex(r"\text{gauge: } \gamma,\ g,\ W^\pm,\ Z", color="#DCE8F8").scale(0.41),
            MathTex(r"\text{scalar: } H", color="#FF7AC8").scale(0.41),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.08)
        content = VGroup(heading, rows).arrange(DOWN, aligned_edge=LEFT, buff=0.13)
        content.move_to(panel).shift(0.08 * UP)
        legend = VGroup(panel, content)
        legend.to_corner(UR, buff=0.16)
        return legend

    def make_field_tags(self, p: dict, c: dict) -> VGroup:
        quark = Text("quark fields", color="#FFD166", weight=BOLD, font="Arial").scale(0.24)
        lepton = Text("lepton fields", color="#7CC7FF", weight=BOLD, font="Arial").scale(0.24)
        gauge = Text("gauge fields", color="#DCE8F8", weight=BOLD, font="Arial").scale(0.24)
        higgs = Text("Higgs field", color="#FF7AC8", weight=BOLD, font="Arial").scale(0.24)
        tags = VGroup(quark, lepton, gauge, higgs).arrange(DOWN, aligned_edge=RIGHT, buff=0.13)
        tags.to_edge(RIGHT, buff=0.48).shift(0.64 * DOWN)
        return tags
