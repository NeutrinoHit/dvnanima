from __future__ import annotations

from pathlib import Path
import configparser
import numpy as np
from manim import *


def load_cfg(path: Path) -> dict:
    cfg = configparser.ConfigParser(inline_comment_prefixes=(";",))
    cfg.read(path)

    def get(section, key, cast=str, fallback=None):
        if fallback is None:
            return cast(cfg[section][key])
        return cast(cfg.get(section, key, fallback=str(fallback)))

    manim_params = {
        "pixel_width": get("manim", "pixel_width", int),
        "pixel_height": get("manim", "pixel_height", int),
        "frame_width": get("manim", "frame_width", float),
        "frame_height": get("manim", "frame_height", float),
        "frame_rate": get("manim", "frame_rate", int, 60),
        "background_color": get("manim", "background_color", str, "#DEDEDE"),
    }

    scene = {
        "axis_origin_x": get("scene", "axis_origin_x", float, -6.2),
        "axis_origin_y": get("scene", "axis_origin_y", float, -3.6),
        "axis_x_len": get("scene", "axis_x_len", float, 12.8),
        "axis_y_len": get("scene", "axis_y_len", float, 7.2),
        "axis_stroke": get("scene", "axis_stroke", float, 12.0),
        "axis_label_scale": get("scene", "axis_label_scale", float, 0.75),
        "title_scale": get("scene", "title_scale", float, 0.82),
        "subtitle_scale": get("scene", "subtitle_scale", float, 0.50),
        "card_top_buff": get("scene", "card_top_buff", float, 0.34),
        "card_axis_clearance": get("scene", "card_axis_clearance", float, 0.25),
        "left_x": get("scene", "left_x", float, -2.8),
        "right_x": get("scene", "right_x", float, 4.1),
        "vertex_x": get("scene", "vertex_x", float, 0.4),
        "vertex_y_low": get("scene", "vertex_y_low", float, -0.7),
        "vertex_y_high": get("scene", "vertex_y_high", float, 1.0),
        "fermion_dx": get("scene", "fermion_dx", float, 1.9),
        "fermion_dy": get("scene", "fermion_dy", float, 2.8),
        "blue_in_y": get("scene", "blue_in_y", float, -2.4),
        "red_in_y": get("scene", "red_in_y", float, 2.5),
        "blue_out_y": get("scene", "blue_out_y", float, 2.5),
        "red_out_y": get("scene", "red_out_y", float, -2.5),
        "fermion_stroke": get("scene", "fermion_stroke", float, 10.0),
        "fermion_tip_len": get("scene", "fermion_tip_len", float, 0.34),
        "mediator_stroke": get("scene", "mediator_stroke", float, 7.0),
        "mediator_amplitude": get("scene", "mediator_amplitude", float, 0.16),
        "mediator_wavelength": get("scene", "mediator_wavelength", float, 0.55),
        "mediator_gap": get("scene", "mediator_gap", float, 0.13),
        "gluon_pitch": get("scene", "gluon_pitch", float, 0.42),
        "gluon_axial_mod": get("scene", "gluon_axial_mod", float, 0.07),
        "axes_time": get("scene", "axes_time", float, 0.9),
        "incoming_time": get("scene", "incoming_time", float, 1.3),
        "exchange_time": get("scene", "exchange_time", float, 1.0),
        "outgoing_time": get("scene", "outgoing_time", float, 1.3),
        "tail_wait": get("scene", "tail_wait", float, 1.0),
        "particle_label_scale": get("scene", "particle_label_scale", float, 0.62),
        "particle_label_offset": get("scene", "particle_label_offset", float, 0.32),
        "mediator_label_scale": get("scene", "mediator_label_scale", float, 0.68),
        "vertex_radius": get("scene", "vertex_radius", float, 0.06),
        "gluon_line_gap": get("scene", "gluon_line_gap", float, 0.09),
    }

    colors = {
        "axis_color": get("colors", "axis_color", str, "#000000"),
        "text_color": get("colors", "text_color", str, "#111111"),
        "blue_fermion": get("colors", "blue_fermion", str, "#16A6FF"),
        "red_fermion": get("colors", "red_fermion", str, "#FF2A1B"),
        "qed_fermion": get("colors", "qed_fermion", str, "#FFFFFF"),
        "qed_mediator": get("colors", "qed_mediator", str, "#FFFFFF"),
        "weak_boson": get("colors", "weak_boson", str, "#8CF8A8"),
        "photon_primary": get("colors", "photon_primary", str, "#FFB000"),
        "photon_secondary": get("colors", "photon_secondary", str, "#FF6A00"),
        "gluon_color": get("colors", "gluon_color", str, "#3AA655"),
        "anti_blue": get("colors", "anti_blue", str, "#FFD84D"),
    }

    return {"manim": manim_params, "scene": scene, "colors": colors}


BASE_DIR = Path(__file__).resolve().parent
CFG = load_cfg(BASE_DIR / "run.cfg")

config.pixel_width = CFG["manim"]["pixel_width"]
config.pixel_height = CFG["manim"]["pixel_height"]
config.frame_width = CFG["manim"]["frame_width"]
config.frame_height = CFG["manim"]["frame_height"]
config.frame_rate = CFG["manim"]["frame_rate"]


def wave_curve(
    start: np.ndarray,
    end: np.ndarray,
    amplitude: float,
    wavelength: float,
    phase: float = 0.0,
    stroke_width: float = 6.0,
    color: str = WHITE,
) -> VMobject:
    vec = end - start
    length = float(np.linalg.norm(vec))
    direction = vec / length
    normal = np.array([-direction[1], direction[0], 0.0])
    samples = max(120, int(72 * length))
    points = []
    for t in np.linspace(0.0, 1.0, samples):
        envelope = np.sin(PI * t)
        wobble = amplitude * envelope * np.sin(TAU * length * t / wavelength + phase)
        points.append(start + direction * (length * t) + normal * wobble)
    curve = VMobject()
    curve.set_points_smoothly(points)
    curve.set_stroke(color=color, width=stroke_width)
    return curve


def gluon_curve(
    start: np.ndarray,
    end: np.ndarray,
    amplitude: float,
    pitch: float,
    axial_mod: float,
    phase: float = 0.0,
    stroke_width: float = 6.0,
    color: str = WHITE,
) -> VMobject:
    vec = end - start
    length = float(np.linalg.norm(vec))
    direction = vec / length
    normal = np.array([-direction[1], direction[0], 0.0])
    loops = max(7, int(round(length / pitch)))
    samples = max(260, 42 * loops)
    points = []
    for t in np.linspace(0.0, 1.0, samples):
        phase_t = TAU * loops * t + phase
        wobble = amplitude * np.sin(phase_t)
        axial = length * t + axial_mod * (np.cos(phase_t) - np.cos(phase))
        points.append(start + direction * axial + normal * wobble)
    curve = VMobject()
    curve.set_points_smoothly(points)
    curve.set_stroke(color=color, width=stroke_width)
    return curve


def line_normal(start: np.ndarray, end: np.ndarray) -> np.ndarray:
    vec = end - start
    length = float(np.linalg.norm(vec))
    direction = vec / length
    return np.array([-direction[1], direction[0], 0.0])


class FeynmanScatteringBase(Scene):
    title_tex = r"e^- e^- \rightarrow e^- e^-"
    subtitle_text = "QED scattering with virtual photon exchange"
    in_label_blue = r"e^-"
    out_label_blue = r"e^-"
    in_label_red = r"e^-"
    out_label_red = r"e^-"
    mediator_label = r"\gamma"

    def fermion_colors(self, c: dict) -> tuple[str, str]:
        return c["blue_fermion"], c["red_fermion"]

    def build_axes(self, p: dict, c: dict) -> tuple[Arrow, Arrow, Dot, Text, Text]:
        origin = np.array([p["axis_origin_x"], p["axis_origin_y"], 0.0])
        x_end = origin + p["axis_x_len"] * RIGHT
        y_end = origin + p["axis_y_len"] * UP
        axis_style = dict(
            buff=0.0,
            color=c["axis_color"],
            stroke_width=p["axis_stroke"],
            tip_length=0.32,
            max_tip_length_to_length_ratio=0.13,
            max_stroke_width_to_length_ratio=1000.0,
        )
        x_axis = Arrow(origin, x_end, **axis_style)
        y_axis = Arrow(origin, y_end, **axis_style)
        # Fill the corner join so the axes visually meet without a gap.
        origin_joint = Dot(point=origin, radius=0.055, color=c["axis_color"])
        x_label = Text("Time", color=c["text_color"]).scale(p["axis_label_scale"])
        y_label = Text("Distance", color=c["text_color"]).scale(p["axis_label_scale"])
        # Keep labels inside frame and above player controls.
        x_label.next_to(x_axis, UP, buff=0.88)
        y_label.rotate(PI / 2).next_to(y_axis, RIGHT, buff=0.34)
        return x_axis, y_axis, origin_joint, x_label, y_label

    def fermion_arrow(self, start: np.ndarray, end: np.ndarray, color: str, p: dict) -> Arrow:
        return Arrow(
            start,
            end,
            buff=0.0,
            color=color,
            stroke_width=p["fermion_stroke"],
            tip_length=p["fermion_tip_len"],
            max_tip_length_to_length_ratio=0.22,
            max_stroke_width_to_length_ratio=1000.0,
        )

    def mediator_bundle(
        self,
        low_vertex: np.ndarray,
        high_vertex: np.ndarray,
        p: dict,
        c: dict,
    ) -> tuple[VGroup, VMobject]:
        primary = wave_curve(
            low_vertex,
            high_vertex,
            amplitude=p["mediator_amplitude"],
            wavelength=p["mediator_wavelength"],
            phase=0.0,
            stroke_width=p["mediator_stroke"],
            color=c["qed_mediator"],
        )
        return VGroup(primary), primary

    def process_card(self, p: dict, c: dict, y_axis: Arrow) -> VGroup:
        title = MathTex(self.title_tex, color=c["text_color"]).scale(p["title_scale"])
        subtitle = Text(self.subtitle_text, color=c["text_color"]).scale(p["subtitle_scale"])
        card = VGroup(title, subtitle).arrange(DOWN, aligned_edge=LEFT, buff=0.16)
        card.to_edge(UP, buff=p["card_top_buff"])
        min_left = y_axis.get_x() + p["card_axis_clearance"]
        if card.get_left()[0] < min_left:
            card.shift((min_left - card.get_left()[0]) * RIGHT)
        return card

    def construct(self):
        p = CFG["scene"]
        c = CFG["colors"]
        self.camera.background_color = CFG["manim"]["background_color"]

        x_axis, y_axis, origin_joint, x_label, y_label = self.build_axes(p, c)
        card = self.process_card(p, c, y_axis)
        fermion_blue_col, fermion_red_col = self.fermion_colors(c)
        self.play(
            GrowArrow(x_axis),
            GrowArrow(y_axis),
            FadeIn(origin_joint),
            FadeIn(x_label, shift=0.12 * UP),
            FadeIn(y_label, shift=0.12 * RIGHT),
            FadeIn(card, shift=0.18 * RIGHT),
            run_time=p["axes_time"],
        )

        low_vertex = np.array([p["vertex_x"], p["vertex_y_low"], 0.0])
        high_vertex = np.array([p["vertex_x"], p["vertex_y_high"], 0.0])
        dx = p["fermion_dx"]
        dy = p["fermion_dy"]
        blue_in_start = low_vertex + np.array([-dx, -dy, 0.0])
        red_in_start = high_vertex + np.array([-dx, dy, 0.0])
        blue_out_end = high_vertex + np.array([dx, dy, 0.0])
        red_out_end = low_vertex + np.array([dx, -dy, 0.0])

        blue_in = self.fermion_arrow(blue_in_start, low_vertex, fermion_blue_col, p)
        red_in = self.fermion_arrow(red_in_start, high_vertex, fermion_red_col, p)
        blue_out = self.fermion_arrow(high_vertex, blue_out_end, fermion_blue_col, p)
        red_out = self.fermion_arrow(low_vertex, red_out_end, fermion_red_col, p)

        n_blue_in = line_normal(blue_in_start, low_vertex)
        n_red_in = line_normal(red_in_start, high_vertex)
        n_blue_out = line_normal(high_vertex, blue_out_end)
        n_red_out = line_normal(low_vertex, red_out_end)

        blue_in_lbl = (
            MathTex(self.in_label_blue, color=c["text_color"])
            .scale(p["particle_label_scale"])
            .move_to(blue_in.point_from_proportion(0.24) - p["particle_label_offset"] * n_blue_in)
        )
        red_in_lbl = (
            MathTex(self.in_label_red, color=c["text_color"])
            .scale(p["particle_label_scale"])
            .move_to(red_in.point_from_proportion(0.24) + p["particle_label_offset"] * n_red_in)
        )
        blue_out_lbl = (
            MathTex(self.out_label_blue, color=c["text_color"])
            .scale(p["particle_label_scale"])
            .move_to(blue_out.point_from_proportion(0.75) + p["particle_label_offset"] * n_blue_out)
        )
        red_out_lbl = (
            MathTex(self.out_label_red, color=c["text_color"])
            .scale(p["particle_label_scale"])
            .move_to(red_out.point_from_proportion(0.75) - p["particle_label_offset"] * n_red_out)
        )

        # Keep particle symbols on consistent rows despite different glyph metrics.
        top_row_y = 0.5 * (red_in_lbl.get_y() + blue_out_lbl.get_y())
        bottom_row_y = 0.5 * (blue_in_lbl.get_y() + red_out_lbl.get_y())
        symmetry_x = low_vertex[0]
        red_in_lbl.set_y(top_row_y)
        blue_out_lbl.set_y(top_row_y)
        blue_in_lbl.set_y(bottom_row_y)
        red_out_lbl.set_y(bottom_row_y)
        red_in_lbl.set_x(2.0 * symmetry_x - blue_out_lbl.get_x())
        blue_in_lbl.set_x(2.0 * symmetry_x - red_out_lbl.get_x())

        vertex_upper = Circle(radius=p["vertex_radius"]).move_to(high_vertex)
        vertex_upper.set_fill(BLACK, opacity=1.0)
        vertex_upper.set_stroke(c["text_color"], width=2.6)
        vertex_upper.set_z_index(10)

        vertex_lower = Circle(radius=p["vertex_radius"]).move_to(low_vertex)
        vertex_lower.set_fill(BLACK, opacity=1.0)
        vertex_lower.set_stroke(c["text_color"], width=2.6)
        vertex_lower.set_z_index(10)

        self.play(
            GrowArrow(blue_in),
            GrowArrow(red_in),
            FadeIn(blue_in_lbl),
            FadeIn(red_in_lbl),
            run_time=p["incoming_time"],
            rate_func=linear,
        )
        self.play(
            FadeIn(vertex_upper),
            FadeIn(vertex_lower),
            run_time=0.25,
        )

        mediator_group, _ = self.mediator_bundle(
            low_vertex, high_vertex, p, c
        )
        mediator_lbl = (
            MathTex(self.mediator_label, color=c["text_color"])
            .scale(p["mediator_label_scale"])
            .next_to(mediator_group, RIGHT, buff=0.26)
        )

        self.play(
            LaggedStart(*[Create(m) for m in mediator_group], lag_ratio=0.16),
            FadeIn(mediator_lbl, shift=0.10 * RIGHT),
            run_time=p["exchange_time"],
        )

        self.play(
            GrowArrow(blue_out),
            GrowArrow(red_out),
            FadeIn(blue_out_lbl),
            FadeIn(red_out_lbl),
            run_time=p["outgoing_time"],
            rate_func=linear,
        )
        self.wait(p["tail_wait"])


class ElectronElectronScattering(FeynmanScatteringBase):
    title_tex = r"e^- e^- \rightarrow e^- e^-"
    subtitle_text = "QED scattering with virtual photon exchange"
    in_label_blue = r"e^-"
    out_label_blue = r"e^-"
    in_label_red = r"e^-"
    out_label_red = r"e^-"
    mediator_label = r"\gamma"

    def fermion_colors(self, c: dict) -> tuple[str, str]:
        return c["qed_fermion"], c["qed_fermion"]


class QuarkQuarkScattering(FeynmanScatteringBase):
    title_tex = r"q\,q \rightarrow q\,q"
    subtitle_text = "QCD scattering with virtual gluon exchange"
    in_label_blue = r"q"
    out_label_blue = r"q"
    in_label_red = r"q"
    out_label_red = r"q"
    mediator_label = r"g"

    def mediator_bundle(
        self,
        low_vertex: np.ndarray,
        high_vertex: np.ndarray,
        p: dict,
        c: dict,
    ) -> tuple[VGroup, VMobject]:
        n = line_normal(high_vertex, low_vertex)
        shift = 0.5 * p["gluon_line_gap"] * n
        red_mode = gluon_curve(
            high_vertex + shift,
            low_vertex + shift,
            amplitude=p["mediator_amplitude"] * 0.95,
            pitch=p["gluon_pitch"],
            axial_mod=p["gluon_axial_mod"],
            phase=0.0,
            stroke_width=p["mediator_stroke"],
            color=c["red_fermion"],
        )
        anti_blue_mode = gluon_curve(
            high_vertex - shift,
            low_vertex - shift,
            amplitude=p["mediator_amplitude"] * 0.95,
            pitch=p["gluon_pitch"],
            axial_mod=p["gluon_axial_mod"],
            phase=0.0,
            stroke_width=p["mediator_stroke"],
            color=c["anti_blue"],
        )
        return VGroup(red_mode, anti_blue_mode), red_mode


class NuMuElectronToNuEMuonScattering(FeynmanScatteringBase):
    title_tex = r"\nu_{\mu}\,e^- \rightarrow \nu_e\,\mu^-"
    subtitle_text = "Charged-current weak scattering with W exchange"
    in_label_blue = r"e^-"
    out_label_blue = r"\mu^-"
    in_label_red = r"\nu_{\mu}"
    out_label_red = r"\nu_e"
    mediator_label = r"W^+"

    def fermion_colors(self, c: dict) -> tuple[str, str]:
        return c["qed_fermion"], c["qed_fermion"]

    def mediator_bundle(
        self,
        low_vertex: np.ndarray,
        high_vertex: np.ndarray,
        p: dict,
        c: dict,
    ) -> tuple[VGroup, VMobject]:
        weak_line = wave_curve(
            high_vertex,
            low_vertex,
            amplitude=p["mediator_amplitude"] * 1.08,
            wavelength=p["mediator_wavelength"] * 0.92,
            phase=0.0,
            stroke_width=p["mediator_stroke"] * 1.02,
            color=c["weak_boson"],
        )
        return VGroup(weak_line), weak_line


class NuMuMuonScatteringViaZ(FeynmanScatteringBase):
    title_tex = r"\nu_{\mu}\,e^- \rightarrow \nu_{\mu}\,e^-"
    subtitle_text = "Neutral-current weak scattering with Z exchange"
    in_label_blue = r"e^-"
    out_label_blue = r"\nu_{\mu}"
    in_label_red = r"\nu_{\mu}"
    out_label_red = r"e^-"
    mediator_label = r"Z^0"

    def fermion_colors(self, c: dict) -> tuple[str, str]:
        return c["qed_fermion"], c["qed_fermion"]

    def mediator_bundle(
        self,
        low_vertex: np.ndarray,
        high_vertex: np.ndarray,
        p: dict,
        c: dict,
    ) -> tuple[VGroup, VMobject]:
        weak_line = wave_curve(
            low_vertex,
            high_vertex,
            amplitude=p["mediator_amplitude"] * 1.08,
            wavelength=p["mediator_wavelength"] * 0.92,
            phase=0.0,
            stroke_width=p["mediator_stroke"] * 1.02,
            color=c["weak_boson"],
        )
        return VGroup(weak_line), weak_line
