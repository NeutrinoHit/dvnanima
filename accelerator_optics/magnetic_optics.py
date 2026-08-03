from __future__ import annotations

import configparser
import os
from pathlib import Path

import numpy as np
from manim import *

try:
    from .trajectory import SectorTrajectory
except ImportError:
    from trajectory import SectorTrajectory


BASE_DIR = Path(__file__).resolve().parent

PROFILE_PRESETS = {
    "shorts": {
        "frame_width": 9.0,
        "frame_height": 16.0,
        "center": (0.45, -2.70),
        "radius": 2.40,
        "entry_start_y": -7.10,
        "exit_end_x": 3.35,
        "straight_end_y": 3.55,
        "kicker_y": 7.35,
        "title_y": 6.55,
        "subtitle_y": 5.82,
        "status_y": 4.72,
        "legend_y": 4.05,
        "final_top_y": 3.62,
        "final_bottom_y": 2.82,
        "field_info_pos": (0.0, 4.02),
        "equation_pos": (0.0, 3.28),
        "vector_legend_pos": (0.0, 2.46),
        "title_scale": 0.82,
        "kicker_scale": 0.34,
        "subtitle_scale": 0.45,
        "status_scale": 0.34,
        "final_scale": 0.67,
    },
    "widescreen": {
        "frame_width": 16.0,
        "frame_height": 9.0,
        "center": (-1.15, -1.45),
        "radius": 2.00,
        "entry_start_y": -4.25,
        "exit_end_x": 4.90,
        "straight_end_y": 2.65,
        "kicker_y": 4.05,
        "title_y": 3.45,
        "subtitle_y": 2.90,
        "status_y": 2.10,
        "legend_y": 1.55,
        "final_top_y": 2.10,
        "final_bottom_y": 1.45,
        "field_info_pos": (4.65, 2.05),
        "equation_pos": (4.65, 1.20),
        "vector_legend_pos": (4.65, 0.30),
        "title_scale": 0.72,
        "kicker_scale": 0.28,
        "subtitle_scale": 0.38,
        "status_scale": 0.28,
        "final_scale": 0.55,
    },
}


def load_cfg(path: Path) -> dict:
    parser = configparser.ConfigParser(inline_comment_prefixes=(";",))
    parser.read(path)

    def get(section: str, key: str, cast=str, fallback=None):
        if fallback is None:
            return cast(parser[section][key])
        return cast(parser.get(section, key, fallback=str(fallback)))

    profile = os.environ.get(
        "DVN_PROFILE",
        get("manim", "profile", str, "shorts"),
    ).strip().lower()
    if profile not in PROFILE_PRESETS:
        profile = "shorts"

    timings = {
        "intro": get("timing", "intro", float, 0.9),
        "magnet_build": get("timing", "magnet_build", float, 1.2),
        "field_off_hold": get("timing", "field_off_hold", float, 0.45),
        "straight_flight": get("timing", "straight_flight", float, 2.2),
        "switch_on": get("timing", "switch_on", float, 0.75),
        "bend_flight": get("timing", "bend_flight", float, 5.2),
        "message": get("timing", "message", float, 1.0),
        "tail": get("timing", "tail", float, 1.7),
    }

    colors = {
        key: get("colors", key, str)
        for key in (
            "background",
            "grid",
            "text",
            "muted",
            "magnet",
            "magnet_edge",
            "field",
            "beam",
            "proton",
            "momentum",
            "force",
            "off",
        )
    }

    return {
        "profile": profile,
        "layout": PROFILE_PRESETS[profile],
        "frame_rate": int(
            os.environ.get(
                "DVN_FRAME_RATE",
                get("manim", "frame_rate", int, 60),
            )
        ),
        "font": get("scene", "font", str, "PT Sans"),
        "timing": timings,
        "colors": colors,
    }


CFG = load_cfg(BASE_DIR / "run.cfg")


def apply_render_geometry() -> None:
    layout = CFG["layout"]
    aspect = layout["frame_width"] / layout["frame_height"]
    long_side = max(int(config.pixel_width), int(config.pixel_height))
    if aspect >= 1.0:
        config.pixel_width = long_side
        config.pixel_height = max(2, int(round(long_side / aspect)))
    else:
        config.pixel_height = long_side
        config.pixel_width = max(2, int(round(long_side * aspect)))
    config.frame_width = layout["frame_width"]
    config.frame_height = layout["frame_height"]
    config.frame_rate = CFG["frame_rate"]
    config.background_color = CFG["colors"]["background"]


apply_render_geometry()


def make_text(
    content: str,
    *,
    color: str,
    scale: float,
    weight: str = "NORMAL",
) -> Text:
    return Text(
        content,
        font=CFG["font"],
        color=color,
        weight=weight,
    ).scale(scale)


def make_status(content: str, color: str, y: float, scale: float) -> VGroup:
    label = make_text(content, color=color, scale=scale, weight="BOLD")
    plate = RoundedRectangle(
        corner_radius=0.18,
        width=label.width + 0.62,
        height=label.height + 0.34,
        stroke_color=color,
        stroke_width=1.5,
        fill_color=color,
        fill_opacity=0.10,
    )
    return VGroup(plate, label).move_to([0.0, y, 0.0])


def make_proton(position: np.ndarray, colors: dict) -> VGroup:
    glow_outer = Circle(
        radius=0.28,
        stroke_width=0,
        fill_color=colors["proton"],
        fill_opacity=0.10,
    )
    glow_inner = Circle(
        radius=0.20,
        stroke_width=0,
        fill_color=colors["proton"],
        fill_opacity=0.20,
    )
    core = Circle(
        radius=0.12,
        stroke_color=colors["text"],
        stroke_width=1.4,
        fill_color=colors["proton"],
        fill_opacity=1.0,
    )
    label = make_text("p⁺", color=colors["background"], scale=0.18, weight="BOLD")
    return VGroup(glow_outer, glow_inner, core, label).move_to(position)


def make_field_symbol(position: np.ndarray, color: str) -> VGroup:
    ring = Circle(radius=0.10, stroke_color=color, stroke_width=1.4)
    dot = Dot(radius=0.026, color=color)
    return VGroup(ring, dot).move_to(position)


def make_field_info(position: tuple[float, float], colors: dict) -> VGroup:
    symbol = make_field_symbol(ORIGIN, colors["field"]).scale(1.35)
    direction = make_text(
        "поле к нам",
        color=colors["text"],
        scale=0.30 if CFG["profile"] == "shorts" else 0.25,
    )
    magnitude = MathTex(
        r"|\mathbf B|=\mathrm{const}",
        color=colors["field"],
    ).scale(0.56 if CFG["profile"] == "shorts" else 0.48)
    return VGroup(symbol, direction, magnitude).arrange(RIGHT, buff=0.20).move_to(
        [position[0], position[1], 0.0]
    )


def make_lorentz_equation(position: tuple[float, float], colors: dict) -> VGroup:
    label = make_text(
        "СИЛА ЛОРЕНЦА",
        color=colors["muted"],
        scale=0.24 if CFG["profile"] == "shorts" else 0.20,
        weight="BOLD",
    )
    equation = MathTex(
        r"\mathbf F_{\!L}",
        "=q",
        r"\mathbf v",
        r"\times",
        r"\mathbf B",
        color=colors["text"],
    ).scale(0.72 if CFG["profile"] == "shorts" else 0.62)
    equation[0].set_color(colors["force"])
    equation[2].set_color(colors["momentum"])
    equation[4].set_color(colors["field"])
    content = VGroup(label, equation).arrange(DOWN, buff=0.10)
    plate = RoundedRectangle(
        corner_radius=0.16,
        width=content.width + 0.50,
        height=content.height + 0.30,
        stroke_color=colors["muted"],
        stroke_width=1.1,
        fill_color=colors["background"],
        fill_opacity=0.72,
    ).set_opacity(0.72)
    return VGroup(plate, content).move_to([position[0], position[1], 0.0])


def make_vector_legend(position: tuple[float, float], colors: dict) -> VGroup:
    def item(color: str, label: str) -> VGroup:
        arrow = Arrow(
            LEFT * 0.42,
            RIGHT * 0.42,
            buff=0.0,
            color=color,
            stroke_width=4.0,
            max_tip_length_to_length_ratio=0.28,
        )
        text = make_text(
            label,
            color=colors["muted"],
            scale=0.27 if CFG["profile"] == "shorts" else 0.23,
        )
        return VGroup(arrow, text).arrange(RIGHT, buff=0.16)

    legend = VGroup(
        item(colors["momentum"], "скорость и импульс"),
        item(colors["force"], "сила Лоренца"),
    ).arrange(RIGHT, buff=0.48)
    return legend.move_to([position[0], position[1], 0.0])


def make_path(points: list[np.ndarray], color: str, width: float, opacity: float) -> VMobject:
    path = VMobject()
    path.set_points_as_corners(points)
    path.set_stroke(color=color, width=width, opacity=opacity)
    return path


class DipoleMagnetScene(Scene):
    def construct(self) -> None:
        layout = CFG["layout"]
        colors = CFG["colors"]
        timing = CFG["timing"]
        self.camera.background_color = colors["background"]

        center_x, center_y = layout["center"]
        radius = layout["radius"]
        center = np.array([center_x, center_y, 0.0])
        trajectory = SectorTrajectory(
            center_x=center_x,
            center_y=center_y,
            radius=radius,
            entry_start_y=layout["entry_start_y"],
            exit_end_x=layout["exit_end_x"],
        )

        background_grid = NumberPlane(
            x_range=[-9, 9, 1],
            y_range=[-9, 9, 1],
            background_line_style={
                "stroke_color": colors["grid"],
                "stroke_width": 0.65,
                "stroke_opacity": 0.18,
            },
            axis_config={"stroke_opacity": 0.0},
        )
        background_grid.set_z_index(-20)
        self.add(background_grid)

        kicker = make_text(
            "МАГНИТНАЯ ОПТИКА УСКОРИТЕЛЯ",
            color=colors["muted"],
            scale=layout["kicker_scale"],
            weight="BOLD",
        ).move_to([0.0, layout["kicker_y"], 0.0])
        title = make_text(
            "Дипольный магнит",
            color=colors["text"],
            scale=layout["title_scale"],
            weight="BOLD",
        ).move_to([0.0, layout["title_y"], 0.0])
        subtitle = make_text(
            "поворот без разгона",
            color=colors["field"],
            scale=layout["subtitle_scale"],
            weight="BOLD",
        ).move_to([0.0, layout["subtitle_y"], 0.0])

        self.play(
            FadeIn(kicker, shift=0.15 * DOWN),
            FadeIn(title, shift=0.18 * UP),
            FadeIn(subtitle),
            run_time=timing["intro"],
        )

        half_gap = 0.78 if CFG["profile"] == "shorts" else 0.68
        field_region = AnnularSector(
            inner_radius=radius - half_gap,
            outer_radius=radius + half_gap,
            angle=PI / 2,
            start_angle=PI / 2,
            arc_center=center,
            fill_color=colors["field"],
            fill_opacity=0.08,
            stroke_color=colors["field"],
            stroke_width=1.2,
            stroke_opacity=0.24,
        )
        outer_pole = Arc(
            radius=radius + half_gap + 0.13,
            start_angle=PI / 2,
            angle=PI / 2,
            arc_center=center,
            stroke_color=colors["magnet_edge"],
            stroke_width=18,
        )
        inner_pole = Arc(
            radius=radius - half_gap - 0.13,
            start_angle=PI / 2,
            angle=PI / 2,
            arc_center=center,
            stroke_color=colors["magnet_edge"],
            stroke_width=18,
        )
        entrance_edge = Line(
            center + np.array([-(radius - half_gap - 0.13), 0.0, 0.0]),
            center + np.array([-(radius + half_gap + 0.13), 0.0, 0.0]),
            color=colors["magnet_edge"],
            stroke_width=18,
        )
        exit_edge = Line(
            center + np.array([0.0, radius - half_gap - 0.13, 0.0]),
            center + np.array([0.0, radius + half_gap + 0.13, 0.0]),
            color=colors["magnet_edge"],
            stroke_width=18,
        )
        pole_shadows = VGroup(
            outer_pole.copy().set_stroke(colors["magnet"], width=27, opacity=0.75),
            inner_pole.copy().set_stroke(colors["magnet"], width=27, opacity=0.75),
            entrance_edge.copy().set_stroke(colors["magnet"], width=27, opacity=0.75),
            exit_edge.copy().set_stroke(colors["magnet"], width=27, opacity=0.75),
        )
        pole_edges = VGroup(outer_pole, inner_pole, entrance_edge, exit_edge)

        field_symbols = VGroup()
        radial_offsets = (-0.43, 0.0, 0.43)
        angles = (104, 123, 142, 161)
        for radial_offset in radial_offsets:
            for angle_degrees in angles:
                angle = angle_degrees * DEGREES
                position = center + (radius + radial_offset) * np.array(
                    [np.cos(angle), np.sin(angle), 0.0]
                )
                field_symbols.add(make_field_symbol(position, colors["field"]))
        field_symbols.set_opacity(0.16)

        magnet_group = VGroup(field_region, pole_shadows, pole_edges, field_symbols)
        magnet_group.set_z_index(1)
        self.play(
            FadeIn(field_region),
            FadeIn(pole_shadows),
            Create(pole_edges),
            FadeIn(field_symbols),
            run_time=timing["magnet_build"],
        )

        off_status = make_status(
            "ПОЛЕ ВЫКЛЮЧЕНО",
            colors["off"],
            layout["status_y"],
            layout["status_scale"],
        )
        self.play(FadeIn(off_status), run_time=0.35)
        self.wait(timing["field_off_hold"])

        straight_end = np.array(
            [trajectory.entrance[0], layout["straight_end_y"], 0.0]
        )
        straight_path = Line(trajectory.entry_start, straight_end)
        straight_particle = make_proton(trajectory.entry_start, colors).set_z_index(8)
        self.add(straight_particle)
        self.play(
            MoveAlongPath(straight_particle, straight_path),
            run_time=timing["straight_flight"],
            rate_func=linear,
        )
        ghost_path = DashedLine(
            trajectory.entry_start,
            straight_end,
            dash_length=0.13,
            dashed_ratio=0.52,
            color=colors["muted"],
            stroke_width=2.0,
        ).set_opacity(0.34)
        self.play(
            FadeOut(straight_particle),
            FadeIn(ghost_path),
            run_time=0.35,
        )

        on_status = make_status(
            "ПОЛЕ ВКЛЮЧЕНО",
            colors["field"],
            layout["status_y"],
            layout["status_scale"],
        )
        self.play(
            Transform(off_status, on_status),
            field_symbols.animate.set_opacity(0.96),
            field_region.animate.set_fill(colors["field"], opacity=0.24),
            run_time=timing["switch_on"],
            rate_func=smooth,
        )
        self.play(
            field_symbols.animate.scale(1.08),
            run_time=0.25,
            rate_func=there_and_back,
        )

        reference_path = make_path(
            trajectory.sample_points(),
            colors["field"],
            width=2.0,
            opacity=0.24,
        ).set_z_index(2)
        self.play(Create(reference_path), run_time=0.65)

        progress = ValueTracker(0.0)
        proton = make_proton(trajectory.entry_start, colors).set_z_index(9)
        proton.add_updater(
            lambda mob: mob.move_to(trajectory.point_and_tangent(progress.get_value())[0])
        )
        trace = TracedPath(
            proton.get_center,
            stroke_color=colors["beam"],
            stroke_width=5.0,
            dissipating_time=None,
        ).set_z_index(4)

        momentum_arrow = always_redraw(
            lambda: Arrow(
                start=trajectory.point_and_tangent(progress.get_value())[0]
                + 0.14 * trajectory.point_and_tangent(progress.get_value())[1],
                end=trajectory.point_and_tangent(progress.get_value())[0]
                + 1.02 * trajectory.point_and_tangent(progress.get_value())[1],
                buff=0.0,
                color=colors["momentum"],
                stroke_width=5.0,
                max_tip_length_to_length_ratio=0.24,
            ).set_z_index(10)
        )
        lorentz_arrow = always_redraw(
            lambda: self.make_lorentz_arrow(trajectory, progress, colors)
        )
        field_info = make_field_info(layout["field_info_pos"], colors)
        equation_card = make_lorentz_equation(layout["equation_pos"], colors)
        vector_legend = make_vector_legend(layout["vector_legend_pos"], colors)

        self.add(trace, proton, momentum_arrow, lorentz_arrow)
        self.play(
            FadeIn(field_info),
            FadeIn(equation_card, shift=0.10 * UP),
            FadeIn(vector_legend),
            run_time=0.70,
        )
        self.play(
            progress.animate.set_value(1.0),
            run_time=timing["bend_flight"],
            rate_func=linear,
        )
        proton.clear_updaters()

        self.play(
            FadeOut(off_status),
            FadeOut(field_info),
            FadeOut(equation_card),
            FadeOut(vector_legend),
            FadeOut(ghost_path),
            run_time=0.45,
        )

        turn_yes = make_text(
            "ПОВОРОТ — ДА",
            color=colors["field"],
            scale=layout["final_scale"],
            weight="BOLD",
        ).move_to([0.0, layout["final_top_y"], 0.0])
        acceleration_no = make_text(
            "РАЗГОН — НЕТ",
            color=colors["momentum"],
            scale=layout["final_scale"],
            weight="BOLD",
        ).move_to([0.0, layout["final_bottom_y"], 0.0])
        divider = Line(
            LEFT * 1.75,
            RIGHT * 1.75,
            color=colors["muted"],
            stroke_width=1.2,
        ).set_opacity(0.42)
        divider.move_to(
            [0.0, 0.5 * (layout["final_top_y"] + layout["final_bottom_y"]), 0.0]
        )
        self.play(
            FadeIn(turn_yes, shift=0.12 * UP),
            Create(divider),
            FadeIn(acceleration_no, shift=0.12 * DOWN),
            run_time=timing["message"],
        )
        self.wait(timing["tail"])

    @staticmethod
    def make_lorentz_arrow(
        trajectory: SectorTrajectory,
        progress: ValueTracker,
        colors: dict,
    ) -> Mobject:
        value = progress.get_value()
        if not trajectory.is_inside_field(value):
            return VGroup()

        position, _ = trajectory.point_and_tangent(value)
        inward_normal = trajectory.inward_normal(value)
        return Arrow(
            start=position + 0.14 * inward_normal,
            end=position + 0.92 * inward_normal,
            buff=0.0,
            color=colors["force"],
            stroke_width=5.0,
            max_tip_length_to_length_ratio=0.26,
        ).set_z_index(10)
