from __future__ import annotations

import configparser
from pathlib import Path

import numpy as np
from manim import *


ROOT = Path(__file__).resolve().parent


def load_cfg() -> dict:
    parser = configparser.ConfigParser()
    parser.read(ROOT / "run.cfg")

    def get(section: str, key: str, cast, fallback):
        return cast(parser.get(section, key, fallback=str(fallback)))

    return {
        "manim": {
            "pixel_width": get("manim", "pixel_width", int, 1600),
            "pixel_height": get("manim", "pixel_height", int, 900),
            "frame_width": get("manim", "frame_width", float, 16.0),
            "frame_height": get("manim", "frame_height", float, 9.0),
            "frame_rate": get("manim", "frame_rate", int, 30),
            "background_color": get(
                "manim", "background_color", str, "#05080d"
            ),
        },
        "scene": {
            "run_time": get("scene", "run_time", float, 10.0),
            "turns": get("scene", "turns", int, 8),
            "track_jump": get("scene", "track_jump", float, 0.72),
            "collision_time": get("scene", "collision_time", float, 0.57),
        },
        "colors": dict(parser["colors"]),
    }


CFG = load_cfg()
config.pixel_width = CFG["manim"]["pixel_width"]
config.pixel_height = CFG["manim"]["pixel_height"]
config.frame_width = CFG["manim"]["frame_width"]
config.frame_height = CFG["manim"]["frame_height"]
config.frame_rate = CFG["manim"]["frame_rate"]


class Project8CRES(Scene):
    """From a trapped electron to a CRES frequency track."""

    def construct(self) -> None:
        scene = CFG["scene"]
        colors = CFG["colors"]
        self.camera.background_color = CFG["manim"]["background_color"]

        cyan = colors["cyan"]
        blue = colors["blue"]
        orange = colors["orange"]
        teal = colors["teal"]
        muted = colors["muted"]
        panel_color = colors["panel"]
        grid_color = colors["grid"]

        heading = Text(
            "Один электрон  →  один частотный трек",
            font="Arial",
            font_size=38,
            weight=BOLD,
            color=WHITE,
        ).to_edge(UP, buff=0.26)

        relation = MathTex(
            r"K\downarrow",
            r"\;\Longrightarrow\;",
            r"\gamma\downarrow",
            r"\;\Longrightarrow\;",
            r"f_c\uparrow",
            font_size=38,
        )
        relation[0].set_color(orange)
        relation[2].set_color(WHITE)
        relation[4].set_color(cyan)
        relation.next_to(heading, DOWN, buff=0.16)

        left_panel = RoundedRectangle(
            width=7.45,
            height=5.75,
            corner_radius=0.18,
            stroke_color=grid_color,
            stroke_width=2,
            fill_color=panel_color,
            fill_opacity=0.92,
        ).move_to(np.array([-3.95, -0.55, 0]))
        right_panel = RoundedRectangle(
            width=7.15,
            height=5.75,
            corner_radius=0.18,
            stroke_color=grid_color,
            stroke_width=2,
            fill_color=panel_color,
            fill_opacity=0.92,
        ).move_to(np.array([4.15, -0.55, 0]))

        trap_label = Text(
            "магнитная ловушка",
            font="Arial",
            font_size=26,
            color=muted,
        ).move_to(np.array([-3.95, 1.95, 0]))
        plot_label = Text(
            "спектрограмма",
            font="Arial",
            font_size=26,
            color=muted,
        ).move_to(np.array([4.15, 1.95, 0]))

        trap = RoundedRectangle(
            width=6.45,
            height=3.35,
            corner_radius=1.2,
            stroke_color=blue,
            stroke_width=2.2,
            fill_color="#07101a",
            fill_opacity=0.82,
        ).move_to(np.array([-3.95, -0.45, 0]))

        field_lines = VGroup()
        for y in np.linspace(-1.55, 0.65, 5):
            line = Arrow(
                start=np.array([-6.75, y, 0]),
                end=np.array([-1.18, y, 0]),
                buff=0,
                max_tip_length_to_length_ratio=0.025,
                stroke_width=1.4,
                color=cyan,
            ).set_opacity(0.22)
            field_lines.add(line)

        coils = VGroup()
        for x, opacity in [(-6.72, 0.8), (-6.36, 0.45), (-1.54, 0.45), (-1.18, 0.8)]:
            coil = Ellipse(
                width=0.34,
                height=3.42,
                stroke_color=teal,
                stroke_width=3.2,
            ).move_to(np.array([x, -0.45, 0]))
            coil.set_fill(opacity=0)
            coil.set_stroke(opacity=opacity)
            coils.add(coil)

        mirror_left = MathTex(r"B_{\max}", font_size=25, color=teal).move_to(
            np.array([-6.52, 1.28, 0])
        )
        mirror_right = MathTex(r"B_{\max}", font_size=25, color=teal).move_to(
            np.array([-1.38, 1.28, 0])
        )
        middle_field = MathTex(r"B_{\min}", font_size=25, color=cyan).move_to(
            np.array([-3.95, 1.14, 0])
        )

        helix = ParametricFunction(
            lambda u: np.array(
                [
                    -6.25 + 4.6 * u,
                    -0.45 + (0.62 - 0.12 * u) * np.sin(TAU * scene["turns"] * u),
                    0,
                ]
            ),
            t_range=[0, 1],
            stroke_color=cyan,
            stroke_width=2.2,
        ).set_fill(opacity=0).set_stroke(opacity=0.28)

        antennas = VGroup()
        for x in np.linspace(-6.05, -1.85, 5):
            for y, sign in [(0.97, 1), (-1.87, -1)]:
                stem = Line(
                    np.array([x, y, 0]),
                    np.array([x, y + sign * 0.17, 0]),
                    color=muted,
                    stroke_width=2.2,
                )
                arms = VGroup(
                    Line(
                        np.array([x, y + sign * 0.17, 0]),
                        np.array([x - 0.11, y + sign * 0.31, 0]),
                        color=cyan,
                        stroke_width=2.2,
                    ),
                    Line(
                        np.array([x, y + sign * 0.17, 0]),
                        np.array([x + 0.11, y + sign * 0.31, 0]),
                        color=cyan,
                        stroke_width=2.2,
                    ),
                )
                antennas.add(stem, arms)

        radio_label = MathTex(
            r"f_c\simeq 27\ \mathrm{GHz}", font_size=27, color=cyan
        ).move_to(np.array([-3.95, -2.73, 0]))

        axes = Axes(
            x_range=[0, 10, 2],
            y_range=[0, 4.5, 1],
            x_length=5.65,
            y_length=3.82,
            tips=False,
            axis_config={
                "color": muted,
                "stroke_width": 1.7,
                "include_ticks": True,
                "tick_size": 0.05,
            },
        ).move_to(np.array([4.2, -0.48, 0]))

        grid = VGroup()
        for x in [2, 4, 6, 8]:
            grid.add(
                Line(
                    axes.c2p(x, 0),
                    axes.c2p(x, 4.5),
                    color=grid_color,
                    stroke_width=1,
                ).set_opacity(0.6)
            )
        for y in [1, 2, 3, 4]:
            grid.add(
                Line(
                    axes.c2p(0, y),
                    axes.c2p(10, y),
                    color=grid_color,
                    stroke_width=1,
                ).set_opacity(0.6)
            )

        x_label = MathTex(r"t", font_size=30, color=WHITE).next_to(
            axes.x_axis, DOWN, buff=0.18
        )
        y_label = MathTex(r"f_c", font_size=30, color=WHITE).next_to(
            axes.y_axis, LEFT, buff=0.18
        )
        y_label.rotate(PI / 2)

        base_frequency = DashedLine(
            axes.c2p(0, 0.62),
            axes.c2p(10, 0.62),
            dash_length=0.08,
            color=muted,
            stroke_width=1.4,
        ).set_opacity(0.42)
        base_label = MathTex(r"f_c(0)", font_size=22, color=muted).next_to(
            axes.c2p(0, 0.62), LEFT, buff=0.08
        )

        progress = ValueTracker(0.0)

        def bounce_parameter(value: float) -> float:
            return 0.5 * (1 - np.cos(TAU * value))

        def electron_point() -> np.ndarray:
            u = bounce_parameter(progress.get_value())
            amplitude = 0.62 - 0.12 * u
            return np.array(
                [
                    -6.25 + 4.6 * u,
                    -0.45 + amplitude * np.sin(TAU * scene["turns"] * u),
                    0,
                ]
            )

        electron = always_redraw(
            lambda: VGroup(
                Circle(radius=0.18, color=orange, fill_color=orange, fill_opacity=0.12)
                .set_stroke(width=0)
                .move_to(electron_point()),
                Circle(radius=0.105, color=orange, fill_color=orange, fill_opacity=0.35)
                .set_stroke(width=0)
                .move_to(electron_point()),
                Dot(electron_point(), radius=0.055, color=WHITE),
            )
        )

        waves = always_redraw(lambda: self._radiation_waves(progress, electron_point(), cyan))

        def track_value(u: float) -> float:
            continuous = 0.62 + 1.18 * u + 0.34 * u**2
            if u >= scene["collision_time"]:
                continuous += scene["track_jump"]
            return continuous

        def partial_track(width: float, opacity: float) -> VMobject:
            current = max(progress.get_value(), 0.002)
            samples = np.linspace(0, current, max(3, int(180 * current)))
            points = [axes.c2p(10 * u, track_value(u)) for u in samples]
            curve = VMobject(stroke_color=cyan, stroke_width=width)
            curve.set_points_as_corners(points)
            curve.set_fill(opacity=0)
            curve.set_stroke(opacity=opacity)
            return curve

        track_glow = always_redraw(lambda: partial_track(12, 0.12))
        track = always_redraw(lambda: partial_track(4.2, 1.0))
        track_dot = always_redraw(
            lambda: Dot(
                axes.c2p(
                    10 * progress.get_value(),
                    track_value(progress.get_value()),
                ),
                radius=0.065,
                color=orange,
            )
        )

        collision_line = DashedLine(
            axes.c2p(10 * scene["collision_time"], 0),
            axes.c2p(10 * scene["collision_time"], 4.5),
            color=orange,
            stroke_width=1.8,
            dash_length=0.09,
        ).set_opacity(0.7)
        collision_label = Text(
            "рассеяние",
            font="Arial",
            font_size=22,
            color=orange,
        ).next_to(collision_line, RIGHT, buff=0.08).shift(UP * 1.25)
        collision_marker = VGroup(collision_line, collision_label).set_opacity(0)

        frequency_arrow = VGroup(
            Arrow(
                axes.c2p(9.12, 2.35),
                axes.c2p(9.12, 3.4),
                buff=0,
                color=cyan,
                stroke_width=3,
            ),
            MathTex(r"f_c\uparrow", font_size=25, color=cyan).next_to(
                axes.c2p(9.12, 3.4), LEFT, buff=0.08
            ),
        ).set_opacity(0)

        self.play(
            FadeIn(heading, shift=DOWN * 0.1),
            FadeIn(relation),
            FadeIn(left_panel, right_panel),
            run_time=0.9,
        )
        self.play(
            FadeIn(trap_label, plot_label),
            FadeIn(trap, field_lines, coils, mirror_left, mirror_right, middle_field),
            FadeIn(helix, antennas, radio_label),
            FadeIn(grid, axes, x_label, y_label, base_frequency, base_label),
            FadeIn(electron, waves, track_glow, track, track_dot),
            run_time=1.1,
        )

        first_leg = scene["run_time"] * scene["collision_time"]
        second_leg = scene["run_time"] - first_leg
        self.play(progress.animate.set_value(scene["collision_time"]), run_time=first_leg, rate_func=linear)
        self.play(
            collision_marker.animate.set_opacity(1),
            Flash(electron_point(), color=orange, flash_radius=0.32, line_length=0.12),
            run_time=0.35,
        )
        self.play(
            progress.animate.set_value(1.0),
            frequency_arrow.animate.set_opacity(1),
            run_time=second_leg,
            rate_func=linear,
        )
        self.wait(0.7)
        self.play(*[FadeOut(mob) for mob in list(self.mobjects)], run_time=0.7)

    @staticmethod
    def _radiation_waves(
        progress: ValueTracker,
        center: np.ndarray,
        color: str,
    ) -> VGroup:
        waves = VGroup()
        phase_time = progress.get_value() * 9.0
        for offset in (0.0, 0.27, 0.54, 0.81):
            phase = (phase_time - offset) % 1.0
            radius = 0.12 + 0.82 * phase
            opacity = 0.38 * (1.0 - phase) ** 1.4
            waves.add(
                Circle(radius=radius, color=color, stroke_width=2.0)
                .set_opacity(opacity)
                .move_to(center)
            )
        return waves
