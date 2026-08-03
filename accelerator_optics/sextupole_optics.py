from __future__ import annotations

import configparser
from pathlib import Path

import numpy as np
from manim import *

try:
    from .magnetic_optics import CFG, make_text
    from .sextupole import ChromaticFocusModel, sextupole_field
except ImportError:
    from magnetic_optics import CFG, make_text
    from sextupole import ChromaticFocusModel, sextupole_field


BASE_DIR = Path(__file__).resolve().parent
MOMENTA = (
    (-0.12, "p < p₀", "momentum"),
    (0.0, "p = p₀", "proton"),
    (0.12, "p > p₀", "field"),
)


def load_timing() -> dict[str, float]:
    parser = configparser.ConfigParser()
    parser.read(BASE_DIR / "run.cfg")
    defaults = {
        "intro": 0.9,
        "field_build": 1.4,
        "field_grow": 1.0,
        "field_hold": 1.4,
        "transition": 0.7,
        "chromatic_build": 1.1,
        "chromatic_flight": 3.8,
        "error_mark": 0.7,
        "correction_build": 1.2,
        "correction_transform": 2.4,
        "message": 0.9,
        "tail": 1.6,
    }
    return {
        key: parser.getfloat("sextupole_timing", key, fallback=value)
        for key, value in defaults.items()
    }


TIMING = load_timing()


def make_background(colors: dict) -> NumberPlane:
    plane = NumberPlane(
        x_range=[-9, 9, 1],
        y_range=[-9, 9, 1],
        background_line_style={
            "stroke_color": colors["grid"],
            "stroke_width": 0.65,
            "stroke_opacity": 0.18,
        },
        axis_config={"stroke_opacity": 0.0},
    )
    plane.set_z_index(-20)
    return plane


def make_sextupole_icon(center: np.ndarray, colors: dict) -> VGroup:
    result = VGroup()
    pole_radius = 2.38
    for index, angle in enumerate((30, 90, 150, 210, 270, 330)):
        theta = angle * DEGREES
        label_text = "S" if index % 2 == 0 else "N"
        pole_color = colors["field"] if label_text == "S" else colors["momentum"]
        pole = RoundedRectangle(
            width=1.15,
            height=0.55,
            corner_radius=0.14,
            stroke_color=pole_color,
            stroke_width=2.0,
            fill_color=colors["magnet"],
            fill_opacity=0.96,
        )
        pole.rotate(theta + PI / 2)
        pole.move_to(center + pole_radius * np.array([np.cos(theta), np.sin(theta), 0.0]))
        label = make_text(
            label_text,
            color=pole_color,
            scale=0.28,
            weight="BOLD",
        ).move_to(pole.get_center())
        result.add(pole, label)
    result.add(
        Circle(
            radius=2.02,
            stroke_color=colors["magnet_edge"],
            stroke_width=1.3,
            stroke_opacity=0.55,
        ).move_to(center)
    )
    return result


def make_field_arrows(center: np.ndarray, colors: dict) -> VGroup:
    arrows = VGroup()
    for x in (-1.45, -0.72, 0.0, 0.72, 1.45):
        for y in (-1.45, -0.72, 0.0, 0.72, 1.45):
            radius = float(np.hypot(x, y))
            if radius < 0.15 or radius > 1.9:
                continue
            field = sextupole_field(x, y)
            direction = field / np.linalg.norm(field)
            length = 0.18 + 0.18 * radius**2
            position = center + np.array([x, y, 0.0])
            arrows.add(
                Arrow(
                    position - 0.5 * length * direction,
                    position + 0.5 * length * direction,
                    buff=0.0,
                    color=colors["field"],
                    stroke_width=3.0,
                    max_tip_length_to_length_ratio=0.30,
                )
            )
    return arrows


def make_field_formula(colors: dict) -> VGroup:
    label = make_text(
        "СЕКСТУПОЛЬНОЕ ПОЛЕ",
        color=colors["muted"],
        scale=0.23,
        weight="BOLD",
    )
    components = MathTex(
        r"B_y=\frac{S}{2}(x^2-y^2),\qquad B_x=Sxy",
        color=colors["field"],
    ).scale(0.54)
    magnitude = MathTex(
        r"|\mathbf B|=\frac{|S|}{2}r^2",
        color=colors["text"],
    ).scale(0.54)
    explanation = make_text(
        "S — вторая производная поля, Тл/м²",
        color=colors["muted"],
        scale=0.23,
    )
    content = VGroup(label, components, magnitude, explanation).arrange(DOWN, buff=0.07)
    plate = RoundedRectangle(
        width=content.width + 0.50,
        height=content.height + 0.28,
        corner_radius=0.16,
        stroke_color=colors["field"],
        stroke_width=1.0,
        fill_color=colors["background"],
        fill_opacity=0.78,
    ).set_opacity(0.84)
    return VGroup(plate, content)


def make_ray_path(
    model: ChromaticFocusModel,
    center_y: float,
    initial_offset: float,
    deviation: float,
    corrected: bool,
    color: str,
) -> VMobject:
    points = model.sample_ray(initial_offset, deviation, corrected=corrected)
    display = [np.array([point[0], center_y + point[1], 0.0]) for point in points]
    path = VMobject().set_points_as_corners(display)
    path.set_stroke(color, width=3.0, opacity=0.78)
    return path


def make_momentum_legend(colors: dict, y: float) -> VGroup:
    items = VGroup()
    for _, label, color_key in MOMENTA:
        dot = Dot(radius=0.07, color=colors[color_key])
        text = make_text(label, color=colors["text"], scale=0.28)
        items.add(VGroup(dot, text).arrange(RIGHT, buff=0.12))
    return items.arrange(RIGHT, buff=0.52).move_to([0.0, y, 0.0])


class SextupoleMagnetScene(Scene):
    def construct(self) -> None:
        colors = CFG["colors"]
        self.camera.background_color = colors["background"]
        self.add(make_background(colors))

        kicker = make_text(
            "МАГНИТНАЯ ОПТИКА УСКОРИТЕЛЯ",
            color=colors["muted"],
            scale=0.34,
            weight="BOLD",
        ).move_to([0.0, 7.35, 0.0])
        title = make_text(
            "Секступольный магнит",
            color=colors["text"],
            scale=0.78,
            weight="BOLD",
        ).move_to([0.0, 6.55, 0.0])
        subtitle = make_text(
            "исправляет импульсную ошибку",
            color=colors["field"],
            scale=0.41,
            weight="BOLD",
        ).move_to([0.0, 5.82, 0.0])
        self.play(
            FadeIn(kicker, shift=0.12 * DOWN),
            FadeIn(title, shift=0.16 * UP),
            FadeIn(subtitle),
            run_time=TIMING["intro"],
        )

        center = np.array([0.0, -0.05, 0.0])
        icon = make_sextupole_icon(center, colors)
        formula = make_field_formula(colors).move_to([0.0, 4.20, 0.0])
        axes = VGroup(
            Arrow([-2.22, -0.05, 0.0], [2.34, -0.05, 0.0], buff=0.0, color=colors["muted"], stroke_width=1.3),
            Arrow([0.0, -2.27, 0.0], [0.0, 2.30, 0.0], buff=0.0, color=colors["muted"], stroke_width=1.3),
            MathTex("x", color=colors["muted"]).scale(0.45).move_to([2.53, -0.05, 0.0]),
            MathTex("y", color=colors["muted"]).scale(0.45).move_to([0.0, 2.50, 0.0]),
        )
        field_arrows = make_field_arrows(center, colors)
        zero = VGroup(
            Dot(center, radius=0.075, color=colors["proton"]),
            MathTex(r"\mathbf B=0", color=colors["proton"]).scale(0.42).next_to(center, DOWN, buff=0.16),
        )
        note = make_text(
            "вдвое дальше — поле вчетверо сильнее",
            color=colors["text"],
            scale=0.34,
            weight="BOLD",
        ).move_to([0.0, -3.25, 0.0])
        self.play(
            FadeIn(formula, shift=0.10 * UP),
            FadeIn(icon),
            Create(axes),
            run_time=TIMING["field_build"],
        )
        self.play(
            LaggedStart(*[GrowArrow(arrow) for arrow in field_arrows], lag_ratio=0.035),
            FadeIn(zero),
            run_time=TIMING["field_grow"],
        )
        self.play(FadeIn(note, shift=0.10 * UP), run_time=0.45)
        self.wait(TIMING["field_hold"])
        self.play(
            FadeOut(VGroup(icon, formula, axes, field_arrows, zero, note)),
            run_time=TIMING["transition"],
        )

        model = ChromaticFocusModel()
        center_y = -0.45
        act_two_title = make_text(
            "КВАДРУПОЛЬ: РАЗНЫЕ ИМПУЛЬСЫ",
            color=colors["text"],
            scale=0.48,
            weight="BOLD",
        ).move_to([0.0, 4.48, 0.0])
        focal_relation = MathTex(
            r"f\propto p",
            color=colors["momentum"],
        ).scale(0.58)
        relation_arrow = Arrow(
            LEFT * 0.34,
            RIGHT * 0.34,
            buff=0.0,
            color=colors["muted"],
            stroke_width=2.8,
        )
        focal_words = make_text(
            "разные фокусы",
            color=colors["momentum"],
            scale=0.31,
            weight="BOLD",
        )
        act_two_formula = VGroup(
            focal_relation,
            relation_arrow,
            focal_words,
        ).arrange(RIGHT, buff=0.28).move_to([0.0, 3.83, 0.0])
        legend = make_momentum_legend(colors, 3.18)
        panel = RoundedRectangle(
            width=7.85,
            height=5.20,
            corner_radius=0.18,
            stroke_color=colors["magnet_edge"],
            stroke_width=1.1,
            fill_color=colors["background"],
            fill_opacity=0.42,
        ).move_to([0.0, center_y, 0.0])
        axis = Line(
            [-3.55, center_y, 0.0],
            [3.45, center_y, 0.0],
            color=colors["muted"],
            stroke_width=1.1,
            stroke_opacity=0.42,
        )
        quadrupole = RoundedRectangle(
            width=0.48,
            height=4.25,
            corner_radius=0.10,
            stroke_color=colors["magnet_edge"],
            stroke_width=2.0,
            fill_color=colors["magnet"],
            fill_opacity=0.82,
        ).move_to([model.quadrupole_position, center_y, 0.0])
        q_label = make_text(
            "Q",
            color=colors["text"],
            scale=0.34,
            weight="BOLD",
        ).move_to([model.quadrupole_position, center_y + 1.85, 0.0])

        paths = []
        path_group = VGroup()
        focus_marks = VGroup()
        upper_paths = []
        for deviation, _, color_key in MOMENTA:
            color = colors[color_key]
            pair = []
            for initial_offset in (-0.72, 0.72):
                path = make_ray_path(
                    model,
                    center_y,
                    initial_offset,
                    deviation,
                    False,
                    color,
                )
                paths.append((path, deviation, initial_offset, color))
                path_group.add(path)
                pair.append(path)
            upper_paths.append((pair[1], deviation, color))
            focus_x = model.focal_position(deviation, corrected=False)
            focus_marks.add(Dot([focus_x, center_y, 0.0], radius=0.095, color=color))

        self.play(
            FadeIn(act_two_title, shift=0.10 * UP),
            FadeIn(act_two_formula),
            FadeIn(legend),
            FadeIn(panel),
            FadeIn(quadrupole),
            FadeIn(q_label),
            FadeIn(axis),
            LaggedStart(*[Create(path) for path in path_group], lag_ratio=0.05),
            run_time=TIMING["chromatic_build"],
        )
        particles = VGroup()
        flights = []
        for path, deviation, color in upper_paths:
            focus_x = model.focal_position(deviation, corrected=False)
            flight_path = make_ray_path(
                model,
                center_y,
                0.72,
                deviation,
                False,
                color,
            )
            points = [point for point in flight_path.points if point[0] <= focus_x]
            if len(points) >= 2:
                flight_path.set_points_as_corners(points)
            particle = Dot(path.get_start(), radius=0.085, color=color).set_z_index(9)
            particles.add(particle)
            flights.append(MoveAlongPath(particle, flight_path, rate_func=linear))
        self.add(particles)
        self.play(*flights, run_time=TIMING["chromatic_flight"])
        error_label = make_text(
            "ХРОМАТИЧЕСКАЯ ОШИБКА",
            color=colors["momentum"],
            scale=0.46,
            weight="BOLD",
        ).move_to([0.0, -3.62, 0.0])
        self.play(
            FadeIn(focus_marks),
            FadeIn(error_label, shift=0.10 * UP),
            run_time=TIMING["error_mark"],
        )

        correction_title = make_text(
            "ДИСПЕРСИЯ + СЕКСТУПОЛЬ",
            color=colors["text"],
            scale=0.49,
            weight="BOLD",
        ).move_to([0.0, 4.48, 0.0])
        dispersion_formula = MathTex(
            r"x=D\delta",
            color=colors["field"],
        ).scale(0.64)
        correction_formula = MathTex(
            r"\Delta G=S D\delta",
            color=colors["force"],
        ).scale(0.64)
        arrow = Arrow(LEFT * 0.42, RIGHT * 0.42, buff=0.0, color=colors["muted"], stroke_width=3.0)
        correction_row = VGroup(dispersion_formula, arrow, correction_formula).arrange(RIGHT, buff=0.35)
        correction_row.move_to([0.0, 3.78, 0.0])
        correction_note = make_text(
            "секступоль ставят там, где D ≠ 0",
            color=colors["muted"],
            scale=0.29,
        ).move_to([0.0, 3.25, 0.0])
        sextupole = RoundedRectangle(
            width=0.48,
            height=4.25,
            corner_radius=0.10,
            stroke_color=colors["force"],
            stroke_width=2.0,
            fill_color=colors["magnet"],
            fill_opacity=0.82,
        ).move_to([model.sextupole_position, center_y, 0.0])
        s_label = make_text(
            "S",
            color=colors["force"],
            scale=0.34,
            weight="BOLD",
        ).move_to([model.sextupole_position, center_y + 1.85, 0.0])

        self.play(
            Transform(act_two_title, correction_title),
            FadeOut(act_two_formula),
            FadeIn(correction_row),
            FadeIn(correction_note),
            FadeIn(sextupole),
            FadeIn(s_label),
            FadeOut(legend),
            FadeOut(error_label),
            FadeOut(particles),
            run_time=TIMING["correction_build"],
        )

        transforms = []
        for path, deviation, initial_offset, color in paths:
            target = make_ray_path(
                model,
                center_y,
                initial_offset,
                deviation,
                True,
                color,
            )
            transforms.append(Transform(path, target))
        corrected_focus_marks = VGroup()
        for deviation, _, color_key in MOMENTA:
            focus_x = model.focal_position(deviation, corrected=True)
            corrected_focus_marks.add(
                Dot([focus_x, center_y, 0.0], radius=0.095, color=colors[color_key])
            )
        self.play(
            *transforms,
            Transform(focus_marks, corrected_focus_marks),
            run_time=TIMING["correction_transform"],
            rate_func=smooth,
        )

        focus_center = float(
            np.mean(
                [model.focal_position(deviation, corrected=True) for deviation, _, _ in MOMENTA]
            )
        )
        focus_ring = Circle(
            radius=0.25,
            stroke_color=colors["proton"],
            stroke_width=2.0,
        ).move_to([focus_center, center_y, 0.0])
        focus_label = make_text(
            "фокусы почти совпали",
            color=colors["proton"],
            scale=0.30,
            weight="BOLD",
        ).move_to([focus_center, center_y + 0.48, 0.0])
        final_message = make_text(
            "ХРОМАТИЧЕСКАЯ ОШИБКА ИСПРАВЛЕНА В ПЕРВОМ ПОРЯДКЕ",
            color=colors["field"],
            scale=0.34,
            weight="BOLD",
        ).move_to([0.0, -3.62, 0.0])
        self.play(
            FadeIn(focus_ring),
            FadeIn(focus_label),
            FadeIn(final_message, shift=0.10 * UP),
            run_time=TIMING["message"],
        )
        self.wait(TIMING["tail"])
