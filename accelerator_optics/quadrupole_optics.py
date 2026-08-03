from __future__ import annotations

import configparser
from pathlib import Path

import numpy as np
from manim import *

try:
    from .magnetic_optics import CFG, make_text
    from .quadrupole import (
        AlternatingGradientLattice,
        QuadrupoleRayModel,
        quadrupole_field,
    )
except ImportError:
    from magnetic_optics import CFG, make_text
    from quadrupole import AlternatingGradientLattice, QuadrupoleRayModel, quadrupole_field


BASE_DIR = Path(__file__).resolve().parent


def load_timing() -> dict[str, float]:
    parser = configparser.ConfigParser()
    parser.read(BASE_DIR / "run.cfg")
    defaults = {
        "intro": 0.9,
        "field_build": 1.4,
        "field_grow": 1.0,
        "field_hold": 1.5,
        "transition": 0.7,
        "rays_build": 1.1,
        "flight": 4.8,
        "focus_mark": 0.8,
        "rotation": 1.1,
        "lattice_transition": 0.7,
        "lattice_build": 1.2,
        "lattice_flight": 4.8,
        "lattice_message": 0.8,
        "tail": 1.8,
    }
    return {
        key: parser.getfloat("quadrupole_timing", key, fallback=value)
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


def make_quadrupole_icon(
    center: np.ndarray,
    colors: dict,
    *,
    scale: float = 1.0,
    labels: bool = True,
) -> VGroup:
    result = VGroup()
    pole_radius = 2.38 * scale
    for index, angle in enumerate((45, 135, 225, 315)):
        theta = angle * DEGREES
        pole_color = colors["field"] if index % 2 == 0 else colors["momentum"]
        pole = RoundedRectangle(
            width=1.45 * scale,
            height=0.64 * scale,
            corner_radius=0.16 * scale,
            stroke_color=pole_color,
            stroke_width=2.0,
            fill_color=colors["magnet"],
            fill_opacity=0.96,
        )
        pole.rotate(theta + PI / 2)
        pole.move_to(center + pole_radius * np.array([np.cos(theta), np.sin(theta), 0.0]))
        result.add(pole)
        if labels:
            label = make_text(
                "S" if index % 2 == 0 else "N",
                color=pole_color,
                scale=0.30 * scale,
                weight="BOLD",
            ).move_to(pole.get_center())
            result.add(label)
    aperture = Circle(
        radius=2.05 * scale,
        stroke_color=colors["magnet_edge"],
        stroke_width=1.3,
        stroke_opacity=0.55,
    ).move_to(center)
    result.add(aperture)
    return result


def make_field_formula(colors: dict) -> VGroup:
    label = make_text(
        "КВАДРУПОЛЬНОЕ ПОЛЕ",
        color=colors["muted"],
        scale=0.24,
        weight="BOLD",
    )
    components = MathTex(
        r"B_x=Gy,\qquad B_y=Gx",
        color=colors["field"],
    ).scale(0.64)
    magnitude = MathTex(
        r"|\mathbf B|=G r",
        color=colors["text"],
    ).scale(0.58)
    explanation = make_text(
        "G — градиент поля",
        color=colors["muted"],
        scale=0.25,
    )
    content = VGroup(label, components, VGroup(magnitude, explanation).arrange(RIGHT, buff=0.32))
    content.arrange(DOWN, buff=0.08)
    plate = RoundedRectangle(
        width=content.width + 0.55,
        height=content.height + 0.30,
        corner_radius=0.16,
        stroke_color=colors["field"],
        stroke_width=1.0,
        fill_color=colors["background"],
        fill_opacity=0.76,
    ).set_opacity(0.82)
    return VGroup(plate, content)


def make_field_arrows(center: np.ndarray, colors: dict) -> VGroup:
    arrows = VGroup()
    for x in (-1.45, -0.72, 0.0, 0.72, 1.45):
        for y in (-1.45, -0.72, 0.0, 0.72, 1.45):
            radius = float(np.hypot(x, y))
            if radius < 0.15 or radius > 1.9:
                continue
            field = quadrupole_field(x, y)
            direction = field / np.linalg.norm(field)
            length = 0.24 + 0.24 * radius
            position = center + np.array([x, y, 0.0])
            arrow = Arrow(
                position - 0.5 * length * direction,
                position + 0.5 * length * direction,
                buff=0.0,
                color=colors["field"],
                stroke_width=3.1,
                max_tip_length_to_length_ratio=0.30,
            )
            arrows.add(arrow)
    return arrows


def make_panel(
    center_y: float,
    title: str,
    relation: str,
    plane_color: str,
    colors: dict,
    model: QuadrupoleRayModel,
    mode: str,
) -> tuple[VGroup, list[VMobject]]:
    frame = RoundedRectangle(
        width=7.75,
        height=3.15,
        corner_radius=0.18,
        stroke_color=colors["magnet_edge"],
        stroke_width=1.1,
        fill_color=colors["background"],
        fill_opacity=0.42,
    ).move_to([0.0, center_y, 0.0])
    title_text = make_text(
        title,
        color=plane_color,
        scale=0.35,
        weight="BOLD",
    ).move_to([-1.30, center_y + 1.19, 0.0])
    relation_tex = MathTex(relation, color=colors["force"]).scale(0.52)
    relation_tex.move_to([2.42, center_y + 1.19, 0.0])
    axis = Line(
        [-3.55, center_y, 0.0],
        [3.55, center_y, 0.0],
        color=colors["muted"],
        stroke_width=1.1,
        stroke_opacity=0.44,
    )
    magnet = Rectangle(
        width=model.length,
        height=2.38,
        stroke_color=colors["magnet_edge"],
        stroke_width=2.0,
        fill_color=colors["magnet"],
        fill_opacity=0.62,
    ).move_to([0.0, center_y, 0.0])
    q_label = make_text(
        "Q",
        color=colors["text"],
        scale=0.38,
        weight="BOLD",
    ).move_to([0.0, center_y + 0.92, 0.0])

    ray_paths: list[VMobject] = []
    rays = VGroup()
    for initial_offset in (-0.38, -0.19, 0.19, 0.38):
        points = model.sample_points(initial_offset, mode)
        display_points = [
            np.array([point[0], center_y + point[1], 0.0]) for point in points
        ]
        path = VMobject().set_points_as_corners(display_points)
        path.set_stroke(plane_color, width=2.7, opacity=0.72)
        ray_paths.append(path)
        rays.add(path)
    return VGroup(frame, axis, magnet, q_label, title_text, relation_tex, rays), ray_paths


def lattice_screen_x(longitudinal: float, lattice: AlternatingGradientLattice) -> float:
    return -3.15 + 6.30 * longitudinal / lattice.total_length


def make_lattice_panel(
    center_y: float,
    label: str,
    plane: str,
    plane_color: str,
    colors: dict,
    lattice: AlternatingGradientLattice,
) -> tuple[VGroup, VMobject, VMobject]:
    frame = RoundedRectangle(
        width=7.75,
        height=2.35,
        corner_radius=0.18,
        stroke_color=colors["magnet_edge"],
        stroke_width=1.1,
        fill_color=colors["background"],
        fill_opacity=0.42,
    ).move_to([0.0, center_y, 0.0])
    axis = Line(
        [-3.45, center_y, 0.0],
        [3.45, center_y, 0.0],
        color=colors["muted"],
        stroke_width=1.0,
        stroke_opacity=0.38,
    )
    title = make_text(
        label,
        color=plane_color,
        scale=0.30,
        weight="BOLD",
    ).move_to([0.0, center_y + 0.92, 0.0])

    magnets = VGroup()
    for index in range(2 * lattice.cells + 1):
        kind = "F" if index % 2 == 0 else "D"
        x_position = lattice_screen_x(index * lattice.drift_length, lattice)
        pole_color = colors["field"] if kind == "F" else colors["momentum"]
        block = RoundedRectangle(
            width=0.30,
            height=1.62,
            corner_radius=0.08,
            stroke_color=pole_color,
            stroke_width=1.5,
            fill_color=colors["magnet"],
            fill_opacity=0.86,
        ).move_to([x_position, center_y, 0.0])
        kind_label = make_text(
            kind,
            color=pole_color,
            scale=0.23,
            weight="BOLD",
        ).move_to([x_position, center_y + 0.56, 0.0])
        magnets.add(block, kind_label)

    envelope = lattice.sample_envelope(plane)
    scale = 1.15
    upper_points = [
        np.array(
            [
                lattice_screen_x(point[0], lattice),
                center_y + scale * point[1],
                0.0,
            ]
        )
        for point in envelope
    ]
    lower_points = [
        np.array([point[0], 2.0 * center_y - point[1], 0.0])
        for point in upper_points
    ]
    upper = VMobject().set_points_as_corners(upper_points)
    lower = VMobject().set_points_as_corners(lower_points)
    upper.set_stroke(plane_color, width=3.0, opacity=0.92)
    lower.set_stroke(plane_color, width=3.0, opacity=0.92)
    fill = VMobject().set_points_as_corners(upper_points + list(reversed(lower_points)))
    fill.close_path()
    fill.set_fill(plane_color, opacity=0.13).set_stroke(width=0)
    return VGroup(frame, axis, magnets, title, fill, upper, lower), upper, lower


class QuadrupoleMagnetScene(Scene):
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
            "Квадрупольный магнит",
            color=colors["text"],
            scale=0.77,
            weight="BOLD",
        ).move_to([0.0, 6.55, 0.0])
        subtitle = make_text(
            "фокусирует — и расфокусирует",
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

        cross_center = np.array([0.0, 0.0, 0.0])
        icon = make_quadrupole_icon(cross_center, colors)
        formula = make_field_formula(colors).move_to([0.0, 4.28, 0.0])
        axes = VGroup(
            Arrow([-2.25, 0.0, 0.0], [2.35, 0.0, 0.0], buff=0.0, color=colors["muted"], stroke_width=1.3),
            Arrow([0.0, -2.25, 0.0], [0.0, 2.35, 0.0], buff=0.0, color=colors["muted"], stroke_width=1.3),
            MathTex("x", color=colors["muted"]).scale(0.45).move_to([2.55, 0.0, 0.0]),
            MathTex("y", color=colors["muted"]).scale(0.45).move_to([0.0, 2.55, 0.0]),
        )
        field_arrows = make_field_arrows(cross_center, colors)
        zero = VGroup(
            Dot(cross_center, radius=0.075, color=colors["proton"]),
            MathTex(r"\mathbf B=0", color=colors["proton"]).scale(0.43).next_to(cross_center, DOWN, buff=0.16),
        )
        note = make_text(
            "дальше от оси — сильнее поле",
            color=colors["text"],
            scale=0.36,
            weight="BOLD",
        ).move_to([0.0, -3.18, 0.0])

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

        cross_section = VGroup(icon, formula, axes, field_arrows, zero, note)
        self.play(FadeOut(cross_section), run_time=TIMING["transition"])

        model = QuadrupoleRayModel()
        focus_panel, focus_paths = make_panel(
            1.85,
            "ГОРИЗОНТАЛЬНАЯ ПЛОСКОСТЬ x",
            r"F_x\propto -x",
            colors["field"],
            colors,
            model,
            "focusing",
        )
        defocus_panel, defocus_paths = make_panel(
            -3.45,
            "ВЕРТИКАЛЬНАЯ ПЛОСКОСТЬ y",
            r"F_y\propto +y",
            colors["momentum"],
            colors,
            model,
            "defocusing",
        )
        self.play(
            FadeIn(focus_panel[:6]),
            FadeIn(defocus_panel[:6]),
            LaggedStart(*[Create(path) for path in focus_paths + defocus_paths], lag_ratio=0.05),
            run_time=TIMING["rays_build"],
        )

        particles = VGroup()
        particle_animations = []
        for index, path in enumerate(focus_paths + defocus_paths):
            color = colors["field"] if index < len(focus_paths) else colors["momentum"]
            particle = Dot(path.get_start(), radius=0.075, color=color).set_z_index(8)
            particles.add(particle)
            particle_animations.append(MoveAlongPath(particle, path, rate_func=linear))
        self.add(particles)
        self.play(*particle_animations, run_time=TIMING["flight"])

        focal_x = model.focal_position
        focus_mark = VGroup(
            Circle(radius=0.20, stroke_color=colors["proton"], stroke_width=2.0).move_to([focal_x, 1.85, 0.0]),
            make_text("фокус", color=colors["proton"], scale=0.28, weight="BOLD").move_to([focal_x, 2.30, 0.0]),
        )
        spreading = make_text(
            "пучок расходится",
            color=colors["momentum"],
            scale=0.31,
            weight="BOLD",
        ).move_to([2.45, -4.83, 0.0])
        self.play(
            FadeIn(focus_mark),
            FadeIn(spreading),
            run_time=TIMING["focus_mark"],
        )

        mini_icon = make_quadrupole_icon(
            np.array([0.0, -6.20, 0.0]),
            colors,
            scale=0.28,
            labels=False,
        )
        rotate_label = make_text(
            "поворот на 90° меняет плоскости местами",
            color=colors["text"],
            scale=0.34,
            weight="BOLD",
        ).move_to([0.0, -7.18, 0.0])
        self.play(FadeIn(mini_icon), FadeIn(rotate_label), run_time=0.45)
        self.play(
            Rotate(mini_icon, angle=PI / 2),
            run_time=TIMING["rotation"],
            rate_func=smooth,
        )
        self.wait(0.35)

        second_act = VGroup(
            focus_panel,
            defocus_panel,
            particles,
            focus_mark,
            spreading,
            mini_icon,
            rotate_label,
        )
        self.play(FadeOut(second_act), run_time=TIMING["lattice_transition"])

        lattice = AlternatingGradientLattice()
        lattice_title = make_text(
            "ЧЕРЕДОВАНИЕ F–D",
            color=colors["text"],
            scale=0.53,
            weight="BOLD",
        ).move_to([0.0, 4.58, 0.0])
        lattice_subtitle = make_text(
            "F — пролёт — D: магниты разделены в пространстве",
            color=colors["muted"],
            scale=0.32,
        ).move_to([0.0, 4.05, 0.0])
        lattice_x, upper_x, lower_x = make_lattice_panel(
            1.85,
            "РАЗМЕР ПУЧКА ПО x",
            "x",
            colors["field"],
            colors,
            lattice,
        )
        lattice_y, upper_y, lower_y = make_lattice_panel(
            -1.28,
            "РАЗМЕР ПУЧКА ПО y",
            "y",
            colors["momentum"],
            colors,
            lattice,
        )
        self.play(
            FadeIn(lattice_title, shift=0.10 * UP),
            FadeIn(lattice_subtitle),
            FadeIn(lattice_x[:5]),
            FadeIn(lattice_y[:5]),
            Create(upper_x),
            Create(lower_x),
            Create(upper_y),
            Create(lower_y),
            run_time=TIMING["lattice_build"],
        )

        progress = ValueTracker(0.0)

        def cursor(panel_y: float) -> Line:
            position = progress.get_value() * lattice.total_length
            x_position = lattice_screen_x(position, lattice)
            return Line(
                [x_position, panel_y - 0.82, 0.0],
                [x_position, panel_y + 0.82, 0.0],
                color=colors["proton"],
                stroke_width=2.0,
            ).set_z_index(9)

        cursor_x = always_redraw(lambda: cursor(1.85))
        cursor_y = always_redraw(lambda: cursor(-1.28))
        bunch_label = make_text(
            "поперечное сечение пучка",
            color=colors["muted"],
            scale=0.28,
        ).move_to([0.0, -3.25, 0.0])

        def bunch_cross_section() -> VGroup:
            position = progress.get_value() * lattice.total_length
            sigma_x = lattice.sigma(position, "x")
            sigma_y = lattice.sigma(position, "y")
            ellipse = Ellipse(
                width=2.60 * sigma_x,
                height=2.60 * sigma_y,
                stroke_color=colors["proton"],
                stroke_width=2.4,
                fill_color=colors["proton"],
                fill_opacity=0.13,
            ).move_to([0.0, -4.05, 0.0])
            center = Dot([0.0, -4.05, 0.0], radius=0.055, color=colors["proton"])
            return VGroup(ellipse, center)

        bunch = always_redraw(bunch_cross_section)
        self.add(cursor_x, cursor_y, bunch_label, bunch)
        self.play(
            progress.animate.set_value(1.0),
            run_time=TIMING["lattice_flight"],
            rate_func=linear,
        )

        bounded = make_text(
            "пучок «дышит», но не разбегается",
            color=colors["text"],
            scale=0.38,
            weight="BOLD",
        ).move_to([0.0, -5.15, 0.0])
        strong_focusing = make_text(
            "СИЛЬНАЯ ФОКУСИРОВКА",
            color=colors["field"],
            scale=0.54,
            weight="BOLD",
        ).move_to([0.0, -5.85, 0.0])
        self.play(
            FadeIn(bounded, shift=0.10 * UP),
            FadeIn(strong_focusing, shift=0.10 * UP),
            run_time=TIMING["lattice_message"],
        )
        self.wait(TIMING["tail"])
