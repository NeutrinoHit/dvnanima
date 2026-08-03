from __future__ import annotations

import configparser
from pathlib import Path

import numpy as np
from manim import *

try:
    from .magnetic_optics import CFG, make_text
    from .quadrupole import AlternatingGradientLattice
    from .ring_model import (
        SymplecticRingTracker,
        TrackingResult,
        generate_momentum_spectrum,
    )
    from .sextupole import ChromaticFocusModel
except ImportError:
    from magnetic_optics import CFG, make_text
    from quadrupole import AlternatingGradientLattice
    from ring_model import SymplecticRingTracker, TrackingResult, generate_momentum_spectrum
    from sextupole import ChromaticFocusModel


BASE_DIR = Path(__file__).resolve().parent
RING_CENTER = np.array([0.0, 1.35, 0.0])
RING_RADIUS = 2.25
MOMENTA = (
    (-0.12, "p < p₀", "momentum"),
    (0.0, "p = p₀", "proton"),
    (0.12, "p > p₀", "field"),
)


def load_timing() -> dict[str, float]:
    parser = configparser.ConfigParser()
    parser.read(BASE_DIR / "run.cfg")
    defaults = {
        "intro": 1.3,
        "component_build": 1.0,
        "component_hold": 1.0,
        "transition": 0.8,
        "dipole_build": 1.3,
        "dipole_flight": 6.0,
        "dipole_hold": 1.2,
        "quadrupole_build": 1.3,
        "quadrupole_explain": 1.8,
        "quadrupole_scan": 6.0,
        "quadrupole_hold": 1.3,
        "chromatic_build": 1.3,
        "chromatic_flight": 4.8,
        "correction_build": 1.3,
        "correction_flight": 4.8,
        "correction_hold": 1.5,
        "ring_build": 1.5,
        "ring_arc": 2.5,
        "ring_magnet": 1.2,
        "ring_finish": 4.0,
        "summary": 1.3,
        "tail": 2.0,
    }
    return {
        key: parser.getfloat("ring_timing", key, fallback=value)
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


def momentum_color(deviation: float, colors: dict, half_width: float = 0.18):
    clipped = float(np.clip(deviation, -half_width, half_width))
    if clipped < 0.0:
        return interpolate_color(
            ManimColor(colors["momentum"]),
            ManimColor(colors["proton"]),
            (clipped + half_width) / half_width,
        )
    return interpolate_color(
        ManimColor(colors["proton"]),
        ManimColor(colors["field"]),
        clipped / half_width,
    )


def make_header(colors: dict) -> tuple[VGroup, Text]:
    kicker = make_text(
        "МАГНИТНАЯ ОПТИКА УСКОРИТЕЛЯ",
        color=colors["muted"],
        scale=0.32,
        weight="BOLD",
    ).move_to([0.0, 7.35, 0.0])
    title = make_text(
        "Как работает накопительное кольцо",
        color=colors["text"],
        scale=0.60,
        weight="BOLD",
    ).move_to([0.0, 6.62, 0.0])
    section = make_text(
        "сначала разберём каждый магнит",
        color=colors["field"],
        scale=0.36,
        weight="BOLD",
    ).move_to([0.0, 5.94, 0.0])
    return VGroup(kicker, title), section


def make_ring_lattice(
    colors: dict,
    half_cells: int,
) -> tuple[VGroup, Circle, VGroup, VGroup, VGroup]:
    orbit = Circle(
        radius=RING_RADIUS,
        stroke_color=colors["beam"],
        stroke_width=2.0,
        stroke_opacity=0.46,
    ).move_to(RING_CENTER)
    aperture = VGroup(
        Circle(
            radius=RING_RADIUS + 0.47,
            stroke_color=colors["off"],
            stroke_width=1.0,
            stroke_opacity=0.26,
        ),
        Circle(
            radius=RING_RADIUS - 0.47,
            stroke_color=colors["off"],
            stroke_width=1.0,
            stroke_opacity=0.26,
        ),
    ).move_to(RING_CENTER)

    segment = TAU / half_cells
    gap = 14 * DEGREES
    dipoles = VGroup()
    for index in range(half_cells):
        start = -PI / 2 + index * segment + gap / 2
        dipole = AnnularSector(
            inner_radius=RING_RADIUS - 0.14,
            outer_radius=RING_RADIUS + 0.14,
            angle=segment - gap,
            start_angle=start,
            fill_color=colors["magnet_edge"],
            fill_opacity=0.88,
            stroke_color=colors["field"],
            stroke_width=1.4,
        ).shift(RING_CENTER)
        dipoles.add(dipole)

    quadrupoles = VGroup()
    sextupoles = VGroup()
    for index in range(half_cells):
        angle = -PI / 2 + index * segment
        direction = np.array([np.cos(angle), np.sin(angle), 0.0])
        quad_color = colors["field"] if index % 2 else colors["momentum"]
        quadrupole = RoundedRectangle(
            width=0.72,
            height=0.34,
            corner_radius=0.09,
            stroke_color=quad_color,
            stroke_width=2.2,
            fill_color=quad_color,
            fill_opacity=0.30,
        )
        quadrupole.rotate(angle + PI / 2)
        quadrupole.move_to(RING_CENTER + RING_RADIUS * direction)
        quadrupoles.add(quadrupole)

        sextupole = RoundedRectangle(
            width=0.46,
            height=0.23,
            corner_radius=0.07,
            stroke_color=colors["force"],
            stroke_width=2.0,
            fill_color=colors["force"],
            fill_opacity=0.58,
        )
        sextupole.rotate(angle + PI / 2)
        sextupole.move_to(RING_CENTER + (RING_RADIUS + 0.40) * direction)
        sextupoles.add(sextupole)

    lattice = VGroup(aperture, orbit, dipoles, quadrupoles, sextupoles)
    return lattice, orbit, dipoles, quadrupoles, sextupoles


def make_component_card(
    number: str,
    title: str,
    explanation: str,
    color: str,
    colors: dict,
) -> VGroup:
    number_circle = Circle(
        radius=0.28,
        stroke_color=color,
        stroke_width=1.8,
        fill_color=color,
        fill_opacity=0.14,
    )
    number_text = make_text(number, color=color, scale=0.30, weight="BOLD")
    badge = VGroup(number_circle, number_text)
    heading = make_text(title, color=color, scale=0.38, weight="BOLD")
    body = make_text(explanation, color=colors["text"], scale=0.29)
    words = VGroup(heading, body).arrange(DOWN, aligned_edge=LEFT, buff=0.10)
    content = VGroup(badge, words).arrange(RIGHT, buff=0.28)
    plate = RoundedRectangle(
        width=7.35,
        height=1.28,
        corner_radius=0.18,
        stroke_color=color,
        stroke_width=1.3,
        fill_color=color,
        fill_opacity=0.07,
    )
    content.move_to(plate.get_center())
    return VGroup(plate, content).move_to([0.0, -2.10, 0.0])


def make_lattice_legend(colors: dict, y: float) -> VGroup:
    def item(marker: Mobject, label: str) -> VGroup:
        text = make_text(label, color=colors["text"], scale=0.24, weight="BOLD")
        return VGroup(marker, text).arrange(RIGHT, buff=0.10)

    dipole_marker = AnnularSector(
        inner_radius=0.18,
        outer_radius=0.28,
        angle=70 * DEGREES,
        start_angle=-35 * DEGREES,
        fill_color=colors["magnet_edge"],
        fill_opacity=0.95,
        stroke_color=colors["field"],
        stroke_width=1.0,
    )
    quad_marker = RoundedRectangle(
        width=0.40,
        height=0.20,
        corner_radius=0.05,
        stroke_color=colors["momentum"],
        fill_color=colors["field"],
        fill_opacity=0.45,
    )
    sext_marker = RoundedRectangle(
        width=0.34,
        height=0.18,
        corner_radius=0.05,
        stroke_color=colors["force"],
        fill_color=colors["force"],
        fill_opacity=0.65,
    )
    return VGroup(
        item(dipole_marker, "ДИПОЛИ"),
        item(quad_marker, "КВАДРУПОЛИ"),
        item(sext_marker, "СЕКСТУПОЛИ"),
    ).arrange(RIGHT, buff=0.42).move_to([0.0, y, 0.0])


def make_field_cross(position: np.ndarray, color: str) -> VGroup:
    ring = Circle(radius=0.13, stroke_color=color, stroke_width=1.5)
    diagonal_one = Line(0.065 * UL, 0.065 * DR, color=color, stroke_width=1.7)
    diagonal_two = Line(0.065 * UR, 0.065 * DL, color=color, stroke_width=1.7)
    return VGroup(ring, diagonal_one, diagonal_two).move_to(position)


def make_dipole_path(offset: float, colors: dict) -> VMobject:
    center = np.array([-1.15, 0.55, 0.0])
    radius = 1.42 + offset
    line_in = [
        np.array([x, center[1] - radius, 0.0])
        for x in np.linspace(-3.45, center[0], 45)
    ]
    arc = [
        center + radius * np.array([np.cos(angle), np.sin(angle), 0.0])
        for angle in np.linspace(-PI / 2, 0.0, 90)
    ]
    line_out = [
        np.array([center[0] + radius, y, 0.0])
        for y in np.linspace(center[1], 2.75, 45)
    ]
    path = VMobject().set_points_as_corners(line_in + arc + line_out)
    path.set_stroke(colors["beam"], width=2.4, opacity=0.42)
    return path


def make_dipole_demonstration(colors: dict) -> tuple[VGroup, list[VMobject], VGroup]:
    center = np.array([-1.15, 0.55, 0.0])
    magnet = AnnularSector(
        inner_radius=1.05,
        outer_radius=1.80,
        angle=PI / 2,
        start_angle=-PI / 2,
        fill_color=colors["magnet_edge"],
        fill_opacity=0.42,
        stroke_color=colors["field"],
        stroke_width=2.0,
    ).shift(center)
    magnet_label = make_text(
        "ДИПОЛЬ",
        color=colors["field"],
        scale=0.32,
        weight="BOLD",
    ).move_to([-0.20, -0.10, 0.0])

    field_symbols = VGroup()
    for angle in (-68, -45, -22):
        theta = angle * DEGREES
        position = center + 1.48 * np.array([np.cos(theta), np.sin(theta), 0.0])
        field_symbols.add(make_field_cross(position, colors["field"]))
    field_label = VGroup(
        make_text("поле в экран", color=colors["muted"], scale=0.25),
        MathTex(r"|\mathbf B|=\mathrm{const}", color=colors["field"]).scale(0.45),
    ).arrange(DOWN, buff=0.06).move_to([-2.52, 1.08, 0.0])
    equation = MathTex(
        r"\mathbf F=q\,\mathbf v\times\mathbf B",
        color=colors["text"],
    ).scale(0.64).move_to([0.0, 4.22, 0.0])

    velocity_in = Arrow(
        [-3.15, -1.13, 0.0],
        [-2.35, -1.13, 0.0],
        buff=0.0,
        color=colors["proton"],
        stroke_width=3.0,
    )
    velocity_in_label = MathTex(r"\mathbf v", color=colors["proton"]).scale(0.45)
    velocity_in_label.next_to(velocity_in, UP, buff=0.08)
    force = Arrow(
        [-1.15, -0.83, 0.0],
        [-1.15, -0.15, 0.0],
        buff=0.0,
        color=colors["force"],
        stroke_width=3.0,
    )
    force_label = MathTex(r"\mathbf F", color=colors["force"]).scale(0.45)
    force_label.next_to(force, RIGHT, buff=0.08)
    velocity_out = Arrow(
        [0.27, 1.80, 0.0],
        [0.27, 2.55, 0.0],
        buff=0.0,
        color=colors["proton"],
        stroke_width=3.0,
    )
    velocity_out_label = MathTex(r"\mathbf v", color=colors["proton"]).scale(0.45)
    velocity_out_label.next_to(velocity_out, RIGHT, buff=0.08)

    paths = [make_dipole_path(offset, colors) for offset in (-0.12, -0.06, 0.0, 0.06, 0.12)]
    particles = VGroup(
        *[
            Dot(path.get_start(), radius=0.065, color=colors["beam"]).set_z_index(8)
            for path in paths
        ]
    )
    static = VGroup(
        magnet,
        magnet_label,
        field_symbols,
        field_label,
        equation,
        velocity_in,
        velocity_in_label,
        force,
        force_label,
        velocity_out,
        velocity_out_label,
    )
    geometry = VGroup(
        magnet,
        magnet_label,
        field_symbols,
        field_label,
        velocity_in,
        velocity_in_label,
        force,
        force_label,
        velocity_out,
        velocity_out_label,
        *paths,
        particles,
    )
    geometry.scale(1.35, about_point=center).shift(0.20 * DOWN)
    return static, paths, particles


def lattice_screen_x(longitudinal: float, lattice: AlternatingGradientLattice) -> float:
    return -3.05 + 6.10 * longitudinal / lattice.total_length


def make_fodo_panel(
    center_y: float,
    plane: str,
    color: str,
    colors: dict,
    lattice: AlternatingGradientLattice,
) -> tuple[VGroup, VGroup, VGroup, VGroup]:
    frame = RoundedRectangle(
        width=7.55,
        height=2.18,
        corner_radius=0.16,
        stroke_color=colors["magnet_edge"],
        stroke_width=1.0,
        fill_color=colors["background"],
        fill_opacity=0.46,
    ).move_to([0.0, center_y, 0.0])
    title = make_text(
        "ВИД СВЕРХУ: ПЛОСКОСТЬ x" if plane == "x" else "ВИД СБОКУ: ПЛОСКОСТЬ y",
        color=color,
        scale=0.27,
        weight="BOLD",
    ).move_to([0.0, center_y + 0.86, 0.0])
    axis = Line(
        [-3.36, center_y, 0.0],
        [3.36, center_y, 0.0],
        color=colors["muted"],
        stroke_width=1.0,
        stroke_opacity=0.38,
    )
    focus_x = VGroup()
    focus_y = VGroup()
    magnets = VGroup()
    for index in range(2 * lattice.cells + 1):
        position = index * lattice.drift_length
        x_position = lattice_screen_x(position, lattice)
        focuses_x = index % 2 == 0
        magnet_color = colors["field"] if focuses_x else colors["momentum"]
        block = RoundedRectangle(
            width=0.30,
            height=1.45,
            corner_radius=0.07,
            stroke_color=magnet_color,
            stroke_width=1.7,
            fill_color=magnet_color,
            fill_opacity=0.16,
        ).move_to([x_position, center_y, 0.0])
        label = MathTex("x" if focuses_x else "y", color=magnet_color).scale(0.37)
        label.move_to([x_position, center_y + 0.51, 0.0])
        pair = VGroup(block, label)
        magnets.add(pair)
        (focus_x if focuses_x else focus_y).add(pair)

    envelope = lattice.sample_envelope(plane)
    upper_points = [
        np.array(
            [
                lattice_screen_x(point[0], lattice),
                center_y + 1.18 * point[1],
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
    upper.set_stroke(color, width=3.0, opacity=0.95)
    lower.set_stroke(color, width=3.0, opacity=0.95)
    fill = VMobject().set_points_as_corners(upper_points + list(reversed(lower_points)))
    fill.close_path()
    fill.set_fill(color, opacity=0.12).set_stroke(width=0)
    static = VGroup(frame, title, axis, magnets)
    envelope_group = VGroup(fill, upper, lower)
    return static, envelope_group, focus_x, focus_y


def make_chromatic_path(
    model: ChromaticFocusModel,
    center_y: float,
    offset: float,
    deviation: float,
    corrected: bool,
    color: str,
) -> VMobject:
    points = model.sample_ray(offset, deviation, corrected=corrected)
    display = [np.array([point[0], center_y + point[1], 0.0]) for point in points]
    path = VMobject().set_points_as_corners(display)
    path.set_stroke(color, width=2.6, opacity=0.82)
    return path


def make_chromatic_panel(
    model: ChromaticFocusModel,
    center_y: float,
    corrected: bool,
    colors: dict,
) -> tuple[VGroup, VGroup, VGroup, VGroup]:
    frame = RoundedRectangle(
        width=7.55,
        height=2.70,
        corner_radius=0.16,
        stroke_color=colors["force"] if corrected else colors["momentum"],
        stroke_width=1.1,
        fill_color=colors["background"],
        fill_opacity=0.46,
    ).move_to([0.0, center_y, 0.0])
    title = make_text(
        "КВАДРУПОЛЬ + СЕКСТУПОЛЬ: ОБЩИЙ ФОКУС"
        if corrected
        else "ТОЛЬКО КВАДРУПОЛЬ: РАЗНЫЕ ФОКУСЫ",
        color=colors["force"] if corrected else colors["momentum"],
        scale=0.27,
        weight="BOLD",
    ).move_to([0.0, center_y + 1.08, 0.0])
    axis = Line(
        [-3.45, center_y, 0.0],
        [3.32, center_y, 0.0],
        color=colors["muted"],
        stroke_width=1.0,
        stroke_opacity=0.38,
    )
    quadrupole = RoundedRectangle(
        width=0.38,
        height=1.75,
        corner_radius=0.08,
        stroke_color=colors["field"],
        stroke_width=1.8,
        fill_color=colors["field"],
        fill_opacity=0.18,
    ).move_to([model.quadrupole_position, center_y, 0.0])
    q_label = make_text(
        "КВАДРУПОЛЬ",
        color=colors["field"],
        scale=0.18,
        weight="BOLD",
    ).move_to([model.quadrupole_position, center_y + 0.73, 0.0])
    elements = VGroup(quadrupole, q_label)
    sextupole_group = VGroup()
    if corrected:
        sextupole = RoundedRectangle(
            width=0.38,
            height=1.75,
            corner_radius=0.08,
            stroke_color=colors["force"],
            stroke_width=1.8,
            fill_color=colors["force"],
            fill_opacity=0.20,
        ).move_to([model.sextupole_position, center_y, 0.0])
        s_label = make_text(
            "СЕКСТУПОЛЬ",
            color=colors["force"],
            scale=0.18,
            weight="BOLD",
        ).move_to([model.sextupole_position, center_y - 0.73, 0.0])
        sextupole_group.add(sextupole, s_label)
        elements.add(sextupole_group)

    paths = VGroup()
    focus_marks = VGroup()
    for deviation, _, color_key in MOMENTA:
        color = colors[color_key]
        for initial_offset in (-0.56, 0.56):
            paths.add(
                make_chromatic_path(
                    model,
                    center_y,
                    initial_offset,
                    deviation,
                    corrected,
                    color,
                )
            )
        focus_x = model.focal_position(deviation, corrected=corrected)
        focus_marks.add(Dot([focus_x, center_y, 0.0], radius=0.070, color=color))
    static = VGroup(frame, title, axis, elements)
    return static, paths, focus_marks, sextupole_group


def make_momentum_legend(colors: dict, y: float) -> VGroup:
    items = VGroup()
    for _, label, color_key in MOMENTA:
        dot = Dot(radius=0.065, color=colors[color_key])
        text = make_text(label, color=colors["text"], scale=0.25)
        items.add(VGroup(dot, text).arrange(RIGHT, buff=0.10))
    return items.arrange(RIGHT, buff=0.52).move_to([0.0, y, 0.0])


def snapshot(result: TrackingResult, sample_value: float) -> tuple[np.ndarray, np.ndarray]:
    maximum = result.states.shape[0] - 1
    value = float(np.clip(sample_value, 0.0, maximum))
    lower = min(int(np.floor(value)), maximum)
    upper = min(lower + 1, maximum)
    fraction = value - lower
    states = (1.0 - fraction) * result.states[lower] + fraction * result.states[upper]
    alive = result.alive[lower].copy()
    return states, alive


def make_ring_cloud(
    result: TrackingResult,
    progress: ValueTracker,
    colors: dict,
) -> VGroup:
    states, alive = snapshot(result, progress.get_value())
    phase = progress.get_value() / result.config.samples_per_turn
    angle = -PI / 2 + TAU * (phase % 1.0)
    radial = np.array([np.cos(angle), np.sin(angle), 0.0])
    tangent = np.array([-np.sin(angle), np.cos(angle), 0.0])
    cloud = VGroup()
    indices = np.linspace(0, states.shape[0] - 1, 25, dtype=int)
    for order, index in enumerate(indices):
        if not alive[index]:
            continue
        state = states[index]
        physical_x = state[0] + result.config.sextupole_dispersion * state[4]
        radial_shift = 0.38 * physical_x / result.config.aperture
        longitudinal_shift = 0.018 * (order - len(indices) / 2)
        position = (
            RING_CENTER
            + (RING_RADIUS + radial_shift) * radial
            + longitudinal_shift * tangent
        )
        cloud.add(
            Dot(
                position,
                radius=0.055,
                color=momentum_color(state[4], colors),
            ).set_z_index(10)
        )
    return cloud


def make_beam_section(
    result: TrackingResult,
    progress: ValueTracker,
    colors: dict,
) -> VGroup:
    states, alive = snapshot(result, progress.get_value())
    current = states[alive]
    sigma_x = float(np.std(current[:, 0])) if current.size else 0.0
    sigma_y = float(np.std(current[:, 2])) if current.size else 0.0
    first_turn = result.states[: result.config.samples_per_turn + 1]
    scale_x = max(float(np.max(np.std(first_turn[:, :, 0], axis=1))), 1.0e-6)
    scale_y = max(float(np.max(np.std(first_turn[:, :, 2], axis=1))), 1.0e-6)
    width = 0.55 + 1.25 * sigma_x / scale_x
    height = 0.55 + 1.25 * sigma_y / scale_y
    ellipse = Ellipse(
        width=width,
        height=height,
        stroke_color=colors["beam"],
        stroke_width=2.0,
        fill_color=colors["beam"],
        fill_opacity=0.14,
    ).move_to([0.0, -2.65, 0.0])
    axes = VGroup(
        Line([-1.12, -2.65, 0.0], [1.12, -2.65, 0.0], color=colors["muted"], stroke_width=0.8, stroke_opacity=0.34),
        Line([0.0, -3.65, 0.0], [0.0, -1.65, 0.0], color=colors["muted"], stroke_width=0.8, stroke_opacity=0.34),
    )
    return VGroup(axes, ellipse)


def make_action_card(content: str, color: str, colors: dict) -> VGroup:
    label = make_text(content, color=color, scale=0.31, weight="BOLD")
    plate = RoundedRectangle(
        width=label.width + 0.62,
        height=0.60,
        corner_radius=0.16,
        stroke_color=color,
        stroke_width=1.3,
        fill_color=color,
        fill_opacity=0.08,
    )
    return VGroup(plate, label).move_to([0.0, -1.05, 0.0])


def make_summary_bars(colors: dict) -> VGroup:
    title = make_text(
        "ОДИН И ТОТ ЖЕ ПУЧОК • 18 ОБОРОТОВ",
        color=colors["muted"],
        scale=0.27,
        weight="BOLD",
    )

    def row(label: str, fraction: float, color: str) -> VGroup:
        label_text = make_text(label, color=colors["text"], scale=0.25)
        label_text.set_width(min(label_text.width, 2.55))
        back = RoundedRectangle(
            width=3.55,
            height=0.27,
            corner_radius=0.08,
            stroke_color=colors["off"],
            stroke_width=1.0,
            fill_color=colors["off"],
            fill_opacity=0.10,
        )
        fill = RoundedRectangle(
            width=3.55 * fraction,
            height=0.27,
            corner_radius=0.08,
            stroke_width=0,
            fill_color=color,
            fill_opacity=0.78,
        )
        fill.align_to(back, LEFT)
        percent = make_text(
            f"{round(100 * fraction)}%",
            color=color,
            scale=0.29,
            weight="BOLD",
        )
        return VGroup(label_text, VGroup(back, fill), percent).arrange(RIGHT, buff=0.18)

    without = row("без секступолей", 46 / 72, colors["momentum"])
    with_correction = row("с секступолями", 1.0, colors["force"])
    return VGroup(title, without, with_correction).arrange(DOWN, buff=0.22).move_to([0.0, -4.45, 0.0])


class StorageRingScene(Scene):
    def construct(self) -> None:
        colors = CFG["colors"]
        self.camera.background_color = colors["background"]
        self.add(make_background(colors))

        tracker = SymplecticRingTracker()
        initial_states = generate_momentum_spectrum()
        without_sextupoles = tracker.track(initial_states, turns=18, sextupoles_on=False)
        with_sextupoles = tracker.track(initial_states, turns=18, sextupoles_on=True)

        header, section = make_header(colors)
        self.play(
            FadeIn(header, shift=0.14 * UP),
            FadeIn(section),
            run_time=TIMING["intro"],
        )

        lattice, orbit, dipoles, quadrupoles, sextupoles = make_ring_lattice(
            colors,
            tracker.config.samples_per_turn,
        )
        self.play(
            FadeIn(lattice[0]),
            Create(orbit),
            run_time=TIMING["component_build"],
        )

        dipole_card = make_component_card(
            "1",
            "ДИПОЛИ",
            "поворачивают траекторию и замыкают орбиту",
            colors["field"],
            colors,
        )
        self.play(
            LaggedStart(*[GrowFromCenter(magnet) for magnet in dipoles], lag_ratio=0.10),
            FadeIn(dipole_card, shift=0.12 * UP),
            run_time=TIMING["component_build"],
        )
        self.wait(TIMING["component_hold"])

        quadrupole_card = make_component_card(
            "2",
            "КВАДРУПОЛИ",
            "поочерёдно фокусируют пучок в плоскостях x и y",
            colors["momentum"],
            colors,
        )
        self.play(
            dipoles.animate.set_opacity(0.36),
            LaggedStart(*[FadeIn(magnet, scale=0.75) for magnet in quadrupoles], lag_ratio=0.10),
            Transform(dipole_card, quadrupole_card),
            run_time=TIMING["component_build"],
        )
        self.wait(TIMING["component_hold"])

        sextupole_card = make_component_card(
            "3",
            "СЕКСТУПОЛИ",
            "исправляют зависимость фокусировки от импульса",
            colors["force"],
            colors,
        )
        self.play(
            quadrupoles.animate.set_opacity(0.36),
            LaggedStart(*[FadeIn(magnet, scale=0.75) for magnet in sextupoles], lag_ratio=0.10),
            Transform(dipole_card, sextupole_card),
            run_time=TIMING["component_build"],
        )
        self.wait(TIMING["component_hold"])

        legend = make_lattice_legend(colors, -3.15)
        assembly_note = make_text(
            "в кольце работают все три системы",
            color=colors["text"],
            scale=0.34,
            weight="BOLD",
        ).move_to([0.0, -3.82, 0.0])
        self.play(
            dipoles.animate.set_opacity(1.0),
            quadrupoles.animate.set_opacity(1.0),
            FadeOut(dipole_card),
            FadeIn(legend),
            FadeIn(assembly_note, shift=0.10 * UP),
            run_time=TIMING["component_build"],
        )
        self.wait(TIMING["component_hold"])

        dipole_section = make_text(
            "1. Диполь: как возникает поворот",
            color=colors["field"],
            scale=0.36,
            weight="BOLD",
        ).move_to(section)
        self.play(
            FadeOut(VGroup(lattice, legend, assembly_note)),
            Transform(section, dipole_section),
            run_time=TIMING["transition"],
        )

        dipole_static, dipole_paths, dipole_particles = make_dipole_demonstration(colors)
        self.play(
            FadeIn(dipole_static, shift=0.08 * UP),
            FadeIn(dipole_particles),
            run_time=TIMING["dipole_build"],
        )
        self.play(
            *[Create(path) for path in dipole_paths],
            *[
                MoveAlongPath(particle, path, rate_func=linear)
                for particle, path in zip(dipole_particles, dipole_paths)
            ],
            run_time=TIMING["dipole_flight"],
        )
        dipole_note = make_text(
            "скорость меняет направление — частицы идут по дуге",
            color=colors["text"],
            scale=0.34,
            weight="BOLD",
        ).move_to([0.0, -3.55, 0.0])
        self.play(FadeIn(dipole_note, shift=0.10 * UP), run_time=0.6)
        self.wait(TIMING["dipole_hold"])

        quadrupole_section = make_text(
            "2. Квадруполи: фокусировка в двух плоскостях",
            color=colors["momentum"],
            scale=0.34,
            weight="BOLD",
        ).move_to(section)
        self.play(
            FadeOut(VGroup(dipole_static, *dipole_paths, dipole_particles, dipole_note)),
            Transform(section, quadrupole_section),
            run_time=TIMING["transition"],
        )

        alternating_lattice = AlternatingGradientLattice(cells=2)
        panel_x, envelope_x, focus_x_top, focus_y_top = make_fodo_panel(
            1.70,
            "x",
            colors["field"],
            colors,
            alternating_lattice,
        )
        panel_y, envelope_y, focus_x_bottom, focus_y_bottom = make_fodo_panel(
            -1.00,
            "y",
            colors["momentum"],
            colors,
            alternating_lattice,
        )
        self.play(FadeIn(panel_x), FadeIn(panel_y), run_time=TIMING["quadrupole_build"])

        first_quad_note = make_text(
            "КВАДРУПОЛЬ x: сжимает по x, расширяет по y",
            color=colors["field"],
            scale=0.31,
            weight="BOLD",
        ).move_to([0.0, -2.65, 0.0])
        self.play(
            Indicate(VGroup(focus_x_top, focus_x_bottom), color=colors["field"], scale_factor=1.07),
            FadeIn(first_quad_note),
            run_time=TIMING["quadrupole_explain"],
        )
        second_quad_note = make_text(
            "СЛЕДУЮЩИЙ КВАДРУПОЛЬ: наоборот",
            color=colors["momentum"],
            scale=0.31,
            weight="BOLD",
        ).move_to(first_quad_note)
        self.play(
            Indicate(VGroup(focus_y_top, focus_y_bottom), color=colors["momentum"], scale_factor=1.07),
            Transform(first_quad_note, second_quad_note),
            run_time=TIMING["quadrupole_explain"],
        )

        scan_progress = ValueTracker(0.0)
        scan_x = always_redraw(
            lambda: Line(
                [lattice_screen_x(scan_progress.get_value(), alternating_lattice), 0.83, 0.0],
                [lattice_screen_x(scan_progress.get_value(), alternating_lattice), 2.52, 0.0],
                color=colors["proton"],
                stroke_width=2.0,
            )
        )
        scan_y = always_redraw(
            lambda: Line(
                [lattice_screen_x(scan_progress.get_value(), alternating_lattice), -1.87, 0.0],
                [lattice_screen_x(scan_progress.get_value(), alternating_lattice), -0.18, 0.0],
                color=colors["proton"],
                stroke_width=2.0,
            )
        )
        section_label = make_text(
            "ПОПЕРЕЧНОЕ СЕЧЕНИЕ ПУЧКА",
            color=colors["muted"],
            scale=0.23,
            weight="BOLD",
        ).move_to([0.0, -3.30, 0.0])
        beam_ellipse = always_redraw(
            lambda: Ellipse(
                width=0.60
                + 1.25
                * alternating_lattice.sigma(scan_progress.get_value(), "x")
                / max(alternating_lattice.sigma(value, "x") for value in np.linspace(0.0, alternating_lattice.total_length, 40)),
                height=0.60
                + 1.25
                * alternating_lattice.sigma(scan_progress.get_value(), "y")
                / max(alternating_lattice.sigma(value, "y") for value in np.linspace(0.0, alternating_lattice.total_length, 40)),
                stroke_color=colors["beam"],
                stroke_width=2.0,
                fill_color=colors["beam"],
                fill_opacity=0.14,
            ).move_to([0.0, -4.15, 0.0])
        )
        self.add(scan_x, scan_y, beam_ellipse)
        self.play(
            FadeOut(first_quad_note),
            FadeIn(section_label),
            FadeIn(envelope_x[0]),
            FadeIn(envelope_y[0]),
            Create(envelope_x[1]),
            Create(envelope_x[2]),
            Create(envelope_y[1]),
            Create(envelope_y[2]),
            scan_progress.animate.set_value(alternating_lattice.total_length),
            run_time=TIMING["quadrupole_scan"],
            rate_func=linear,
        )
        quadrupole_note = make_text(
            "чередующаяся цепочка удерживает пучок по x и по y",
            color=colors["text"],
            scale=0.32,
            weight="BOLD",
        ).move_to([0.0, -5.10, 0.0])
        self.play(FadeIn(quadrupole_note, shift=0.10 * UP), run_time=0.6)
        self.wait(TIMING["quadrupole_hold"])

        chromatic_section = make_text(
            "3. Зачем к квадруполям добавляют секступоли",
            color=colors["force"],
            scale=0.34,
            weight="BOLD",
        ).move_to(section)
        quadrupole_act = VGroup(
            panel_x,
            panel_y,
            envelope_x,
            envelope_y,
            scan_x,
            scan_y,
            beam_ellipse,
            section_label,
            quadrupole_note,
        )
        self.play(
            FadeOut(quadrupole_act),
            Transform(section, chromatic_section),
            run_time=TIMING["transition"],
        )

        chromatic_model = ChromaticFocusModel()
        chromatic_legend = make_momentum_legend(colors, 4.98)
        top_static, top_paths, top_focus, _ = make_chromatic_panel(
            chromatic_model,
            2.15,
            False,
            colors,
        )
        relation = MathTex(r"f\propto p", color=colors["momentum"]).scale(0.55)
        relation.move_to([0.0, 4.35, 0.0])
        self.play(
            FadeIn(chromatic_legend),
            FadeIn(relation),
            FadeIn(top_static),
            run_time=TIMING["chromatic_build"],
        )
        self.play(
            LaggedStart(*[Create(path) for path in top_paths], lag_ratio=0.04),
            run_time=TIMING["chromatic_flight"],
        )
        error_note = make_text(
            "разные импульсы → разные фокусы",
            color=colors["momentum"],
            scale=0.30,
            weight="BOLD",
        ).move_to([0.0, 0.52, 0.0])
        self.play(FadeIn(top_focus), FadeIn(error_note), run_time=0.6)

        bottom_static, bottom_paths, bottom_focus, bottom_sextupole = make_chromatic_panel(
            chromatic_model,
            -2.05,
            True,
            colors,
        )
        dispersion = VGroup(
            MathTex(r"x=D\delta", color=colors["field"]).scale(0.48),
            make_text("разные импульсы попадают в разные места", color=colors["muted"], scale=0.23),
        ).arrange(RIGHT, buff=0.24).move_to([0.0, -0.35, 0.0])
        self.play(
            FadeIn(bottom_static),
            FadeIn(dispersion),
            run_time=0.75 * TIMING["correction_build"],
        )
        self.play(
            Indicate(bottom_sextupole, color=colors["force"], scale_factor=1.10),
            run_time=0.75 * TIMING["correction_build"],
        )
        self.play(
            LaggedStart(*[Create(path) for path in bottom_paths], lag_ratio=0.04),
            run_time=TIMING["correction_flight"],
        )
        correction_note = make_text(
            "СЕКСТУПОЛЬ НЕ ЗАМЕНЯЕТ КВАДРУПОЛЬ — ОН ПОПРАВЛЯЕТ ЕГО ФОКУСИРОВКУ",
            color=colors["force"],
            scale=0.27,
            weight="BOLD",
        ).move_to([0.0, -3.72, 0.0])
        self.play(
            FadeIn(bottom_focus),
            FadeIn(correction_note, shift=0.10 * UP),
            run_time=0.7,
        )
        self.wait(TIMING["correction_hold"])

        ring_section = make_text(
            "4. Теперь проследим один оборот",
            color=colors["field"],
            scale=0.36,
            weight="BOLD",
        ).move_to(section)
        chromatic_act = VGroup(
            chromatic_legend,
            relation,
            top_static,
            top_paths,
            top_focus,
            error_note,
            bottom_static,
            bottom_paths,
            bottom_focus,
            dispersion,
            correction_note,
        )
        self.play(
            FadeOut(chromatic_act),
            Transform(section, ring_section),
            run_time=TIMING["transition"],
        )

        final_lattice, _, final_dipoles, final_quadrupoles, final_sextupoles = make_ring_lattice(
            colors,
            tracker.config.samples_per_turn,
        )
        final_legend = make_lattice_legend(colors, 4.98)
        progress = ValueTracker(0.0)
        ring_cloud = always_redraw(lambda: make_ring_cloud(with_sextupoles, progress, colors))
        beam_section = always_redraw(lambda: make_beam_section(with_sextupoles, progress, colors))
        beam_section_label = make_text(
            "ПОПЕРЕЧНОЕ СЕЧЕНИЕ ПУЧКА",
            color=colors["muted"],
            scale=0.23,
            weight="BOLD",
        ).move_to([0.0, -3.85, 0.0])
        action = make_action_card("ДИПОЛЬ: ПОВОРАЧИВАЕТ", colors["field"], colors)
        self.play(
            FadeIn(final_lattice),
            FadeIn(final_legend),
            FadeIn(action),
            FadeIn(beam_section_label),
            run_time=TIMING["ring_build"],
        )
        self.add(ring_cloud, beam_section)

        for segment_index in range(2):
            dipole_action = make_action_card(
                "ДИПОЛЬ: ПОВОРАЧИВАЕТ",
                colors["field"],
                colors,
            )
            if segment_index:
                self.play(FadeOut(action), FadeIn(dipole_action), run_time=0.25)
                action = dipole_action
            self.play(
                Indicate(final_dipoles[segment_index], color=colors["field"], scale_factor=1.04),
                progress.animate.set_value(segment_index + 0.78),
                run_time=TIMING["ring_arc"],
                rate_func=linear,
            )

            boundary = segment_index + 1
            focuses_x = boundary % 2 == 1
            quad_color = colors["field"] if focuses_x else colors["momentum"]
            quad_action = make_action_card(
                "КВАДРУПОЛЬ: ФОКУСИРУЕТ ПО x" if focuses_x else "КВАДРУПОЛЬ: ФОКУСИРУЕТ ПО y",
                quad_color,
                colors,
            )
            self.play(FadeOut(action), FadeIn(quad_action), run_time=0.25)
            action = quad_action
            self.play(
                Indicate(final_quadrupoles[boundary], color=quad_color, scale_factor=1.10),
                progress.animate.set_value(float(boundary)),
                run_time=TIMING["ring_magnet"],
            )

            sext_action = make_action_card(
                "СЕКСТУПОЛЬ: ИСПРАВЛЯЕТ ИМПУЛЬСНУЮ ОШИБКУ",
                colors["force"],
                colors,
            )
            self.play(FadeOut(action), FadeIn(sext_action), run_time=0.25)
            action = sext_action
            self.play(
                Indicate(final_sextupoles[boundary], color=colors["force"], scale_factor=1.15),
                run_time=TIMING["ring_magnet"],
            )

        finish_action = make_action_card(
            "ОСТАЛЬНАЯ ЧАСТЬ ОБОРОТА",
            colors["beam"],
            colors,
        )
        self.play(FadeOut(action), FadeIn(finish_action), run_time=0.25)
        action = finish_action
        self.play(
            progress.animate.set_value(float(tracker.config.samples_per_turn)),
            run_time=TIMING["ring_finish"],
            rate_func=linear,
        )

        completed = make_action_card("ОДИН ПОЛНЫЙ ОБОРОТ", colors["proton"], colors)
        self.play(FadeOut(action), FadeIn(completed), run_time=0.35)
        action = completed
        summary = make_summary_bars(colors)
        self.play(
            FadeOut(beam_section),
            FadeOut(beam_section_label),
            FadeOut(action),
            FadeIn(summary, shift=0.12 * UP),
            run_time=TIMING["summary"],
        )
        final_note = make_text(
            "ДИПОЛИ ПОВОРАЧИВАЮТ • КВАДРУПОЛИ ФОКУСИРУЮТ • СЕКСТУПОЛИ КОРРЕКТИРУЮТ",
            color=colors["text"],
            scale=0.27,
            weight="BOLD",
        ).move_to([0.0, -5.72, 0.0])
        self.play(FadeIn(final_note, shift=0.10 * UP), run_time=0.7)
        self.wait(TIMING["tail"])
