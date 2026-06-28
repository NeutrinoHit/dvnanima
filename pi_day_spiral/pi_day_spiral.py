from __future__ import annotations

from pathlib import Path
import configparser
import numpy as np
from manim import *

# Local source of pi digits for the short. This is deterministic and enough for the scene.
PI_DIGITS = (
    "314159265358979323846264338327950288419716939937510"
    "58209749445923078164062862089986280348253421170679"
    "82148086513282306647093844609550582231725359408128"
)


def load_cfg(path: Path) -> dict:
    cfg = configparser.ConfigParser(inline_comment_prefixes=(";",))
    cfg.read(path)

    def get(section, key, cast=str, fallback=None):
        if fallback is None:
            return cast(cfg[section][key])
        return cast(cfg.get(section, key, fallback=str(fallback)))

    def get_bool(section, key, fallback=False):
        return cfg.getboolean(section, key, fallback=fallback)

    manim_params = {
        "pixel_width": get("manim", "pixel_width", int, 1080),
        "pixel_height": get("manim", "pixel_height", int, 1920),
        "frame_width": get("manim", "frame_width", float, 9.0),
        "frame_height": get("manim", "frame_height", float, 16.0),
        "frame_rate": get("manim", "frame_rate", int, 60),
        "background_color": get("manim", "background_color", str, "#000000"),
    }

    scene = {
        "n_digits": get("scene", "n_digits", int, 84),
        "digit_font_size": get("scene", "digit_font_size", float, 34.0),
        "stream_spacing": get("scene", "stream_spacing", float, 0.42),
        "pi_symbol_scale": get("scene", "pi_symbol_scale", float, 24.0),
        "pi_shift_x": get("scene", "pi_shift_x", float, 0.0),
        "pi_shift_y": get("scene", "pi_shift_y", float, -1.05),
        "contour_start": get("scene", "contour_start", float, 0.08),
        "contour_stroke_width": get("scene", "contour_stroke_width", float, 5.0),
        "show_outline": get_bool("scene", "show_outline", True),
        "box_width": get("scene", "box_width", float, 2.6),
        "box_height": get("scene", "box_height", float, 1.55),
        "box_center_x": get("scene", "box_center_x", float, 0.0),
        "box_center_y": get("scene", "box_center_y", float, 5.65),
        "box_lid_height": get("scene", "box_lid_height", float, 0.46),
        "burst_width": get("scene", "burst_width", float, 1.4),
        "burst_height": get("scene", "burst_height", float, 1.4),
        "curve_noise": get("scene", "curve_noise", float, 1.1),
        "spiral_turns_min": get("scene", "spiral_turns_min", float, 1.6),
        "spiral_turns_max": get("scene", "spiral_turns_max", float, 3.8),
        "spiral_phase_step": get("scene", "spiral_phase_step", float, 0.42),
        "spiral_lift": get("scene", "spiral_lift", float, 1.1),
        "pop_scale": get("scene", "pop_scale", float, 1.65),
        "pop_fraction": get("scene", "pop_fraction", float, 0.18),
        "box_intro_time": get("scene", "box_intro_time", float, 1.6),
        "outline_draw_time": get("scene", "outline_draw_time", float, 1.8),
        "lid_open_time": get("scene", "lid_open_time", float, 0.9),
        "stream_time": get("scene", "stream_time", float, 21.0),
        "final_pulse_time": get("scene", "final_pulse_time", float, 3.4),
        "hold_time": get("scene", "hold_time", float, 1.3),
        "seed": get("scene", "seed", int, 314),
    }

    colors = {
        "digits": get("colors", "digits", str, "#F6F3EB"),
        "highlight": get("colors", "highlight", str, "#FFD166"),
        "outline": get("colors", "outline", str, "#FF5A5F"),
        "box": get("colors", "box", str, "#D9485F"),
        "lid": get("colors", "lid", str, "#F25F5C"),
        "ribbon": get("colors", "ribbon", str, "#FFD166"),
        "bow": get("colors", "bow", str, "#7FDBB6"),
        "spark": get("colors", "spark", str, "#B7E5FF"),
    }

    return {"manim": manim_params, "scene": scene, "colors": colors}


BASE_DIR = Path(__file__).resolve().parent
CFG = load_cfg(BASE_DIR / "run.cfg")

config.pixel_width = CFG["manim"]["pixel_width"]
config.pixel_height = CFG["manim"]["pixel_height"]
config.frame_width = CFG["manim"]["frame_width"]
config.frame_height = CFG["manim"]["frame_height"]
config.frame_rate = CFG["manim"]["frame_rate"]


def make_digit_prototypes(font_size: float, color: str) -> dict[str, Text]:
    prototypes: dict[str, Text] = {}
    for char in sorted(set(PI_DIGITS)):
        prototypes[char] = Text(char, font_size=font_size, color=color, weight=BOLD)
    return prototypes


def longest_path(mob: Mobject) -> VMobject:
    paths = [m for m in mob.family_members_with_points() if isinstance(m, VMobject) and m.get_num_points() > 0]
    if not paths:
        raise ValueError("No vector path found for pi glyph")
    return max(paths, key=lambda path: path.get_num_points())


def ordered_contour(path: VMobject, n_dense: int = 4000) -> np.ndarray:
    dense_props = np.linspace(0.0, 1.0, 4000, endpoint=False)
    dense = np.array([path.point_from_proportion(float(t)) for t in dense_props])
    ys = dense[:, 1]
    y_cut = np.max(ys) - 0.08 * (np.max(ys) - np.min(ys))
    top_band = np.where(ys >= y_cut)[0]
    if len(top_band) == 0:
        start_idx = 0
    else:
        start_idx = int(top_band[np.argmin(dense[top_band, 0])])

    look_ahead = (start_idx + 8) % len(dense)
    forward_dx = dense[look_ahead, 0] - dense[start_idx, 0]
    if forward_dx < 0:
        dense = dense[::-1]
        start_idx = len(dense) - 1 - start_idx

    dense = np.roll(dense, -start_idx, axis=0)
    return dense


def sample_contour_points(path: VMobject, n_points: int, start_prop: float) -> list[np.ndarray]:
    dense = ordered_contour(path)
    dense = np.vstack([dense, dense[0]])
    segment_lengths = np.linalg.norm(np.diff(dense, axis=0), axis=1)
    cumulative = np.concatenate([[0.0], np.cumsum(segment_lengths)])
    total_length = float(cumulative[-1])

    out: list[np.ndarray] = []
    for s in np.linspace(0.0, total_length, n_points, endpoint=False):
        idx = int(np.searchsorted(cumulative, s, side="right") - 1)
        idx = max(0, min(idx, len(segment_lengths) - 1))
        ds = max(float(segment_lengths[idx]), 1e-8)
        alpha = (s - cumulative[idx]) / ds
        out.append((1.0 - alpha) * dense[idx] + alpha * dense[idx + 1])
    return out


def subpath(path: VMobject, t0: float, t1: float, samples: int = 48) -> VMobject:
    curve = VMobject()
    curve.set_points_smoothly(
        [path.point_from_proportion(t) for t in np.linspace(t0, t1, samples)]
    )
    return curve


def spiral_flight_path(
    start: np.ndarray,
    end: np.ndarray,
    turns: float,
    phase: float,
    lift: float,
    samples: int = 120,
) -> VMobject:
    delta = end - start
    length = float(np.linalg.norm(delta[:2]))
    if length <= 1e-8:
        return Line(start, end)

    unit = delta / max(np.linalg.norm(delta), 1e-8)
    normal = np.array([-unit[1], unit[0], 0.0])
    r0 = 0.42 * length
    theta0 = phase
    initial_offset = r0 * (np.cos(theta0) * normal + 0.28 * np.sin(theta0) * unit)

    points: list[np.ndarray] = []
    for alpha in np.linspace(0.0, 1.0, samples):
        theta = phase + turns * TAU * alpha
        radius = r0 * (1.0 - alpha) ** 1.15
        swirl = radius * (np.cos(theta) * normal + 0.28 * np.sin(theta) * unit)
        lift_vec = lift * np.sin(PI * alpha) * UP
        points.append(start + alpha * delta + swirl - (1.0 - alpha) * initial_offset + lift_vec)

    path = VMobject()
    path.set_points_smoothly(points)
    return path


def guiding_spiral_path(
    start: np.ndarray,
    center: np.ndarray,
    end: np.ndarray,
    box_width: float,
    box_height: float,
    turns: float = 2.15,
    samples: int = 220,
) -> VMobject:
    w = box_width
    h = box_height
    points = [
        start,
        center + np.array([-0.05 * w, 0.92 * h, 0.0]),
        center + np.array([0.78 * w, 0.82 * h, 0.0]),
        center + np.array([0.98 * w, -0.10 * h, 0.0]),
        center + np.array([0.58 * w, -1.42 * h, 0.0]),
        center + np.array([-0.78 * w, -1.18 * h, 0.0]),
        center + np.array([-0.98 * w, 0.28 * h, 0.0]),
        center + np.array([-0.28 * w, 1.18 * h, 0.0]),
        center + np.array([0.92 * w, 1.32 * h, 0.0]),
        center + np.array([1.58 * w, 0.55 * h, 0.0]),
        center + np.array([1.58 * w, -0.72 * h, 0.0]),
        center + np.array([1.08 * w, -1.85 * h, 0.0]),
        center + np.array([-0.02 * w, -2.48 * h, 0.0]),
        end + np.array([0.65, 0.82, 0.0]),
        end + np.array([0.16, 0.12, 0.0]),
        end,
    ]

    path = VMobject()
    path.set_points_smoothly(points)
    return path


def prefix_path_from_points(points: np.ndarray, alpha: float) -> VMobject:
    alpha = float(np.clip(alpha, 0.0, 1.0))
    if alpha <= 0.0:
        return Line(points[0], points[0] + 1e-4 * RIGHT)

    dense = np.vstack([points, points[-1]])
    segment_lengths = np.linalg.norm(np.diff(dense, axis=0), axis=1)
    cumulative = np.concatenate([[0.0], np.cumsum(segment_lengths)])
    total_length = float(cumulative[-1])
    target_s = alpha * total_length
    idx = int(np.searchsorted(cumulative, target_s, side="right") - 1)
    idx = max(0, min(idx, len(segment_lengths) - 1))
    ds = max(float(segment_lengths[idx]), 1e-8)
    local_alpha = (target_s - cumulative[idx]) / ds
    end_point = (1.0 - local_alpha) * dense[idx] + local_alpha * dense[idx + 1]

    path_points = list(dense[: idx + 1])
    path_points.append(end_point)
    curve = VMobject()
    curve.set_points_smoothly(path_points)
    return curve


def cumulative_lengths(points: np.ndarray) -> np.ndarray:
    return np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))])


def point_at_distance(points: np.ndarray, lengths: np.ndarray, s: float) -> np.ndarray:
    s = float(np.clip(s, 0.0, lengths[-1]))
    idx = int(np.searchsorted(lengths, s, side="right") - 1)
    idx = max(0, min(idx, len(points) - 2))
    ds = max(float(lengths[idx + 1] - lengths[idx]), 1e-8)
    alpha = (s - lengths[idx]) / ds
    return (1.0 - alpha) * points[idx] + alpha * points[idx + 1]


def festive_box(center: np.ndarray, width: float, height: float, lid_height: float, colors: dict) -> tuple[VGroup, VMobject, np.ndarray]:
    base = RoundedRectangle(corner_radius=0.18, width=width, height=height, color=colors["box"], stroke_width=4)
    base.set_fill(colors["box"], opacity=0.28)
    base.move_to(center)

    lid = RoundedRectangle(corner_radius=0.16, width=1.06 * width, height=lid_height, color=colors["lid"], stroke_width=4)
    lid.set_fill(colors["lid"], opacity=0.35)
    lid.move_to(base.get_top() + 0.5 * lid_height * DOWN)

    ribbon_v = Line(base.get_top(), base.get_bottom(), color=colors["ribbon"], stroke_width=9)
    ribbon_h = Line(base.get_left(), base.get_right(), color=colors["ribbon"], stroke_width=9)

    bow_center = lid.get_top() + 0.04 * DOWN
    bow_left = ArcBetweenPoints(bow_center + 0.05 * LEFT, bow_center + 0.48 * LEFT + 0.14 * UP, angle=TAU / 3)
    bow_right = ArcBetweenPoints(bow_center + 0.05 * RIGHT, bow_center + 0.48 * RIGHT + 0.14 * UP, angle=-TAU / 3)
    bow_knot = Dot(bow_center, radius=0.08, color=colors["bow"])
    for part in (bow_left, bow_right):
        part.set_stroke(colors["bow"], width=6)

    box_group = VGroup(base, ribbon_v, ribbon_h, lid, bow_left, bow_right, bow_knot)
    opening = base.get_top() + 0.02 * DOWN
    return box_group, lid, opening


class PiDaySpiralShorts(Scene):
    def construct(self):
        p = CFG["scene"]
        c = CFG["colors"]
        self.camera.background_color = CFG["manim"]["background_color"]
        rng = np.random.default_rng(p["seed"])

        prototypes = make_digit_prototypes(p["digit_font_size"], c["digits"])
        digit_chars = PI_DIGITS[: p["n_digits"]]

        pi_outline = MathTex(r"\pi", color=c["outline"])
        pi_outline.scale(p["pi_symbol_scale"])
        pi_outline.shift(p["pi_shift_x"] * RIGHT + p["pi_shift_y"] * UP)
        pi_outline.set_fill(opacity=0.0)
        pi_outline.set_stroke(c["outline"], width=p["contour_stroke_width"], opacity=0.92)

        contour_path = longest_path(pi_outline)
        contour_dense = ordered_contour(contour_path)
        targets = sample_contour_points(contour_path, len(digit_chars), p["contour_start"])
        target_alphas = np.linspace(0.0, 1.0, len(digit_chars), endpoint=False)

        box_center = np.array([p["box_center_x"], p["box_center_y"], 0.0])
        gift_box, lid, opening = festive_box(box_center, p["box_width"], p["box_height"], p["box_lid_height"], c)

        spark_left = Star(n=5, outer_radius=0.16, color=c["spark"]).move_to(box_center + np.array([-1.7, 0.35, 0.0]))
        spark_right = Star(n=5, outer_radius=0.13, color=c["spark"]).move_to(box_center + np.array([1.55, 0.55, 0.0]))
        spark_top = Star(n=5, outer_radius=0.11, color=c["spark"]).move_to(box_center + np.array([0.0, 1.0, 0.0]))
        sparks = VGroup(spark_left, spark_right, spark_top).set_opacity(0.0)

        if p["show_outline"]:
            self.add(pi_outline)

        self.play(FadeIn(gift_box, shift=0.2 * DOWN), FadeIn(sparks), run_time=p["box_intro_time"])
        self.play(
            lid.animate.shift(5.8 * RIGHT + 2.2 * UP).rotate(110 * DEGREES).fade(1.0),
            sparks.animate.set_opacity(1.0),
            run_time=p["lid_open_time"],
        )
        self.play(
            Create(pi_outline) if not p["show_outline"] else Indicate(pi_outline, color=c["outline"], scale_factor=1.02),
            run_time=p["outline_draw_time"],
        )

        digit_mobs: list[Mobject] = []
        route_start = opening + 0.02 * UP
        entry_point = contour_dense[0]
        shared_spiral = guiding_spiral_path(
            route_start,
            box_center,
            entry_point,
            box_width=p["box_width"],
            box_height=p["box_height"],
        )
        shared_points = np.array([shared_spiral.point_from_proportion(t) for t in np.linspace(0.0, 1.0, 220)])
        contour_route = contour_dense[np.linspace(0, len(contour_dense) - 1, 900).astype(int)]
        master_points = np.vstack([shared_points, contour_route[1:]])
        master_lengths = cumulative_lengths(master_points)
        spiral_length = cumulative_lengths(shared_points)[-1]
        contour_length = cumulative_lengths(contour_route)[-1]
        progress = ValueTracker(0.0)
        spacing = p["stream_spacing"]
        reveal_distance = 0.26
        shrink_distance = 1.25

        for idx, (char, target_alpha) in enumerate(zip(digit_chars, target_alphas)):
            prototype = prototypes[char]
            mob = prototype.copy().set_opacity(0.0)
            mob.move_to(route_start)
            self.add(mob)
            digit_mobs.append(mob)

            stop_distance = spiral_length + target_alpha * contour_length
            offset_distance = idx * spacing

            def make_updater(template: Mobject, stop_s: float, offset_s: float):
                def updater(m: Mobject) -> Mobject:
                    current_s = progress.get_value() - offset_s
                    if current_s <= 0.0:
                        m.become(template.copy().scale(p["pop_scale"]).move_to(route_start).set_opacity(0.0))
                        return m

                    traveled_s = min(current_s, stop_s)
                    point = point_at_distance(master_points, master_lengths, traveled_s)
                    opacity = smooth(min(traveled_s / reveal_distance, 1.0))
                    scale_alpha = smooth(min(traveled_s / shrink_distance, 1.0))
                    scale = interpolate(p["pop_scale"], 1.0, scale_alpha)
                    m.become(template.copy().scale(scale).move_to(point).set_opacity(opacity))
                    return m

                return updater

            mob.add_updater(make_updater(prototype, stop_distance, offset_distance))

        final_digits = VGroup(*digit_mobs)
        pulse = pi_outline.copy().set_stroke(c["highlight"], width=p["contour_stroke_width"] * 1.35, opacity=0.0)
        self.add(pulse)

        self.play(
            progress.animate.set_value(spiral_length + contour_length + (len(digit_chars) - 1) * spacing),
            sparks.animate.set_opacity(0.55),
            rate_func=linear,
            run_time=p["stream_time"],
        )
        for mob in digit_mobs:
            mob.clear_updaters()
        self.play(
            pulse.animate.set_opacity(0.65),
            final_digits.animate.set_color(c["highlight"]),
            run_time=0.5 * p["final_pulse_time"],
            rate_func=there_and_back,
        )
        self.play(
            pulse.animate.set_stroke(width=p["contour_stroke_width"] * 1.7).set_opacity(0.0),
            final_digits.animate.set_color(c["digits"]),
            run_time=0.5 * p["final_pulse_time"],
            rate_func=there_and_back,
        )
        self.wait(p["hold_time"])
