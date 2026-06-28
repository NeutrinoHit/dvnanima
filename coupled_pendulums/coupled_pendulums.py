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
        "pixel_width": get("manim", "pixel_width", int, 1080),
        "pixel_height": get("manim", "pixel_height", int, 1920),
        "frame_width": get("manim", "frame_width", float, 9.0),
        "frame_height": get("manim", "frame_height", float, 16.0),
        "frame_rate": get("manim", "frame_rate", int, 60),
        "background_color": get("manim", "background_color", str, "#000000"),
    }

    pendulum = {
        "length": get("pendulum", "length", float, 2.8),
        "mass_value": get("pendulum", "mass_value", float, 1.0),
        "gravity": get("pendulum", "gravity", float, 12.0),
        "coupling_k": get("pendulum", "coupling_k", float, 1.6),
        "anchor_separation": get("pendulum", "anchor_separation", float, 3.0),
        "support_y": get("pendulum", "support_y", float, 2.4),
        "beam_overhang": get("pendulum", "beam_overhang", float, 0.7),
        "bob_radius": get("pendulum", "bob_radius", float, 0.22),
        "anchor_radius": get("pendulum", "anchor_radius", float, 0.055),
        "rod_stroke": get("pendulum", "rod_stroke", float, 6.0),
        "spring_width": get("pendulum", "spring_width", float, 0.18),
        "spring_turns": get("pendulum", "spring_turns", int, 8),
        "spring_stroke": get("pendulum", "spring_stroke", float, 3.2),
        "ghost_opacity": get("pendulum", "ghost_opacity", float, 0.18),
        "mode_amplitude": get("pendulum", "mode_amplitude", float, 0.34),
        "initial_left": get("pendulum", "initial_left", float, 0.34),
        "initial_right": get("pendulum", "initial_right", float, 0.0),
        "title_scale": get("pendulum", "title_scale", float, 0.76),
        "subtitle_scale": get("pendulum", "subtitle_scale", float, 0.42),
        "formula_scale": get("pendulum", "formula_scale", float, 0.58),
        "state_scale": get("pendulum", "state_scale", float, 0.56),
        "top_buff": get("pendulum", "top_buff", float, 0.35),
        "intro_time": get("pendulum", "intro_time", float, 0.9),
        "build_time": get("pendulum", "build_time", float, 1.5),
        "mode_ramp_time": get("pendulum", "mode_ramp_time", float, 0.45),
        "mode_periods": get("pendulum", "mode_periods", float, 1.75),
        "mode_fade_time": get("pendulum", "mode_fade_time", float, 0.35),
        "prep_time": get("pendulum", "prep_time", float, 0.75),
        "free_beats": get("pendulum", "free_beats", float, 1.0),
        "tail_wait": get("pendulum", "tail_wait", float, 1.0),
    }

    colors = {
        "text": get("colors", "text", str, "#FFFFFF"),
        "muted": get("colors", "muted", str, "#BDBDBD"),
        "beam": get("colors", "beam", str, "#FFFFFF"),
        "rods": get("colors", "rods", str, "#F2F2F2"),
        "spring": get("colors", "spring", str, "#9FB3C8"),
        "bobs": get("colors", "bobs", str, "#79C7FF"),
        "bob_stroke": get("colors", "bob_stroke", str, "#FFFFFF"),
        "highlight": get("colors", "highlight", str, "#FFD166"),
        "accent": get("colors", "accent", str, "#FF7B72"),
        "guide": get("colors", "guide", str, "#808080"),
    }

    return {"manim": manim_params, "pendulum": pendulum, "colors": colors}


BASE_DIR = Path(__file__).resolve().parent
CFG = load_cfg(BASE_DIR / "run.cfg")


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


apply_render_geometry(CFG["manim"])


def spring_path(start: np.ndarray, end: np.ndarray, width: float, turns: int) -> VMobject:
    vec = end - start
    length = float(np.linalg.norm(vec))
    if length <= 1e-8:
        return Line(start, end)

    direction = vec / length
    normal = np.array([-direction[1], direction[0], 0.0])
    lead = min(0.12 * length, 0.10)
    usable = max(length - 2.0 * lead, 1e-6)
    points = [start, start + direction * lead]

    for idx in range(2 * turns + 1):
        t = idx / (2 * turns)
        offset = 0.0 if idx in (0, 2 * turns) else width * (1 if idx % 2 else -1)
        points.append(start + direction * (lead + usable * t) + normal * offset)

    points.extend([end - direction * lead, end])
    curve = VMobject()
    curve.set_points_as_corners(points)
    return curve


def bob_position(anchor: np.ndarray, length: float, theta: float) -> np.ndarray:
    return anchor + np.array([length * np.sin(theta), -length * np.cos(theta), 0.0])


def mode_frequencies(gravity: float, length: float, coupling_k: float, mass_value: float) -> tuple[float, float]:
    omega_sym = np.sqrt(max(gravity / length, 1e-8))
    omega_anti = np.sqrt(max(gravity / length + 2.0 * coupling_k / mass_value, 1e-8))
    return omega_sym, omega_anti


class CoupledPendulums(Scene):
    def construct(self):
        p = CFG["pendulum"]
        c = CFG["colors"]
        self.camera.background_color = CFG["manim"]["background_color"]

        length = p["length"]
        anchor_sep = p["anchor_separation"]
        support_y = p["support_y"]
        anchors = [
            np.array([-anchor_sep / 2.0, support_y, 0.0]),
            np.array([anchor_sep / 2.0, support_y, 0.0]),
        ]

        omega_sym, omega_anti = mode_frequencies(
            p["gravity"], length, p["coupling_k"], p["mass_value"]
        )
        plus_coeff = 0.5 * (p["initial_left"] + p["initial_right"])
        minus_coeff = 0.5 * (p["initial_left"] - p["initial_right"])

        time_tracker = ValueTracker(0.0)
        amplitude_tracker = ValueTracker(0.0)
        prep_left = ValueTracker(0.0)
        prep_right = ValueTracker(0.0)
        state = {"kind": "idle", "profile": np.zeros(2), "omega": omega_sym}

        def angles() -> np.ndarray:
            t = time_tracker.get_value()
            if state["kind"] == "mode":
                return amplitude_tracker.get_value() * state["profile"] * np.cos(state["omega"] * t)
            if state["kind"] == "prep":
                return np.array([prep_left.get_value(), prep_right.get_value()])
            if state["kind"] == "free":
                theta_plus = plus_coeff * np.cos(omega_sym * t)
                theta_minus = minus_coeff * np.cos(omega_anti * t)
                return np.array([theta_plus + theta_minus, theta_plus - theta_minus])
            return np.zeros(2)

        def bob_center(idx: int) -> np.ndarray:
            return bob_position(anchors[idx], length, angles()[idx])

        def spring_endpoints() -> tuple[np.ndarray, np.ndarray]:
            left = bob_center(0)
            right = bob_center(1)
            direction = right - left
            norm = float(np.linalg.norm(direction))
            if norm <= 1e-8:
                unit = RIGHT.copy()
            else:
                unit = direction / norm
            return left + unit * p["bob_radius"], right - unit * p["bob_radius"]

        title = Text("Two Coupled Pendulums", color=c["text"]).scale(p["title_scale"])
        subtitle = Text("Normal modes in the small-angle limit", color=c["muted"]).scale(
            p["subtitle_scale"]
        )
        eq_modes = MathTex(
            r"(\theta_1,\theta_2)\propto (1,1)\quad \mathrm{or}\quad (1,-1)",
            color=c["text"],
        ).scale(p["formula_scale"])
        eq_freq = MathTex(
            r"\omega_+^2=\frac{g}{\ell},\qquad \omega_-^2=\frac{g}{\ell}+\frac{2k}{m}",
            color=c["highlight"],
        ).scale(p["formula_scale"])
        header = VGroup(title, subtitle, eq_modes, eq_freq).arrange(DOWN, buff=0.12)
        header.to_edge(UP, buff=p["top_buff"])

        state_label = MathTex(r"\text{Normal modes}", color=c["highlight"]).scale(p["state_scale"])
        state_label.next_to(header, DOWN, buff=0.18)

        beam = Line(
            anchors[0] + p["beam_overhang"] * LEFT,
            anchors[1] + p["beam_overhang"] * RIGHT,
            color=c["beam"],
            stroke_width=p["rod_stroke"],
        )
        support_dots = VGroup(
            *[
                Dot(point=anchor, radius=p["anchor_radius"], color=c["beam"])
                for anchor in anchors
            ]
        )

        ghost_rods = VGroup(
            *[
                Line(anchor, bob_position(anchor, length, 0.0))
                .set_stroke(c["guide"], width=p["rod_stroke"] * 0.7, opacity=p["ghost_opacity"])
                for anchor in anchors
            ]
        )
        ghost_bobs = VGroup(
            *[
                Circle(radius=p["bob_radius"])
                .set_fill(c["guide"], opacity=0.08)
                .set_stroke(c["guide"], width=1.6, opacity=p["ghost_opacity"])
                .move_to(bob_position(anchor, length, 0.0))
                for anchor in anchors
            ]
        )
        ghost_spring = spring_path(
            bob_position(anchors[0], length, 0.0) + p["bob_radius"] * RIGHT,
            bob_position(anchors[1], length, 0.0) - p["bob_radius"] * RIGHT,
            p["spring_width"],
            p["spring_turns"],
        ).set_stroke(c["guide"], width=p["spring_stroke"], opacity=p["ghost_opacity"])

        rods = VGroup(
            *[
                always_redraw(
                    lambda idx=idx: Line(anchors[idx], bob_center(idx)).set_stroke(
                        c["rods"], width=p["rod_stroke"]
                    )
                )
                for idx in range(2)
            ]
        )
        bobs = VGroup(
            *[
                always_redraw(
                    lambda idx=idx: Circle(radius=p["bob_radius"])
                    .set_fill(c["bobs"], opacity=1.0)
                    .set_stroke(c["bob_stroke"], width=2.0)
                    .move_to(bob_center(idx))
                )
                for idx in range(2)
            ]
        )
        spring = always_redraw(
            lambda: spring_path(
                *spring_endpoints(),
                p["spring_width"],
                p["spring_turns"],
            ).set_stroke(c["spring"], width=p["spring_stroke"])
        )

        self.play(
            FadeIn(header, shift=0.18 * DOWN),
            FadeIn(state_label, shift=0.12 * DOWN),
            Create(beam),
            FadeIn(support_dots, scale=0.7),
            run_time=p["intro_time"],
        )
        self.play(
            FadeIn(ghost_rods),
            FadeIn(ghost_bobs),
            Create(ghost_spring),
            run_time=0.35,
        )
        self.play(
            LaggedStart(*[Create(rod) for rod in rods], lag_ratio=0.08),
            Create(spring),
            LaggedStart(*[FadeIn(bob) for bob in bobs], lag_ratio=0.08),
            run_time=p["build_time"],
        )
        self.add(rods, spring, bobs)

        modes = [
            (
                np.array([1.0, 1.0]),
                omega_sym,
                r"\text{in phase: }\theta_1=\theta_2=A\cos(\omega_+ t)",
                c["highlight"],
            ),
            (
                np.array([1.0, -1.0]),
                omega_anti,
                r"\text{out of phase: }\theta_1=-\theta_2=A\cos(\omega_- t)",
                c["accent"],
            ),
        ]

        current_label = state_label
        for profile, omega, label_tex, color in modes:
            state["kind"] = "mode"
            state["profile"] = profile
            state["omega"] = omega
            time_tracker.set_value(0.0)
            amplitude_tracker.set_value(0.0)

            next_label = MathTex(label_tex, color=color).scale(p["state_scale"])
            next_label.next_to(header, DOWN, buff=0.18)
            self.play(Transform(current_label, next_label), run_time=0.35)
            self.play(amplitude_tracker.animate.set_value(p["mode_amplitude"]), run_time=p["mode_ramp_time"])

            mode_time = p["mode_periods"] * TAU / omega
            self.play(time_tracker.animate.set_value(mode_time), run_time=mode_time, rate_func=linear)
            self.play(amplitude_tracker.animate.set_value(0.0), run_time=p["mode_fade_time"])

        state["kind"] = "prep"
        time_tracker.set_value(0.0)
        amplitude_tracker.set_value(0.0)
        prep_left.set_value(0.0)
        prep_right.set_value(0.0)

        prep_label = MathTex(
            r"\theta_1(0)=A,\qquad \theta_2(0)=0,\qquad \dot\theta_1(0)=\dot\theta_2(0)=0",
            color=c["accent"],
        ).scale(p["state_scale"])
        prep_label.next_to(header, DOWN, buff=0.18)
        self.play(Transform(current_label, prep_label), run_time=0.35)
        self.play(
            prep_left.animate.set_value(p["initial_left"]),
            prep_right.animate.set_value(p["initial_right"]),
            run_time=p["prep_time"],
        )

        release_label = MathTex(
            r"\text{release: superposition of the two normal modes}",
            color=c["highlight"],
        ).scale(p["state_scale"])
        release_label.next_to(header, DOWN, buff=0.18)
        self.play(Transform(current_label, release_label), run_time=0.35)

        state["kind"] = "free"
        time_tracker.set_value(0.0)
        beat_time = p["free_beats"] * TAU / max(abs(omega_anti - omega_sym), 1e-6)
        self.play(time_tracker.animate.set_value(beat_time), run_time=max(6.0, beat_time), rate_func=linear)
        self.wait(p["tail_wait"])
