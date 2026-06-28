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

    chain = {
        "n_masses": get("chain", "n_masses", int, 7),
        "mode_list": get("chain", "mode_list", str, "0,1,2,3"),
        "mass_value": get("chain", "mass_value", float, 1.0),
        "onsite_k": get("chain", "onsite_k", float, 5.0),
        "coupling_k": get("chain", "coupling_k", float, 3.5),
        "mode_amplitude": get("chain", "mode_amplitude", float, 0.45),
        "kick_index": get("chain", "kick_index", int, 3),
        "kick_amplitude": get("chain", "kick_amplitude", float, 0.70),
        "left_x": get("chain", "left_x", float, -3.2),
        "right_x": get("chain", "right_x", float, 3.2),
        "support_y": get("chain", "support_y", float, 1.5),
        "mass_eq_y": get("chain", "mass_eq_y", float, -1.2),
        "rail_overhang": get("chain", "rail_overhang", float, 0.35),
        "mass_radius": get("chain", "mass_radius", float, 0.18),
        "vertical_spring_width": get("chain", "vertical_spring_width", float, 0.14),
        "vertical_spring_turns": get("chain", "vertical_spring_turns", int, 9),
        "coupling_spring_height": get("chain", "coupling_spring_height", float, 0.12),
        "coupling_spring_turns": get("chain", "coupling_spring_turns", int, 6),
        "rail_stroke": get("chain", "rail_stroke", float, 6.0),
        "spring_stroke": get("chain", "spring_stroke", float, 3.0),
        "coupling_stroke": get("chain", "coupling_stroke", float, 2.8),
        "title_scale": get("chain", "title_scale", float, 0.58),
        "formula_scale": get("chain", "formula_scale", float, 0.46),
        "state_scale": get("chain", "state_scale", float, 0.52),
        "top_buff": get("chain", "top_buff", float, 0.30),
        "intro_time": get("chain", "intro_time", float, 1.0),
        "build_time": get("chain", "build_time", float, 1.8),
        "mode_ramp_time": get("chain", "mode_ramp_time", float, 0.45),
        "mode_periods": get("chain", "mode_periods", float, 2.0),
        "mode_fade_time": get("chain", "mode_fade_time", float, 0.35),
        "prep_time": get("chain", "prep_time", float, 0.6),
        "kick_periods": get("chain", "kick_periods", float, 3.2),
        "tail_wait": get("chain", "tail_wait", float, 0.8),
    }

    colors = {
        "text": get("colors", "text", str, "#FFFFFF"),
        "rail": get("colors", "rail", str, "#FFFFFF"),
        "vertical_springs": get("colors", "vertical_springs", str, "#E5E5E5"),
        "coupling_springs": get("colors", "coupling_springs", str, "#A6A6A6"),
        "masses": get("colors", "masses", str, "#79C7FF"),
        "mass_stroke": get("colors", "mass_stroke", str, "#FFFFFF"),
        "highlight": get("colors", "highlight", str, "#FFD166"),
        "kick": get("colors", "kick", str, "#FF7B72"),
    }

    return {"manim": manim_params, "chain": chain, "colors": colors}


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


def parse_int_list(raw: str) -> list[int]:
    out = []
    for token in raw.replace(";", ",").split(","):
        token = token.strip()
        if not token:
            continue
        try:
            out.append(int(token))
        except ValueError:
            continue
    return out


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


def free_end_profile(mode: int, n_masses: int) -> np.ndarray:
    j = np.arange(1, n_masses + 1)
    profile = np.cos(mode * PI * (j - 0.5) / n_masses)
    profile /= np.max(np.abs(profile))
    return profile


def free_end_frequency(mode: int, n_masses: int, mass_value: float, onsite_k: float, coupling_k: float) -> float:
    return np.sqrt((onsite_k + 4.0 * coupling_k * np.sin(mode * PI / (2.0 * n_masses)) ** 2) / mass_value)


def normalized_free_end_basis(n_masses: int) -> np.ndarray:
    basis = np.zeros((n_masses, n_masses))
    j = np.arange(1, n_masses + 1)
    for mode in range(n_masses):
        if mode == 0:
            basis[mode] = np.full(n_masses, 1.0 / np.sqrt(n_masses))
        else:
            basis[mode] = np.sqrt(2.0 / n_masses) * np.cos(mode * PI * (j - 0.5) / n_masses)
    return basis


class CoupledOscillators1D(Scene):
    def construct(self):
        p = CFG["chain"]
        c = CFG["colors"]
        self.camera.background_color = CFG["manim"]["background_color"]

        n = max(2, p["n_masses"])
        modes = [mode for mode in parse_int_list(p["mode_list"]) if 0 <= mode < n]
        if not modes:
            modes = [0, 1, 2] if n >= 3 else list(range(n))

        mass_value = p["mass_value"]
        onsite_k = p["onsite_k"]
        coupling_k = p["coupling_k"]
        kick_index = max(0, min(n - 1, p["kick_index"]))

        left_x = p["left_x"]
        right_x = p["right_x"]
        x_positions = np.linspace(left_x, right_x, n)
        support_y = p["support_y"]
        mass_eq_y = p["mass_eq_y"]

        basis = normalized_free_end_basis(n)
        omegas = np.array(
            [free_end_frequency(mode, n, mass_value, onsite_k, coupling_k) for mode in range(n)]
        )
        kick_initial = np.zeros(n)
        kick_initial[kick_index] = p["kick_amplitude"]
        modal_coeff = basis @ kick_initial

        phase_tracker = ValueTracker(0.0)
        amplitude_tracker = ValueTracker(0.0)
        prep_tracker = ValueTracker(0.0)
        state = {"kind": "idle", "profile": np.zeros(n), "omega": 1.0}

        def displacements() -> np.ndarray:
            if state["kind"] == "mode":
                return amplitude_tracker.get_value() * state["profile"] * np.cos(phase_tracker.get_value())
            if state["kind"] == "prep":
                disp = np.zeros(n)
                disp[kick_index] = prep_tracker.get_value()
                return disp
            if state["kind"] == "kick":
                t = phase_tracker.get_value()
                cos_terms = np.cos(omegas * t)
                return basis.T @ (modal_coeff * cos_terms)
            return np.zeros(n)

        def mass_center(idx: int) -> np.ndarray:
            return np.array([x_positions[idx], mass_eq_y + displacements()[idx], 0.0])

        rail = Line(
            np.array([left_x - p["rail_overhang"], support_y, 0.0]),
            np.array([right_x + p["rail_overhang"], support_y, 0.0]),
            color=c["rail"],
            stroke_width=p["rail_stroke"],
        )

        title = Text("Coupled Vertical Oscillators", color=c["text"]).scale(p["title_scale"])
        eq_motion = MathTex(
            r"m\ddot q_j=-k_0 q_j-k_c(2q_j-q_{j-1}-q_{j+1})",
            color=c["text"],
        ).scale(p["formula_scale"])
        eq_mode = MathTex(
            r"q_j^{(s)}(t)=A_s\cos\!\left[\frac{s\pi}{N}\!\left(j-\frac12\right)\right]\cos(\omega_s t)",
            color=c["text"],
        ).scale(p["formula_scale"])
        eq_freq = MathTex(
            r"\omega_s^2=\frac{k_0}{m}+\frac{4k_c}{m}\sin^2\!\left(\frac{s\pi}{2N}\right)",
            color=c["highlight"],
        ).scale(p["formula_scale"])
        header = VGroup(title, eq_motion, eq_mode, eq_freq).arrange(DOWN, buff=0.12)
        header.to_edge(UP, buff=p["top_buff"])

        state_label = MathTex(r"\text{Modes}", color=c["highlight"]).scale(p["state_scale"])
        state_label.next_to(header, DOWN, buff=0.18)

        masses = VGroup()
        for idx in range(n):
            masses.add(
                always_redraw(
                    lambda j=idx: Circle(radius=p["mass_radius"])
                    .set_fill(c["masses"], opacity=1.0)
                    .set_stroke(c["mass_stroke"], width=2.0)
                    .move_to(mass_center(j))
                )
            )

        vertical_springs = VGroup()
        for idx in range(n):
            anchor = np.array([x_positions[idx], support_y, 0.0])
            vertical_springs.add(
                always_redraw(
                    lambda j=idx, a=anchor: spring_path(
                        a,
                        mass_center(j) + p["mass_radius"] * UP,
                        p["vertical_spring_width"],
                        p["vertical_spring_turns"],
                    ).set_stroke(c["vertical_springs"], width=p["spring_stroke"])
                )
            )

        coupling_springs = VGroup()
        for idx in range(n - 1):
            coupling_springs.add(
                always_redraw(
                    lambda j=idx: spring_path(
                        mass_center(j) + p["mass_radius"] * RIGHT,
                        mass_center(j + 1) - p["mass_radius"] * RIGHT,
                        p["coupling_spring_height"],
                        p["coupling_spring_turns"],
                    ).set_stroke(c["coupling_springs"], width=p["coupling_stroke"])
                )
            )

        self.play(
            FadeIn(header, shift=0.18 * DOWN),
            FadeIn(state_label, shift=0.12 * DOWN),
            Create(rail),
            run_time=p["intro_time"],
        )
        self.play(
            LaggedStart(*[Create(s) for s in vertical_springs], lag_ratio=0.04),
            LaggedStart(*[Create(s) for s in coupling_springs], lag_ratio=0.04),
            LaggedStart(*[FadeIn(m) for m in masses], lag_ratio=0.05),
            run_time=p["build_time"],
        )

        self.add(vertical_springs, coupling_springs, masses)

        current_label = state_label
        for mode in modes:
            profile = free_end_profile(mode, n)
            omega = omegas[mode]
            state["kind"] = "mode"
            state["profile"] = profile
            state["omega"] = omega
            phase_tracker.set_value(0.0)
            amplitude_tracker.set_value(0.0)

            next_label = MathTex(rf"\text{{mode }} s={mode}", color=c["highlight"]).scale(p["state_scale"])
            next_label.next_to(header, DOWN, buff=0.18)
            self.play(Transform(current_label, next_label), run_time=0.35)

            self.play(amplitude_tracker.animate.set_value(p["mode_amplitude"]), run_time=p["mode_ramp_time"])
            mode_time = p["mode_periods"] * TAU
            self.play(
                phase_tracker.animate.set_value(mode_time),
                run_time=p["mode_periods"] * TAU / omega,
                rate_func=linear,
            )
            self.play(amplitude_tracker.animate.set_value(0.0), run_time=p["mode_fade_time"])

        state["kind"] = "prep"
        phase_tracker.set_value(0.0)
        amplitude_tracker.set_value(0.0)
        prep_tracker.set_value(0.0)

        kick_label = MathTex(
            rf"q_j(0)=A\,\delta_{{j,{kick_index + 1}}},\quad \dot q_j(0)=0",
            color=c["kick"],
        ).scale(p["state_scale"])
        kick_label.next_to(header, DOWN, buff=0.18)
        self.play(Transform(current_label, kick_label), run_time=0.35)
        self.play(prep_tracker.animate.set_value(p["kick_amplitude"]), run_time=p["prep_time"])

        state["kind"] = "kick"
        prep_tracker.set_value(0.0)
        phase_tracker.set_value(0.0)
        kick_time = p["kick_periods"] * TAU / np.min(omegas)
        self.play(
            phase_tracker.animate.set_value(kick_time),
            run_time=max(5.0, kick_time),
            rate_func=linear,
        )
        self.wait(p["tail_wait"])
