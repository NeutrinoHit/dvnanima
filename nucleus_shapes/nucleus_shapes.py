from __future__ import annotations

from pathlib import Path
import configparser
import re

import numpy as np
from manim import *


SHAPE_PRESETS: dict[str, np.ndarray] = {
    "sphere": np.array([0.00, 0.00, 0.00, 1.00]),
    "prolate": np.array([0.34, 0.00, 0.00, 0.95]),
    "oblate": np.array([-0.26, 0.00, 0.00, 0.98]),
    "triaxial": np.array([0.16, 0.20, 0.00, 0.96]),
    "pear": np.array([0.15, 0.02, 0.24, 0.95]),
}


def load_cfg(path: Path) -> dict:
    cfg = configparser.ConfigParser(inline_comment_prefixes=(";",))
    cfg.read(path)

    def get(section, key, cast=str, fallback=None):
        if fallback is None:
            return cast(cfg[section][key])
        return cast(cfg.get(section, key, fallback=str(fallback)))

    manim_params = {
        "pixel_width": get("manim", "pixel_width", int, 1440),
        "pixel_height": get("manim", "pixel_height", int, 1800),
        "frame_width": get("manim", "frame_width", float, 8.0),
        "frame_height": get("manim", "frame_height", float, 10.0),
        "frame_rate": get("manim", "frame_rate", int, 60),
        "background_color": get("manim", "background_color", str, "#000000"),
    }

    nucleus = {
        "nucleus_radius": get("nucleus", "nucleus_radius", float, 1.95),
        "surface_opacity": get("nucleus", "surface_opacity", float, 1.0),
        "surface_stroke_width": get("nucleus", "surface_stroke_width", float, 0.0),
        "surface_stroke_opacity": get("nucleus", "surface_stroke_opacity", float, 0.0),
        "surface_resolution_u": get("nucleus", "surface_resolution_u", int, 32),
        "surface_resolution_v": get("nucleus", "surface_resolution_v", int, 64),
        "sheen_factor": get("nucleus", "sheen_factor", float, 0.0),
        "sheen_direction_x": get("nucleus", "sheen_direction_x", float, -0.8),
        "sheen_direction_y": get("nucleus", "sheen_direction_y", float, -0.25),
        "sheen_direction_z": get("nucleus", "sheen_direction_z", float, 1.0),
        "shape_sequence": get(
            "nucleus",
            "shape_sequence",
            str,
            "sphere, prolate, oblate, triaxial, pear",
        ),
        "camera_phi": get("nucleus", "camera_phi", float, 72.0),
        "camera_theta": get("nucleus", "camera_theta", float, -22.0),
        "camera_zoom": get("nucleus", "camera_zoom", float, 2.05),
        "ambient_rotation_rate": get("nucleus", "ambient_rotation_rate", float, 0.0),
        "spin_rate": get("nucleus", "spin_rate", float, 0.32),
        "light_x": get("nucleus", "light_x", float, -6.0),
        "light_y": get("nucleus", "light_y", float, -4.0),
        "light_z": get("nucleus", "light_z", float, 9.0),
        "shading_factor": get("nucleus", "shading_factor", float, 0.36),
        "intro_time": get("nucleus", "intro_time", float, 0.8),
        "hold_time": get("nucleus", "hold_time", float, 0.45),
        "transition_time": get("nucleus", "transition_time", float, 1.5),
        "tail_wait": get("nucleus", "tail_wait", float, 0.6),
    }

    colors = {
        "surface": get("colors", "surface", str, "#D8DCE3"),
    }

    return {"manim": manim_params, "nucleus": nucleus, "colors": colors}


BASE_DIR = Path(__file__).resolve().parent
CFG = load_cfg(BASE_DIR / "run.cfg")


def apply_render_geometry(manim_params: dict) -> None:
    def even(value: int) -> int:
        return value if value % 2 == 0 else value + 1

    aspect = manim_params["frame_width"] / manim_params["frame_height"]
    long_side = max(int(config.pixel_width), int(config.pixel_height))
    if aspect >= 1.0:
        config.pixel_width = even(long_side)
        config.pixel_height = even(max(2, int(round(long_side / aspect))))
    else:
        config.pixel_height = even(long_side)
        config.pixel_width = even(max(2, int(round(long_side * aspect))))
    config.frame_width = manim_params["frame_width"]
    config.frame_height = manim_params["frame_height"]
    config.frame_rate = manim_params["frame_rate"]


apply_render_geometry(CFG["manim"])


def parse_shape_sequence(raw: str) -> list[str]:
    sequence: list[str] = []
    for token in re.split(r"[,;]+", raw):
        name = token.strip().lower().replace("-", "_").replace(" ", "_")
        if name in SHAPE_PRESETS:
            sequence.append(name)
    return sequence or ["sphere", "prolate", "oblate", "triaxial", "pear"]


def interpolate_shape_params(progress: float, sequence: list[str]) -> np.ndarray:
    if len(sequence) == 1:
        return SHAPE_PRESETS[sequence[0]]
    clamped = np.clip(progress, 0.0, len(sequence) - 1.0)
    left = int(np.floor(clamped))
    right = min(left + 1, len(sequence) - 1)
    alpha = clamped - left
    return (1.0 - alpha) * SHAPE_PRESETS[sequence[left]] + alpha * SHAPE_PRESETS[sequence[right]]


def deformation_factor(direction: np.ndarray, params: np.ndarray) -> float:
    x, y, z = direction
    beta2, beta22, beta3, scale = params
    p2 = 0.5 * (3.0 * z * z - 1.0)
    p3 = 0.5 * (5.0 * z * z * z - 3.0 * z)
    triaxial = x * x - y * y
    factor = scale * (1.0 + beta2 * p2 + beta22 * triaxial + beta3 * p3)
    return max(float(factor), 0.38)


def rotation_matrix_z(angle: float) -> np.ndarray:
    c = float(np.cos(angle))
    s = float(np.sin(angle))
    return np.array(
        [
            [c, -s, 0.0],
            [s, c, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )


def normalized(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-8:
        return np.array([0.0, 0.0, 1.0], dtype=float)
    return vector / norm


def surface_point(
    theta: float,
    phi: float,
    params: np.ndarray,
    nucleus_radius: float,
    spin_angle: float,
) -> np.ndarray:
    direction = np.array(
        [
            np.sin(theta) * np.cos(phi),
            np.sin(theta) * np.sin(phi),
            np.cos(theta),
        ],
        dtype=float,
    )
    radius = nucleus_radius * deformation_factor(direction, params)
    return rotation_matrix_z(spin_angle) @ (direction * radius)


def build_surface(params: np.ndarray, p: dict, c: dict, appearance: float, spin_angle: float) -> Surface:
    sheen_direction = normalized(
        np.array(
            [
                p["sheen_direction_x"],
                p["sheen_direction_y"],
                p["sheen_direction_z"],
            ],
            dtype=float,
        )
    )
    surface = Surface(
        lambda u, v: surface_point(u, v, params, p["nucleus_radius"], spin_angle),
        u_range=[0.001, PI - 0.001],
        v_range=[0.0, TAU],
        resolution=(p["surface_resolution_u"], p["surface_resolution_v"]),
        checkerboard_colors=[c["surface"], c["surface"]],
        fill_opacity=appearance * p["surface_opacity"],
    )
    surface.set_fill(c["surface"], opacity=appearance * p["surface_opacity"])
    surface.set_stroke(
        color=c["surface"],
        width=p["surface_stroke_width"],
        opacity=appearance * p["surface_stroke_opacity"],
    )
    surface.set_shade_in_3d(True)
    surface.set_sheen(p["sheen_factor"], direction=sheen_direction)
    return surface


class NucleusShapeMorphs(ThreeDScene):
    def construct(self):
        p = CFG["nucleus"]
        c = CFG["colors"]
        self.camera.background_color = CFG["manim"]["background_color"]
        self.set_camera_orientation(
            phi=p["camera_phi"] * DEGREES,
            theta=p["camera_theta"] * DEGREES,
            zoom=p["camera_zoom"],
        )
        self.camera.shading_factor = p["shading_factor"]
        self.camera.light_source.move_to(
            np.array([p["light_x"], p["light_y"], p["light_z"]], dtype=float)
        )

        sequence = parse_shape_sequence(p["shape_sequence"])
        appearance = ValueTracker(0.0)
        progress = ValueTracker(0.0)
        spin = ValueTracker(0.0)
        if abs(p["spin_rate"]) > 1e-8:
            spin.add_updater(lambda mob, dt: mob.increment_value(p["spin_rate"] * dt))
            self.add(spin)

        def current_params() -> np.ndarray:
            return interpolate_shape_params(progress.get_value(), sequence)

        surface = always_redraw(
            lambda: build_surface(
                current_params(),
                p,
                c,
                appearance.get_value(),
                spin.get_value(),
            )
        )

        self.add(surface)
        self.play(appearance.animate.set_value(1.0), run_time=p["intro_time"], rate_func=smooth)

        if abs(p["ambient_rotation_rate"]) > 1e-8:
            self.begin_ambient_camera_rotation(rate=p["ambient_rotation_rate"])

        self.wait(p["hold_time"])
        for idx in range(1, len(sequence)):
            self.play(
                progress.animate.set_value(float(idx)),
                run_time=p["transition_time"],
                rate_func=smooth,
            )
            self.wait(p["hold_time"])

        self.wait(p["tail_wait"])
        if abs(p["ambient_rotation_rate"]) > 1e-8:
            self.stop_ambient_camera_rotation()
