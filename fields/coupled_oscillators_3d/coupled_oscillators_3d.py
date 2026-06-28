from __future__ import annotations

from pathlib import Path
import configparser
import re
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

    lattice = {
        "nx": get("lattice", "nx", int, 4),
        "ny": get("lattice", "ny", int, 4),
        "nz": get("lattice", "nz", int, 4),
        "mode_list": get("lattice", "mode_list", str, "1x1x1,2x1x1,1x2x1,1x1x2"),
        "mass_value": get("lattice", "mass_value", float, 1.0),
        "onsite_k": get("lattice", "onsite_k", float, 4.0),
        "coupling_k": get("lattice", "coupling_k", float, 2.5),
        "mode_amplitude": get("lattice", "mode_amplitude", float, 0.28),
        "kick_i": get("lattice", "kick_i", int, 1),
        "kick_j": get("lattice", "kick_j", int, 1),
        "kick_k": get("lattice", "kick_k", int, 1),
        "kick_amplitude": get("lattice", "kick_amplitude", float, 0.34),
        "span_x": get("lattice", "span_x", float, 3.5),
        "span_y": get("lattice", "span_y", float, 3.5),
        "span_z": get("lattice", "span_z", float, 3.5),
        "disp_axis_x": get("lattice", "disp_axis_x", float, 0.0),
        "disp_axis_y": get("lattice", "disp_axis_y", float, 1.0),
        "disp_axis_z": get("lattice", "disp_axis_z", float, 0.35),
        "spring_width": get("lattice", "spring_width", float, 0.04),
        "spring_turns": get("lattice", "spring_turns", int, 4),
        "spring_stroke": get("lattice", "spring_stroke", float, 1.7),
        "border_stroke": get("lattice", "border_stroke", float, 2.6),
        "node_radius": get("lattice", "node_radius", float, 0.05),
        "node_opacity": get("lattice", "node_opacity", float, 0.92),
        "title_scale": get("lattice", "title_scale", float, 0.54),
        "formula_scale": get("lattice", "formula_scale", float, 0.36),
        "state_scale": get("lattice", "state_scale", float, 0.48),
        "top_buff": get("lattice", "top_buff", float, 0.2),
        "camera_phi": get("lattice", "camera_phi", float, 68.0),
        "camera_theta": get("lattice", "camera_theta", float, -48.0),
        "camera_zoom": get("lattice", "camera_zoom", float, 1.05),
        "intro_time": get("lattice", "intro_time", float, 1.0),
        "build_time": get("lattice", "build_time", float, 1.8),
        "mode_ramp_time": get("lattice", "mode_ramp_time", float, 0.4),
        "mode_periods": get("lattice", "mode_periods", float, 1.8),
        "mode_fade_time": get("lattice", "mode_fade_time", float, 0.3),
        "prep_time": get("lattice", "prep_time", float, 0.8),
        "kick_periods": get("lattice", "kick_periods", float, 3.2),
        "tail_wait": get("lattice", "tail_wait", float, 0.8),
    }

    colors = {
        "text": get("colors", "text", str, "#FFFFFF"),
        "border": get("colors", "border", str, "#D7D7D7"),
        "springs_x": get("colors", "springs_x", str, "#8BD3FF"),
        "springs_y": get("colors", "springs_y", str, "#A7F3D0"),
        "springs_z": get("colors", "springs_z", str, "#F2F2F2"),
        "highlight": get("colors", "highlight", str, "#FFD166"),
        "kick": get("colors", "kick", str, "#FF7B72"),
        "node": get("colors", "node", str, "#B7E0FF"),
    }

    return {"manim": manim_params, "lattice": lattice, "colors": colors}


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


def parse_mode_triples(raw: str) -> list[tuple[int, int, int]]:
    triples: list[tuple[int, int, int]] = []
    for token in re.split(r"[,;]+", raw):
        token = token.strip().lower()
        if not token:
            continue
        match = re.match(r"(\d+)\s*[x*]\s*(\d+)\s*[x*]\s*(\d+)", token)
        if not match:
            continue
        triples.append((int(match.group(1)), int(match.group(2)), int(match.group(3))))
    return triples


def normalized_basis(nx: int, ny: int, nz: int) -> np.ndarray:
    i = np.arange(1, nx + 1)[:, None, None]
    j = np.arange(1, ny + 1)[None, :, None]
    k = np.arange(1, nz + 1)[None, None, :]
    basis = np.zeros((nx, ny, nz, nx, ny, nz))
    norm = np.sqrt(2.0 / (nx + 1)) * np.sqrt(2.0 / (ny + 1)) * np.sqrt(2.0 / (nz + 1))
    for mx in range(1, nx + 1):
        sin_x = np.sin(mx * PI * i / (nx + 1))
        for my in range(1, ny + 1):
            sin_y = np.sin(my * PI * j / (ny + 1))
            for mz in range(1, nz + 1):
                sin_z = np.sin(mz * PI * k / (nz + 1))
                basis[mx - 1, my - 1, mz - 1] = norm * (sin_x * sin_y * sin_z)
    return basis


def mode_profile(mx: int, my: int, mz: int, nx: int, ny: int, nz: int) -> np.ndarray:
    profile = normalized_basis(nx, ny, nz)[mx - 1, my - 1, mz - 1]
    peak = float(np.max(np.abs(profile)))
    return profile / peak if peak > 0 else profile


def mode_frequency(
    mx: int,
    my: int,
    mz: int,
    nx: int,
    ny: int,
    nz: int,
    mass_value: float,
    onsite_k: float,
    coupling_k: float,
) -> float:
    sx = np.sin(mx * PI / (2.0 * (nx + 1)))
    sy = np.sin(my * PI / (2.0 * (ny + 1)))
    sz = np.sin(mz * PI / (2.0 * (nz + 1)))
    return np.sqrt((onsite_k + 4.0 * coupling_k * (sx**2 + sy**2 + sz**2)) / mass_value)


def spring_path_3d(start: np.ndarray, end: np.ndarray, width: float, turns: int) -> VMobject:
    vec = end - start
    length = float(np.linalg.norm(vec))
    if length <= 1e-8:
        return Line(start, end)

    direction = vec / length
    helper = OUT
    if abs(np.dot(direction, helper)) > 0.92:
        helper = RIGHT
    normal = np.cross(direction, helper)
    normal_norm = np.linalg.norm(normal)
    if normal_norm <= 1e-8:
        helper = UP
        normal = np.cross(direction, helper)
        normal_norm = np.linalg.norm(normal)
    normal /= max(normal_norm, 1e-8)

    lead = min(0.15 * length, 0.08)
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


class CoupledOscillators3D(ThreeDScene):
    def construct(self):
        p = CFG["lattice"]
        c = CFG["colors"]
        self.camera.background_color = CFG["manim"]["background_color"]
        self.set_camera_orientation(
            phi=p["camera_phi"] * DEGREES,
            theta=p["camera_theta"] * DEGREES,
            zoom=p["camera_zoom"],
        )

        nx = max(2, p["nx"])
        ny = max(2, p["ny"])
        nz = max(2, p["nz"])
        mode_triples = [
            triple for triple in parse_mode_triples(p["mode_list"])
            if 1 <= triple[0] <= nx and 1 <= triple[1] <= ny and 1 <= triple[2] <= nz
        ]
        if not mode_triples:
            mode_triples = [(1, 1, 1), (2, 1, 1), (1, 2, 1), (1, 1, 2)]

        mass_value = p["mass_value"]
        onsite_k = p["onsite_k"]
        coupling_k = p["coupling_k"]
        kick_i = max(0, min(nx - 1, p["kick_i"]))
        kick_j = max(0, min(ny - 1, p["kick_j"]))
        kick_k = max(0, min(nz - 1, p["kick_k"]))

        x_positions = np.linspace(-0.5 * p["span_x"], 0.5 * p["span_x"], nx)
        y_positions = np.linspace(-0.5 * p["span_y"], 0.5 * p["span_y"], ny)
        z_positions = np.linspace(-0.5 * p["span_z"], 0.5 * p["span_z"], nz)

        disp_axis = np.array([p["disp_axis_x"], p["disp_axis_y"], p["disp_axis_z"]], dtype=float)
        disp_norm = float(np.linalg.norm(disp_axis))
        if disp_norm <= 1e-8:
            disp_axis = np.array([0.0, 1.0, 0.35])
            disp_norm = float(np.linalg.norm(disp_axis))
        disp_axis /= disp_norm

        basis = normalized_basis(nx, ny, nz)
        omega_grid = np.zeros((nx, ny, nz))
        for mx in range(1, nx + 1):
            for my in range(1, ny + 1):
                for mz in range(1, nz + 1):
                    omega_grid[mx - 1, my - 1, mz - 1] = mode_frequency(
                        mx,
                        my,
                        mz,
                        nx,
                        ny,
                        nz,
                        mass_value,
                        onsite_k,
                        coupling_k,
                    )

        kick_initial = np.zeros((nx, ny, nz))
        kick_initial[kick_i, kick_j, kick_k] = p["kick_amplitude"]
        modal_coeffs = np.sum(basis * kick_initial[None, None, None, :, :, :], axis=(3, 4, 5))

        phase_tracker = ValueTracker(0.0)
        amplitude_tracker = ValueTracker(0.0)
        prep_tracker = ValueTracker(0.0)
        state = {"kind": "idle", "profile": np.zeros((nx, ny, nz)), "omega": 1.0}

        def displacement_field() -> np.ndarray:
            if state["kind"] == "mode":
                return amplitude_tracker.get_value() * state["profile"] * np.cos(phase_tracker.get_value())
            if state["kind"] == "prep":
                disp = np.zeros((nx, ny, nz))
                disp[kick_i, kick_j, kick_k] = prep_tracker.get_value()
                return disp
            if state["kind"] == "kick":
                t = phase_tracker.get_value()
                cos_terms = np.cos(omega_grid * t)
                return np.sum(
                    (modal_coeffs * cos_terms)[:, :, :, None, None, None] * basis,
                    axis=(0, 1, 2),
                )
            return np.zeros((nx, ny, nz))

        def equilibrium_point(i_idx: int, j_idx: int, k_idx: int) -> np.ndarray:
            return np.array([x_positions[i_idx], y_positions[j_idx], z_positions[k_idx]])

        def point(i_idx: int, j_idx: int, k_idx: int, disp: np.ndarray | None = None) -> np.ndarray:
            field = displacement_field() if disp is None else disp
            return equilibrium_point(i_idx, j_idx, k_idx) + disp_axis * field[i_idx, j_idx, k_idx]

        title = Text("Coupled 3D Oscillators", color=c["text"]).scale(p["title_scale"])
        eq_motion = MathTex(
            r"m\ddot q_{ijk}=-k_0 q_{ijk}-k_c\left(6q_{ijk}-\sum_{\mathrm{n.n.}} q\right)",
            color=c["text"],
        ).scale(p["formula_scale"])
        eq_mode = MathTex(
            r"q_{ijk}^{(a,b,c)}(t)=A\sin\!\left(\frac{a\pi i}{N_x+1}\right)\sin\!\left(\frac{b\pi j}{N_y+1}\right)\sin\!\left(\frac{c\pi k}{N_z+1}\right)\cos(\omega_{abc} t)",
            color=c["text"],
        ).scale(p["formula_scale"])
        eq_freq = MathTex(
            r"\omega_{abc}^2=\frac{k_0}{m}+\frac{4k_c}{m}\left[\sin^2\!\left(\frac{a\pi}{2(N_x+1)}\right)+\sin^2\!\left(\frac{b\pi}{2(N_y+1)}\right)+\sin^2\!\left(\frac{c\pi}{2(N_z+1)}\right)\right]",
            color=c["highlight"],
        ).scale(p["formula_scale"] * 0.92)
        eq_kick = MathTex(
            r"q_{ijk}(0)=A\,\delta_{i,i_0}\delta_{j,j_0}\delta_{k,k_0},\quad \dot q_{ijk}(0)=0",
            color=c["kick"],
        ).scale(p["formula_scale"])
        header = VGroup(title, eq_motion, eq_mode, eq_freq, eq_kick).arrange(DOWN, buff=0.07)
        header.to_edge(UP, buff=p["top_buff"])
        self.add_fixed_in_frame_mobjects(header)

        xmin, xmax = x_positions[0], x_positions[-1]
        ymin, ymax = y_positions[0], y_positions[-1]
        zmin, zmax = z_positions[0], z_positions[-1]
        corners = {
            "000": np.array([xmin, ymin, zmin]),
            "001": np.array([xmin, ymin, zmax]),
            "010": np.array([xmin, ymax, zmin]),
            "011": np.array([xmin, ymax, zmax]),
            "100": np.array([xmax, ymin, zmin]),
            "101": np.array([xmax, ymin, zmax]),
            "110": np.array([xmax, ymax, zmin]),
            "111": np.array([xmax, ymax, zmax]),
        }
        box_edges = VGroup(
            *[
                Line3D(corners[a], corners[b], thickness=0.01, color=c["border"])
                for a, b in [
                    ("000", "100"), ("000", "010"), ("000", "001"),
                    ("111", "011"), ("111", "101"), ("111", "110"),
                    ("100", "110"), ("100", "101"),
                    ("010", "110"), ("010", "011"),
                    ("001", "101"), ("001", "011"),
                ]
            ]
        )

        def make_lattice() -> VGroup:
            disp = displacement_field()
            group = VGroup()

            for i_idx in range(nx - 1):
                for j_idx in range(ny):
                    for k_idx in range(nz):
                        spring = spring_path_3d(
                            point(i_idx, j_idx, k_idx, disp),
                            point(i_idx + 1, j_idx, k_idx, disp),
                            p["spring_width"],
                            p["spring_turns"],
                        )
                        spring.set_stroke(c["springs_x"], width=p["spring_stroke"], opacity=0.88)
                        group.add(spring)

            for i_idx in range(nx):
                for j_idx in range(ny - 1):
                    for k_idx in range(nz):
                        spring = spring_path_3d(
                            point(i_idx, j_idx, k_idx, disp),
                            point(i_idx, j_idx + 1, k_idx, disp),
                            p["spring_width"],
                            p["spring_turns"],
                        )
                        spring.set_stroke(c["springs_y"], width=p["spring_stroke"], opacity=0.88)
                        group.add(spring)

            for i_idx in range(nx):
                for j_idx in range(ny):
                    for k_idx in range(nz - 1):
                        spring = spring_path_3d(
                            point(i_idx, j_idx, k_idx, disp),
                            point(i_idx, j_idx, k_idx + 1, disp),
                            p["spring_width"],
                            p["spring_turns"],
                        )
                        spring.set_stroke(c["springs_z"], width=p["spring_stroke"], opacity=0.86)
                        group.add(spring)

            for i_idx in range(nx):
                for j_idx in range(ny):
                    for k_idx in range(nz):
                        pos = point(i_idx, j_idx, k_idx, disp)
                        group.add(
                            Dot3D(
                                point=pos,
                                radius=p["node_radius"],
                                color=c["node"],
                                resolution=(5, 5),
                            ).set_opacity(p["node_opacity"])
                        )
            return group

        lattice = always_redraw(make_lattice)
        kick_marker = always_redraw(
            lambda: Dot3D(
                point=point(kick_i, kick_j, kick_k),
                radius=p["node_radius"] * 1.12,
                color=c["kick"],
                resolution=(6, 6),
            ).set_opacity(p["node_opacity"] if state["kind"] in {"prep", "kick"} else 0.0)
        )

        self.play(FadeIn(header, shift=0.12 * DOWN), run_time=p["intro_time"])
        self.play(Create(box_edges), Create(lattice), FadeIn(kick_marker), run_time=p["build_time"])

        for mx, my, mz in mode_triples:
            state["kind"] = "mode"
            state["profile"] = mode_profile(mx, my, mz, nx, ny, nz)
            state["omega"] = mode_frequency(mx, my, mz, nx, ny, nz, mass_value, onsite_k, coupling_k)
            phase_tracker.set_value(0.0)
            self.play(amplitude_tracker.animate.set_value(p["mode_amplitude"]), run_time=p["mode_ramp_time"])
            self.play(
                phase_tracker.animate.set_value(TAU * p["mode_periods"]),
                run_time=p["mode_periods"] * TAU / state["omega"],
                rate_func=linear,
            )
            self.play(amplitude_tracker.animate.set_value(0.0), run_time=p["mode_fade_time"])

        state["kind"] = "prep"
        prep_tracker.set_value(0.0)
        self.play(prep_tracker.animate.set_value(p["kick_amplitude"]), run_time=p["prep_time"])

        state["kind"] = "kick"
        phase_tracker.set_value(0.0)
        omega_min = float(np.min(omega_grid))
        self.play(
            prep_tracker.animate.set_value(0.0),
            phase_tracker.animate.set_value(TAU * p["kick_periods"]),
            run_time=p["kick_periods"] * TAU / omega_min,
            rate_func=linear,
        )
        self.wait(p["tail_wait"])
