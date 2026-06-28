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

    def get_bool(section, key, fallback=False):
        return cfg.getboolean(section, key, fallback=fallback)

    profile = get("manim", "profile", str, "shorts").strip().lower()
    if profile not in {"shorts", "widescreen"}:
        profile = "shorts"

    manim_defaults = {
        "shorts": {
            "pixel_width": 1080,
            "pixel_height": 1920,
            "frame_width": 9.0,
            "frame_height": 16.0,
        },
        "widescreen": {
            "pixel_width": 1920,
            "pixel_height": 1080,
            "frame_width": 16.0,
            "frame_height": 9.0,
        },
    }[profile]

    manim_params = {
        "profile": profile,
        # `profile` is the top-level switch for delivery format. If manual geometry
        # is needed later, introduce a separate `custom` profile instead of silently
        # mixing explicit dimensions with profile presets.
        "pixel_width": manim_defaults["pixel_width"],
        "pixel_height": manim_defaults["pixel_height"],
        "frame_width": manim_defaults["frame_width"],
        "frame_height": manim_defaults["frame_height"],
        "frame_rate": get("manim", "frame_rate", int, 60),
        "background_color": get("manim", "background_color", str, "#000000"),
    }

    layout_defaults = {
        "shorts": {
            "lattice_shift_x": 0.22,
            "lattice_shift_y": 0.85,
            "top_buff": 0.12,
            "camera_zoom": 1.02,
            "frame_center_y": 1.35,
            "frame_center_z": -0.08,
        },
        "widescreen": {
            "lattice_shift_x": 0.28,
            "lattice_shift_y": 0.15,
            "top_buff": 0.14,
            "camera_zoom": 1.02,
            "frame_center_y": 0.45,
            "frame_center_z": -0.12,
        },
    }[profile]

    lattice = {
        "nx": get("lattice", "nx", int, 9),
        "ny": get("lattice", "ny", int, 9),
        "mode_list": get("lattice", "mode_list", str, "1x1,1x2,2x1,2x2"),
        "mass_value": get("lattice", "mass_value", float, 1.0),
        "onsite_k": get("lattice", "onsite_k", float, 5.0),
        "coupling_k": get("lattice", "coupling_k", float, 3.0),
        "mode_amplitude": get("lattice", "mode_amplitude", float, 0.48),
        "kick_i": get("lattice", "kick_i", int, 4),
        "kick_j": get("lattice", "kick_j", int, 4),
        "kick_amplitude": get("lattice", "kick_amplitude", float, 0.55),
        "span_x": get("lattice", "span_x", float, 5.6),
        "span_y": get("lattice", "span_y", float, 5.6),
        "lattice_shift_x": get("lattice", "lattice_shift_x", float, layout_defaults["lattice_shift_x"]),
        "lattice_shift_y": get("lattice", "lattice_shift_y", float, layout_defaults["lattice_shift_y"]),
        "lattice_shift_z": get("lattice", "lattice_shift_z", float, 0.0),
        "support_z": get("lattice", "support_z", float, 0.9),
        "show_mesh": get_bool("lattice", "show_mesh", False),
        "show_coupling_springs": get_bool("lattice", "show_coupling_springs", True),
        "show_vertical_springs": get_bool("lattice", "show_vertical_springs", True),
        "mesh_stroke": get("lattice", "mesh_stroke", float, 2.6),
        "spring_width": get("lattice", "spring_width", float, 0.06),
        "spring_turns": get("lattice", "spring_turns", int, 5),
        "spring_stroke": get("lattice", "spring_stroke", float, 2.0),
        "vertical_spring_width": get("lattice", "vertical_spring_width", float, 0.035),
        "vertical_spring_turns": get("lattice", "vertical_spring_turns", int, 6),
        "vertical_spring_stroke": get("lattice", "vertical_spring_stroke", float, 1.8),
        "border_stroke": get("lattice", "border_stroke", float, 4.0),
        "node_radius": get("lattice", "node_radius", float, 0.045),
        "node_opacity": get("lattice", "node_opacity", float, 0.9),
        "title_scale": get("lattice", "title_scale", float, 0.56),
        "formula_scale": get("lattice", "formula_scale", float, 0.43),
        "state_scale": get("lattice", "state_scale", float, 0.50),
        "top_buff": get("lattice", "top_buff", float, layout_defaults["top_buff"]),
        "camera_phi": get("lattice", "camera_phi", float, 68.0),
        "camera_theta": get("lattice", "camera_theta", float, -42.0),
        "camera_zoom": get("lattice", "camera_zoom", float, layout_defaults["camera_zoom"]),
        "frame_center_x": get("lattice", "frame_center_x", float, 0.0),
        "frame_center_y": get("lattice", "frame_center_y", float, layout_defaults["frame_center_y"]),
        "frame_center_z": get("lattice", "frame_center_z", float, layout_defaults["frame_center_z"]),
        "intro_time": get("lattice", "intro_time", float, 1.0),
        "build_time": get("lattice", "build_time", float, 1.8),
        "mode_ramp_time": get("lattice", "mode_ramp_time", float, 0.45),
        "mode_periods": get("lattice", "mode_periods", float, 2.0),
        "mode_fade_time": get("lattice", "mode_fade_time", float, 0.35),
        "prep_time": get("lattice", "prep_time", float, 0.8),
        "kick_periods": get("lattice", "kick_periods", float, 3.6),
        "continuum_amplitude": get("lattice", "continuum_amplitude", float, 0.42),
        "continuum_speed": get("lattice", "continuum_speed", float, 2.2),
        "continuum_mass": get("lattice", "continuum_mass", float, 0.0),
        "continuum_sigma": get("lattice", "continuum_sigma", float, 0.42),
        "continuum_height_scale": get("lattice", "continuum_height_scale", float, 1.0),
        "continuum_grid": get("lattice", "continuum_grid", int, 48),
        "continuum_time_samples": get("lattice", "continuum_time_samples", int, 120),
        "continuum_resolution": get("lattice", "continuum_resolution", int, 28),
        "continuum_fade_time": get("lattice", "continuum_fade_time", float, 1.4),
        "field_header_time": get("lattice", "field_header_time", float, 0.8),
        "continuum_hold_time": get("lattice", "continuum_hold_time", float, 0.5),
        "continuum_runtime": get("lattice", "continuum_runtime", float, 4.6 * TAU / 2.2),
        "continuum_periods": get("lattice", "continuum_periods", float, 2.2),
        "tail_wait": get("lattice", "tail_wait", float, 0.8),
    }

    colors = {
        "text": get("colors", "text", str, "#FFFFFF"),
        "border": get("colors", "border", str, "#FFFFFF"),
        "support": get("colors", "support", str, "#DADADA"),
        "mesh": get("colors", "mesh", str, "#79C7FF"),
        "mesh_dim": get("colors", "mesh_dim", str, "#3E5466"),
        "vertical_springs": get("colors", "vertical_springs", str, "#E5E5E5"),
        "highlight": get("colors", "highlight", str, "#FFD166"),
        "kick": get("colors", "kick", str, "#FF7B72"),
        "node": get("colors", "node", str, "#9FD7FF"),
        "continuum_fill": get("colors", "continuum_fill", str, "#65B9FF"),
        "continuum_stroke": get("colors", "continuum_stroke", str, "#BCE5FF"),
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


def parse_mode_pairs(raw: str) -> list[tuple[int, int]]:
    pairs: list[tuple[int, int]] = []
    for token in re.split(r"[,;]+", raw):
        token = token.strip().lower()
        if not token:
            continue
        match = re.match(r"(\d+)\s*[x*]\s*(\d+)", token)
        if not match:
            continue
        pairs.append((int(match.group(1)), int(match.group(2))))
    return pairs


def normalized_basis(nx: int, ny: int) -> np.ndarray:
    i = np.arange(1, nx + 1)[:, None]
    j = np.arange(1, ny + 1)[None, :]
    basis = np.zeros((nx, ny, nx, ny))
    norm = np.sqrt(2.0 / (nx + 1)) * np.sqrt(2.0 / (ny + 1))
    for mx in range(1, nx + 1):
        sin_x = np.sin(mx * PI * i / (nx + 1))
        for my in range(1, ny + 1):
            sin_y = np.sin(my * PI * j / (ny + 1))
            basis[mx - 1, my - 1] = norm * (sin_x @ sin_y)
    return basis


def mode_profile(mx: int, my: int, nx: int, ny: int) -> np.ndarray:
    profile = normalized_basis(nx, ny)[mx - 1, my - 1]
    peak = np.max(np.abs(profile))
    return profile / peak if peak > 0 else profile


def mode_frequency(mx: int, my: int, nx: int, ny: int, mass_value: float, onsite_k: float, coupling_k: float) -> float:
    sx = np.sin(mx * PI / (2.0 * (nx + 1)))
    sy = np.sin(my * PI / (2.0 * (ny + 1)))
    return np.sqrt((onsite_k + 4.0 * coupling_k * (sx**2 + sy**2)) / mass_value)


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


def approximate_j0(z_values: np.ndarray, angle_samples: int = 96) -> np.ndarray:
    theta = np.linspace(0.0, TAU, max(24, angle_samples), endpoint=False)
    cos_theta = np.cos(theta)
    return np.mean(np.cos(z_values[..., None] * cos_theta), axis=-1)


def precompute_continuum_klein_gordon(
    r_max: float,
    radial_samples: int,
    k_samples: int,
    time_samples: int,
    duration: float,
    amplitude: float,
    sigma: float,
    mass: float,
    speed: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    radial_count = max(64, radial_samples)
    momentum_count = max(160, k_samples)
    safe_sigma = max(float(sigma), 1e-3)
    r_grid = np.linspace(0.0, max(float(r_max), 1e-3), radial_count)
    k_max = 8.0 / safe_sigma + 2.5 * mass / max(float(speed), 1e-3)
    k_grid = np.linspace(0.0, k_max, momentum_count)

    kr = np.outer(r_grid, k_grid)
    j0_matrix = approximate_j0(kr)
    gaussian = np.exp(-0.5 * safe_sigma**2 * k_grid**2)
    omega = np.sqrt((speed * k_grid) ** 2 + mass**2)
    weights = k_grid * gaussian

    t_grid = np.linspace(0.0, duration, max(2, time_samples))
    radial_frames = np.zeros((len(t_grid), len(r_grid)))
    for idx, t_val in enumerate(t_grid):
        integrand = j0_matrix * (weights * np.cos(omega * t_val))[None, :]
        radial_frames[idx] = amplitude * safe_sigma**2 * np.trapezoid(integrand, x=k_grid, axis=1)

    return r_grid, t_grid, radial_frames


def radial_sample(r_grid: np.ndarray, radial_profile: np.ndarray, radius: float) -> float:
    if radius <= r_grid[0]:
        return float(radial_profile[0])
    if radius >= r_grid[-1]:
        return float(radial_profile[-1])

    idx = int(np.searchsorted(r_grid, radius, side="right") - 1)
    idx = max(0, min(idx, len(r_grid) - 2))
    dr = max(float(r_grid[idx + 1] - r_grid[idx]), 1e-8)
    tau = (radius - r_grid[idx]) / dr
    return float((1.0 - tau) * radial_profile[idx] + tau * radial_profile[idx + 1])


class CoupledOscillators2D(ThreeDScene):
    def construct(self):
        p = CFG["lattice"]
        c = CFG["colors"]
        self.camera.background_color = CFG["manim"]["background_color"]
        self.set_camera_orientation(
            phi=p["camera_phi"] * DEGREES,
            theta=p["camera_theta"] * DEGREES,
            zoom=p["camera_zoom"],
            frame_center=np.array([p["frame_center_x"], p["frame_center_y"], p["frame_center_z"]]),
        )

        nx = max(2, p["nx"])
        ny = max(2, p["ny"])
        mode_pairs = [pair for pair in parse_mode_pairs(p["mode_list"]) if 1 <= pair[0] <= nx and 1 <= pair[1] <= ny]
        if not mode_pairs:
            mode_pairs = [(1, 1), (1, 2), (2, 1), (2, 2)]

        mass_value = p["mass_value"]
        onsite_k = p["onsite_k"]
        coupling_k = p["coupling_k"]
        kick_i = max(0, min(nx - 1, p["kick_i"]))
        kick_j = max(0, min(ny - 1, p["kick_j"]))

        x_positions = np.linspace(-0.5 * p["span_x"], 0.5 * p["span_x"], nx)
        y_positions = np.linspace(-0.5 * p["span_y"], 0.5 * p["span_y"], ny)
        lattice_shift = np.array([p["lattice_shift_x"], p["lattice_shift_y"], p["lattice_shift_z"]])

        basis = normalized_basis(nx, ny)
        omega_grid = np.zeros((nx, ny))
        for mx in range(1, nx + 1):
            for my in range(1, ny + 1):
                omega_grid[mx - 1, my - 1] = mode_frequency(mx, my, nx, ny, mass_value, onsite_k, coupling_k)

        kick_initial = np.zeros((nx, ny))
        kick_initial[kick_i, kick_j] = p["kick_amplitude"]
        modal_coeffs = np.sum(basis * kick_initial[None, None, :, :], axis=(2, 3))

        phase_tracker = ValueTracker(0.0)
        amplitude_tracker = ValueTracker(0.0)
        prep_tracker = ValueTracker(0.0)
        state = {"kind": "idle", "profile": np.zeros((nx, ny)), "omega": 1.0}

        def displacement_field() -> np.ndarray:
            if state["kind"] == "mode":
                return amplitude_tracker.get_value() * state["profile"] * np.cos(phase_tracker.get_value())
            if state["kind"] == "prep":
                disp = np.zeros((nx, ny))
                disp[kick_i, kick_j] = prep_tracker.get_value()
                return disp
            if state["kind"] == "kick":
                t = phase_tracker.get_value()
                cos_terms = np.cos(omega_grid * t)
                return np.sum(
                    (modal_coeffs * cos_terms)[:, :, None, None] * basis,
                    axis=(0, 1),
                )
            return np.zeros((nx, ny))

        def point(i_idx: int, j_idx: int, disp: np.ndarray | None = None) -> np.ndarray:
            field = displacement_field() if disp is None else disp
            return np.array([x_positions[i_idx], y_positions[j_idx], field[i_idx, j_idx]]) + lattice_shift

        def support_point(i_idx: int, j_idx: int) -> np.ndarray:
            return np.array([x_positions[i_idx], y_positions[j_idx], p["support_z"]]) + lattice_shift

        title = Text("Coupled 2D Oscillators", color=c["text"]).scale(p["title_scale"])
        eq_motion = MathTex(
            r"m\ddot q_{ij}=-k_0 q_{ij}-k_c(4q_{ij}-q_{i+1,j}-q_{i-1,j}-q_{i,j+1}-q_{i,j-1})",
            color=c["text"],
        ).scale(p["formula_scale"])
        eq_mode = MathTex(
            r"q_{ij}^{(m,n)}(t)=A_{mn}\sin\!\left(\frac{m\pi i}{N_x+1}\right)\sin\!\left(\frac{n\pi j}{N_y+1}\right)\cos(\omega_{mn} t)",
            color=c["text"],
        ).scale(p["formula_scale"])
        eq_freq = MathTex(
            r"\omega_{mn}^2=\frac{k_0}{m}+\frac{4k_c}{m}\left[\sin^2\!\left(\frac{m\pi}{2(N_x+1)}\right)+\sin^2\!\left(\frac{n\pi}{2(N_y+1)}\right)\right]",
            color=c["highlight"],
        ).scale(p["formula_scale"])
        eq_kick = MathTex(
            r"q_{ij}(0)=A\,\delta_{i,i_0}\delta_{j,j_0},\quad \dot q_{ij}(0)=0",
            color=c["kick"],
        ).scale(p["formula_scale"])
        discrete_header = VGroup(title, eq_motion, eq_mode, eq_freq, eq_kick).arrange(DOWN, buff=0.08)
        discrete_header.to_edge(UP, buff=p["top_buff"])
        self.add_fixed_in_frame_mobjects(discrete_header)

        continuum_title = Text("Field", color=c["text"]).scale(p["title_scale"])
        continuum_motion_label = Text("Equation of motion", color=c["text"]).scale(p["state_scale"] * 0.9)
        continuum_pde = MathTex(
            r"(\partial_t^2-c^2\nabla^2+m^2)\phi=0",
            color=c["text"],
        ).scale(p["formula_scale"])
        continuum_kick_label = Text("Field kick", color=c["kick"]).scale(p["state_scale"] * 0.9)
        continuum_init = MathTex(
            r"\phi(0,r)=A e^{-r^2/(2\sigma^2)},\quad \dot\phi(0,r)=0",
            color=c["kick"],
        ).scale(p["formula_scale"])
        continuum_solution = MathTex(
            r"\phi(t,r)=A\sigma^2\int_0^\infty k\,dk\,J_0(kr)e^{-\sigma^2 k^2/2}\cos\!\left(t\sqrt{c^2k^2+m^2}\right)",
            color=c["highlight"],
        ).scale(p["formula_scale"] * 0.9)
        continuum_header = VGroup(
            continuum_title,
            continuum_motion_label,
            continuum_pde,
            continuum_kick_label,
            continuum_init,
            continuum_solution,
        ).arrange(DOWN, buff=0.08)
        continuum_header.to_edge(UP, buff=p["top_buff"])
        continuum_header.set_opacity(0.0)
        self.add_fixed_in_frame_mobjects(continuum_header)
        field_label = Text("Initial Gaussian packet", color=c["highlight"]).scale(p["state_scale"])
        field_label.next_to(continuum_header, DOWN, buff=0.16)
        field_label.set_opacity(0.0)
        self.add_fixed_in_frame_mobjects(field_label)

        border = Polygon(
            np.array([x_positions[0], y_positions[0], 0.0]) + lattice_shift,
            np.array([x_positions[-1], y_positions[0], 0.0]) + lattice_shift,
            np.array([x_positions[-1], y_positions[-1], 0.0]) + lattice_shift,
            np.array([x_positions[0], y_positions[-1], 0.0]) + lattice_shift,
            color=c["border"],
            stroke_width=p["border_stroke"],
        )
        support_border = Polygon(
            np.array([x_positions[0], y_positions[0], p["support_z"]]) + lattice_shift,
            np.array([x_positions[-1], y_positions[0], p["support_z"]]) + lattice_shift,
            np.array([x_positions[-1], y_positions[-1], p["support_z"]]) + lattice_shift,
            np.array([x_positions[0], y_positions[-1], p["support_z"]]) + lattice_shift,
            color=c["support"],
            stroke_width=p["border_stroke"] * 0.8,
        )

        mesh_opacity = ValueTracker(1.0)
        continuum_opacity = ValueTracker(0.0)

        def make_mesh() -> VGroup:
            disp = displacement_field()
            group = VGroup()
            alpha = mesh_opacity.get_value()

            if p["show_vertical_springs"]:
                for i_idx in range(nx):
                    for j_idx in range(ny):
                        spring = spring_path_3d(
                            support_point(i_idx, j_idx),
                            point(i_idx, j_idx, disp),
                            p["vertical_spring_width"],
                            p["vertical_spring_turns"],
                        )
                        spring.set_stroke(c["vertical_springs"], width=p["vertical_spring_stroke"], opacity=0.92 * alpha)
                        group.add(spring)

            if p["show_mesh"]:
                for j_idx in range(ny):
                    line = VMobject()
                    line.set_points_as_corners([point(i_idx, j_idx, disp) for i_idx in range(nx)])
                    line.set_stroke(c["mesh"], width=p["mesh_stroke"], opacity=0.95 * alpha)
                    group.add(line)

                for i_idx in range(nx):
                    line = VMobject()
                    line.set_points_as_corners([point(i_idx, j_idx, disp) for j_idx in range(ny)])
                    line.set_stroke(c["mesh_dim"], width=p["mesh_stroke"] * 0.95, opacity=0.95 * alpha)
                    group.add(line)

            if p["show_coupling_springs"]:
                for i_idx in range(nx - 1):
                    for j_idx in range(ny):
                        spring = spring_path_3d(
                            point(i_idx, j_idx, disp),
                            point(i_idx + 1, j_idx, disp),
                            p["spring_width"],
                            p["spring_turns"],
                        )
                        spring.set_stroke(c["mesh"], width=p["spring_stroke"], opacity=0.95 * alpha)
                        group.add(spring)

                for i_idx in range(nx):
                    for j_idx in range(ny - 1):
                        spring = spring_path_3d(
                            point(i_idx, j_idx, disp),
                            point(i_idx, j_idx + 1, disp),
                            p["spring_width"],
                            p["spring_turns"],
                        )
                        spring.set_stroke(c["mesh_dim"], width=p["spring_stroke"], opacity=0.95 * alpha)
                        group.add(spring)

            return group

        continuum_duration = max(p["continuum_runtime"], 0.1)
        continuum_tracker = ValueTracker(0.0)
        radial_max = float(
            np.sqrt(
                max(abs(x_positions[0]), abs(x_positions[-1])) ** 2
                + max(abs(y_positions[0]), abs(y_positions[-1])) ** 2
            )
        )
        continuum_r, continuum_t, continuum_frames = precompute_continuum_klein_gordon(
            radial_max,
            p["continuum_grid"],
            p["continuum_grid"] * 6,
            p["continuum_time_samples"],
            continuum_duration,
            p["continuum_amplitude"],
            p["continuum_sigma"],
            p["continuum_mass"],
            p["continuum_speed"],
        )

        def continuum_height(x_val: float, y_val: float) -> float:
            t_now = float(np.clip(continuum_tracker.get_value(), continuum_t[0], continuum_t[-1]))
            tidx = int(np.searchsorted(continuum_t, t_now, side="right") - 1)
            tidx = max(0, min(tidx, len(continuum_t) - 2))
            dt = max(float(continuum_t[tidx + 1] - continuum_t[tidx]), 1e-8)
            tau = (t_now - continuum_t[tidx]) / dt
            radius = float(np.sqrt((x_val - lattice_shift[0]) ** 2 + (y_val - lattice_shift[1]) ** 2))
            f0 = radial_sample(continuum_r, continuum_frames[tidx], radius)
            f1 = radial_sample(continuum_r, continuum_frames[tidx + 1], radius)
            return lattice_shift[2] + p["continuum_height_scale"] * ((1.0 - tau) * f0 + tau * f1)

        def make_continuum_surface() -> Surface:
            surface = Surface(
                lambda u, v: np.array([u, v, continuum_height(u, v)]),
                u_range=[x_positions[0], x_positions[-1]],
                v_range=[y_positions[0], y_positions[-1]],
                resolution=(p["continuum_resolution"], p["continuum_resolution"]),
            )
            alpha = continuum_opacity.get_value()
            surface.set_fill(c["continuum_fill"], opacity=0.72 * alpha)
            surface.set_stroke(c["continuum_stroke"], width=0.9, opacity=0.85 * alpha)
            return surface

        mesh = always_redraw(make_mesh)
        continuum_surface = always_redraw(make_continuum_surface)
        kick_marker = always_redraw(
            lambda: Dot3D(
                point=point(kick_i, kick_j),
                radius=p["node_radius"],
                color=c["kick"],
                resolution=(5, 5),
            ).set_opacity(p["node_opacity"] * mesh_opacity.get_value() if state["kind"] in {"prep", "kick"} else 0.0)
        )
        self.play(FadeIn(discrete_header, shift=0.12 * DOWN), run_time=p["intro_time"])
        self.play(Create(support_border), Create(border), Create(mesh), FadeIn(kick_marker), run_time=p["build_time"])

        state_label: Mobject | None = None

        for mx, my in mode_pairs:
            state["kind"] = "mode"
            state["profile"] = mode_profile(mx, my, nx, ny)
            state["omega"] = mode_frequency(mx, my, nx, ny, mass_value, onsite_k, coupling_k)
            phase_tracker.set_value(0.0)
            next_state = Text(
                f"Mode ({mx},{my}), w = {state['omega']:.2f}",
                color=c["highlight"],
            ).scale(p["state_scale"])
            next_state.next_to(discrete_header, DOWN, buff=0.18)
            self.add_fixed_in_frame_mobjects(next_state)
            animations = [amplitude_tracker.animate.set_value(p["mode_amplitude"])]
            if state_label is None:
                animations.append(FadeIn(next_state, shift=0.08 * DOWN))
            else:
                animations.extend(
                    [
                        FadeOut(state_label, shift=0.06 * UP),
                        FadeIn(next_state, shift=0.06 * DOWN),
                    ]
                )
            self.play(*animations, run_time=p["mode_ramp_time"])
            state_label = next_state
            self.play(
                phase_tracker.animate.set_value(TAU * p["mode_periods"]),
                run_time=p["mode_periods"] * TAU / state["omega"],
                rate_func=linear,
            )
            self.play(amplitude_tracker.animate.set_value(0.0), run_time=p["mode_fade_time"])

        state["kind"] = "prep"
        prep_tracker.set_value(0.0)
        kick_state = Text(
            f"Local kick at ({kick_i+1},{kick_j+1})",
            color=c["kick"],
        ).scale(p["state_scale"])
        kick_state.next_to(discrete_header, DOWN, buff=0.18)
        self.add_fixed_in_frame_mobjects(kick_state)
        if state_label is None:
            self.play(FadeIn(kick_state, shift=0.06 * DOWN), run_time=0.2)
        else:
            self.play(
                FadeOut(state_label, shift=0.06 * UP),
                FadeIn(kick_state, shift=0.06 * DOWN),
                run_time=0.2,
            )
        state_label = kick_state
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
        continuum_tracker.set_value(0.0)
        self.add(continuum_surface)
        self.play(
            FadeOut(discrete_header),
            FadeOut(state_label) if state_label is not None else AnimationGroup(),
            run_time=0.4,
        )

        state["kind"] = "prep"
        prep_tracker.set_value(0.0)
        self.play(
            prep_tracker.animate.set_value(p["kick_amplitude"]),
            continuum_opacity.animate.set_value(0.26),
            run_time=p["prep_time"],
        )
        self.play(
            prep_tracker.animate.set_value(0.0),
            mesh_opacity.animate.set_value(0.0),
            border.animate.set_opacity(0.0),
            support_border.animate.set_opacity(0.0),
            continuum_opacity.animate.set_value(1.0),
            run_time=p["continuum_fade_time"],
        )
        self.wait(p["continuum_hold_time"])
        self.play(
            continuum_header.animate.set_opacity(1.0),
            field_label.animate.set_opacity(1.0),
            run_time=p["field_header_time"],
        )
        evolving_field_label = Text("Field wave evolution", color=c["highlight"]).scale(p["state_scale"])
        evolving_field_label.move_to(field_label)
        evolving_field_label.set_opacity(0.0)
        self.add_fixed_in_frame_mobjects(evolving_field_label)
        self.play(
            field_label.animate.set_opacity(0.0),
            run_time=0.15,
        )
        self.play(
            evolving_field_label.animate.set_opacity(1.0),
            continuum_tracker.animate.set_value(continuum_duration),
            run_time=continuum_duration,
            rate_func=linear,
        )
        self.wait(p["tail_wait"])
