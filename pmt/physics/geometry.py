from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from pmt.physics.types import ELEMENTARY_CHARGE, GeometrySpec, SimulationConfig


@dataclass(frozen=True)
class GeometryData:
    fixed_mask: np.ndarray
    fixed_values: np.ndarray
    electrode_id: np.ndarray

    boundary_mask: np.ndarray
    cathode_mask: np.ndarray
    receiver_mask: np.ndarray
    focus_mask: np.ndarray

    emission_mode: str
    emission_x_m: float
    emission_y_min_m: float
    emission_y_max_m: float

    photocathode_center_x_m: float
    photocathode_center_y_m: float
    emission_radius_m: float
    emission_theta_min_rad: float
    emission_theta_max_rad: float
    emission_offset_m: float

    launch_point_x_m: float
    launch_point_y_m: float


def make_grid(cfg: SimulationConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float]:
    x_axis = np.linspace(-0.5 * cfg.grid.lx_m, 0.5 * cfg.grid.lx_m, cfg.grid.nx)
    y_axis = np.linspace(-0.5 * cfg.grid.ly_m, 0.5 * cfg.grid.ly_m, cfg.grid.ny)
    x_grid, y_grid = np.meshgrid(x_axis, y_axis, indexing="ij")
    dx = float(x_axis[1] - x_axis[0])
    dy = float(y_axis[1] - y_axis[0])
    return x_axis, y_axis, x_grid, y_grid, dx, dy


def _rotated_rectangle_mask(
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    center_x_m: float,
    center_y_m: float,
    length_m: float,
    thickness_m: float,
    angle_deg: float,
) -> np.ndarray:
    theta = math.radians(angle_deg)
    c = math.cos(theta)
    s = math.sin(theta)
    rel_x = x_grid - center_x_m
    rel_y = y_grid - center_y_m
    along = c * rel_x + s * rel_y
    normal = -s * rel_x + c * rel_y
    return (np.abs(along) <= 0.5 * length_m) & (np.abs(normal) <= 0.5 * thickness_m)


def _build_cathode_mask(geometry: GeometrySpec, x_grid: np.ndarray, y_grid: np.ndarray, dx: float, dy: float) -> tuple[np.ndarray, dict[str, float]]:
    shape = geometry.cathode_shape.strip().lower()

    emission = {
        "mode": "line",
        "x_m": min(
            geometry.line_cathode_x_m + 0.5 * geometry.line_cathode_thickness_m + 1.5 * dx,
            float(np.max(x_grid) - dx),
        ),
        "y_min_m": -0.5 * geometry.line_cathode_height_m,
        "y_max_m": 0.5 * geometry.line_cathode_height_m,
        "photocathode_center_x_m": geometry.line_cathode_x_m,
        "photocathode_center_y_m": 0.0,
        "radius_m": float("nan"),
        "theta_min_rad": float("nan"),
        "theta_max_rad": float("nan"),
        "offset_m": max(0.75 * min(dx, dy), 1e-6),
    }

    if shape != "hemisphere":
        mask = (
            (np.abs(x_grid - geometry.line_cathode_x_m) <= 0.5 * geometry.line_cathode_thickness_m)
            & (np.abs(y_grid) <= 0.5 * geometry.line_cathode_height_m)
        )
        return mask, emission

    radius = max(0.5 * geometry.photocathode_diameter_m, 2.0 * min(dx, dy))
    thickness = min(max(geometry.photocathode_thickness_m, min(dx, dy)), 0.85 * radius)
    inner_radius = radius - thickness

    xc = geometry.photocathode_center_x_m
    yc = geometry.photocathode_center_y_m

    rel_x = x_grid - xc
    rel_y = y_grid - yc
    radial = np.sqrt(rel_x * rel_x + rel_y * rel_y)
    mask = (radial <= radius) & (radial >= inner_radius) & (x_grid <= xc)

    theta_min = math.radians(geometry.photocathode_active_theta_min_deg)
    theta_max = math.radians(geometry.photocathode_active_theta_max_deg)
    theta_min = float(np.clip(theta_min, -0.5 * math.pi + 1e-4, 0.5 * math.pi - 1e-4))
    theta_max = float(np.clip(theta_max, -0.5 * math.pi + 1e-4, 0.5 * math.pi - 1e-4))
    if theta_min > theta_max:
        theta_min, theta_max = theta_max, theta_min

    emission.update(
        {
            "mode": "hemisphere",
            "photocathode_center_x_m": xc,
            "photocathode_center_y_m": yc,
            "radius_m": max(inner_radius - 0.6 * min(dx, dy), 0.5 * inner_radius),
            "theta_min_rad": theta_min,
            "theta_max_rad": theta_max,
            "offset_m": max(0.75 * min(dx, dy), 1e-6),
        }
    )
    return mask, emission


def _build_receiver_mask(geometry: GeometrySpec, x_grid: np.ndarray, y_grid: np.ndarray) -> np.ndarray:
    kind = geometry.receiver_kind.strip().lower()
    if kind == "point":
        dx = x_grid - geometry.receiver_point_x_m
        dy = y_grid - geometry.receiver_point_y_m
        return (dx * dx + dy * dy) <= geometry.receiver_radius_m * geometry.receiver_radius_m

    return _rotated_rectangle_mask(
        x_grid,
        y_grid,
        center_x_m=geometry.plate_center_x_m,
        center_y_m=geometry.plate_center_y_m,
        length_m=geometry.plate_length_m,
        thickness_m=geometry.plate_thickness_m,
        angle_deg=geometry.plate_angle_deg,
    )


def _build_focus_mask(geometry: GeometrySpec, x_grid: np.ndarray, y_grid: np.ndarray) -> np.ndarray:
    if not geometry.focus_enabled:
        return np.zeros_like(x_grid, dtype=bool)

    kind = geometry.focus_kind.strip().lower()
    if kind == "point":
        dx = x_grid - geometry.focus_center_x_m
        dy = y_grid - geometry.focus_center_y_m
        radius = 0.5 * max(geometry.focus_length_m, geometry.focus_thickness_m)
        return (dx * dx + dy * dy) <= radius * radius

    return _rotated_rectangle_mask(
        x_grid,
        y_grid,
        center_x_m=geometry.focus_center_x_m,
        center_y_m=geometry.focus_center_y_m,
        length_m=geometry.focus_length_m,
        thickness_m=geometry.focus_thickness_m,
        angle_deg=geometry.focus_angle_deg,
    )


def build_geometry(cfg: SimulationConfig, x_grid: np.ndarray, y_grid: np.ndarray, dx: float, dy: float) -> GeometryData:
    boundary_mask = np.zeros_like(x_grid, dtype=bool)
    boundary_mask[0, :] = True
    boundary_mask[-1, :] = True
    boundary_mask[:, 0] = True
    boundary_mask[:, -1] = True

    cathode_mask, emission = _build_cathode_mask(cfg.geometry, x_grid, y_grid, dx=dx, dy=dy)
    if not cfg.geometry.enable_cathode:
        cathode_mask = np.zeros_like(x_grid, dtype=bool)

    receiver_mask = _build_receiver_mask(cfg.geometry, x_grid, y_grid)
    if not cfg.geometry.enable_receiver:
        receiver_mask = np.zeros_like(x_grid, dtype=bool)
    focus_mask = _build_focus_mask(cfg.geometry, x_grid, y_grid)

    fixed_mask = boundary_mask | cathode_mask | receiver_mask | focus_mask
    fixed_values = np.full_like(x_grid, cfg.electrodes.background_voltage, dtype=float)
    fixed_values[boundary_mask] = cfg.electrodes.background_voltage
    fixed_values[cathode_mask] = cfg.electrodes.cathode_voltage
    fixed_values[receiver_mask] = cfg.electrodes.anode_voltage
    fixed_values[focus_mask] = cfg.electrodes.focus_voltage

    electrode_id = np.zeros_like(x_grid, dtype=np.uint8)
    electrode_id[boundary_mask] = 4
    electrode_id[cathode_mask] = 1
    electrode_id[receiver_mask] = 2
    electrode_id[focus_mask] = 3

    return GeometryData(
        fixed_mask=fixed_mask,
        fixed_values=fixed_values,
        electrode_id=electrode_id,
        boundary_mask=boundary_mask,
        cathode_mask=cathode_mask,
        receiver_mask=receiver_mask,
        focus_mask=focus_mask,
        emission_mode=str(emission["mode"]),
        emission_x_m=float(emission["x_m"]),
        emission_y_min_m=float(emission["y_min_m"]),
        emission_y_max_m=float(emission["y_max_m"]),
        photocathode_center_x_m=float(emission["photocathode_center_x_m"]),
        photocathode_center_y_m=float(emission["photocathode_center_y_m"]),
        emission_radius_m=float(emission["radius_m"]),
        emission_theta_min_rad=float(emission["theta_min_rad"]),
        emission_theta_max_rad=float(emission["theta_max_rad"]),
        emission_offset_m=float(emission["offset_m"]),
        launch_point_x_m=cfg.geometry.launch_point_x_m,
        launch_point_y_m=cfg.geometry.launch_point_y_m,
    )


def launch_electrons(
    cfg: SimulationConfig,
    geometry: GeometryData,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    electrons = cfg.electrons

    energy_ev = rng.normal(electrons.initial_energy_ev, electrons.initial_energy_spread_ev, size=electrons.count)
    energy_ev = np.maximum(0.01, energy_ev)
    speed = np.sqrt(2.0 * energy_ev * ELEMENTARY_CHARGE / electrons.mass_kg)

    angle_spread_rad = math.radians(electrons.angle_spread_deg)
    launch_angle_rad = math.radians(electrons.launch_angle_deg)

    if cfg.scene.launch_mode == "point":
        x0 = np.full(electrons.count, geometry.launch_point_x_m, dtype=float)
        y0 = np.full(electrons.count, geometry.launch_point_y_m, dtype=float)
        if electrons.emission_jitter_m > 0.0:
            y0 += rng.normal(0.0, electrons.emission_jitter_m, size=electrons.count)

        if cfg.scene.kind == "point_field" or (
            abs(electrons.point_radial_velocity_m_s) > 0.0
            or abs(electrons.point_transverse_velocity_m_s) > 0.0
            or electrons.point_transverse_velocity_spread_m_s > 0.0
        ):
            cx = cfg.central.center_x_m
            cy = cfg.central.center_y_m

            rx = cx - x0
            ry = cy - y0
            r = np.hypot(rx, ry)
            inv_r = np.zeros_like(r)
            good = r > 1e-12
            inv_r[good] = 1.0 / r[good]

            erx = np.where(good, rx * inv_r, 1.0)
            ery = np.where(good, ry * inv_r, 0.0)
            etx = -ery
            ety = erx

            vr = np.full(electrons.count, electrons.point_radial_velocity_m_s, dtype=float)
            vt = np.full(electrons.count, electrons.point_transverse_velocity_m_s, dtype=float)
            if electrons.point_transverse_velocity_spread_m_s > 0.0:
                vt += rng.normal(0.0, electrons.point_transverse_velocity_spread_m_s, size=electrons.count)

            vx = vr * erx + vt * etx
            vy = vr * ery + vt * ety
        else:
            angle = launch_angle_rad + rng.normal(0.0, angle_spread_rad, size=electrons.count)
            vx = speed * np.cos(angle)
            vy = speed * np.sin(angle)
        return np.column_stack((x0, y0)), np.column_stack((vx, vy))

    if geometry.emission_mode == "hemisphere" and np.isfinite(geometry.emission_radius_m):
        theta = rng.uniform(
            geometry.emission_theta_min_rad,
            geometry.emission_theta_max_rad,
            size=electrons.count,
        )
        if electrons.emission_jitter_m > 0.0:
            jitter_theta = electrons.emission_jitter_m / max(geometry.emission_radius_m, 1e-9)
            theta += rng.normal(0.0, jitter_theta, size=electrons.count)
            theta = np.clip(theta, geometry.emission_theta_min_rad, geometry.emission_theta_max_rad)

        xc = geometry.photocathode_center_x_m
        yc = geometry.photocathode_center_y_m
        r_emit = geometry.emission_radius_m

        x_surface = xc - r_emit * np.cos(theta)
        y_surface = yc + r_emit * np.sin(theta)

        # Inward normal from the inner hemisphere surface toward PMT volume.
        normal = np.column_stack((np.cos(theta), -np.sin(theta)))
        tangent = np.column_stack((np.sin(theta), np.cos(theta)))

        delta = launch_angle_rad + rng.normal(0.0, angle_spread_rad, size=electrons.count)
        direction = normal * np.cos(delta)[:, None] + tangent * np.sin(delta)[:, None]

        positions = np.column_stack((x_surface, y_surface)) + geometry.emission_offset_m * normal
        velocities = speed[:, None] * direction
        return positions, velocities

    y0 = rng.uniform(geometry.emission_y_min_m, geometry.emission_y_max_m, size=electrons.count)
    if electrons.emission_jitter_m > 0.0:
        y0 += rng.normal(0.0, electrons.emission_jitter_m, size=electrons.count)

    angle = launch_angle_rad + rng.normal(0.0, angle_spread_rad, size=electrons.count)
    vx = speed * np.cos(angle)
    vy = speed * np.sin(angle)

    positions = np.column_stack((np.full(electrons.count, geometry.emission_x_m), y0))
    velocities = np.column_stack((vx, vy))
    return positions, velocities
