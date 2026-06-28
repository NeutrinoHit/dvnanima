from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
import math
import tomllib

import numpy as np


DEFAULT_CONFIG_PATH = Path(__file__).with_name("pmt.toml")
ELEMENTARY_CHARGE = 1.602176634e-19
ELECTRON_MASS = 9.1093837015e-31
CM_TO_M = 1.0e-2
_ZERO_TOL = 1e-14


@dataclass(frozen=True)
class GridConfig:
    nx: int
    ny: int
    lx: float
    ly: float


@dataclass(frozen=True)
class ParticleConfig:
    count: int
    steps: int
    dt: float
    mass: float
    charge: float
    initial_energy_ev: float
    initial_energy_spread_ev: float
    angle_spread_deg: float
    emission_jitter_y: float
    integration_substeps: int


@dataclass(frozen=True)
class SimulationConfig:
    grid: GridConfig
    particles: ParticleConfig
    cathode_voltage: float
    dynode_voltage: float
    background_voltage: float
    cathode_x: float
    cathode_height: float
    cathode_thickness: float
    dynode_center_x: float
    dynode_center_y: float
    dynode_length: float
    dynode_thickness: float
    dynode_angle_deg: float
    bz_t: float
    sor_omega: float
    sor_tolerance: float
    sor_max_iterations: int
    seed: int
    # Geometry mode:
    # - "line": legacy flat photocathode (cathode_x/cathode_height/cathode_thickness)
    # - "hemisphere": spherical photocathode shell in 2D cross-section
    photocathode_shape: str = "line"
    photocathode_diameter: float = 0.508
    photocathode_thickness: float = 0.006
    photocathode_center_x: float = -0.08
    photocathode_center_y: float = 0.0
    photocathode_active_theta_min_deg: float = -80.0
    photocathode_active_theta_max_deg: float = 80.0
    length_unit_label: str = "m"


@dataclass(frozen=True)
class GeometryData:
    fixed_mask: np.ndarray
    fixed_values: np.ndarray
    electrode_id: np.ndarray
    cathode_mask: np.ndarray
    dynode_mask: np.ndarray
    emission_mode: str
    emission_x: float
    emission_y_min: float
    emission_y_max: float
    photocathode_center_x: float
    photocathode_center_y: float
    emission_radius: float
    emission_theta_min: float
    emission_theta_max: float
    emission_offset: float


def _read_length_m(data: dict, key_cm: str, key_legacy: str, default_cm: float) -> float:
    if key_cm in data:
        return float(data[key_cm]) * CM_TO_M
    if key_legacy in data:
        legacy = float(data[key_legacy])
        # Legacy files stored lengths directly in meters.
        return legacy
    return float(default_cm) * CM_TO_M


def _read_grid_length_m(data: dict, key_cm: str, key_legacy: str, default_cm: float) -> float:
    if key_cm in data:
        return float(data[key_cm]) * CM_TO_M
    if key_legacy in data:
        value = float(data[key_legacy])
        # Heuristic:
        # - old configs had domain in meters (~0.01...1.0),
        # - new configs use centimeters (tens to hundreds).
        return value if abs(value) < 5.0 else value * CM_TO_M
    return float(default_cm) * CM_TO_M


def load_pmt_config(path: str | Path | None = None) -> SimulationConfig:
    config_path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    data = tomllib.loads(config_path.read_text(encoding="utf-8"))

    grid_data = data["grid"]
    particle_data = data["particles"]
    simulation_data = data["simulation"]

    return SimulationConfig(
        grid=GridConfig(
            nx=int(grid_data["nx"]),
            ny=int(grid_data["ny"]),
            lx=_read_grid_length_m(grid_data, "lx_cm", "lx", default_cm=90.0),
            ly=_read_grid_length_m(grid_data, "ly_cm", "ly", default_cm=70.0),
        ),
        particles=ParticleConfig(
            count=int(particle_data["count"]),
            steps=int(particle_data["steps"]),
            dt=float(particle_data["dt"]),
            mass=float(particle_data.get("mass", ELECTRON_MASS)),
            charge=float(particle_data.get("charge", -ELEMENTARY_CHARGE)),
            initial_energy_ev=float(particle_data.get("initial_energy_ev", 0.35)),
            initial_energy_spread_ev=float(particle_data.get("initial_energy_spread_ev", 0.10)),
            angle_spread_deg=float(particle_data.get("angle_spread_deg", 10.0)),
            emission_jitter_y=float(particle_data.get("emission_jitter_y", 0.0)),
            integration_substeps=max(1, int(particle_data.get("integration_substeps", 4))),
        ),
        cathode_voltage=float(simulation_data.get("cathode_voltage", 0.0)),
        dynode_voltage=float(simulation_data.get("dynode_voltage", 1600.0)),
        background_voltage=float(simulation_data.get("background_voltage", 0.0)),
        cathode_x=_read_length_m(simulation_data, "cathode_x_cm", "cathode_x", default_cm=-26.0),
        cathode_height=_read_length_m(simulation_data, "cathode_height_cm", "cathode_height", default_cm=50.8),
        cathode_thickness=_read_length_m(simulation_data, "cathode_thickness_cm", "cathode_thickness", default_cm=0.6),
        dynode_center_x=_read_length_m(simulation_data, "dynode_center_x_cm", "dynode_center_x", default_cm=6.0),
        dynode_center_y=_read_length_m(simulation_data, "dynode_center_y_cm", "dynode_center_y", default_cm=0.0),
        dynode_length=_read_length_m(simulation_data, "dynode_length_cm", "dynode_length", default_cm=7.5),
        dynode_thickness=_read_length_m(simulation_data, "dynode_thickness_cm", "dynode_thickness", default_cm=1.2),
        dynode_angle_deg=float(simulation_data.get("dynode_angle_deg", -18.0)),
        bz_t=float(simulation_data.get("bz_t", 0.0)),
        sor_omega=float(simulation_data.get("sor_omega", 1.93)),
        sor_tolerance=float(simulation_data.get("sor_tolerance", 1e-8)),
        sor_max_iterations=int(simulation_data.get("sor_max_iterations", 16000)),
        seed=int(simulation_data.get("seed", 12345)),
        photocathode_shape=str(simulation_data.get("photocathode_shape", "hemisphere")).strip().lower(),
        photocathode_diameter=_read_length_m(
            simulation_data,
            "photocathode_diameter_cm",
            "photocathode_diameter",
            default_cm=50.8,
        ),
        photocathode_thickness=_read_length_m(
            simulation_data,
            "photocathode_thickness_cm",
            "photocathode_thickness",
            default_cm=0.6,
        ),
        photocathode_center_x=_read_length_m(
            simulation_data,
            "photocathode_center_x_cm",
            "photocathode_center_x",
            default_cm=-8.0,
        ),
        photocathode_center_y=_read_length_m(
            simulation_data,
            "photocathode_center_y_cm",
            "photocathode_center_y",
            default_cm=0.0,
        ),
        photocathode_active_theta_min_deg=float(simulation_data.get("photocathode_active_theta_min_deg", -80.0)),
        photocathode_active_theta_max_deg=float(simulation_data.get("photocathode_active_theta_max_deg", 80.0)),
        length_unit_label="m",
    )


def _scaled_count(value: int, scale: float, minimum: int) -> int:
    return max(minimum, int(round(float(value) * float(scale))))


def make_preview_config(
    cfg: SimulationConfig,
    grid_scale: float = 0.5,
    particle_scale: float = 0.35,
    step_scale: float = 0.35,
) -> SimulationConfig:
    preview_grid = replace(
        cfg.grid,
        nx=_scaled_count(cfg.grid.nx, grid_scale, minimum=96),
        ny=_scaled_count(cfg.grid.ny, grid_scale, minimum=72),
    )
    preview_particles = replace(
        cfg.particles,
        count=_scaled_count(cfg.particles.count, particle_scale, minimum=64),
        steps=_scaled_count(cfg.particles.steps, step_scale, minimum=280),
    )
    return replace(cfg, grid=preview_grid, particles=preview_particles)


def make_grid(cfg: SimulationConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float]:
    x_axis = np.linspace(-0.5 * cfg.grid.lx, 0.5 * cfg.grid.lx, cfg.grid.nx)
    y_axis = np.linspace(-0.5 * cfg.grid.ly, 0.5 * cfg.grid.ly, cfg.grid.ny)
    x_grid, y_grid = np.meshgrid(x_axis, y_axis, indexing="ij")
    dx = float(x_axis[1] - x_axis[0])
    dy = float(y_axis[1] - y_axis[0])
    return x_axis, y_axis, x_grid, y_grid, dx, dy


def _rotated_rectangle_mask(
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    center_x: float,
    center_y: float,
    length: float,
    thickness: float,
    angle_deg: float,
) -> np.ndarray:
    theta = math.radians(angle_deg)
    c = math.cos(theta)
    s = math.sin(theta)
    rel_x = x_grid - center_x
    rel_y = y_grid - center_y
    along = c * rel_x + s * rel_y
    normal = -s * rel_x + c * rel_y
    return (np.abs(along) <= 0.5 * length) & (np.abs(normal) <= 0.5 * thickness)


def build_cathode_dynode_geometry(
    cfg: SimulationConfig,
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    dx: float,
) -> GeometryData:
    dy = float(y_grid[0, 1] - y_grid[0, 0]) if y_grid.shape[1] > 1 else dx
    shape = str(cfg.photocathode_shape).strip().lower()
    emission_mode = "line"
    emission_x = min(
        cfg.cathode_x + 0.5 * cfg.cathode_thickness + 1.5 * dx,
        float(np.max(x_grid) - dx),
    )
    emission_y_min = -0.5 * cfg.cathode_height
    emission_y_max = 0.5 * cfg.cathode_height
    photocathode_center_x = cfg.cathode_x
    photocathode_center_y = 0.0
    emission_radius = float("nan")
    emission_theta_min = float("nan")
    emission_theta_max = float("nan")
    emission_offset = max(0.75 * dx, 1e-6)

    if shape == "hemisphere":
        radius = max(0.5 * cfg.photocathode_diameter, 2.0 * dx)
        thickness = min(max(cfg.photocathode_thickness, dx), 0.85 * radius)
        inner_radius = radius - thickness
        xc = cfg.photocathode_center_x
        yc = cfg.photocathode_center_y

        rel_x = x_grid - xc
        rel_y = y_grid - yc
        radial = np.sqrt(rel_x * rel_x + rel_y * rel_y)

        cathode_mask = (radial <= radius) & (radial >= inner_radius) & (x_grid <= xc)

        theta_min = math.radians(cfg.photocathode_active_theta_min_deg)
        theta_max = math.radians(cfg.photocathode_active_theta_max_deg)
        theta_min = float(np.clip(theta_min, -0.5 * math.pi + 1e-4, 0.5 * math.pi - 1e-4))
        theta_max = float(np.clip(theta_max, -0.5 * math.pi + 1e-4, 0.5 * math.pi - 1e-4))
        if theta_min > theta_max:
            theta_min, theta_max = theta_max, theta_min

        emission_mode = "hemisphere"
        photocathode_center_x = xc
        photocathode_center_y = yc
        emission_radius = max(inner_radius - 0.6 * min(dx, dy), 0.5 * inner_radius)
        emission_theta_min = theta_min
        emission_theta_max = theta_max
        emission_offset = max(0.75 * min(dx, dy), 1e-6)
    else:
        cathode_mask = (
            (np.abs(x_grid - cfg.cathode_x) <= 0.5 * cfg.cathode_thickness)
            & (np.abs(y_grid) <= 0.5 * cfg.cathode_height)
        )

    dynode_mask = _rotated_rectangle_mask(
        x_grid,
        y_grid,
        center_x=cfg.dynode_center_x,
        center_y=cfg.dynode_center_y,
        length=cfg.dynode_length,
        thickness=cfg.dynode_thickness,
        angle_deg=cfg.dynode_angle_deg,
    )

    fixed_mask = np.zeros_like(x_grid, dtype=bool)
    fixed_mask[0, :] = True
    fixed_mask[-1, :] = True
    fixed_mask[:, 0] = True
    fixed_mask[:, -1] = True
    fixed_mask |= cathode_mask
    fixed_mask |= dynode_mask

    fixed_values = np.full_like(x_grid, cfg.background_voltage, dtype=float)
    fixed_values[cathode_mask] = cfg.cathode_voltage
    fixed_values[dynode_mask] = cfg.dynode_voltage

    electrode_id = np.zeros_like(x_grid, dtype=np.uint8)
    electrode_id[fixed_mask] = 3
    electrode_id[cathode_mask] = 1
    electrode_id[dynode_mask] = 2

    return GeometryData(
        fixed_mask=fixed_mask,
        fixed_values=fixed_values,
        electrode_id=electrode_id,
        cathode_mask=cathode_mask,
        dynode_mask=dynode_mask,
        emission_mode=emission_mode,
        emission_x=emission_x,
        emission_y_min=emission_y_min,
        emission_y_max=emission_y_max,
        photocathode_center_x=photocathode_center_x,
        photocathode_center_y=photocathode_center_y,
        emission_radius=emission_radius,
        emission_theta_min=emission_theta_min,
        emission_theta_max=emission_theta_max,
        emission_offset=emission_offset,
    )


def solve_laplace_red_black_sor(
    fixed_mask: np.ndarray,
    fixed_values: np.ndarray,
    dx: float,
    dy: float,
    omega: float,
    tolerance: float,
    max_iterations: int,
    initial_phi: np.ndarray | None = None,
) -> tuple[np.ndarray, int, float]:
    if fixed_mask.shape != fixed_values.shape:
        raise ValueError("fixed_mask and fixed_values must have the same shape")

    phi = np.array(fixed_values, dtype=float, copy=True)
    if initial_phi is not None:
        phi[:] = np.asarray(initial_phi, dtype=float)
        phi[fixed_mask] = fixed_values[fixed_mask]

    free_mask = ~fixed_mask
    if not np.any(free_mask):
        return phi, 0, 0.0

    nx, ny = phi.shape
    if nx < 3 or ny < 3:
        return phi, 0, 0.0

    interior_free = free_mask[1:-1, 1:-1]
    if not np.any(interior_free):
        return phi, 0, 0.0

    ii, jj = np.indices(interior_free.shape)
    red_mask = ((ii + jj) & 1) == 0
    black_mask = ~red_mask
    red_free = interior_free & red_mask
    black_free = interior_free & black_mask

    inv_dx2 = 1.0 / (dx * dx)
    inv_dy2 = 1.0 / (dy * dy)
    denom_inv = 1.0 / (2.0 * (inv_dx2 + inv_dy2))

    phi_interior = phi[1:-1, 1:-1]
    residual = float("inf")

    for iteration in range(1, max_iterations + 1):
        max_delta = 0.0
        for update_mask in (red_free, black_free):
            if not np.any(update_mask):
                continue
            candidate = (
                (phi[2:, 1:-1] + phi[:-2, 1:-1]) * inv_dx2
                + (phi[1:-1, 2:] + phi[1:-1, :-2]) * inv_dy2
            ) * denom_inv
            delta = omega * (candidate - phi_interior)
            local = np.max(np.abs(delta[update_mask]))
            if local > max_delta:
                max_delta = float(local)
            phi_interior[update_mask] += delta[update_mask]

        residual = max_delta
        if residual < tolerance:
            break

    phi[fixed_mask] = fixed_values[fixed_mask]
    return phi, iteration, residual


def compute_electric_field(phi: np.ndarray, dx: float, dy: float) -> tuple[np.ndarray, np.ndarray]:
    ex = np.empty_like(phi, dtype=float)
    ey = np.empty_like(phi, dtype=float)

    ex[1:-1, :] = -(phi[2:, :] - phi[:-2, :]) / (2.0 * dx)
    ex[0, :] = -(phi[1, :] - phi[0, :]) / dx
    ex[-1, :] = -(phi[-1, :] - phi[-2, :]) / dx

    ey[:, 1:-1] = -(phi[:, 2:] - phi[:, :-2]) / (2.0 * dy)
    ey[:, 0] = -(phi[:, 1] - phi[:, 0]) / dy
    ey[:, -1] = -(phi[:, -1] - phi[:, -2]) / dy

    return ex, ey


def _bilinear_indices(
    x: np.ndarray,
    y: np.ndarray,
    x_axis: np.ndarray,
    y_axis: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)

    x_min = float(x_axis[0])
    y_min = float(y_axis[0])
    dx = float(x_axis[1] - x_axis[0])
    dy = float(y_axis[1] - y_axis[0])

    tx = (x_arr - x_min) / dx
    ty = (y_arr - y_min) / dy

    i0 = np.floor(tx).astype(np.int64)
    j0 = np.floor(ty).astype(np.int64)

    nx = x_axis.shape[0]
    ny = y_axis.shape[0]
    i0 = np.clip(i0, 0, nx - 2)
    j0 = np.clip(j0, 0, ny - 2)

    wx = tx - i0
    wy = ty - j0

    inside = (
        (x_arr >= float(x_axis[0]))
        & (x_arr <= float(x_axis[-1]))
        & (y_arr >= float(y_axis[0]))
        & (y_arr <= float(y_axis[-1]))
    )
    return i0, j0, wx, wy, inside


def bilinear_interpolate_scalar(
    field: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    x_axis: np.ndarray,
    y_axis: np.ndarray,
    outside_value: float = float("nan"),
) -> np.ndarray:
    i0, j0, wx, wy, inside = _bilinear_indices(x, y, x_axis, y_axis)

    f00 = field[i0, j0]
    f10 = field[i0 + 1, j0]
    f01 = field[i0, j0 + 1]
    f11 = field[i0 + 1, j0 + 1]

    values = (
        (1.0 - wx) * (1.0 - wy) * f00
        + wx * (1.0 - wy) * f10
        + (1.0 - wx) * wy * f01
        + wx * wy * f11
    )

    if not np.all(inside):
        values = np.array(values, copy=True)
        values[~inside] = outside_value

    return values


def interpolate_electric_field(
    ex: np.ndarray,
    ey: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    x_axis: np.ndarray,
    y_axis: np.ndarray,
    outside_value: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    i0, j0, wx, wy, inside = _bilinear_indices(x, y, x_axis, y_axis)

    def interp(field: np.ndarray) -> np.ndarray:
        f00 = field[i0, j0]
        f10 = field[i0 + 1, j0]
        f01 = field[i0, j0 + 1]
        f11 = field[i0 + 1, j0 + 1]
        out = (
            (1.0 - wx) * (1.0 - wy) * f00
            + wx * (1.0 - wy) * f10
            + (1.0 - wx) * wy * f01
            + wx * wy * f11
        )
        if not np.all(inside):
            out = np.array(out, copy=True)
            out[~inside] = outside_value
        return out

    return interp(ex), interp(ey), inside


def boris_push_2d(
    velocity: np.ndarray,
    electric_field: np.ndarray,
    bz_t: float,
    q_over_m: float,
    dt: float,
) -> np.ndarray:
    v = np.asarray(velocity, dtype=float)
    e = np.asarray(electric_field, dtype=float)
    if v.shape[-1] != 2 or e.shape[-1] != 2:
        raise ValueError("velocity and electric_field must end with dimension 2")

    half_impulse = 0.5 * q_over_m * dt
    v_minus = v + half_impulse * e

    if abs(bz_t) <= _ZERO_TOL:
        v_plus = v_minus
    else:
        t = half_impulse * bz_t
        s = 2.0 * t / (1.0 + t * t)

        vmx = v_minus[..., 0]
        vmy = v_minus[..., 1]

        v_prime_x = vmx + vmy * t
        v_prime_y = vmy - vmx * t

        v_plus_x = vmx + v_prime_y * s
        v_plus_y = vmy - v_prime_x * s
        v_plus = np.stack((v_plus_x, v_plus_y), axis=-1)

    return v_plus + half_impulse * e


def launch_photoelectrons(
    cfg: SimulationConfig,
    geometry: GeometryData,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    particles = cfg.particles
    energy_ev = rng.normal(particles.initial_energy_ev, particles.initial_energy_spread_ev, size=particles.count)
    energy_ev = np.maximum(0.01, energy_ev)
    speed = np.sqrt(2.0 * energy_ev * ELEMENTARY_CHARGE / particles.mass)
    angle_spread = math.radians(particles.angle_spread_deg)

    if geometry.emission_mode == "hemisphere" and np.isfinite(geometry.emission_radius):
        theta = rng.uniform(geometry.emission_theta_min, geometry.emission_theta_max, size=particles.count)
        if particles.emission_jitter_y > 0.0:
            jitter_theta = particles.emission_jitter_y / max(geometry.emission_radius, 1e-9)
            theta += rng.normal(0.0, jitter_theta, size=particles.count)
            theta = np.clip(theta, geometry.emission_theta_min, geometry.emission_theta_max)

        xc = geometry.photocathode_center_x
        yc = geometry.photocathode_center_y
        r_emit = geometry.emission_radius

        x_surface = xc - r_emit * np.cos(theta)
        y_surface = yc + r_emit * np.sin(theta)

        normal = np.column_stack((np.cos(theta), -np.sin(theta)))
        tangent = np.column_stack((np.sin(theta), np.cos(theta)))

        delta = rng.normal(0.0, angle_spread, size=particles.count)
        direction = normal * np.cos(delta)[:, None] + tangent * np.sin(delta)[:, None]

        positions = np.column_stack((x_surface, y_surface)) + geometry.emission_offset * normal
        velocities = speed[:, None] * direction
        return positions, velocities

    y0 = rng.uniform(geometry.emission_y_min, geometry.emission_y_max, size=particles.count)
    if particles.emission_jitter_y > 0.0:
        y0 += rng.normal(0.0, particles.emission_jitter_y, size=particles.count)

    angle = rng.normal(0.0, angle_spread, size=particles.count)
    vx = speed * np.cos(angle)
    vy = speed * np.sin(angle)

    positions = np.column_stack((np.full(particles.count, geometry.emission_x), y0))
    velocities = np.column_stack((vx, vy))
    return positions, velocities


def positions_to_cell_indices(
    x: np.ndarray,
    y: np.ndarray,
    x_axis: np.ndarray,
    y_axis: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)

    dx = float(x_axis[1] - x_axis[0])
    dy = float(y_axis[1] - y_axis[0])
    i = np.rint((x_arr - float(x_axis[0])) / dx).astype(np.int64)
    j = np.rint((y_arr - float(y_axis[0])) / dy).astype(np.int64)

    inside = (i >= 0) & (i < x_axis.shape[0]) & (j >= 0) & (j < y_axis.shape[0])
    i = np.clip(i, 0, x_axis.shape[0] - 1)
    j = np.clip(j, 0, y_axis.shape[0] - 1)
    return i, j, inside


def trace_particles(
    x_axis: np.ndarray,
    y_axis: np.ndarray,
    ex: np.ndarray,
    ey: np.ndarray,
    dynode_mask: np.ndarray,
    cathode_mask: np.ndarray,
    positions0: np.ndarray,
    velocities0: np.ndarray,
    steps: int,
    dt: float,
    charge: float,
    mass: float,
    bz_t: float,
    integration_substeps: int = 1,
) -> dict:
    count = positions0.shape[0]
    positions = np.full((steps + 1, count, 2), np.nan, dtype=float)
    velocities = np.full((steps + 1, count, 2), np.nan, dtype=float)
    alive = np.zeros((steps + 1, count), dtype=bool)

    positions[0] = positions0
    velocities[0] = velocities0
    alive[0] = True

    impact_mask = np.zeros(count, dtype=bool)
    impact_step = np.full(count, -1, dtype=np.int32)
    impact_position = np.full((count, 2), np.nan, dtype=float)
    impact_velocity = np.full((count, 2), np.nan, dtype=float)

    q_over_m = charge / mass
    substeps = max(1, int(integration_substeps))
    dt_sub = dt / substeps
    last_recorded = 0

    for step in range(steps):
        live = alive[step]
        if not np.any(live):
            break

        live_idx = np.flatnonzero(live)
        pos_curr = np.array(positions[step, live_idx], copy=True)
        vel_curr = np.array(velocities[step, live_idx], copy=True)
        alive_curr = np.ones(live_idx.shape[0], dtype=bool)

        for _ in range(substeps):
            local_idx = np.flatnonzero(alive_curr)
            if local_idx.size == 0:
                break

            pos_live = pos_curr[local_idx]
            vel_live = vel_curr[local_idx]

            ex_s, ey_s, _ = interpolate_electric_field(
                ex,
                ey,
                pos_live[:, 0],
                pos_live[:, 1],
                x_axis,
                y_axis,
                outside_value=0.0,
            )
            field = np.column_stack((ex_s, ey_s))

            vel_new_live = boris_push_2d(vel_live, field, bz_t=bz_t, q_over_m=q_over_m, dt=dt_sub)
            pos_new_live = pos_live + dt_sub * vel_new_live

            i_cell, j_cell, inside_new = positions_to_cell_indices(
                pos_new_live[:, 0],
                pos_new_live[:, 1],
                x_axis,
                y_axis,
            )

            hit_dynode = np.zeros(local_idx.shape[0], dtype=bool)
            hit_cathode = np.zeros(local_idx.shape[0], dtype=bool)
            inside_idx = np.flatnonzero(inside_new)
            if inside_idx.size:
                ii = i_cell[inside_idx]
                jj = j_cell[inside_idx]
                hit_dynode[inside_idx] = dynode_mask[ii, jj]
                hit_cathode[inside_idx] = cathode_mask[ii, jj]

            pos_curr[local_idx] = pos_new_live
            vel_curr[local_idx] = vel_new_live

            die = (~inside_new) | hit_dynode | hit_cathode
            if np.any(hit_dynode):
                global_hit_idx = live_idx[local_idx[hit_dynode]]
                newly_hit = ~impact_mask[global_hit_idx]
                if np.any(newly_hit):
                    gidx = global_hit_idx[newly_hit]
                    impact_mask[gidx] = True
                    impact_step[gidx] = step + 1
                    impact_position[gidx] = pos_new_live[hit_dynode][newly_hit]
                    impact_velocity[gidx] = vel_new_live[hit_dynode][newly_hit]

            alive_curr[local_idx[die]] = False

        positions[step + 1, live_idx] = pos_curr
        velocities[step + 1, live_idx] = vel_curr
        alive[step + 1, live_idx] = alive_curr

        last_recorded = step + 1

    time_axis = np.arange(last_recorded + 1, dtype=float) * dt
    return {
        "time_axis": time_axis,
        "positions": positions[: last_recorded + 1],
        "velocities": velocities[: last_recorded + 1],
        "alive": alive[: last_recorded + 1],
        "impact_mask": impact_mask,
        "impact_step": impact_step,
        "impact_position": impact_position,
        "impact_velocity": impact_velocity,
    }


def kinetic_energy_ev(velocity: np.ndarray, mass: float) -> np.ndarray:
    speed2 = np.sum(np.asarray(velocity, dtype=float) ** 2, axis=-1)
    return 0.5 * mass * speed2 / ELEMENTARY_CHARGE


def build_simulation_bundle(cfg: SimulationConfig | None = None) -> dict:
    resolved_cfg = cfg if cfg is not None else load_pmt_config()

    x_axis, y_axis, x_grid, y_grid, dx, dy = make_grid(resolved_cfg)
    geometry = build_cathode_dynode_geometry(resolved_cfg, x_grid, y_grid, dx)

    potential, solver_iterations, solver_residual = solve_laplace_red_black_sor(
        fixed_mask=geometry.fixed_mask,
        fixed_values=geometry.fixed_values,
        dx=dx,
        dy=dy,
        omega=resolved_cfg.sor_omega,
        tolerance=resolved_cfg.sor_tolerance,
        max_iterations=resolved_cfg.sor_max_iterations,
    )
    ex, ey = compute_electric_field(potential, dx=dx, dy=dy)
    ex = np.array(ex, copy=True)
    ey = np.array(ey, copy=True)
    ex[geometry.fixed_mask] = 0.0
    ey[geometry.fixed_mask] = 0.0

    rng = np.random.default_rng(resolved_cfg.seed)
    launch_positions, launch_velocities = launch_photoelectrons(resolved_cfg, geometry, rng)

    tracks = trace_particles(
        x_axis=x_axis,
        y_axis=y_axis,
        ex=ex,
        ey=ey,
        dynode_mask=geometry.dynode_mask,
        cathode_mask=geometry.cathode_mask,
        positions0=launch_positions,
        velocities0=launch_velocities,
        steps=resolved_cfg.particles.steps,
        dt=resolved_cfg.particles.dt,
        charge=resolved_cfg.particles.charge,
        mass=resolved_cfg.particles.mass,
        bz_t=resolved_cfg.bz_t,
        integration_substeps=resolved_cfg.particles.integration_substeps,
    )

    impact_count = int(np.count_nonzero(tracks["impact_mask"]))
    emitted_count = int(resolved_cfg.particles.count)
    collection_efficiency = impact_count / max(1, emitted_count)

    return {
        "cfg": resolved_cfg,
        "x_axis": x_axis,
        "y_axis": y_axis,
        "potential": potential,
        "ex": ex,
        "ey": ey,
        "fixed_mask": geometry.fixed_mask,
        "electrode_id": geometry.electrode_id,
        "cathode_mask": geometry.cathode_mask,
        "dynode_mask": geometry.dynode_mask,
        "solver_iterations": int(solver_iterations),
        "solver_residual": float(solver_residual),
        "launch_positions": launch_positions,
        "launch_velocities": launch_velocities,
        "time_axis": tracks["time_axis"],
        "particle_positions": tracks["positions"],
        "particle_velocities": tracks["velocities"],
        "particle_alive": tracks["alive"],
        "particle_impact_mask": tracks["impact_mask"],
        "particle_impact_step": tracks["impact_step"],
        "particle_impact_position": tracks["impact_position"],
        "particle_impact_velocity": tracks["impact_velocity"],
        "collection_efficiency": float(collection_efficiency),
    }


__all__ = [
    "DEFAULT_CONFIG_PATH",
    "ELEMENTARY_CHARGE",
    "ELECTRON_MASS",
    "CM_TO_M",
    "GridConfig",
    "ParticleConfig",
    "SimulationConfig",
    "GeometryData",
    "load_pmt_config",
    "make_preview_config",
    "make_grid",
    "build_cathode_dynode_geometry",
    "solve_laplace_red_black_sor",
    "compute_electric_field",
    "bilinear_interpolate_scalar",
    "interpolate_electric_field",
    "boris_push_2d",
    "launch_photoelectrons",
    "trace_particles",
    "kinetic_energy_ev",
    "build_simulation_bundle",
]
