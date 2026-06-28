from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
import tomllib
from typing import Callable

import numpy as np


DEFAULT_CONFIG_PATH = Path(__file__).with_name("radiating_charge.toml")
_ZERO_TOL = 1e-12


@dataclass(frozen=True)
class GridConfig:
    nx: int
    ny: int
    lx: float
    ly: float
    z_obs: float
    mask_radius: float


@dataclass(frozen=True)
class TimeConfig:
    t_start: float
    t_end: float
    trajectory_dt: float
    animation_num_samples: int


@dataclass(frozen=True)
class SourceConfig:
    charge: float
    c: float
    eps0: float


@dataclass(frozen=True)
class ObservableConfig:
    lower_surface: str
    upper_surface: str
    keys: tuple[str, ...]


@dataclass(frozen=True)
class RenderConfig:
    lower_height_target: float
    upper_height_target: float
    lower_visual_gain: float
    upper_visual_gain: float
    upper_plane_shift: float
    display_transform: str
    playback_time_scale: float


@dataclass(frozen=True)
class TrajectoryConfig:
    model: str


@dataclass(frozen=True)
class CoulombTrajectoryConfig:
    mass: float
    center_charge: float
    softening: float
    center_x: float
    center_y: float
    center_z: float
    x0: float
    y0: float
    z0: float
    vx0: float
    vy0: float
    vz0: float


@dataclass(frozen=True)
class OscillatorTrajectoryConfig:
    mass: float
    omega0: float
    gamma: float
    center_x: float
    center_y: float
    center_z: float
    x0: float
    y0: float
    z0: float
    vx0: float
    vy0: float
    vz0: float


@dataclass(frozen=True)
class RadiatingChargeConfig:
    grid: GridConfig
    time: TimeConfig
    source: SourceConfig
    observables: ObservableConfig
    render: RenderConfig
    trajectory: TrajectoryConfig
    coulomb: CoulombTrajectoryConfig
    oscillator: OscillatorTrajectoryConfig


@dataclass(frozen=True)
class TrajectorySolution:
    t_nodes: np.ndarray
    r_nodes: np.ndarray
    v_nodes: np.ndarray
    a_nodes: np.ndarray

    def eval(self, t_query: np.ndarray | float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        t_arr = np.asarray(t_query, dtype=float)
        flat = t_arr.reshape(-1)
        t_clamped = np.clip(flat, float(self.t_nodes[0]), float(self.t_nodes[-1]))

        r_cols = [np.interp(t_clamped, self.t_nodes, self.r_nodes[:, idx]) for idx in range(3)]
        v_cols = [np.interp(t_clamped, self.t_nodes, self.v_nodes[:, idx]) for idx in range(3)]
        a_cols = [np.interp(t_clamped, self.t_nodes, self.a_nodes[:, idx]) for idx in range(3)]
        r = np.stack(r_cols, axis=1).reshape(t_arr.shape + (3,))
        v = np.stack(v_cols, axis=1).reshape(t_arr.shape + (3,))
        a = np.stack(a_cols, axis=1).reshape(t_arr.shape + (3,))
        return r, v, a


ALL_OBSERVABLE_KEYS = (
    "ex",
    "ey",
    "ez",
    "bx",
    "by",
    "bz",
    "sx",
    "sy",
    "sz",
    "e_mag",
    "b_mag",
    "s_mag",
    "ex_rad",
    "ey_rad",
    "ez_rad",
    "bx_rad",
    "by_rad",
    "bz_rad",
    "sx_rad",
    "sy_rad",
    "sz_rad",
    "e_rad_mag",
    "b_rad_mag",
    "s_rad_mag",
)


def load_radiating_charge_config(path: str | Path | None = None) -> RadiatingChargeConfig:
    config_path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    data = tomllib.loads(config_path.read_text(encoding="utf-8"))

    trajectory_model = str(data["trajectory"].get("model", "coulomb")).strip().lower()
    if trajectory_model not in {"coulomb", "oscillator"}:
        raise ValueError("trajectory.model must be `coulomb` or `oscillator`.")

    keys_raw = data["observables"].get("keys", ["e_mag", "bz", "s_mag"])
    keys = tuple(str(item) for item in keys_raw)
    for key in keys:
        if key not in ALL_OBSERVABLE_KEYS:
            raise ValueError(f"Unknown observables.keys item `{key}`. Allowed: {ALL_OBSERVABLE_KEYS}")

    return RadiatingChargeConfig(
        grid=GridConfig(
            nx=int(data["grid"]["nx"]),
            ny=int(data["grid"]["ny"]),
            lx=float(data["grid"]["lx"]),
            ly=float(data["grid"]["ly"]),
            z_obs=float(data["grid"].get("z_obs", 0.0)),
            mask_radius=float(data["grid"].get("mask_radius", 0.0)),
        ),
        time=TimeConfig(
            t_start=float(data["time"]["t_start"]),
            t_end=float(data["time"]["t_end"]),
            trajectory_dt=float(data["time"].get("trajectory_dt", 0.002)),
            animation_num_samples=int(data["time"]["animation_num_samples"]),
        ),
        source=SourceConfig(
            charge=float(data["source"]["charge"]),
            c=float(data["source"].get("c", 1.0)),
            eps0=float(data["source"].get("eps0", 1.0)),
        ),
        observables=ObservableConfig(
            lower_surface=str(data["observables"].get("lower_surface", "e_mag")),
            upper_surface=str(data["observables"].get("upper_surface", "s_mag")),
            keys=keys,
        ),
        render=RenderConfig(
            lower_height_target=float(data["render"].get("lower_height_target", 1.0)),
            upper_height_target=float(data["render"].get("upper_height_target", 1.0)),
            lower_visual_gain=float(data["render"].get("lower_visual_gain", 1.0)),
            upper_visual_gain=float(data["render"].get("upper_visual_gain", 1.0)),
            upper_plane_shift=float(data["render"].get("upper_plane_shift", 2.0)),
            display_transform=str(data["render"].get("display_transform", "linear")),
            playback_time_scale=float(data["render"].get("playback_time_scale", 1.0)),
        ),
        trajectory=TrajectoryConfig(model=trajectory_model),
        coulomb=CoulombTrajectoryConfig(
            mass=float(data["trajectory"]["coulomb"].get("mass", 1.0)),
            center_charge=float(data["trajectory"]["coulomb"].get("center_charge", 0.0)),
            softening=float(data["trajectory"]["coulomb"].get("softening", 0.0)),
            center_x=float(data["trajectory"]["coulomb"].get("center_x", 0.0)),
            center_y=float(data["trajectory"]["coulomb"].get("center_y", 0.0)),
            center_z=float(data["trajectory"]["coulomb"].get("center_z", 0.0)),
            x0=float(data["trajectory"]["coulomb"]["x0"]),
            y0=float(data["trajectory"]["coulomb"].get("y0", 0.0)),
            z0=float(data["trajectory"]["coulomb"].get("z0", 0.0)),
            vx0=float(data["trajectory"]["coulomb"]["vx0"]),
            vy0=float(data["trajectory"]["coulomb"].get("vy0", 0.0)),
            vz0=float(data["trajectory"]["coulomb"].get("vz0", 0.0)),
        ),
        oscillator=OscillatorTrajectoryConfig(
            mass=float(data["trajectory"]["oscillator"].get("mass", 1.0)),
            omega0=float(data["trajectory"]["oscillator"].get("omega0", 1.0)),
            gamma=float(data["trajectory"]["oscillator"].get("gamma", 0.0)),
            center_x=float(data["trajectory"]["oscillator"].get("center_x", 0.0)),
            center_y=float(data["trajectory"]["oscillator"].get("center_y", 0.0)),
            center_z=float(data["trajectory"]["oscillator"].get("center_z", 0.0)),
            x0=float(data["trajectory"]["oscillator"]["x0"]),
            y0=float(data["trajectory"]["oscillator"].get("y0", 0.0)),
            z0=float(data["trajectory"]["oscillator"].get("z0", 0.0)),
            vx0=float(data["trajectory"]["oscillator"]["vx0"]),
            vy0=float(data["trajectory"]["oscillator"].get("vy0", 0.0)),
            vz0=float(data["trajectory"]["oscillator"].get("vz0", 0.0)),
        ),
    )


def _scaled_count(value: int, scale: float, minimum: int) -> int:
    return max(minimum, int(round(float(value) * float(scale))))


def make_preview_config(
    cfg: RadiatingChargeConfig,
    grid_scale: float = 0.5,
    time_scale: float = 0.35,
    playback_scale: float = 0.35,
) -> RadiatingChargeConfig:
    preview_grid = replace(
        cfg.grid,
        nx=_scaled_count(cfg.grid.nx, grid_scale, minimum=96),
        ny=_scaled_count(cfg.grid.ny, grid_scale, minimum=72),
    )
    preview_time = replace(
        cfg.time,
        animation_num_samples=_scaled_count(cfg.time.animation_num_samples, time_scale, minimum=72),
        trajectory_dt=cfg.time.trajectory_dt / max(time_scale, 1e-6),
    )
    preview_render = replace(
        cfg.render,
        playback_time_scale=cfg.render.playback_time_scale * float(playback_scale),
    )
    return replace(cfg, grid=preview_grid, time=preview_time, render=preview_render)


def make_grid(cfg: RadiatingChargeConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x_axis = np.linspace(-0.5 * cfg.grid.lx, 0.5 * cfg.grid.lx, cfg.grid.nx, endpoint=False)
    y_axis = np.linspace(-0.5 * cfg.grid.ly, 0.5 * cfg.grid.ly, cfg.grid.ny, endpoint=False)
    x_grid, y_grid = np.meshgrid(x_axis, y_axis, indexing="ij")
    return x_axis, y_axis, x_grid, y_grid


def make_animation_times(cfg: RadiatingChargeConfig) -> np.ndarray:
    return np.linspace(cfg.time.t_start, cfg.time.t_end, cfg.time.animation_num_samples)


def _max_light_delay(cfg: RadiatingChargeConfig) -> float:
    half_x = 0.5 * cfg.grid.lx
    half_y = 0.5 * cfg.grid.ly
    observer_radius = float(np.sqrt(half_x**2 + half_y**2 + cfg.grid.z_obs**2))

    if cfg.trajectory.model == "coulomb":
        model = cfg.coulomb
    else:
        model = cfg.oscillator

    span = float(max(cfg.time.t_end - cfg.time.t_start, 0.0))
    pos0 = np.array([model.x0, model.y0, model.z0], dtype=float)
    vel0 = np.array([model.vx0, model.vy0, model.vz0], dtype=float)
    center = np.array([model.center_x, model.center_y, model.center_z], dtype=float)

    speed0 = float(np.linalg.norm(vel0))
    source_extent = float(np.linalg.norm(pos0 - center)) + speed0 * span * 2.5 + float(np.linalg.norm(center))
    return (observer_radius + source_extent + 1.0) / max(cfg.source.c, 1e-9)


def _rk4_step(
    state: np.ndarray,
    dt: float,
    rhs,
) -> np.ndarray:
    k1 = rhs(state)
    k2 = rhs(state + 0.5 * dt * k1)
    k3 = rhs(state + 0.5 * dt * k2)
    k4 = rhs(state + dt * k3)
    return state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def _coulomb_acceleration(
    pos: np.ndarray,
    vel: np.ndarray,
    cfg: RadiatingChargeConfig,
) -> np.ndarray:
    src = cfg.source
    model = cfg.coulomb
    ke = 1.0 / (4.0 * np.pi * src.eps0)
    center = np.array([model.center_x, model.center_y, model.center_z], dtype=float)
    rel = pos - center
    r2 = float(np.dot(rel, rel))
    denom = (r2 + model.softening**2) ** 1.5
    if denom <= _ZERO_TOL:
        return np.zeros(3, dtype=float)
    coeff = ke * src.charge * model.center_charge / model.mass
    return coeff * rel / denom


def _oscillator_acceleration(
    pos: np.ndarray,
    vel: np.ndarray,
    cfg: RadiatingChargeConfig,
) -> np.ndarray:
    model = cfg.oscillator
    center = np.array([model.center_x, model.center_y, model.center_z], dtype=float)
    rel = pos - center
    return -(model.omega0**2) * rel - model.gamma * vel


def acceleration_from_state(
    pos: np.ndarray,
    vel: np.ndarray,
    cfg: RadiatingChargeConfig,
) -> np.ndarray:
    if cfg.trajectory.model == "coulomb":
        return _coulomb_acceleration(pos, vel, cfg)
    if cfg.trajectory.model == "oscillator":
        return _oscillator_acceleration(pos, vel, cfg)
    raise ValueError(f"Unsupported trajectory model `{cfg.trajectory.model}`")


def solve_trajectory(cfg: RadiatingChargeConfig, t_min: float, t_max: float) -> TrajectorySolution:
    dt = float(cfg.time.trajectory_dt)
    if dt <= 0.0:
        raise ValueError("time.trajectory_dt must be positive")
    if t_max <= t_min:
        raise ValueError("Trajectory time interval must satisfy t_max > t_min")

    if cfg.trajectory.model == "coulomb":
        model = cfg.coulomb
    else:
        model = cfg.oscillator

    state = np.array([model.x0, model.y0, model.z0, model.vx0, model.vy0, model.vz0], dtype=float)
    t_ref = float(cfg.time.t_start)
    if not (t_min <= t_ref <= t_max):
        raise ValueError("Expected t_min <= time.t_start <= t_max in solve_trajectory().")

    def rhs(local_state: np.ndarray) -> np.ndarray:
        pos = local_state[:3]
        vel = local_state[3:]
        acc = acceleration_from_state(pos, vel, cfg)
        return np.array([vel[0], vel[1], vel[2], acc[0], acc[1], acc[2]], dtype=float)

    def build_time_grid(start: float, end: float, step_abs: float) -> np.ndarray:
        if np.isclose(start, end):
            return np.array([start], dtype=float)
        direction = 1.0 if end > start else -1.0
        step = direction * abs(step_abs)
        values = [start]
        t_val = start
        while (end - t_val) * direction > abs(step):
            t_val += step
            values.append(t_val)
        if not np.isclose(values[-1], end):
            values.append(end)
        return np.array(values, dtype=float)

    forward_times = build_time_grid(t_ref, t_max, dt)
    backward_times = build_time_grid(t_ref, t_min, dt)

    forward_states = np.zeros((len(forward_times), 6), dtype=float)
    forward_states[0] = state
    local_state = state.copy()
    for idx in range(1, len(forward_times)):
        step_dt = float(forward_times[idx] - forward_times[idx - 1])
        local_state = _rk4_step(local_state, step_dt, rhs)
        forward_states[idx] = local_state

    backward_states = np.zeros((len(backward_times), 6), dtype=float)
    backward_states[0] = state
    local_state = state.copy()
    for idx in range(1, len(backward_times)):
        step_dt = float(backward_times[idx] - backward_times[idx - 1])
        local_state = _rk4_step(local_state, step_dt, rhs)
        backward_states[idx] = local_state

    times = np.concatenate([backward_times[::-1], forward_times[1:]])
    states = np.vstack([backward_states[::-1], forward_states[1:]])

    positions = states[:, :3]
    velocities = states[:, 3:]
    accelerations = np.array([acceleration_from_state(pos, vel, cfg) for pos, vel in zip(positions, velocities, strict=True)])

    return TrajectorySolution(
        t_nodes=times,
        r_nodes=positions,
        v_nodes=velocities,
        a_nodes=accelerations,
    )


def solve_retarded_times(
    trajectory: TrajectorySolution,
    x_obs: np.ndarray,
    y_obs: np.ndarray,
    z_obs: float,
    t_obs: float,
    c: float,
    initial_guess: np.ndarray | None = None,
    max_iter: int = 14,
    tol: float = 1e-9,
) -> tuple[np.ndarray, np.ndarray]:
    if c <= 0.0:
        raise ValueError("c must be positive")

    shape = x_obs.shape
    x_flat = np.asarray(x_obs, dtype=float).reshape(-1)
    y_flat = np.asarray(y_obs, dtype=float).reshape(-1)
    z_flat = np.full_like(x_flat, float(z_obs))

    t_min = float(trajectory.t_nodes[0])
    if initial_guess is None:
        r_now, _, _ = trajectory.eval(float(t_obs))
        r0 = np.asarray(r_now, dtype=float).reshape(3)
        dist_now = np.sqrt((x_flat - r0[0]) ** 2 + (y_flat - r0[1]) ** 2 + (z_flat - r0[2]) ** 2)
        tr = float(t_obs) - dist_now / c
    else:
        tr = np.asarray(initial_guess, dtype=float).reshape(-1)

    tr = np.clip(tr, t_min, float(t_obs))

    residual = np.full_like(tr, np.inf, dtype=float)
    for _ in range(max_iter):
        r_ret, v_ret, _ = trajectory.eval(tr)
        rx = x_flat - r_ret[:, 0]
        ry = y_flat - r_ret[:, 1]
        rz = z_flat - r_ret[:, 2]
        dist = np.sqrt(rx**2 + ry**2 + rz**2)
        dist = np.maximum(dist, 1e-12)

        nx = rx / dist
        ny = ry / dist
        nz = rz / dist

        beta_x = v_ret[:, 0] / c
        beta_y = v_ret[:, 1] / c
        beta_z = v_ret[:, 2] / c
        one_minus = 1.0 - (nx * beta_x + ny * beta_y + nz * beta_z)
        one_minus = np.where(np.abs(one_minus) < 1e-8, np.sign(one_minus) * 1e-8 + (one_minus == 0.0) * 1e-8, one_minus)

        residual = float(t_obs) - tr - dist / c
        delta = residual / one_minus
        tr_next = np.clip(tr + delta, t_min, float(t_obs))

        if np.max(np.abs(tr_next - tr)) < tol and np.max(np.abs(residual)) < tol:
            tr = tr_next
            break
        tr = tr_next

    r_ret, _, _ = trajectory.eval(tr)
    rx = x_flat - r_ret[:, 0]
    ry = y_flat - r_ret[:, 1]
    rz = z_flat - r_ret[:, 2]
    dist = np.sqrt(rx**2 + ry**2 + rz**2)
    residual = float(t_obs) - tr - dist / c

    return tr.reshape(shape), residual.reshape(shape)


def electric_field_components(
    trajectory: TrajectorySolution,
    t_ret: np.ndarray,
    x_obs: np.ndarray,
    y_obs: np.ndarray,
    z_obs: float,
    charge: float,
    c: float,
    eps0: float,
) -> tuple[np.ndarray, np.ndarray]:
    ke = 1.0 / (4.0 * np.pi * eps0)

    t_flat = np.asarray(t_ret, dtype=float).reshape(-1)
    x_flat = np.asarray(x_obs, dtype=float).reshape(-1)
    y_flat = np.asarray(y_obs, dtype=float).reshape(-1)
    z_flat = np.full_like(x_flat, float(z_obs))

    r_ret, v_ret, a_ret = trajectory.eval(t_flat)

    R = np.stack((x_flat - r_ret[:, 0], y_flat - r_ret[:, 1], z_flat - r_ret[:, 2]), axis=1)
    dist = np.linalg.norm(R, axis=1)
    dist = np.maximum(dist, 1e-12)
    n = R / dist[:, None]

    beta = v_ret / c
    beta2 = np.sum(beta * beta, axis=1)
    beta_dot = a_ret / c
    n_minus_beta = n - beta

    one_minus = 1.0 - np.sum(n * beta, axis=1)
    one_minus = np.where(np.abs(one_minus) < 1e-8, np.sign(one_minus) * 1e-8 + (one_minus == 0.0) * 1e-8, one_minus)

    denom_vel = (one_minus**3) * (dist**2)
    vel_term = ((1.0 - beta2)[:, None] * n_minus_beta) / denom_vel[:, None]

    cross_inner = np.cross(n_minus_beta, beta_dot)
    acc_term_num = np.cross(n, cross_inner)
    denom_acc = c * (one_minus**3) * dist
    acc_term = acc_term_num / denom_acc[:, None]

    e_vel = ke * charge * vel_term
    e_acc = ke * charge * acc_term
    return e_vel.reshape(t_ret.shape + (3,)), e_acc.reshape(t_ret.shape + (3,))


def evaluate_lw_fields(
    trajectory: TrajectorySolution,
    t_ret: np.ndarray,
    x_obs: np.ndarray,
    y_obs: np.ndarray,
    z_obs: float,
    charge: float,
    c: float,
    eps0: float,
    mask_radius: float = 0.0,
) -> dict[str, np.ndarray]:
    t_flat = np.asarray(t_ret, dtype=float).reshape(-1)
    x_flat = np.asarray(x_obs, dtype=float).reshape(-1)
    y_flat = np.asarray(y_obs, dtype=float).reshape(-1)
    z_flat = np.full_like(x_flat, float(z_obs))

    r_ret, _, _ = trajectory.eval(t_flat)
    R = np.stack((x_flat - r_ret[:, 0], y_flat - r_ret[:, 1], z_flat - r_ret[:, 2]), axis=1)
    dist = np.linalg.norm(R, axis=1)

    e_vel, e_acc = electric_field_components(trajectory, t_ret, x_obs, y_obs, z_obs, charge, c, eps0)
    e_total = e_vel + e_acc

    n = R / np.maximum(dist[:, None], 1e-12)
    b_vel_flat = np.cross(n, e_vel.reshape(-1, 3)) / c
    b_acc_flat = np.cross(n, e_acc.reshape(-1, 3)) / c
    b_vel = b_vel_flat.reshape(e_total.shape)
    b_acc = b_acc_flat.reshape(e_total.shape)
    b_total = b_vel + b_acc

    mu0 = 1.0 / (eps0 * c**2)
    s_total = np.cross(e_total.reshape(-1, 3), b_total.reshape(-1, 3)) / mu0
    s_total = s_total.reshape(e_total.shape)
    s_rad = np.cross(e_acc.reshape(-1, 3), b_acc.reshape(-1, 3)) / mu0
    s_rad = s_rad.reshape(e_total.shape)

    shape = np.asarray(t_ret).shape
    fields = {
        "ex": e_total[..., 0],
        "ey": e_total[..., 1],
        "ez": e_total[..., 2],
        "bx": b_total[..., 0],
        "by": b_total[..., 1],
        "bz": b_total[..., 2],
        "sx": s_total[..., 0],
        "sy": s_total[..., 1],
        "sz": s_total[..., 2],
        "e_mag": np.linalg.norm(e_total.reshape(-1, 3), axis=1).reshape(shape),
        "b_mag": np.linalg.norm(b_total.reshape(-1, 3), axis=1).reshape(shape),
        "s_mag": np.linalg.norm(s_total.reshape(-1, 3), axis=1).reshape(shape),
        "ex_rad": e_acc[..., 0],
        "ey_rad": e_acc[..., 1],
        "ez_rad": e_acc[..., 2],
        "bx_rad": b_acc[..., 0],
        "by_rad": b_acc[..., 1],
        "bz_rad": b_acc[..., 2],
        "sx_rad": s_rad[..., 0],
        "sy_rad": s_rad[..., 1],
        "sz_rad": s_rad[..., 2],
        "e_rad_mag": np.linalg.norm(e_acc.reshape(-1, 3), axis=1).reshape(shape),
        "b_rad_mag": np.linalg.norm(b_acc.reshape(-1, 3), axis=1).reshape(shape),
        "s_rad_mag": np.linalg.norm(s_rad.reshape(-1, 3), axis=1).reshape(shape),
    }

    if mask_radius > 0.0:
        mask = dist.reshape(shape) < mask_radius
        for key in ALL_OBSERVABLE_KEYS:
            fields[key] = np.where(mask, 0.0, fields[key])

    return fields


def build_simulation_bundle(
    cfg: RadiatingChargeConfig | None = None,
    progress_callback: Callable[[int, int, float, float], None] | None = None,
) -> dict:
    cfg = cfg or load_radiating_charge_config()
    x_axis, y_axis, x_grid, y_grid = make_grid(cfg)
    times = make_animation_times(cfg)

    t_min = float(times[0] - _max_light_delay(cfg))
    t_max = float(times[-1])
    trajectory = solve_trajectory(cfg, t_min=t_min, t_max=t_max)

    trajectory_xyz, trajectory_v, trajectory_a = trajectory.eval(times)

    frames = {key: [] for key in cfg.observables.keys}
    residual_max = []

    t_ret_guess: np.ndarray | None = None
    for frame_idx, t_val in enumerate(times, start=1):
        t_ret, residual = solve_retarded_times(
            trajectory,
            x_grid,
            y_grid,
            cfg.grid.z_obs,
            float(t_val),
            cfg.source.c,
            initial_guess=t_ret_guess,
        )
        t_ret_guess = t_ret
        frame_residual = float(np.max(np.abs(residual)))
        residual_max.append(frame_residual)

        field_snapshot = evaluate_lw_fields(
            trajectory,
            t_ret,
            x_grid,
            y_grid,
            cfg.grid.z_obs,
            cfg.source.charge,
            cfg.source.c,
            cfg.source.eps0,
            mask_radius=cfg.grid.mask_radius,
        )
        for key in cfg.observables.keys:
            frames[key].append(field_snapshot[key].astype(np.float32))

        if progress_callback is not None:
            progress_callback(frame_idx, len(times), float(t_val), frame_residual)

    centers = np.zeros((len(times), 1, 2), dtype=np.float32)
    centers[:, 0, 0] = trajectory_xyz[:, 0]
    centers[:, 0, 1] = trajectory_xyz[:, 1]

    return {
        "cfg": cfg,
        "x_axis": x_axis.astype(np.float32),
        "y_axis": y_axis.astype(np.float32),
        "times": times.astype(np.float32),
        "source_centers": centers,
        "trajectory_xyz": trajectory_xyz.astype(np.float32),
        "trajectory_v": trajectory_v.astype(np.float32),
        "trajectory_a": trajectory_a.astype(np.float32),
        "retarded_residual_max": np.asarray(residual_max, dtype=np.float32),
        "frames": {key: np.stack(values, axis=0).astype(np.float32) for key, values in frames.items()},
    }
