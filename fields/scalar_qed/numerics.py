from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
import tomllib

import numpy as np


DEFAULT_CONFIG_PATH = Path(__file__).with_name("scalar_qed.toml")
_ERF_P = 0.3275911
_ERF_A1 = 0.254829592
_ERF_A2 = -0.284496736
_ERF_A3 = 1.421413741
_ERF_A4 = -1.453152027
_ERF_A5 = 1.061405429
_ZERO_TOL = 1e-12


@dataclass(frozen=True)
class GridConfig:
    nx: int
    ny: int
    lx: float
    ly: float


@dataclass(frozen=True)
class TimeConfig:
    t_start: float
    t_end: float
    simulation_num_samples: int
    animation_num_samples: int


@dataclass(frozen=True)
class PhysicsConfig:
    mass: float
    charge_q: float


@dataclass(frozen=True)
class ObservableConfig:
    lower_surface: str
    upper_surface: str


@dataclass(frozen=True)
class RenderConfig:
    lower_height_target: float
    upper_height_target: float
    lower_visual_gain: float
    upper_visual_gain: float
    upper_plane_shift: float
    playback_time_scale: float


@dataclass(frozen=True)
class PacketConfig:
    name: str
    charge_sign: int
    charge_ratio: float
    norm: float
    sigma: float
    px: float
    py: float
    x0: float
    y0: float
    phase0: float = 0.0


@dataclass(frozen=True)
class PacketFrame:
    center: np.ndarray
    momentum: np.ndarray
    velocity: np.ndarray
    energy: float
    gamma: float
    tau: float
    sigma_eff: float


@dataclass(frozen=True)
class ScalarQEDConfig:
    grid: GridConfig
    time: TimeConfig
    physics: PhysicsConfig
    observables: ObservableConfig
    render: RenderConfig
    packets: tuple[PacketConfig, ...]


def load_scalar_qed_config(path: str | Path | None = None) -> ScalarQEDConfig:
    config_path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    data = tomllib.loads(config_path.read_text(encoding="utf-8"))
    time_data = data["time"]
    simulation_num_samples = int(time_data.get("simulation_num_samples", time_data.get("num_samples", 0)))
    if simulation_num_samples < 2:
        raise ValueError("scalar_qed.toml must define time.simulation_num_samples >= 2 (or legacy time.num_samples).")
    animation_num_samples = int(time_data.get("animation_num_samples", time_data.get("output_num_samples", simulation_num_samples)))
    if animation_num_samples < 2:
        raise ValueError("scalar_qed.toml must define time.animation_num_samples >= 2 when provided.")

    packets_data = data.get("packets", [])
    packets = tuple(_load_packet(item, idx) for idx, item in enumerate(packets_data))
    if not packets:
        raise ValueError("scalar_qed.toml must define at least one packet in [[packets]].")

    return ScalarQEDConfig(
        grid=GridConfig(
            nx=int(data["grid"]["nx"]),
            ny=int(data["grid"]["ny"]),
            lx=float(data["grid"]["lx"]),
            ly=float(data["grid"]["ly"]),
        ),
        time=TimeConfig(
            t_start=float(time_data["t_start"]),
            t_end=float(time_data["t_end"]),
            simulation_num_samples=simulation_num_samples,
            animation_num_samples=animation_num_samples,
        ),
        physics=PhysicsConfig(
            mass=float(data["physics"]["mass"]),
            charge_q=float(data["physics"]["charge_q"]),
        ),
        observables=ObservableConfig(
            lower_surface=str(data["observables"]["lower_surface"]),
            upper_surface=str(data["observables"]["upper_surface"]),
        ),
        render=RenderConfig(
            lower_height_target=float(data["render"]["lower_height_target"]),
            upper_height_target=float(data["render"]["upper_height_target"]),
            lower_visual_gain=float(data["render"].get("lower_visual_gain", 1.0)),
            upper_visual_gain=float(data["render"].get("upper_visual_gain", 1.0)),
            upper_plane_shift=float(data["render"]["upper_plane_shift"]),
            playback_time_scale=float(data["render"].get("playback_time_scale", 1.0)),
        ),
        packets=packets,
    )


def _scaled_count(value: int, scale: float, minimum: int) -> int:
    return max(minimum, int(round(float(value) * float(scale))))


def make_preview_config(
    cfg: ScalarQEDConfig,
    grid_scale: float = 0.45,
    time_scale: float = 0.35,
    playback_scale: float = 0.35,
) -> ScalarQEDConfig:
    """Build a lighter config for quick parameter tuning without touching physics extents."""
    preview_grid = replace(
        cfg.grid,
        nx=_scaled_count(cfg.grid.nx, grid_scale, minimum=72),
        ny=_scaled_count(cfg.grid.ny, grid_scale, minimum=54),
    )
    preview_time = replace(
        cfg.time,
        simulation_num_samples=_scaled_count(cfg.time.simulation_num_samples, time_scale, minimum=72),
        animation_num_samples=min(
            _scaled_count(cfg.time.animation_num_samples, time_scale, minimum=48),
            _scaled_count(cfg.time.simulation_num_samples, time_scale, minimum=72),
        ),
    )
    preview_render = replace(
        cfg.render,
        playback_time_scale=cfg.render.playback_time_scale * float(playback_scale),
    )
    return replace(cfg, grid=preview_grid, time=preview_time, render=preview_render)


def _load_packet(item: dict, idx: int) -> PacketConfig:
    charge_ratio_raw = item.get("charge_ratio")
    charge_sign_raw = int(item.get("charge_sign", 1))
    phase0 = float(item.get("phase0", 0.0))

    if charge_ratio_raw is not None:
        charge_ratio = float(charge_ratio_raw)
        charge_sign = 1 if charge_ratio > 0.0 else -1 if charge_ratio < 0.0 else charge_sign_raw
        norm = float(np.sqrt(abs(charge_ratio) / 2.0))
    else:
        norm = float(item["norm"])
        charge_sign = 1 if charge_sign_raw >= 0 else -1
        charge_ratio = 2.0 * charge_sign * norm**2

    return PacketConfig(
        name=str(item.get("name", f"packet_{idx + 1}")),
        charge_sign=charge_sign,
        charge_ratio=charge_ratio,
        norm=norm,
        sigma=float(item["sigma"]),
        px=float(item["px"]),
        py=float(item["py"]),
        x0=float(item["x0"]),
        y0=float(item["y0"]),
        phase0=phase0,
    )


def gaussian_erf(x: np.ndarray) -> np.ndarray:
    abs_x = np.abs(x)
    t = 1.0 / (1.0 + _ERF_P * abs_x)
    poly = ((((_ERF_A5 * t + _ERF_A4) * t + _ERF_A3) * t + _ERF_A2) * t + _ERF_A1) * t
    return np.sign(x) * (1.0 - poly * np.exp(-(abs_x**2)))


def packet_charge_sign(packet: PacketConfig) -> float:
    return 1.0 if packet.charge_sign >= 0 else -1.0


def packet_initial_momentum(packet: PacketConfig) -> np.ndarray:
    return np.array([packet.px, packet.py], dtype=float)


def momentum_energy(momentum: np.ndarray, mass: float) -> float:
    return float(np.sqrt(mass**2 + float(np.dot(momentum, momentum))))


def momentum_gamma(momentum: np.ndarray, mass: float) -> float:
    return momentum_energy(momentum, mass) / mass


def momentum_velocity(momentum: np.ndarray, mass: float) -> np.ndarray:
    return np.asarray(momentum, dtype=float) / momentum_energy(momentum, mass)


def packet_energy(packet: PacketConfig, mass: float) -> float:
    return momentum_energy(packet_initial_momentum(packet), mass)


def packet_gamma(packet: PacketConfig, mass: float) -> float:
    return momentum_gamma(packet_initial_momentum(packet), mass)


def packet_velocity(packet: PacketConfig, mass: float) -> np.ndarray:
    return momentum_velocity(packet_initial_momentum(packet), mass)


def packet_charge(packet: PacketConfig, mass: float, charge_q: float) -> float:
    return packet.charge_ratio * mass * charge_q


def packet_minkowski_momentum(packet: PacketConfig, mass: float) -> np.ndarray:
    momentum = packet_initial_momentum(packet)
    return np.array([momentum_energy(momentum, mass), momentum[0], momentum[1]], dtype=float)


def minkowski_dot(momentum_a: np.ndarray, momentum_b: np.ndarray) -> float:
    return float(momentum_a[0] * momentum_b[0] - momentum_a[1] * momentum_b[1] - momentum_a[2] * momentum_b[2])


def make_grid(cfg: ScalarQEDConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x_axis = np.linspace(-0.5 * cfg.grid.lx, 0.5 * cfg.grid.lx, cfg.grid.nx, endpoint=False)
    y_axis = np.linspace(-0.5 * cfg.grid.ly, 0.5 * cfg.grid.ly, cfg.grid.ny, endpoint=False)
    x_grid, y_grid = np.meshgrid(x_axis, y_axis, indexing="ij")
    return x_axis, y_axis, x_grid, y_grid


def make_simulation_times(cfg: ScalarQEDConfig) -> np.ndarray:
    return np.linspace(cfg.time.t_start, cfg.time.t_end, cfg.time.simulation_num_samples)


def make_animation_times(cfg: ScalarQEDConfig) -> np.ndarray:
    return np.linspace(cfg.time.t_start, cfg.time.t_end, cfg.time.animation_num_samples)


def packet_center(
    packet: PacketConfig,
    mass: float,
    t: float,
    time_origin: float = 0.0,
    center_override: np.ndarray | None = None,
) -> np.ndarray:
    if center_override is not None:
        return np.asarray(center_override, dtype=float)
    return np.array([packet.x0, packet.y0], dtype=float) + packet_velocity(packet, mass) * (t - time_origin)


def packet_frame(
    packet: PacketConfig,
    mass: float,
    t: float,
    time_origin: float = 0.0,
    center_override: np.ndarray | None = None,
) -> PacketFrame:
    momentum = packet_initial_momentum(packet)
    energy = momentum_energy(momentum, mass)
    gamma = energy / mass
    velocity = momentum / energy
    center = packet_center(packet, mass, t, time_origin=time_origin, center_override=center_override)
    elapsed = t - time_origin
    tau = 2.0 * packet.sigma**2 * elapsed / (mass * gamma)
    sigma_eff = packet.sigma / np.sqrt(1.0 + tau**2)
    return PacketFrame(
        center=np.array(center, dtype=float),
        momentum=np.array(momentum, dtype=float),
        velocity=np.array(velocity, dtype=float),
        energy=energy,
        gamma=gamma,
        tau=tau,
        sigma_eff=float(sigma_eff),
    )


def packet_r_star_sq(
    packet: PacketConfig,
    mass: float,
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    t: float,
    frame: PacketFrame | None = None,
    time_origin: float = 0.0,
) -> np.ndarray:
    active_frame = frame or packet_frame(packet, mass, t, time_origin=time_origin)
    dx = x_grid - active_frame.center[0]
    dy = y_grid - active_frame.center[1]
    radius_sq = dx**2 + dy**2

    speed_sq = float(np.dot(active_frame.velocity, active_frame.velocity))
    if speed_sq <= _ZERO_TOL:
        return radius_sq

    longitudinal = (active_frame.velocity[0] * dx + active_frame.velocity[1] * dy) ** 2 / speed_sq
    return radius_sq + (active_frame.gamma**2 - 1.0) * longitudinal


def packet_phase_argument(
    packet: PacketConfig,
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    t: float,
    frame: PacketFrame,
    time_origin: float = 0.0,
) -> np.ndarray:
    elapsed = t - time_origin
    dx = x_grid - frame.center[0]
    dy = y_grid - frame.center[1]
    return frame.momentum[0] * dx + frame.momentum[1] * dy - frame.energy * elapsed + packet.phase0


def packet_free_field(
    packet: PacketConfig,
    mass: float,
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    t: float,
    frame: PacketFrame | None = None,
    time_origin: float = 0.0,
) -> np.ndarray:
    active_frame = frame or packet_frame(packet, mass, t, time_origin=time_origin)
    r_star_sq = packet_r_star_sq(packet, mass, x_grid, y_grid, t, frame=active_frame, time_origin=time_origin)
    prefactor = packet.norm * (2.0 * packet.sigma**2 / np.pi) ** 0.75
    complex_scale = 1.0 + 1j * active_frame.tau
    envelope = prefactor * np.power(complex_scale, -1.5) * np.exp(-(packet.sigma**2 * r_star_sq) / complex_scale)

    sign = packet_charge_sign(packet)
    if sign < 0.0:
        envelope = np.conjugate(envelope)

    phase = packet_phase_argument(packet, x_grid, y_grid, t, active_frame, time_origin=time_origin)
    return envelope * np.exp(1j * sign * phase)


def packet_second_order_field(
    packet: PacketConfig,
    physics: PhysicsConfig,
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    t: float,
    theta: np.ndarray,
    frame: PacketFrame | None = None,
    time_origin: float = 0.0,
) -> np.ndarray:
    phi_free = packet_free_field(packet, physics.mass, x_grid, y_grid, t, frame=frame, time_origin=time_origin)
    return -1j * packet_charge_sign(packet) * physics.charge_q**2 * theta * phi_free


def packet_full_field(
    packet: PacketConfig,
    physics: PhysicsConfig,
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    t: float,
    theta: np.ndarray,
    frame: PacketFrame | None = None,
    time_origin: float = 0.0,
) -> np.ndarray:
    phi_free = packet_free_field(packet, physics.mass, x_grid, y_grid, t, frame=frame, time_origin=time_origin)
    return phi_free * np.exp(-1j * packet_charge_sign(packet) * physics.charge_q**2 * theta)


def packet_density_profile(
    packet: PacketConfig,
    mass: float,
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    t: float,
    frame: PacketFrame | None = None,
    time_origin: float = 0.0,
) -> np.ndarray:
    active_frame = frame or packet_frame(packet, mass, t, time_origin=time_origin)
    r_star_sq = packet_r_star_sq(packet, mass, x_grid, y_grid, t, frame=active_frame, time_origin=time_origin)
    return (2.0 * active_frame.sigma_eff**2 / np.pi) ** 1.5 * np.exp(-2.0 * active_frame.sigma_eff**2 * r_star_sq)


def packet_density(
    packet: PacketConfig,
    mass: float,
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    t: float,
    frame: PacketFrame | None = None,
    time_origin: float = 0.0,
) -> np.ndarray:
    return packet.norm**2 * packet_density_profile(packet, mass, x_grid, y_grid, t, frame=frame, time_origin=time_origin)


def smoothed_coulomb_potential(radius: np.ndarray, sigma_eff: np.ndarray | float) -> np.ndarray:
    safe_radius = np.maximum(radius, _ZERO_TOL)
    sigma_arr = np.asarray(sigma_eff, dtype=float)
    argument = np.sqrt(2.0) * sigma_arr * safe_radius
    potential = gaussian_erf(argument) / (4.0 * np.pi * safe_radius)
    zero_limit = sigma_arr / (np.sqrt(2.0) * np.pi**1.5)
    return np.where(radius > 1e-9, potential, zero_limit)


def packet_potential_kernel(
    packet: PacketConfig,
    mass: float,
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    t: float,
    frame: PacketFrame | None = None,
    time_origin: float = 0.0,
) -> np.ndarray:
    active_frame = frame or packet_frame(packet, mass, t, time_origin=time_origin)
    radius = np.sqrt(packet_r_star_sq(packet, mass, x_grid, y_grid, t, frame=active_frame, time_origin=time_origin))
    return smoothed_coulomb_potential(radius, active_frame.sigma_eff)


def packet_potential(
    packet: PacketConfig,
    mass: float,
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    t: float,
    frame: PacketFrame | None = None,
    time_origin: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    active_frame = frame or packet_frame(packet, mass, t, time_origin=time_origin)
    potential_kernel = packet_potential_kernel(packet, mass, x_grid, y_grid, t, frame=active_frame, time_origin=time_origin)
    scale = packet.charge_ratio
    return (
        potential_kernel,
        scale * active_frame.energy * potential_kernel,
        scale * active_frame.momentum[0] * potential_kernel,
        scale * active_frame.momentum[1] * potential_kernel,
    )


def packet_current(
    packet: PacketConfig,
    mass: float,
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    t: float,
    frame: PacketFrame | None = None,
    time_origin: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    active_frame = frame or packet_frame(packet, mass, t, time_origin=time_origin)
    density_profile = packet_density_profile(packet, mass, x_grid, y_grid, t, frame=active_frame, time_origin=time_origin)
    density = packet.norm**2 * density_profile
    scale = packet.charge_ratio
    return density, scale * active_frame.energy * density_profile, scale * active_frame.momentum[0] * density_profile, scale * active_frame.momentum[1] * density_profile


def bilinear_sample_field(
    x_axis: np.ndarray,
    y_axis: np.ndarray,
    field: np.ndarray,
    x_query: np.ndarray,
    y_query: np.ndarray,
) -> np.ndarray:
    x_step = float(x_axis[1] - x_axis[0])
    y_step = float(y_axis[1] - y_axis[0])

    x_float = (x_query - x_axis[0]) / x_step
    y_float = (y_query - y_axis[0]) / y_step

    ix = np.floor(x_float).astype(int)
    iy = np.floor(y_float).astype(int)
    tx = x_float - ix
    ty = y_float - iy

    ix = np.clip(ix, 0, len(x_axis) - 2)
    iy = np.clip(iy, 0, len(y_axis) - 2)
    tx = np.where(x_query <= x_axis[0], 0.0, np.where(x_query >= x_axis[-1], 1.0, tx))
    ty = np.where(y_query <= y_axis[0], 0.0, np.where(y_query >= y_axis[-1], 1.0, ty))

    v00 = field[ix, iy]
    v10 = field[ix + 1, iy]
    v01 = field[ix, iy + 1]
    v11 = field[ix + 1, iy + 1]
    return (1.0 - tx) * (1.0 - ty) * v00 + tx * (1.0 - ty) * v10 + (1.0 - tx) * ty * v01 + tx * ty * v11


def bilinear_sample(x_axis: np.ndarray, y_axis: np.ndarray, field: np.ndarray, x_val: float, y_val: float) -> float:
    if x_val <= x_axis[0]:
        ix = 0
        tx = 0.0
    elif x_val >= x_axis[-1]:
        ix = len(x_axis) - 2
        tx = 1.0
    else:
        ix = int(np.searchsorted(x_axis, x_val, side="right") - 1)
        dx = x_axis[ix + 1] - x_axis[ix]
        tx = (x_val - x_axis[ix]) / dx

    if y_val <= y_axis[0]:
        iy = 0
        ty = 0.0
    elif y_val >= y_axis[-1]:
        iy = len(y_axis) - 2
        ty = 1.0
    else:
        iy = int(np.searchsorted(y_axis, y_val, side="right") - 1)
        dy = y_axis[iy + 1] - y_axis[iy]
        ty = (y_val - y_axis[iy]) / dy

    v00 = field[ix, iy]
    v10 = field[ix + 1, iy]
    v01 = field[ix, iy + 1]
    v11 = field[ix + 1, iy + 1]
    return float((1.0 - tx) * (1.0 - ty) * v00 + tx * (1.0 - ty) * v10 + (1.0 - tx) * ty * v01 + tx * ty * v11)


def _build_straight_trajectories(cfg: ScalarQEDConfig, times: np.ndarray) -> np.ndarray:
    centers = np.zeros((len(times), len(cfg.packets), 2), dtype=float)
    time_origin = float(times[0])
    for packet_idx, packet in enumerate(cfg.packets):
        start = np.array([packet.x0, packet.y0], dtype=float)
        velocity = packet_velocity(packet, cfg.physics.mass)
        for time_idx, t_value in enumerate(times):
            centers[time_idx, packet_idx] = start + velocity * (float(t_value) - time_origin)
    return centers


def _packet_frames_at_time(
    cfg: ScalarQEDConfig,
    times: np.ndarray,
    centers: np.ndarray,
    time_idx: int,
) -> list[PacketFrame]:
    t_value = float(times[time_idx])
    time_origin = float(times[0])
    return _packet_frames_for_centers(cfg, t_value, centers[time_idx], time_origin)


def _packet_frames_for_centers(
    cfg: ScalarQEDConfig,
    t_value: float,
    centers_at_time: np.ndarray,
    time_origin: float,
) -> list[PacketFrame]:
    return [
        packet_frame(
            packet,
            cfg.physics.mass,
            t_value,
            time_origin=time_origin,
            center_override=centers_at_time[packet_idx],
        )
        for packet_idx, packet in enumerate(cfg.packets)
    ]


def _packet_snapshot_data(
    cfg: ScalarQEDConfig,
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    t: float,
    frames: list[PacketFrame],
    time_origin: float,
) -> list[dict[str, np.ndarray]]:
    packet_data: list[dict[str, np.ndarray]] = []
    for packet, frame in zip(cfg.packets, frames, strict=True):
        potential_kernel, a0, ax, ay = packet_potential(packet, cfg.physics.mass, x_grid, y_grid, t, frame=frame, time_origin=time_origin)
        density, j0, jx, jy = packet_current(packet, cfg.physics.mass, x_grid, y_grid, t, frame=frame, time_origin=time_origin)
        packet_data.append(
            {
                "potential_kernel": potential_kernel,
                "density": density,
                "j0": j0,
                "jx": jx,
                "jy": jy,
                "a0": a0,
                "ax": ax,
                "ay": ay,
            }
        )
    return packet_data


def _theta_sources(
    cfg: ScalarQEDConfig,
    packet_data: list[dict[str, np.ndarray]],
    frames: list[PacketFrame],
    interaction_only: bool = False,
) -> list[np.ndarray]:
    momenta = [packet_minkowski_momentum(packet, cfg.physics.mass) for packet in cfg.packets]
    potentials = [item["potential_kernel"] for item in packet_data]

    sources: list[np.ndarray] = []
    for packet_idx, (packet_a, momentum_a) in enumerate(zip(cfg.packets, momenta, strict=True)):
        source = np.zeros_like(potentials[0], dtype=float)
        for other_idx, (packet_b, momentum_b, potential_b) in enumerate(zip(cfg.packets, momenta, potentials, strict=True)):
            if interaction_only and other_idx == packet_idx:
                continue
            source += packet_charge_sign(packet_a) * packet_b.charge_ratio * minkowski_dot(momentum_a, momentum_b) * potential_b
        sources.append(source)
    return sources


def _advance_theta_fields(
    x_axis: np.ndarray,
    y_axis: np.ndarray,
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    theta_prev: list[np.ndarray],
    sources_prev: list[np.ndarray],
    sources_curr: list[np.ndarray],
    frames_curr: list[PacketFrame],
    dt: float,
) -> list[np.ndarray]:
    theta_curr: list[np.ndarray] = []
    for frame, theta_field, source_prev, source_curr in zip(frames_curr, theta_prev, sources_prev, sources_curr, strict=True):
        x_back = x_grid - frame.velocity[0] * dt
        y_back = y_grid - frame.velocity[1] * dt
        theta_adv = bilinear_sample_field(x_axis, y_axis, theta_field, x_back, y_back)
        source_prev_adv = bilinear_sample_field(x_axis, y_axis, source_prev, x_back, y_back)
        theta_curr.append(theta_adv + 0.5 * dt * (source_prev_adv + source_curr) / frame.energy)
    return theta_curr


def _solve_theta_history(
    cfg: ScalarQEDConfig,
    x_axis: np.ndarray,
    y_axis: np.ndarray,
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    times: np.ndarray,
    centers_track: np.ndarray,
) -> tuple[list[list[np.ndarray]], list[list[np.ndarray]]]:
    time_origin = float(times[0])
    frames_prev = _packet_frames_at_time(cfg, times, centers_track, 0)
    packet_data_prev = _packet_snapshot_data(cfg, x_grid, y_grid, float(times[0]), frames_prev, time_origin)

    theta_total_prev = [np.zeros_like(x_grid, dtype=float) for _ in cfg.packets]
    theta_int_prev = [np.zeros_like(x_grid, dtype=float) for _ in cfg.packets]
    sources_total_prev = _theta_sources(cfg, packet_data_prev, frames_prev, interaction_only=False)
    sources_int_prev = _theta_sources(cfg, packet_data_prev, frames_prev, interaction_only=True)

    theta_total_history: list[list[np.ndarray]] = [[field.copy() for field in theta_total_prev]]
    theta_int_history: list[list[np.ndarray]] = [[field.copy() for field in theta_int_prev]]

    for time_idx, (prev_t, curr_t) in enumerate(zip(times[:-1], times[1:], strict=True), start=1):
        frames_curr = _packet_frames_at_time(cfg, times, centers_track, time_idx)
        packet_data_curr = _packet_snapshot_data(cfg, x_grid, y_grid, float(curr_t), frames_curr, time_origin)
        sources_total_curr = _theta_sources(cfg, packet_data_curr, frames_curr, interaction_only=False)
        sources_int_curr = _theta_sources(cfg, packet_data_curr, frames_curr, interaction_only=True)
        dt = float(curr_t - prev_t)

        theta_total_curr = _advance_theta_fields(
            x_axis,
            y_axis,
            x_grid,
            y_grid,
            theta_total_prev,
            sources_total_prev,
            sources_total_curr,
            frames_curr,
            dt,
        )
        theta_int_curr = _advance_theta_fields(
            x_axis,
            y_axis,
            x_grid,
            y_grid,
            theta_int_prev,
            sources_int_prev,
            sources_int_curr,
            frames_curr,
            dt,
        )

        theta_total_history.append([field.copy() for field in theta_total_curr])
        theta_int_history.append([field.copy() for field in theta_int_curr])

        theta_total_prev = theta_total_curr
        theta_int_prev = theta_int_curr
        sources_total_prev = sources_total_curr
        sources_int_prev = sources_int_curr

    return theta_total_history, theta_int_history


def _field_gradient_at_point(
    x_axis: np.ndarray,
    y_axis: np.ndarray,
    field: np.ndarray,
    point: np.ndarray,
) -> np.ndarray:
    dx = float(x_axis[1] - x_axis[0])
    dy = float(y_axis[1] - y_axis[0])
    x_val = float(point[0])
    y_val = float(point[1])
    dfdx = (bilinear_sample(x_axis, y_axis, field, x_val + 0.5 * dx, y_val) - bilinear_sample(x_axis, y_axis, field, x_val - 0.5 * dx, y_val)) / dx
    dfdy = (bilinear_sample(x_axis, y_axis, field, x_val, y_val + 0.5 * dy) - bilinear_sample(x_axis, y_axis, field, x_val, y_val - 0.5 * dy)) / dy
    return np.array([dfdx, dfdy], dtype=float)


def _transverse_projector(packet: PacketConfig) -> np.ndarray:
    momentum = packet_initial_momentum(packet)
    norm = float(np.linalg.norm(momentum))
    if norm <= _ZERO_TOL:
        return np.eye(2, dtype=float)
    direction = momentum / norm
    return np.eye(2, dtype=float) - np.outer(direction, direction)


def _build_eikonal_trajectories(
    cfg: ScalarQEDConfig,
    times: np.ndarray,
    x_axis: np.ndarray,
    y_axis: np.ndarray,
    theta_int_history: list[list[np.ndarray]],
    reference_centers: np.ndarray,
) -> np.ndarray:
    corrected_centers = np.zeros_like(reference_centers, dtype=float)
    corrected_centers[0] = reference_centers[0]

    base_velocities = [packet_velocity(packet, cfg.physics.mass) for packet in cfg.packets]
    base_energies = [packet_energy(packet, cfg.physics.mass) for packet in cfg.packets]
    projectors = [_transverse_projector(packet) for packet in cfg.packets]
    coupling_sq = cfg.physics.charge_q**2

    for time_idx in range(len(times) - 1):
        dt = float(times[time_idx + 1] - times[time_idx])
        for packet_idx, packet in enumerate(cfg.packets):
            reference_point = reference_centers[time_idx, packet_idx]
            grad_theta = _field_gradient_at_point(x_axis, y_axis, theta_int_history[time_idx][packet_idx], reference_point)
            delta_p = -coupling_sq * (projectors[packet_idx] @ grad_theta)
            local_velocity = base_velocities[packet_idx] + delta_p / base_energies[packet_idx]
            corrected_centers[time_idx + 1, packet_idx] = corrected_centers[time_idx, packet_idx] + dt * local_velocity

    return corrected_centers


def _compose_snapshot(
    cfg: ScalarQEDConfig,
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    t: float,
    packet_data: list[dict[str, np.ndarray]],
    theta_total_fields: list[np.ndarray],
    theta_int_fields: list[np.ndarray],
    frames: list[PacketFrame],
    time_origin: float,
) -> dict[str, np.ndarray]:
    phi_free_packets: list[np.ndarray] = []
    phi_linear_packets: list[np.ndarray] = []
    phi_full_packets: list[np.ndarray] = []

    for packet, theta_total, frame in zip(cfg.packets, theta_total_fields, frames, strict=True):
        phi_free = packet_free_field(packet, cfg.physics.mass, x_grid, y_grid, t, frame=frame, time_origin=time_origin)
        phi_linear = phi_free + (-1j * packet_charge_sign(packet) * cfg.physics.charge_q**2 * theta_total * phi_free)
        phi_full = phi_free * np.exp(-1j * packet_charge_sign(packet) * cfg.physics.charge_q**2 * theta_total)
        phi_free_packets.append(phi_free)
        phi_linear_packets.append(phi_linear)
        phi_full_packets.append(phi_full)

    phi_free_total = sum(phi_free_packets, np.zeros_like(x_grid, dtype=complex))
    phi_linear_total = sum(phi_linear_packets, np.zeros_like(x_grid, dtype=complex))
    phi_total = sum(phi_full_packets, np.zeros_like(x_grid, dtype=complex))

    density_total = sum((item["density"] for item in packet_data), np.zeros_like(x_grid, dtype=float))
    j0_total = sum((item["j0"] for item in packet_data), np.zeros_like(x_grid, dtype=float))
    jx_total = sum((item["jx"] for item in packet_data), np.zeros_like(x_grid, dtype=float))
    jy_total = sum((item["jy"] for item in packet_data), np.zeros_like(x_grid, dtype=float))
    a0_total = sum((item["a0"] for item in packet_data), np.zeros_like(x_grid, dtype=float))
    ax_total = sum((item["ax"] for item in packet_data), np.zeros_like(x_grid, dtype=float))
    ay_total = sum((item["ay"] for item in packet_data), np.zeros_like(x_grid, dtype=float))
    theta_total_sum = sum(theta_total_fields, np.zeros_like(x_grid, dtype=float))
    theta_int_sum = sum(theta_int_fields, np.zeros_like(x_grid, dtype=float))

    snapshot = {
        "phi_real": phi_total.real,
        "phi_imag": phi_total.imag,
        "phi_abs2": np.abs(phi_total) ** 2,
        "phi_phase": np.angle(phi_total),
        "phi_free_real": phi_free_total.real,
        "phi_free_imag": phi_free_total.imag,
        "phi_free_abs2": np.abs(phi_free_total) ** 2,
        "phi_linear_real": phi_linear_total.real,
        "phi_linear_imag": phi_linear_total.imag,
        "phi_linear_abs2": np.abs(phi_linear_total) ** 2,
        "density": density_total,
        "j0": j0_total,
        "jx": jx_total,
        "jy": jy_total,
        "rho": cfg.physics.charge_q * j0_total,
        "a0": a0_total,
        "ax": ax_total,
        "ay": ay_total,
        "a0_physical": cfg.physics.charge_q * a0_total,
        "ax_physical": cfg.physics.charge_q * ax_total,
        "ay_physical": cfg.physics.charge_q * ay_total,
        "theta_sum": theta_total_sum,
        "theta_int_sum": theta_int_sum,
        "phase_sum": cfg.physics.charge_q**2 * theta_total_sum,
        "phase_int_sum": cfg.physics.charge_q**2 * theta_int_sum,
    }

    for idx, (packet, theta_total, theta_int, phi_packet, phi_free_packet, packet_snapshot, frame) in enumerate(
        zip(cfg.packets, theta_total_fields, theta_int_fields, phi_full_packets, phi_free_packets, packet_data, frames, strict=True),
        start=1,
    ):
        signed_phase = packet_charge_sign(packet) * cfg.physics.charge_q**2 * theta_total
        signed_phase_int = packet_charge_sign(packet) * cfg.physics.charge_q**2 * theta_int
        snapshot[f"theta_{idx}"] = theta_total
        snapshot[f"theta_int_{idx}"] = theta_int
        snapshot[f"phase_{idx}"] = signed_phase
        snapshot[f"phase_int_{idx}"] = signed_phase_int
        snapshot[f"phi_{idx}_real"] = phi_packet.real
        snapshot[f"phi_{idx}_imag"] = phi_packet.imag
        snapshot[f"phi_{idx}_abs2"] = np.abs(phi_packet) ** 2
        snapshot[f"phi_{idx}_free_real"] = phi_free_packet.real
        snapshot[f"phi_{idx}_free_imag"] = phi_free_packet.imag
        snapshot[f"phi_{idx}_free_abs2"] = np.abs(phi_free_packet) ** 2
        snapshot[f"density_{idx}"] = packet_snapshot["density"]
        snapshot[f"j0_{idx}"] = packet_snapshot["j0"]
        snapshot[f"a0_{idx}"] = packet_snapshot["a0"]
        snapshot[f"center_{idx}_x"] = np.full_like(x_grid, frame.center[0], dtype=float)
        snapshot[f"center_{idx}_y"] = np.full_like(x_grid, frame.center[1], dtype=float)

    return snapshot


def resolve_observable(snapshot: dict[str, np.ndarray], key: str) -> np.ndarray:
    if key not in snapshot:
        raise KeyError(f"Unknown observable `{key}`. Available keys: {sorted(snapshot.keys())}")
    return snapshot[key]


def _run_simulation(
    cfg: ScalarQEDConfig,
    x_axis: np.ndarray,
    y_axis: np.ndarray,
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    times: np.ndarray,
) -> tuple[np.ndarray, list[list[np.ndarray]], list[list[np.ndarray]]]:
    straight_centers = _build_straight_trajectories(cfg, times)
    theta_total_history, theta_int_history = _solve_theta_history(cfg, x_axis, y_axis, x_grid, y_grid, times, straight_centers)
    corrected_centers = _build_eikonal_trajectories(cfg, times, x_axis, y_axis, theta_int_history, straight_centers)
    return corrected_centers, theta_total_history, theta_int_history


def build_simulation_bundle(cfg: ScalarQEDConfig | None = None) -> dict:
    cfg = cfg or load_scalar_qed_config()
    x_axis, y_axis, x_grid, y_grid = make_grid(cfg)
    simulation_times = make_simulation_times(cfg)
    corrected_centers, theta_total_history, theta_int_history = _run_simulation(
        cfg,
        x_axis,
        y_axis,
        x_grid,
        y_grid,
        simulation_times,
    )
    return {
        "cfg": cfg,
        "x_axis": x_axis,
        "y_axis": y_axis,
        "simulation_times": simulation_times,
        "corrected_centers": corrected_centers,
        "theta_total_history": theta_total_history,
        "theta_int_history": theta_int_history,
    }


def snapshot_fields(
    cfg: ScalarQEDConfig,
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    t: float,
) -> dict[str, np.ndarray]:
    x_axis = x_grid[:, 0]
    y_axis = y_grid[0, :]
    times = np.linspace(cfg.time.t_start, t, max(2, int(np.ceil(abs(t - cfg.time.t_start) * 24.0)) + 1))
    time_origin = float(times[0])
    corrected_centers, theta_total_history, theta_int_history = _run_simulation(cfg, x_axis, y_axis, x_grid, y_grid, times)

    time_idx = len(times) - 1
    frames = _packet_frames_at_time(cfg, times, corrected_centers, time_idx)
    packet_data = _packet_snapshot_data(cfg, x_grid, y_grid, float(times[time_idx]), frames, time_origin)
    return _compose_snapshot(
        cfg,
        x_grid,
        y_grid,
        float(times[time_idx]),
        packet_data,
        theta_total_history[time_idx],
        theta_int_history[time_idx],
        frames,
        time_origin,
    )


def _time_interpolation_state(times: np.ndarray, t_value: float) -> tuple[int, float]:
    if t_value <= times[0]:
        return 0, 0.0
    if t_value >= times[-1]:
        return len(times) - 1, 0.0
    idx = int(np.searchsorted(times, t_value, side="right") - 1)
    idx = max(0, min(idx, len(times) - 2))
    dt = max(float(times[idx + 1] - times[idx]), 1e-9)
    return idx, (t_value - times[idx]) / dt


def _sample_packet_field_history(
    times: np.ndarray,
    history: list[list[np.ndarray]],
    t_value: float,
) -> list[np.ndarray]:
    idx, tau = _time_interpolation_state(times, t_value)
    if tau <= _ZERO_TOL or idx == len(times) - 1:
        return [field.copy() for field in history[idx]]
    return [
        (1.0 - tau) * field_prev + tau * field_next
        for field_prev, field_next in zip(history[idx], history[idx + 1], strict=True)
    ]


def build_animation_bundle_from_simulation(simulation_bundle: dict) -> dict:
    cfg: ScalarQEDConfig = simulation_bundle["cfg"]
    x_axis: np.ndarray = simulation_bundle["x_axis"]
    y_axis: np.ndarray = simulation_bundle["y_axis"]
    simulation_times: np.ndarray = simulation_bundle["simulation_times"]
    corrected_centers: np.ndarray = simulation_bundle["corrected_centers"]
    theta_total_history: list[list[np.ndarray]] = simulation_bundle["theta_total_history"]
    theta_int_history: list[list[np.ndarray]] = simulation_bundle["theta_int_history"]
    x_grid, y_grid = np.meshgrid(x_axis, y_axis, indexing="ij")
    animation_times = make_animation_times(cfg)
    time_origin = float(simulation_times[0])

    lower_frames: list[np.ndarray] = []
    upper_frames: list[np.ndarray] = []
    phi_real_frames: list[np.ndarray] = []
    phi_abs2_frames: list[np.ndarray] = []
    charge_frames: list[np.ndarray] = []
    a0_frames: list[np.ndarray] = []
    packet_centers: list[np.ndarray] = []

    for t_value in animation_times:
        centers_at_time = sample_frames(simulation_times, corrected_centers, float(t_value))
        frames = _packet_frames_for_centers(cfg, float(t_value), centers_at_time, time_origin)
        packet_data = _packet_snapshot_data(cfg, x_grid, y_grid, float(t_value), frames, time_origin)
        theta_total_fields = _sample_packet_field_history(simulation_times, theta_total_history, float(t_value))
        theta_int_fields = _sample_packet_field_history(simulation_times, theta_int_history, float(t_value))
        snapshot = _compose_snapshot(
            cfg,
            x_grid,
            y_grid,
            float(t_value),
            packet_data,
            theta_total_fields,
            theta_int_fields,
            frames,
            time_origin,
        )
        lower_frames.append(resolve_observable(snapshot, cfg.observables.lower_surface))
        upper_frames.append(resolve_observable(snapshot, cfg.observables.upper_surface))
        phi_real_frames.append(snapshot["phi_real"])
        phi_abs2_frames.append(snapshot["phi_abs2"])
        charge_frames.append(snapshot["rho"])
        a0_frames.append(snapshot["a0"])
        packet_centers.append(np.array([frame.center for frame in frames], dtype=float))

    return {
        "cfg": cfg,
        "x_axis": x_axis,
        "y_axis": y_axis,
        "simulation_times": simulation_times,
        "times": animation_times,
        "animation_times": animation_times,
        "lower_frames": np.array(lower_frames),
        "upper_frames": np.array(upper_frames),
        "phi_real_frames": np.array(phi_real_frames),
        "phi_abs2_frames": np.array(phi_abs2_frames),
        "charge_frames": np.array(charge_frames),
        "a0_frames": np.array(a0_frames),
        "packet_centers": np.array(packet_centers),
    }


def build_animation_bundle(cfg: ScalarQEDConfig | None = None) -> dict:
    return build_animation_bundle_from_simulation(build_simulation_bundle(cfg))


def sample_frames(times: np.ndarray, frames: np.ndarray, t_value: float) -> np.ndarray:
    idx, tau = _time_interpolation_state(times, t_value)
    if tau <= _ZERO_TOL or idx == len(times) - 1:
        return frames[idx]
    return (1.0 - tau) * frames[idx] + tau * frames[idx + 1]
