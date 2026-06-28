from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from fields.common.helpers import smooth_window_2d


@dataclass(frozen=True)
class ScalarEDSimulationConfig:
    nx: int = 192
    ny: int = 128
    lx: float = 20.0
    ly: float = 12.0
    dt: float = 0.006
    output_dt: float = 0.024
    t_final: float = 16.0
    mass: float = 1.0
    charge_e: float = 0.18
    sigma: float = 0.1
    amplitude: float = 0.35
    carrier_k: float = 1.0
    initial_offset: float = 4.5
    impact_parameter: float = 1.6
    damping_edge_fraction: float = 0.14
    damping_strength: float = 0.0


@dataclass(frozen=True)
class PacketSeed:
    x0: float
    y0: float
    kx: float
    ky: float
    amplitude: float
    sigma: float
    charge_sign: int


def packet_velocity(seed: PacketSeed, mass: float) -> np.ndarray:
    k2 = seed.kx**2 + seed.ky**2
    omega = np.sqrt(k2 + mass**2)
    return np.array([seed.kx / omega, seed.ky / omega])


def packet_profile(x_grid: np.ndarray, y_grid: np.ndarray, seed: PacketSeed) -> np.ndarray:
    dx = x_grid - seed.x0
    dy = y_grid - seed.y0
    envelope = seed.amplitude * np.exp(-(dx**2 + dy**2) / (2.0 * seed.sigma**2))
    # The sign is tied to the charge-frequency branch so that the user-facing
    # wave-vector sets the physical group velocity in the same way for both charges.
    phase = -seed.charge_sign * (seed.kx * dx + seed.ky * dy)
    return envelope * np.exp(1j * phase)


def packet_pi_initial(x_grid: np.ndarray, y_grid: np.ndarray, seed: PacketSeed, mass: float) -> np.ndarray:
    phi = packet_profile(x_grid, y_grid, seed)
    k2 = seed.kx**2 + seed.ky**2
    omega = np.sqrt(k2 + mass**2)
    return 1j * seed.charge_sign * omega * phi


def spectral_laplacian(field: np.ndarray, kx_grid: np.ndarray, ky_grid: np.ndarray) -> np.ndarray:
    field_hat = np.fft.fft2(field)
    lap_hat = -(kx_grid**2 + ky_grid**2) * field_hat
    return np.fft.ifft2(lap_hat)


def charge_density(phi: np.ndarray, pi: np.ndarray, charge_e: float) -> np.ndarray:
    return (-1j * charge_e * (np.conjugate(phi) * pi - np.conjugate(pi) * phi)).real


def solve_a0(rho: np.ndarray, kx_grid: np.ndarray, ky_grid: np.ndarray) -> np.ndarray:
    rho_zero_mean = rho - np.mean(rho)
    rho_hat = np.fft.fft2(rho_zero_mean)
    k2 = kx_grid**2 + ky_grid**2
    a0_hat = np.zeros_like(rho_hat, dtype=complex)
    mask = k2 > 1e-12
    a0_hat[mask] = rho_hat[mask] / k2[mask]
    a0_hat[~mask] = 0.0
    return np.fft.ifft2(a0_hat).real


def spectral_gradient(
    field: np.ndarray, kx_grid: np.ndarray, ky_grid: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    field_hat = np.fft.fft2(field)
    grad_x = np.fft.ifft2(1j * kx_grid * field_hat)
    grad_y = np.fft.ifft2(1j * ky_grid * field_hat)
    return grad_x, grad_y


def initial_fields(cfg: ScalarEDSimulationConfig, x_grid: np.ndarray, y_grid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    half_impact = 0.5 * cfg.impact_parameter
    left = PacketSeed(
        -cfg.initial_offset,
        half_impact,
        cfg.carrier_k,
        0.0,
        cfg.amplitude,
        cfg.sigma,
        +1,
    )
    right = PacketSeed(
        cfg.initial_offset,
        -half_impact,
        -cfg.carrier_k,
        0.0,
        cfg.amplitude,
        cfg.sigma,
        -1,
    )
    phi0 = packet_profile(x_grid, y_grid, left) + packet_profile(x_grid, y_grid, right)
    pi0 = packet_pi_initial(x_grid, y_grid, left, cfg.mass) + packet_pi_initial(x_grid, y_grid, right, cfg.mass)
    return phi0, pi0


def rhs(
    phi: np.ndarray,
    pi: np.ndarray,
    cfg: ScalarEDSimulationConfig,
    kx_grid: np.ndarray,
    ky_grid: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rho = charge_density(phi, pi, cfg.charge_e)
    a0 = solve_a0(rho, kx_grid, ky_grid)
    dphi = pi - 1j * cfg.charge_e * a0 * phi
    dpi = spectral_laplacian(phi, kx_grid, ky_grid) - cfg.mass**2 * phi - 1j * cfg.charge_e * a0 * pi
    return dphi, dpi, a0


def rk4_step(
    phi: np.ndarray,
    pi: np.ndarray,
    dt: float,
    cfg: ScalarEDSimulationConfig,
    kx_grid: np.ndarray,
    ky_grid: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    k1_phi, k1_pi, _ = rhs(phi, pi, cfg, kx_grid, ky_grid)
    k2_phi, k2_pi, _ = rhs(phi + 0.5 * dt * k1_phi, pi + 0.5 * dt * k1_pi, cfg, kx_grid, ky_grid)
    k3_phi, k3_pi, _ = rhs(phi + 0.5 * dt * k2_phi, pi + 0.5 * dt * k2_pi, cfg, kx_grid, ky_grid)
    k4_phi, k4_pi, _ = rhs(phi + dt * k3_phi, pi + dt * k3_pi, cfg, kx_grid, ky_grid)

    phi_next = phi + (dt / 6.0) * (k1_phi + 2.0 * k2_phi + 2.0 * k3_phi + k4_phi)
    pi_next = pi + (dt / 6.0) * (k1_pi + 2.0 * k2_pi + 2.0 * k3_pi + k4_pi)
    return phi_next, pi_next


def simulate_scalar_ed(cfg: ScalarEDSimulationConfig | None = None) -> dict:
    cfg = cfg or ScalarEDSimulationConfig()
    num_steps = max(1, int(round(cfg.t_final / cfg.dt)))
    output_stride = max(1, int(round(cfg.output_dt / cfg.dt)))

    x_axis = np.linspace(-0.5 * cfg.lx, 0.5 * cfg.lx, cfg.nx, endpoint=False)
    y_axis = np.linspace(-0.5 * cfg.ly, 0.5 * cfg.ly, cfg.ny, endpoint=False)
    dx = float(x_axis[1] - x_axis[0])
    dy = float(y_axis[1] - y_axis[0])
    x_grid, y_grid = np.meshgrid(x_axis, y_axis, indexing="ij")

    kx = 2.0 * np.pi * np.fft.fftfreq(cfg.nx, d=dx)
    ky = 2.0 * np.pi * np.fft.fftfreq(cfg.ny, d=dy)
    kx_grid, ky_grid = np.meshgrid(kx, ky, indexing="ij")
    damping = smooth_window_2d(x_axis, y_axis, cfg.damping_edge_fraction, cfg.damping_strength)

    phi, pi = initial_fields(cfg, x_grid, y_grid)
    _, _, a0 = rhs(phi, pi, cfg, kx_grid, ky_grid)

    density_frames = [np.abs(phi) ** 2]
    a0_frames = [a0.copy()]
    charge_frames = [charge_density(phi, pi, cfg.charge_e)]
    times = [0.0]

    matter_energy = []
    coulomb_energy = []
    total_energy = []

    def append_energies(phi_field: np.ndarray, pi_field: np.ndarray, a0_field: np.ndarray) -> None:
        grad_x, grad_y = spectral_gradient(phi_field, kx_grid, ky_grid)
        a0_grad_x, a0_grad_y = spectral_gradient(a0_field, kx_grid, ky_grid)
        density_matter = (
            0.5 * np.abs(pi_field) ** 2
            + 0.5 * np.abs(grad_x) ** 2
            + 0.5 * np.abs(grad_y) ** 2
            + 0.5 * cfg.mass**2 * np.abs(phi_field) ** 2
        )
        density_coulomb = 0.5 * (a0_grad_x.real**2 + a0_grad_y.real**2)
        e_matter = float(np.sum(density_matter) * dx * dy)
        e_coulomb = float(np.sum(density_coulomb) * dx * dy)
        matter_energy.append(e_matter)
        coulomb_energy.append(e_coulomb)
        total_energy.append(e_matter + e_coulomb)

    append_energies(phi, pi, a0)

    for step in range(1, num_steps + 1):
        phi, pi = rk4_step(phi, pi, cfg.dt, cfg, kx_grid, ky_grid)
        if cfg.damping_strength > 0.0:
            phi *= damping
            pi *= damping

        if not np.isfinite(phi).all() or not np.isfinite(pi).all():
            raise FloatingPointError(
                "Scalar electrodynamics simulation became non-finite. Reduce dt, lower charge_e, or widen the packets/domain."
            )

        if step % output_stride == 0 or step == num_steps:
            _, _, a0 = rhs(phi, pi, cfg, kx_grid, ky_grid)
            density_frames.append(np.abs(phi) ** 2)
            a0_frames.append(a0.copy())
            charge_frames.append(charge_density(phi, pi, cfg.charge_e))
            append_energies(phi, pi, a0)
            times.append(step * cfg.dt)

    return {
        "cfg": cfg,
        "x_axis": x_axis,
        "y_axis": y_axis,
        "times": np.array(times),
        "density_frames": np.array(density_frames),
        "a0_frames": np.array(a0_frames),
        "charge_frames": np.array(charge_frames),
        "matter_energy": np.array(matter_energy),
        "coulomb_energy": np.array(coulomb_energy),
        "total_energy": np.array(total_energy),
    }


def sample_frames(times: np.ndarray, frames: np.ndarray, t_value: float) -> np.ndarray:
    if t_value <= times[0]:
        return frames[0]
    if t_value >= times[-1]:
        return frames[-1]
    idx = int(np.searchsorted(times, t_value, side="right") - 1)
    idx = max(0, min(idx, len(times) - 2))
    dt = max(float(times[idx + 1] - times[idx]), 1e-9)
    tau = (t_value - times[idx]) / dt
    return (1.0 - tau) * frames[idx] + tau * frames[idx + 1]


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


def signed_charge_centers(charge_density_field: np.ndarray, x_axis: np.ndarray, y_axis: np.ndarray) -> dict[str, tuple[float, float]]:
    x_grid = x_axis[:, None]
    y_grid = y_axis[None, :]
    positive = np.maximum(charge_density_field, 0.0)
    negative = np.maximum(-charge_density_field, 0.0)

    def center(weight: np.ndarray) -> tuple[float, float]:
        norm = max(float(np.sum(weight)), 1e-15)
        cx = float(np.sum(weight * x_grid) / norm)
        cy = float(np.sum(weight * y_grid) / norm)
        return cx, cy

    return {"positive": center(positive), "negative": center(negative)}
