from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PacketParameters:
    amplitude: float
    sigma: float
    mass: float
    kx: float
    ky: float
    x0: float
    y0: float
    phase0: float = 0.0


@dataclass(frozen=True)
class SceneTimeConfig:
    t_start: float
    t_end: float


def packet_diagnostics(params: PacketParameters) -> dict:
    k_vec = np.array([params.kx, params.ky], dtype=float)
    k0 = float(np.linalg.norm(k_vec))
    omega0 = float(np.sqrt(k0**2 + params.mass**2))
    e_parallel = k_vec / max(k0, 1e-8)
    e_perp = np.array([-e_parallel[1], e_parallel[0]])
    v_group = k_vec / omega0
    alpha_parallel = params.mass**2 / omega0**3
    alpha_perp = 1.0 / omega0
    return {
        "k_vec": k_vec,
        "k0": k0,
        "omega0": omega0,
        "e_parallel": e_parallel,
        "e_perp": e_perp,
        "v_group": v_group,
        "alpha_parallel": alpha_parallel,
        "alpha_perp": alpha_perp,
    }


def packet_center(params: PacketParameters, t: float) -> np.ndarray:
    data = packet_diagnostics(params)
    return np.array([params.x0, params.y0]) + data["v_group"] * t


def packet_field(x_grid, y_grid, t: float, params: PacketParameters, full_phase: bool = True) -> np.ndarray:
    data = packet_diagnostics(params)
    center = packet_center(params, t)
    dx = np.asarray(x_grid) - center[0]
    dy = np.asarray(y_grid) - center[1]

    delta_parallel = dx * data["e_parallel"][0] + dy * data["e_parallel"][1]
    delta_perp = dx * data["e_perp"][0] + dy * data["e_perp"][1]

    sigma_sq = params.sigma**2
    sigma_parallel_sq = sigma_sq + (data["alpha_parallel"]**2 * t**2) / sigma_sq
    sigma_perp_sq = sigma_sq + (data["alpha_perp"]**2 * t**2) / sigma_sq
    amplitude_factor = (
        params.amplitude
        * sigma_sq
        / ((sigma_sq**2 + data["alpha_parallel"]**2 * t**2) ** 0.25 * (sigma_sq**2 + data["alpha_perp"]**2 * t**2) ** 0.25)
    )
    envelope = np.exp(-0.5 * delta_parallel**2 / sigma_parallel_sq - 0.5 * delta_perp**2 / sigma_perp_sq)

    base_x = np.asarray(x_grid) - params.x0
    base_y = np.asarray(y_grid) - params.y0
    carrier = params.kx * base_x + params.ky * base_y - data["omega0"] * t + params.phase0

    if full_phase:
        carrier += 0.5 * data["alpha_parallel"] * t * delta_parallel**2 / max(sigma_sq**2 + data["alpha_parallel"]**2 * t**2, 1e-9)
        carrier += 0.5 * data["alpha_perp"] * t * delta_perp**2 / max(sigma_sq**2 + data["alpha_perp"]**2 * t**2, 1e-9)
        carrier -= 0.5 * np.arctan2(data["alpha_parallel"] * t, sigma_sq)
        carrier -= 0.5 * np.arctan2(data["alpha_perp"] * t, sigma_sq)

    return amplitude_factor * envelope * np.cos(carrier)


def superposed_field(x_grid, y_grid, t: float, packets: list[PacketParameters], full_phase: bool = True) -> np.ndarray:
    total = np.zeros_like(np.asarray(x_grid), dtype=float)
    for params in packets:
        total += packet_field(x_grid, y_grid, t, params, full_phase=full_phase)
    return total
