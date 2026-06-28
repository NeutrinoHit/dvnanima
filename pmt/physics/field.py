from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pmt.physics.geometry import GeometryData
from pmt.physics.types import SimulationConfig


@dataclass(frozen=True)
class FieldSolution:
    potential: np.ndarray
    ex: np.ndarray
    ey: np.ndarray
    solver_iterations: int
    solver_residual: float


def solve_laplace_red_black_sor(
    fixed_mask: np.ndarray,
    fixed_values: np.ndarray,
    dx: float,
    dy: float,
    omega: float,
    tolerance: float,
    max_iterations: int,
) -> tuple[np.ndarray, int, float]:
    if fixed_mask.shape != fixed_values.shape:
        raise ValueError("fixed_mask and fixed_values must have identical shapes")

    phi = np.array(fixed_values, dtype=float, copy=True)
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


def analytic_central_field(
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    center_x_m: float,
    center_y_m: float,
    kappa: float,
    softening_m: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    dx = x_grid - center_x_m
    dy = y_grid - center_y_m
    r2 = dx * dx + dy * dy + softening_m * softening_m
    sqrt_r2 = np.sqrt(r2)

    potential = kappa / sqrt_r2
    coeff = kappa / (r2 * sqrt_r2)
    ex = coeff * dx
    ey = coeff * dy
    return potential, ex, ey


def build_field(
    cfg: SimulationConfig,
    geometry: GeometryData,
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    dx: float,
    dy: float,
) -> FieldSolution:
    mode = cfg.scene.physics_mode.strip().lower()

    if mode in {"central", "analytic", "point", "point_field"}:
        potential, ex, ey = analytic_central_field(
            x_grid=x_grid,
            y_grid=y_grid,
            center_x_m=cfg.central.center_x_m,
            center_y_m=cfg.central.center_y_m,
            kappa=cfg.central.kappa,
            softening_m=cfg.central.softening_m,
        )
        return FieldSolution(
            potential=potential,
            ex=ex,
            ey=ey,
            solver_iterations=0,
            solver_residual=0.0,
        )

    potential, iterations, residual = solve_laplace_red_black_sor(
        fixed_mask=geometry.fixed_mask,
        fixed_values=geometry.fixed_values,
        dx=dx,
        dy=dy,
        omega=cfg.solver.sor_omega,
        tolerance=cfg.solver.sor_tolerance,
        max_iterations=cfg.solver.sor_max_iterations,
    )
    ex, ey = compute_electric_field(potential, dx=dx, dy=dy)

    ex = np.array(ex, copy=True)
    ey = np.array(ey, copy=True)
    ex[geometry.fixed_mask] = 0.0
    ey[geometry.fixed_mask] = 0.0

    return FieldSolution(
        potential=potential,
        ex=ex,
        ey=ey,
        solver_iterations=int(iterations),
        solver_residual=float(residual),
    )
