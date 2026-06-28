from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pmt.physics.types import ELEMENTARY_CHARGE


_ZERO_TOL = 1e-14


@dataclass(frozen=True)
class TrackResult:
    time_axis_s: np.ndarray
    positions_m: np.ndarray
    velocities_m_s: np.ndarray
    alive: np.ndarray

    status: np.ndarray
    impact_step: np.ndarray
    impact_position_m: np.ndarray
    impact_velocity_m_s: np.ndarray


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


def interpolate_field(
    ex: np.ndarray,
    ey: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    x_axis: np.ndarray,
    y_axis: np.ndarray,
    outside_value: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    i0, j0, wx, wy, inside = _bilinear_indices(x, y, x_axis, y_axis)

    def _interp(field: np.ndarray) -> np.ndarray:
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

    return _interp(ex), _interp(ey), inside


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


def boris_push_2d(
    velocity: np.ndarray,
    electric_field: np.ndarray,
    bz_t: float,
    q_over_m: float,
    dt: float,
) -> np.ndarray:
    v = np.asarray(velocity, dtype=float)
    e = np.asarray(electric_field, dtype=float)

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


def trace_particles(
    x_axis: np.ndarray,
    y_axis: np.ndarray,
    ex: np.ndarray,
    ey: np.ndarray,
    receiver_mask: np.ndarray,
    cathode_mask: np.ndarray,
    focus_mask: np.ndarray,
    positions0: np.ndarray,
    velocities0: np.ndarray,
    steps: int,
    dt_s: float,
    charge_c: float,
    mass_kg: float,
    bz_t: float,
    integration_substeps: int = 1,
) -> TrackResult:
    count = int(positions0.shape[0])
    positions = np.full((steps + 1, count, 2), np.nan, dtype=float)
    velocities = np.full((steps + 1, count, 2), np.nan, dtype=float)
    alive = np.zeros((steps + 1, count), dtype=bool)

    positions[0] = positions0
    velocities[0] = velocities0
    alive[0] = True

    status = np.zeros(count, dtype=np.int8)
    impact_step = np.full(count, -1, dtype=np.int32)
    impact_position = np.full((count, 2), np.nan, dtype=float)
    impact_velocity = np.full((count, 2), np.nan, dtype=float)

    q_over_m = charge_c / mass_kg
    substeps = max(1, int(integration_substeps))
    dt_sub = dt_s / substeps
    last_recorded = 0

    for step in range(steps):
        live = status == 0
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

            ex_s, ey_s, _ = interpolate_field(
                ex,
                ey,
                pos_live[:, 0],
                pos_live[:, 1],
                x_axis,
                y_axis,
                outside_value=0.0,
            )
            field = np.column_stack((ex_s, ey_s))

            vel_new = boris_push_2d(vel_live, field, bz_t=bz_t, q_over_m=q_over_m, dt=dt_sub)
            pos_new = pos_live + dt_sub * vel_new

            i_cell, j_cell, inside_new = positions_to_cell_indices(
                pos_new[:, 0],
                pos_new[:, 1],
                x_axis,
                y_axis,
            )

            hit_receiver = np.zeros(local_idx.shape[0], dtype=bool)
            hit_cathode = np.zeros(local_idx.shape[0], dtype=bool)
            hit_focus = np.zeros(local_idx.shape[0], dtype=bool)

            inside_local = np.flatnonzero(inside_new)
            if inside_local.size:
                ii = i_cell[inside_local]
                jj = j_cell[inside_local]
                hit_receiver[inside_local] = receiver_mask[ii, jj]
                hit_cathode[inside_local] = cathode_mask[ii, jj]
                hit_focus[inside_local] = focus_mask[ii, jj]

            pos_curr[local_idx] = pos_new
            vel_curr[local_idx] = vel_new

            die_collect = hit_receiver
            die_outside = ~inside_new
            die_cathode = hit_cathode
            die_focus = hit_focus
            die_any = die_collect | die_outside | die_cathode | die_focus

            if np.any(die_any):
                global_idx = live_idx[local_idx[die_any]]
                never_terminated = status[global_idx] == 0
                if np.any(never_terminated):
                    gidx = global_idx[never_terminated]
                    local_terminated = np.flatnonzero(die_any)[never_terminated]

                    reason_collect = die_collect[local_terminated]
                    reason_outside = die_outside[local_terminated]
                    reason_cathode = die_cathode[local_terminated]
                    reason_focus = die_focus[local_terminated]

                    status[gidx[reason_collect]] = 1
                    status[gidx[~reason_collect & reason_outside]] = 2
                    status[gidx[~reason_collect & ~reason_outside & reason_cathode]] = 3
                    status[gidx[~reason_collect & ~reason_outside & ~reason_cathode & reason_focus]] = 4

                    impact_step[gidx] = step + 1
                    impact_position[gidx] = pos_new[local_terminated]
                    impact_velocity[gidx] = vel_new[local_terminated]

            alive_curr[local_idx[die_any]] = False

        positions[step + 1, live_idx] = pos_curr
        velocities[step + 1, live_idx] = vel_curr
        alive[step + 1, live_idx] = alive_curr
        last_recorded = step + 1

    status[status == 0] = 5
    time_axis = np.arange(last_recorded + 1, dtype=float) * dt_s

    return TrackResult(
        time_axis_s=time_axis,
        positions_m=positions[: last_recorded + 1],
        velocities_m_s=velocities[: last_recorded + 1],
        alive=alive[: last_recorded + 1],
        status=status,
        impact_step=impact_step,
        impact_position_m=impact_position,
        impact_velocity_m_s=impact_velocity,
    )


def kinetic_energy_ev(velocity_m_s: np.ndarray, mass_kg: float) -> np.ndarray:
    speed2 = np.sum(np.asarray(velocity_m_s, dtype=float) ** 2, axis=-1)
    return 0.5 * mass_kg * speed2 / ELEMENTARY_CHARGE
