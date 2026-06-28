from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any

import numpy as np

from pmt.physics.field import FieldSolution, build_field
from pmt.physics.geometry import GeometryData, build_geometry, launch_electrons, make_grid
from pmt.physics.trajectories import TrackResult, trace_particles
from pmt.physics.types import SimulationConfig


@dataclass(frozen=True)
class SceneStats:
    electron_count: int
    collected_count: int
    miss_count: int
    collection_efficiency: float
    solver_iterations: int
    solver_residual: float
    compute_seconds: float


@dataclass(frozen=True)
class SceneResult:
    config: SimulationConfig
    x_axis_m: np.ndarray
    y_axis_m: np.ndarray
    x_grid_m: np.ndarray
    y_grid_m: np.ndarray

    geometry: GeometryData
    field: FieldSolution
    tracks: TrackResult

    launch_positions_m: np.ndarray
    launch_velocities_m_s: np.ndarray
    stats: SceneStats


def _compute_stats(cfg: SimulationConfig, field: FieldSolution, tracks: TrackResult, elapsed_s: float) -> SceneStats:
    electron_count = int(cfg.electrons.count)
    collected_count = int(np.count_nonzero(tracks.status == 1))
    miss_count = int(electron_count - collected_count)
    efficiency = float(collected_count / max(electron_count, 1))

    return SceneStats(
        electron_count=electron_count,
        collected_count=collected_count,
        miss_count=miss_count,
        collection_efficiency=efficiency,
        solver_iterations=int(field.solver_iterations),
        solver_residual=float(field.solver_residual),
        compute_seconds=float(elapsed_s),
    )


def run_scene(cfg: SimulationConfig) -> SceneResult:
    started = time.perf_counter()

    x_axis, y_axis, x_grid, y_grid, dx, dy = make_grid(cfg)
    geometry = build_geometry(cfg, x_grid=x_grid, y_grid=y_grid, dx=dx, dy=dy)
    field = build_field(cfg, geometry=geometry, x_grid=x_grid, y_grid=y_grid, dx=dx, dy=dy)

    rng = np.random.default_rng(cfg.electrons.seed)
    launch_positions, launch_velocities = launch_electrons(cfg, geometry=geometry, rng=rng)

    tracks = trace_particles(
        x_axis=x_axis,
        y_axis=y_axis,
        ex=field.ex,
        ey=field.ey,
        receiver_mask=geometry.receiver_mask,
        cathode_mask=geometry.cathode_mask,
        focus_mask=geometry.focus_mask,
        positions0=launch_positions,
        velocities0=launch_velocities,
        steps=cfg.time.steps,
        dt_s=cfg.time.dt_s,
        charge_c=cfg.electrons.charge_c,
        mass_kg=cfg.electrons.mass_kg,
        bz_t=cfg.time.bz_t,
        integration_substeps=cfg.time.integration_substeps,
    )

    elapsed = time.perf_counter() - started
    stats = _compute_stats(cfg, field=field, tracks=tracks, elapsed_s=elapsed)

    return SceneResult(
        config=cfg,
        x_axis_m=x_axis,
        y_axis_m=y_axis,
        x_grid_m=x_grid,
        y_grid_m=y_grid,
        geometry=geometry,
        field=field,
        tracks=tracks,
        launch_positions_m=launch_positions,
        launch_velocities_m_s=launch_velocities,
        stats=stats,
    )


def scene_result_to_dict(result: SceneResult) -> dict[str, Any]:
    return {
        "x_axis_m": result.x_axis_m,
        "y_axis_m": result.y_axis_m,
        "potential": result.field.potential,
        "ex": result.field.ex,
        "ey": result.field.ey,
        "electrode_id": result.geometry.electrode_id,
        "fixed_mask": result.geometry.fixed_mask,
        "time_axis_s": result.tracks.time_axis_s,
        "launch_positions_m": result.launch_positions_m,
        "launch_velocities_m_s": result.launch_velocities_m_s,
        "positions_m": result.tracks.positions_m,
        "velocities_m_s": result.tracks.velocities_m_s,
        "alive": result.tracks.alive,
        "status": result.tracks.status,
        "impact_step": result.tracks.impact_step,
        "impact_position_m": result.tracks.impact_position_m,
        "impact_velocity_m_s": result.tracks.impact_velocity_m_s,
    }
