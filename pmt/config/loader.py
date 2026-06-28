from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path
import re
import tomllib
from typing import Any

from pmt.physics.types import (
    CM_TO_M,
    ELECTRON_MASS,
    ELEMENTARY_CHARGE,
    CentralFieldSpec,
    ElectronSpec,
    ElectrodeSpec,
    GeometrySpec,
    GridSpec,
    RenderSpec,
    SceneSpec,
    SimulationConfig,
    SolverSpec,
    TimeSpec,
)


DEFAULT_BASE_CONFIG = Path(__file__).with_name("base.toml")


def _as_float(value: Any, default: float) -> float:
    if value is None:
        return float(default)
    return float(value)


def _as_int(value: Any, default: int) -> int:
    if value is None:
        return int(default)
    return int(value)


def _as_bool(value: Any, default: bool) -> bool:
    if value is None:
        return bool(default)
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off"}:
        return False
    return bool(default)


def _read_length_m(data: dict[str, Any], key_cm: str, key_m: str, default_cm: float) -> float:
    if key_m in data:
        return float(data[key_m])
    if key_cm in data:
        return float(data[key_cm]) * CM_TO_M
    return float(default_cm) * CM_TO_M


def load_toml(path: str | Path) -> dict[str, Any]:
    resolved = Path(path)
    text = resolved.read_text(encoding="utf-8")
    parsed = tomllib.loads(text)
    if not isinstance(parsed, dict):
        raise ValueError(f"TOML root must be a table: {resolved}")
    return parsed


def deep_merge_dicts(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge_dicts(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def merge_base_and_scene(base_path: str | Path, scene_path: str | Path) -> dict[str, Any]:
    base_cfg = load_toml(base_path)
    scene_cfg = load_toml(scene_path)
    return deep_merge_dicts(base_cfg, scene_cfg)


def discover_scene_files(scene_dir: str | Path) -> list[Path]:
    root = Path(scene_dir)
    return sorted(path for path in root.glob("*.toml") if path.is_file())


def slugify_name(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
    cleaned = cleaned.strip("._-")
    return cleaned or "scene"


def infer_scene_name(merged_config: dict[str, Any], scene_path: str | Path | None = None) -> str:
    scene = merged_config.get("scene", {})
    if isinstance(scene, dict):
        name = str(scene.get("name", "")).strip()
        if name:
            return name
    if scene_path is not None:
        return Path(scene_path).stem
    return "scene"


def merged_config_to_runtime(merged: dict[str, Any], scene_path: str | Path | None = None) -> SimulationConfig:
    scene_data = merged.get("scene", {}) if isinstance(merged.get("scene"), dict) else {}
    grid_data = merged.get("grid", {}) if isinstance(merged.get("grid"), dict) else {}

    electrons_data = merged.get("electrons", {}) if isinstance(merged.get("electrons"), dict) else {}
    particles_data = merged.get("particles", {}) if isinstance(merged.get("particles"), dict) else {}
    if not electrons_data and particles_data:
        electrons_data = particles_data

    time_data = merged.get("time", {}) if isinstance(merged.get("time"), dict) else {}
    if not time_data and particles_data:
        time_data = {
            "dt": particles_data.get("dt"),
            "steps": particles_data.get("steps"),
            "integration_substeps": particles_data.get("integration_substeps"),
            "bz_t": merged.get("simulation", {}).get("bz_t") if isinstance(merged.get("simulation"), dict) else None,
        }

    electrodes_data = merged.get("electrodes", {}) if isinstance(merged.get("electrodes"), dict) else {}
    geometry_data = merged.get("geometry", {}) if isinstance(merged.get("geometry"), dict) else {}
    central_data = merged.get("central", {}) if isinstance(merged.get("central"), dict) else {}
    render_data = merged.get("render", {}) if isinstance(merged.get("render"), dict) else {}
    solver_data = merged.get("solver", {}) if isinstance(merged.get("solver"), dict) else {}
    simulation_data = merged.get("simulation", {}) if isinstance(merged.get("simulation"), dict) else {}

    scene_name = infer_scene_name(merged, scene_path)

    scene = SceneSpec(
        name=scene_name,
        title=str(scene_data.get("title", scene_name.replace("_", " ").title())),
        kind=str(scene_data.get("kind", scene_name)).strip().lower(),
        physics_mode=str(scene_data.get("physics_mode", "numerical")).strip().lower(),
        launch_mode=str(scene_data.get("launch_mode", "surface")).strip().lower(),
    )

    grid = GridSpec(
        nx=_as_int(grid_data.get("nx"), 280),
        ny=_as_int(grid_data.get("ny"), 220),
        lx_m=_read_length_m(grid_data, "lx_cm", "lx_m", 120.0),
        ly_m=_read_length_m(grid_data, "ly_cm", "ly_m", 90.0),
    )

    time = TimeSpec(
        dt_s=_as_float(time_data.get("dt"), 5.0e-12),
        steps=_as_int(time_data.get("steps"), 8000),
        integration_substeps=max(1, _as_int(time_data.get("integration_substeps"), 4)),
        bz_t=_as_float(time_data.get("bz_t", simulation_data.get("bz_t")), 0.0),
    )

    electrons = ElectronSpec(
        count=max(1, _as_int(electrons_data.get("count"), 400)),
        mass_kg=_as_float(electrons_data.get("mass", electrons_data.get("mass_kg")), ELECTRON_MASS),
        charge_c=_as_float(electrons_data.get("charge", electrons_data.get("charge_c")), -ELEMENTARY_CHARGE),
        initial_energy_ev=_as_float(electrons_data.get("initial_energy_ev"), 0.35),
        initial_energy_spread_ev=max(0.0, _as_float(electrons_data.get("initial_energy_spread_ev"), 0.12)),
        angle_spread_deg=max(0.0, _as_float(electrons_data.get("angle_spread_deg"), 8.0)),
        emission_jitter_m=_read_length_m(electrons_data, "emission_jitter_cm", "emission_jitter_m", 0.10),
        seed=_as_int(electrons_data.get("seed", simulation_data.get("seed")), 12345),
        launch_angle_deg=_as_float(electrons_data.get("launch_angle_deg"), 0.0),
        point_radial_velocity_m_s=_as_float(electrons_data.get("point_radial_velocity_m_s"), 0.0),
        point_transverse_velocity_m_s=_as_float(electrons_data.get("point_transverse_velocity_m_s"), 0.0),
        point_transverse_velocity_spread_m_s=max(
            0.0,
            _as_float(electrons_data.get("point_transverse_velocity_spread_m_s"), 0.0),
        ),
    )

    electrodes = ElectrodeSpec(
        background_voltage=_as_float(
            electrodes_data.get("background_voltage", simulation_data.get("background_voltage")),
            0.0,
        ),
        cathode_voltage=_as_float(
            electrodes_data.get("cathode_voltage", simulation_data.get("cathode_voltage")),
            0.0,
        ),
        anode_voltage=_as_float(
            electrodes_data.get("anode_voltage", simulation_data.get("dynode_voltage")),
            1200.0,
        ),
        focus_voltage=_as_float(electrodes_data.get("focus_voltage"), 600.0),
    )

    geometry = GeometrySpec(
        enable_cathode=_as_bool(geometry_data.get("enable_cathode"), True),
        enable_receiver=_as_bool(geometry_data.get("enable_receiver"), True),
        cathode_shape=str(
            geometry_data.get("cathode_shape", simulation_data.get("photocathode_shape", "hemisphere"))
        )
        .strip()
        .lower(),
        photocathode_diameter_m=_read_length_m(
            geometry_data,
            "photocathode_diameter_cm",
            "photocathode_diameter_m",
            _as_float(simulation_data.get("photocathode_diameter_cm"), 50.8),
        ),
        photocathode_thickness_m=_read_length_m(
            geometry_data,
            "photocathode_thickness_cm",
            "photocathode_thickness_m",
            _as_float(simulation_data.get("photocathode_thickness_cm"), 0.6),
        ),
        photocathode_center_x_m=_read_length_m(
            geometry_data,
            "photocathode_center_x_cm",
            "photocathode_center_x_m",
            _as_float(simulation_data.get("photocathode_center_x_cm"), -20.0),
        ),
        photocathode_center_y_m=_read_length_m(
            geometry_data,
            "photocathode_center_y_cm",
            "photocathode_center_y_m",
            _as_float(simulation_data.get("photocathode_center_y_cm"), 0.0),
        ),
        photocathode_active_theta_min_deg=_as_float(
            geometry_data.get(
                "photocathode_active_theta_min_deg",
                simulation_data.get("photocathode_active_theta_min_deg"),
            ),
            -80.0,
        ),
        photocathode_active_theta_max_deg=_as_float(
            geometry_data.get(
                "photocathode_active_theta_max_deg",
                simulation_data.get("photocathode_active_theta_max_deg"),
            ),
            80.0,
        ),
        line_cathode_x_m=_read_length_m(
            geometry_data,
            "line_cathode_x_cm",
            "line_cathode_x_m",
            _as_float(simulation_data.get("cathode_x_cm"), -26.0),
        ),
        line_cathode_height_m=_read_length_m(
            geometry_data,
            "line_cathode_height_cm",
            "line_cathode_height_m",
            _as_float(simulation_data.get("cathode_height_cm"), 50.8),
        ),
        line_cathode_thickness_m=_read_length_m(
            geometry_data,
            "line_cathode_thickness_cm",
            "line_cathode_thickness_m",
            _as_float(simulation_data.get("cathode_thickness_cm"), 0.6),
        ),
        launch_point_x_m=_read_length_m(geometry_data, "launch_point_x_cm", "launch_point_x_m", -5.0),
        launch_point_y_m=_read_length_m(geometry_data, "launch_point_y_cm", "launch_point_y_m", 0.0),
        receiver_kind=str(geometry_data.get("receiver_kind", "plate")).strip().lower(),
        receiver_point_x_m=_read_length_m(geometry_data, "receiver_point_x_cm", "receiver_point_x_m", 8.0),
        receiver_point_y_m=_read_length_m(geometry_data, "receiver_point_y_cm", "receiver_point_y_m", 0.0),
        receiver_radius_m=_read_length_m(geometry_data, "receiver_radius_cm", "receiver_radius_m", 0.25),
        plate_center_x_m=_read_length_m(
            geometry_data,
            "plate_center_x_cm",
            "plate_center_x_m",
            _as_float(simulation_data.get("dynode_center_x_cm"), 8.0),
        ),
        plate_center_y_m=_read_length_m(
            geometry_data,
            "plate_center_y_cm",
            "plate_center_y_m",
            _as_float(simulation_data.get("dynode_center_y_cm"), 0.0),
        ),
        plate_length_m=_read_length_m(
            geometry_data,
            "plate_length_cm",
            "plate_length_m",
            _as_float(simulation_data.get("dynode_length_cm"), 4.0),
        ),
        plate_thickness_m=_read_length_m(
            geometry_data,
            "plate_thickness_cm",
            "plate_thickness_m",
            _as_float(simulation_data.get("dynode_thickness_cm"), 0.8),
        ),
        plate_angle_deg=_as_float(
            geometry_data.get("plate_angle_deg", simulation_data.get("dynode_angle_deg")),
            -14.0,
        ),
        focus_enabled=_as_bool(geometry_data.get("focus_enabled"), False),
        focus_kind=str(geometry_data.get("focus_kind", "plate")).strip().lower(),
        focus_center_x_m=_read_length_m(geometry_data, "focus_center_x_cm", "focus_center_x_m", 2.0),
        focus_center_y_m=_read_length_m(geometry_data, "focus_center_y_cm", "focus_center_y_m", 0.0),
        focus_length_m=_read_length_m(geometry_data, "focus_length_cm", "focus_length_m", 3.0),
        focus_thickness_m=_read_length_m(geometry_data, "focus_thickness_cm", "focus_thickness_m", 0.5),
        focus_angle_deg=_as_float(geometry_data.get("focus_angle_deg"), -10.0),
    )

    central = CentralFieldSpec(
        center_x_m=_read_length_m(central_data, "center_x_cm", "center_x_m", 0.0),
        center_y_m=_read_length_m(central_data, "center_y_cm", "center_y_m", 0.0),
        kappa=_as_float(central_data.get("kappa"), 1.0),
        softening_m=max(1e-9, _read_length_m(central_data, "softening_cm", "softening_m", 0.3)),
    )

    render = RenderSpec(
        figure_width_in=_as_float(render_data.get("figure_width"), 12.0),
        figure_height_in=_as_float(render_data.get("figure_height"), 8.0),
        dpi=max(40, _as_int(render_data.get("dpi"), 180)),
        background=str(render_data.get("background", "black")),
        trajectory_color=str(render_data.get("trajectory_color", "white")),
        field_line_color=str(render_data.get("field_line_color", "#ff4d4d")),
        electrode_cathode_color=str(render_data.get("electrode_cathode_color", "#66b3ff")),
        electrode_receiver_color=str(render_data.get("electrode_receiver_color", "#ff9f66")),
        electrode_focus_color=str(render_data.get("electrode_focus_color", "#8cff99")),
        scalar_map=str(render_data.get("scalar_map", "potential")).strip().lower(),
        colormap=str(render_data.get("colormap", "magma")),
        show_field_lines=_as_bool(render_data.get("show_field_lines"), True),
        field_line_density=max(0.0, _as_float(render_data.get("field_line_density"), 1.0)),
        field_line_mode=str(render_data.get("field_line_mode", "stream")).strip().lower(),
        normalize_field_arrows=_as_bool(render_data.get("normalize_field_arrows"), False),
        field_arrow_length_cm=max(0.1, _as_float(render_data.get("field_arrow_length_cm"), 2.0)),
        show_equipotential_lines=_as_bool(render_data.get("show_equipotential_lines"), False),
        equipotential_count=max(0, _as_int(render_data.get("equipotential_count"), 0)),
        equipotential_color=str(render_data.get("equipotential_color", "#9fb8ff")),
        equipotential_linewidth=max(0.1, _as_float(render_data.get("equipotential_linewidth"), 1.0)),
        trajectory_alpha=min(1.0, max(0.0, _as_float(render_data.get("trajectory_alpha"), 0.38))),
        trajectory_linewidth=max(0.1, _as_float(render_data.get("trajectory_linewidth"), 0.9)),
        max_trajectories=max(1, _as_int(render_data.get("max_trajectories"), 240)),
    )

    solver = SolverSpec(
        sor_omega=_as_float(solver_data.get("sor_omega", simulation_data.get("sor_omega")), 1.93),
        sor_tolerance=_as_float(solver_data.get("sor_tolerance", simulation_data.get("sor_tolerance")), 1e-8),
        sor_max_iterations=_as_int(
            solver_data.get("sor_max_iterations", simulation_data.get("sor_max_iterations")),
            20000,
        ),
    )

    return SimulationConfig(
        scene=scene,
        grid=grid,
        time=time,
        electrons=electrons,
        electrodes=electrodes,
        geometry=geometry,
        central=central,
        render=render,
        solver=solver,
    )


def runtime_config_to_dict(config: SimulationConfig) -> dict[str, Any]:
    return asdict(config)


def merged_config_to_pretty_json(merged: dict[str, Any]) -> str:
    return json.dumps(merged, ensure_ascii=False, indent=2, sort_keys=True)
