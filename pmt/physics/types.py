from __future__ import annotations

from dataclasses import dataclass


CM_TO_M = 1.0e-2
M_TO_CM = 1.0e2
ELEMENTARY_CHARGE = 1.602176634e-19
ELECTRON_MASS = 9.1093837015e-31


@dataclass(frozen=True)
class SceneSpec:
    name: str
    title: str
    kind: str
    physics_mode: str
    launch_mode: str


@dataclass(frozen=True)
class GridSpec:
    nx: int
    ny: int
    lx_m: float
    ly_m: float


@dataclass(frozen=True)
class TimeSpec:
    dt_s: float
    steps: int
    integration_substeps: int
    bz_t: float


@dataclass(frozen=True)
class ElectronSpec:
    count: int
    mass_kg: float
    charge_c: float
    initial_energy_ev: float
    initial_energy_spread_ev: float
    angle_spread_deg: float
    emission_jitter_m: float
    seed: int
    launch_angle_deg: float
    point_radial_velocity_m_s: float
    point_transverse_velocity_m_s: float
    point_transverse_velocity_spread_m_s: float


@dataclass(frozen=True)
class ElectrodeSpec:
    background_voltage: float
    cathode_voltage: float
    anode_voltage: float
    focus_voltage: float


@dataclass(frozen=True)
class GeometrySpec:
    enable_cathode: bool
    enable_receiver: bool

    cathode_shape: str
    photocathode_diameter_m: float
    photocathode_thickness_m: float
    photocathode_center_x_m: float
    photocathode_center_y_m: float
    photocathode_active_theta_min_deg: float
    photocathode_active_theta_max_deg: float

    line_cathode_x_m: float
    line_cathode_height_m: float
    line_cathode_thickness_m: float

    launch_point_x_m: float
    launch_point_y_m: float

    receiver_kind: str
    receiver_point_x_m: float
    receiver_point_y_m: float
    receiver_radius_m: float

    plate_center_x_m: float
    plate_center_y_m: float
    plate_length_m: float
    plate_thickness_m: float
    plate_angle_deg: float

    focus_enabled: bool
    focus_kind: str
    focus_center_x_m: float
    focus_center_y_m: float
    focus_length_m: float
    focus_thickness_m: float
    focus_angle_deg: float


@dataclass(frozen=True)
class CentralFieldSpec:
    center_x_m: float
    center_y_m: float
    kappa: float
    softening_m: float


@dataclass(frozen=True)
class RenderSpec:
    figure_width_in: float
    figure_height_in: float
    dpi: int

    background: str
    trajectory_color: str
    field_line_color: str
    electrode_cathode_color: str
    electrode_receiver_color: str
    electrode_focus_color: str

    scalar_map: str
    colormap: str
    show_field_lines: bool
    field_line_density: float
    field_line_mode: str
    normalize_field_arrows: bool
    field_arrow_length_cm: float

    show_equipotential_lines: bool
    equipotential_count: int
    equipotential_color: str
    equipotential_linewidth: float

    trajectory_alpha: float
    trajectory_linewidth: float
    max_trajectories: int


@dataclass(frozen=True)
class SolverSpec:
    sor_omega: float
    sor_tolerance: float
    sor_max_iterations: int


@dataclass(frozen=True)
class SimulationConfig:
    scene: SceneSpec
    grid: GridSpec
    time: TimeSpec
    electrons: ElectronSpec
    electrodes: ElectrodeSpec
    geometry: GeometrySpec
    central: CentralFieldSpec
    render: RenderSpec
    solver: SolverSpec
