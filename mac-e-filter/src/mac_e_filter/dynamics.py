"""Full relativistic Lorentz dynamics and trajectory diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.constants import c, elementary_charge, electron_mass
from scipy.integrate import cumulative_trapezoid, solve_ivp
from scipy.optimize import brentq

from .fields import AxisymmetricMagneticField

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class Particle:
    name: str
    mass_kg: float
    charge_c: float

    def __post_init__(self) -> None:
        if self.mass_kg <= 0.0:
            raise ValueError("mass_kg must be positive")
        if self.charge_c == 0.0:
            raise ValueError("charge_c must be non-zero")


ELECTRON = Particle(
    name="electron",
    mass_kg=electron_mass,
    charge_c=-elementary_charge,
)


@dataclass(frozen=True)
class IntegrationSettings:
    """Numerical controls for the adaptive DOP853 reference integration."""

    relative_tolerance: float = 1.0e-10
    position_atol_m: float = 1.0e-10
    normalized_momentum_atol: float = 1.0e-12
    max_gyro_phase_rad: float = 0.15
    field_bound_t: float = 1.5e-3
    output_samples: int = 3001

    def __post_init__(self) -> None:
        if self.relative_tolerance <= 0.0:
            raise ValueError("relative_tolerance must be positive")
        if self.position_atol_m <= 0.0:
            raise ValueError("position_atol_m must be positive")
        if self.normalized_momentum_atol <= 0.0:
            raise ValueError("normalized_momentum_atol must be positive")
        if self.max_gyro_phase_rad <= 0.0:
            raise ValueError("max_gyro_phase_rad must be positive")
        if self.field_bound_t <= 0.0:
            raise ValueError("field_bound_t must be positive")
        if self.output_samples < 2:
            raise ValueError("output_samples must be at least two")

    def max_step_s(self, particle: Particle) -> float:
        # gamma >= 1, so |q| B / m is a conservative upper bound on the
        # relativistic gyrofrequency.
        return (
            self.max_gyro_phase_rad
            * particle.mass_kg
            / (abs(particle.charge_c) * self.field_bound_t)
        )


@dataclass(frozen=True)
class TrajectoryDiagnostics:
    velocity_m_s: FloatArray
    magnetic_field_t: FloatArray
    magnetic_field_magnitude_t: FloatArray
    gamma: FloatArray
    kinetic_energy_ev: FloatArray
    pitch_angle_deg: FloatArray
    magnetic_moment_j_per_t: FloatArray
    local_gyrofrequency_rad_s: FloatArray


@dataclass(frozen=True)
class TrajectorySolution:
    time_s: FloatArray
    position_m: FloatArray
    normalized_momentum: FloatArray
    diagnostics: TrajectoryDiagnostics
    solver_statistics: dict[str, Any]

    def summary(self) -> dict[str, float]:
        energy = self.diagnostics.kinetic_energy_ev
        magnetic_moment = self.diagnostics.magnetic_moment_j_per_t
        energy_relative_span = (
            float(np.ptp(energy) / abs(energy[0]))
            if energy[0] != 0.0
            else 0.0
        )
        magnetic_moment_relative_span = (
            float(np.ptp(magnetic_moment) / abs(magnetic_moment[0]))
            if magnetic_moment[0] != 0.0
            else 0.0
        )
        radius = np.hypot(self.position_m[:, 0], self.position_m[:, 1])
        return {
            "flight_time_s": float(self.time_s[-1] - self.time_s[0]),
            "kinetic_energy_initial_ev": float(energy[0]),
            "kinetic_energy_relative_span": energy_relative_span,
            "magnetic_moment_relative_span": magnetic_moment_relative_span,
            "pitch_angle_initial_deg": float(
                self.diagnostics.pitch_angle_deg[0]
            ),
            "pitch_angle_final_deg": float(
                self.diagnostics.pitch_angle_deg[-1]
            ),
            "radius_max_m": float(radius.max()),
            "field_min_t": float(
                self.diagnostics.magnetic_field_magnitude_t.min()
            ),
            "field_max_t": float(
                self.diagnostics.magnetic_field_magnitude_t.max()
            ),
        }


@dataclass(frozen=True)
class AdiabaticTransportSolution:
    """Guiding-centre transport along a prescribed magnetic field line.

    The first adiabatic invariant ``p_perp**2 / B`` and total relativistic
    momentum are conserved.  This is deliberately a separate data type from
    :class:`TrajectorySolution`: it must not be mistaken for a sampled
    full-orbit Lorentz solution.
    """

    path_position_m: FloatArray
    path_length_m: FloatArray
    time_s: FloatArray
    magnetic_field_t: FloatArray
    magnetic_field_magnitude_t: FloatArray
    pitch_angle_deg: FloatArray
    parallel_speed_m_s: FloatArray
    perpendicular_speed_m_s: FloatArray
    larmor_radius_m: FloatArray
    gyro_phase_rad: FloatArray
    cumulative_turns: FloatArray
    magnetic_moment_j_per_t: FloatArray
    adiabaticity_per_radian: FloatArray


@dataclass(frozen=True)
class AdiabaticElectrostaticTransportSolution:
    """Relativistic adiabatic transport in prescribed static ``Phi`` and ``B``.

    Total energy ``gamma m c**2 + q Phi`` and ``p_perp**2 / B`` are imposed
    pointwise.  The trajectory either reaches the end of the supplied field
    line or turns where ``p_parallel = 0`` and returns along that line.
    """

    path_position_m: FloatArray
    path_length_m: FloatArray
    time_s: FloatArray
    magnetic_field_t: FloatArray
    magnetic_field_magnitude_t: FloatArray
    electric_potential_v: FloatArray
    kinetic_energy_ev: FloatArray
    total_energy_ev: FloatArray
    pitch_angle_deg: FloatArray
    parallel_speed_m_s: FloatArray
    perpendicular_speed_m_s: FloatArray
    parallel_normalized_momentum: FloatArray
    perpendicular_normalized_momentum: FloatArray
    larmor_radius_m: FloatArray
    gyro_phase_rad: FloatArray
    cumulative_turns: FloatArray
    magnetic_moment_j_per_t: FloatArray
    outcome: str
    turning_position_m: FloatArray | None


def kinetic_energy_ev_to_normalized_momentum(
    kinetic_energy_ev: float,
    particle: Particle = ELECTRON,
) -> float:
    """Return ``|p|/(m c)`` for a kinetic energy in electron-volts."""

    if kinetic_energy_ev < 0.0:
        raise ValueError("kinetic_energy_ev must be non-negative")
    gamma = 1.0 + (
        kinetic_energy_ev * elementary_charge / (particle.mass_kg * c**2)
    )
    return float(np.sqrt((gamma - 1.0) * (gamma + 1.0)))


def relativistic_speed_from_kinetic_energy_ev(
    kinetic_energy_ev: float,
    particle: Particle = ELECTRON,
) -> float:
    """Return the speed corresponding to a relativistic kinetic energy."""

    normalized_momentum = kinetic_energy_ev_to_normalized_momentum(
        kinetic_energy_ev,
        particle,
    )
    gamma = np.sqrt(1.0 + normalized_momentum**2)
    return float(c * normalized_momentum / gamma)


def normalized_momentum_from_angles(
    kinetic_energy_ev: float,
    pitch_to_positive_z_deg: float,
    gyro_phase_deg: float = 0.0,
    particle: Particle = ELECTRON,
) -> FloatArray:
    """Construct ``p/(m c)`` from energy and direction relative to +z."""

    if not 0.0 <= pitch_to_positive_z_deg < 90.0:
        raise ValueError("pitch_to_positive_z_deg must be in [0, 90)")
    magnitude = kinetic_energy_ev_to_normalized_momentum(
        kinetic_energy_ev, particle
    )
    pitch = np.deg2rad(pitch_to_positive_z_deg)
    phase = np.deg2rad(gyro_phase_deg)
    return magnitude * np.array(
        [
            np.sin(pitch) * np.cos(phase),
            np.sin(pitch) * np.sin(phase),
            np.cos(pitch),
        ]
    )


def adiabatic_pitch_angle_deg(
    initial_pitch_deg: ArrayLike,
    initial_field_t: float,
    final_field_t: float,
) -> FloatArray:
    """Map pitch angles using conservation of ``p_perp^2 / B``.

    Total momentum is assumed constant. A ``ValueError`` is raised if the
    requested final field lies beyond the corresponding magnetic-mirror point.
    This relation is a guiding-centre diagnostic, not the full-orbit solver.
    """

    if initial_field_t <= 0.0 or final_field_t < 0.0:
        raise ValueError("Magnetic-field magnitudes must be positive")
    initial_pitch = np.asarray(initial_pitch_deg, dtype=float)
    if np.any((initial_pitch < 0.0) | (initial_pitch > 90.0)):
        raise ValueError("Pitch angles must lie in [0, 90] degrees")
    final_sine_squared = (
        np.sin(np.deg2rad(initial_pitch)) ** 2
        * final_field_t
        / initial_field_t
    )
    if np.any(final_sine_squared > 1.0 + 64.0 * np.finfo(float).eps):
        raise ValueError("The final field is beyond the magnetic-mirror point")
    return np.rad2deg(
        np.arcsin(np.sqrt(np.clip(final_sine_squared, 0.0, 1.0)))
    )


def decompose_velocity(
    velocity_m_s: ArrayLike,
    magnetic_field_t: ArrayLike,
) -> tuple[FloatArray, FloatArray]:
    """Return velocity components parallel and perpendicular to local ``B``."""

    velocity = np.asarray(velocity_m_s, dtype=float)
    magnetic_field = np.asarray(magnetic_field_t, dtype=float)
    if velocity.shape != magnetic_field.shape or velocity.shape[-1:] != (3,):
        raise ValueError(
            "velocity_m_s and magnetic_field_t must have equal (..., 3) shapes"
        )
    field_magnitude = np.linalg.norm(
        magnetic_field,
        axis=-1,
        keepdims=True,
    )
    if np.any(field_magnitude == 0.0):
        raise ValueError("Cannot decompose velocity where magnetic field is zero")
    field_unit = magnetic_field / field_magnitude
    parallel = (
        np.sum(velocity * field_unit, axis=-1, keepdims=True) * field_unit
    )
    perpendicular = velocity - parallel
    return parallel, perpendicular


def adiabatic_transport_along_path(
    path_position_m: ArrayLike,
    magnetic_field_t: ArrayLike,
    *,
    kinetic_energy_ev: float,
    initial_pitch_deg: float,
    particle: Particle = ELECTRON,
) -> AdiabaticTransportSolution:
    """Transport an electron along a sampled field line adiabatically.

    The path and field are explicit numerical inputs.  The calculation keeps
    the relativistic momentum magnitude fixed and conserves
    ``p_perp**2 / |B|``.  A magnetic-mirror crossing raises ``ValueError``.
    """

    position = np.asarray(path_position_m, dtype=float)
    magnetic_field = np.asarray(magnetic_field_t, dtype=float)
    if (
        position.ndim != 2
        or position.shape[1] != 3
        or magnetic_field.shape != position.shape
    ):
        raise ValueError(
            "path_position_m and magnetic_field_t must have equal (N, 3) shapes"
        )
    if position.shape[0] < 2:
        raise ValueError("The sampled path must contain at least two points")
    if not 0.0 <= initial_pitch_deg < 90.0:
        raise ValueError("initial_pitch_deg must be in [0, 90)")

    segment_length = np.linalg.norm(np.diff(position, axis=0), axis=1)
    if np.any(segment_length <= 0.0):
        raise ValueError("Successive path samples must be distinct")
    path_length = np.concatenate(
        (np.array([0.0]), np.cumsum(segment_length))
    )

    field_magnitude = np.linalg.norm(magnetic_field, axis=1)
    if np.any(field_magnitude <= 0.0):
        raise ValueError("Magnetic-field magnitude must be positive")
    initial_field = float(field_magnitude[0])
    sine_squared = (
        np.sin(np.deg2rad(initial_pitch_deg)) ** 2
        * field_magnitude
        / initial_field
    )
    if np.any(sine_squared > 1.0 + 64.0 * np.finfo(float).eps):
        mirror_index = int(np.argmax(sine_squared > 1.0))
        raise ValueError(
            "The sampled path crosses the magnetic-mirror condition at "
            f"index {mirror_index}"
        )
    sine_pitch = np.sqrt(np.clip(sine_squared, 0.0, 1.0))
    cosine_pitch = np.sqrt(np.maximum(0.0, 1.0 - sine_pitch**2))
    pitch_angle = np.rad2deg(np.arcsin(sine_pitch))

    normalized_momentum = kinetic_energy_ev_to_normalized_momentum(
        kinetic_energy_ev,
        particle,
    )
    gamma = float(np.sqrt(1.0 + normalized_momentum**2))
    speed = float(c * normalized_momentum / gamma)
    parallel_speed = speed * cosine_pitch
    perpendicular_speed = speed * sine_pitch
    if np.any(parallel_speed <= 0.0):
        raise ValueError("Longitudinal speed vanishes at a mirror point")

    inverse_parallel_speed = 1.0 / parallel_speed
    time = cumulative_trapezoid(
        inverse_parallel_speed,
        path_length,
        initial=0.0,
    )
    gyrofrequency = (
        abs(particle.charge_c) * field_magnitude / (gamma * particle.mass_kg)
    )
    gyro_phase = cumulative_trapezoid(
        gyrofrequency,
        time,
        initial=0.0,
    )
    cumulative_turns = gyro_phase / (2.0 * np.pi)

    physical_momentum = particle.mass_kg * c * normalized_momentum
    perpendicular_momentum = physical_momentum * sine_pitch
    larmor_radius = perpendicular_momentum / (
        abs(particle.charge_c) * field_magnitude
    )
    magnetic_moment = perpendicular_momentum**2 / (
        2.0 * particle.mass_kg * field_magnitude
    )

    log_field = np.log(field_magnitude)
    log_field_gradient = np.gradient(log_field, path_length, edge_order=2)
    adiabaticity_per_radian = (
        parallel_speed * np.abs(log_field_gradient) / gyrofrequency
    )

    return AdiabaticTransportSolution(
        path_position_m=position,
        path_length_m=path_length,
        time_s=time,
        magnetic_field_t=magnetic_field,
        magnetic_field_magnitude_t=field_magnitude,
        pitch_angle_deg=pitch_angle,
        parallel_speed_m_s=parallel_speed,
        perpendicular_speed_m_s=perpendicular_speed,
        larmor_radius_m=larmor_radius,
        gyro_phase_rad=gyro_phase,
        cumulative_turns=cumulative_turns,
        magnetic_moment_j_per_t=magnetic_moment,
        adiabaticity_per_radian=adiabaticity_per_radian,
    )


def adiabatic_electrostatic_transport_along_path(
    path_position_m: ArrayLike,
    magnetic_field_t: ArrayLike,
    electric_potential_v: ArrayLike,
    *,
    kinetic_energy_ev: float,
    initial_pitch_deg: float,
    particle: Particle = ELECTRON,
) -> AdiabaticElectrostaticTransportSolution:
    r"""Transport a particle adiabatically through prescribed static fields.

    The exact relativistic algebra within the guiding-centre model is

    .. math::

       \gamma(s)mc^2 + q\Phi(s) = \mathcal E,\qquad
       \frac{p_\perp^2(s)}{B(s)}
       =\frac{p_{\perp 0}^2}{B_0}.

    A reflection point is located by a root solve of
    ``p_parallel**2 = p**2 - p_perp**2``.  The integrable time singularity at
    the turning point is evaluated after a square-root change of variable.
    """

    position = np.asarray(path_position_m, dtype=float)
    magnetic_field = np.asarray(magnetic_field_t, dtype=float)
    potential = np.asarray(electric_potential_v, dtype=float)
    if (
        position.ndim != 2
        or position.shape[1] != 3
        or magnetic_field.shape != position.shape
        or potential.shape != (position.shape[0],)
    ):
        raise ValueError(
            "position and B must have equal (N, 3) shapes and Phi shape (N,)"
        )
    if position.shape[0] < 3:
        raise ValueError("The sampled path must contain at least three points")
    if not 0.0 <= initial_pitch_deg < 90.0:
        raise ValueError("initial_pitch_deg must be in [0, 90)")

    segment_length = np.linalg.norm(np.diff(position, axis=0), axis=1)
    if np.any(segment_length <= 0.0):
        raise ValueError("Successive path samples must be distinct")
    source_path_length = np.concatenate(
        (np.array([0.0]), np.cumsum(segment_length))
    )
    field_magnitude = np.linalg.norm(magnetic_field, axis=1)
    if np.any(field_magnitude <= 0.0):
        raise ValueError("Magnetic-field magnitude must be positive")

    initial_u = kinetic_energy_ev_to_normalized_momentum(
        kinetic_energy_ev,
        particle,
    )
    initial_gamma = float(np.sqrt(1.0 + initial_u**2))
    rest_energy_j = particle.mass_kg * c**2
    conserved_total_energy_j = (
        initial_gamma * rest_energy_j
        + particle.charge_c * potential[0]
    )
    gamma = (
        conserved_total_energy_j - particle.charge_c * potential
    ) / rest_energy_j
    total_u_squared = gamma**2 - 1.0
    initial_perpendicular_u_squared = (
        initial_u * np.sin(np.deg2rad(initial_pitch_deg))
    ) ** 2
    perpendicular_u_squared = (
        initial_perpendicular_u_squared
        * field_magnitude
        / field_magnitude[0]
    )
    parallel_u_squared = total_u_squared - perpendicular_u_squared
    if parallel_u_squared[0] <= 0.0:
        raise ValueError("Initial longitudinal momentum must be positive")

    crossing_candidates = np.flatnonzero(parallel_u_squared <= 0.0)
    turning_position: FloatArray | None = None
    outcome = "transmitted"
    if crossing_candidates.size:
        crossing_index = int(crossing_candidates[0])
        if crossing_index == 0:
            raise ValueError("Reflection occurs at the initial point")
        lower_index = crossing_index - 1

        def local_parallel_u_squared(fraction: float) -> float:
            local_field = (
                (1.0 - fraction) * field_magnitude[lower_index]
                + fraction * field_magnitude[crossing_index]
            )
            local_potential = (
                (1.0 - fraction) * potential[lower_index]
                + fraction * potential[crossing_index]
            )
            local_gamma = (
                conserved_total_energy_j
                - particle.charge_c * local_potential
            ) / rest_energy_j
            local_perpendicular = (
                initial_perpendicular_u_squared
                * local_field
                / field_magnitude[0]
            )
            return float(local_gamma**2 - 1.0 - local_perpendicular)

        turning_fraction = brentq(
            local_parallel_u_squared,
            0.0,
            1.0,
            xtol=4.0 * np.finfo(float).eps,
            rtol=4.0 * np.finfo(float).eps,
        )
        turning_position = (
            (1.0 - turning_fraction) * position[lower_index]
            + turning_fraction * position[crossing_index]
        )
        turning_field = (
            (1.0 - turning_fraction) * magnetic_field[lower_index]
            + turning_fraction * magnetic_field[crossing_index]
        )
        turning_potential = float(
            (1.0 - turning_fraction) * potential[lower_index]
            + turning_fraction * potential[crossing_index]
        )
        turning_segment_length = (
            turning_fraction * segment_length[lower_index]
        )
        outbound_position = np.vstack(
            (position[:crossing_index], turning_position)
        )
        outbound_field = np.vstack(
            (magnetic_field[:crossing_index], turning_field)
        )
        outbound_potential = np.concatenate(
            (potential[:crossing_index], np.array([turning_potential]))
        )
        outbound_path_length = np.concatenate(
            (
                source_path_length[:crossing_index],
                np.array(
                    [
                        source_path_length[lower_index]
                        + turning_segment_length
                    ]
                ),
            )
        )
        outcome = "reflected"
    else:
        outbound_position = position
        outbound_field = magnetic_field
        outbound_potential = potential
        outbound_path_length = source_path_length

    outbound_field_magnitude = np.linalg.norm(outbound_field, axis=1)
    outbound_gamma = (
        conserved_total_energy_j
        - particle.charge_c * outbound_potential
    ) / rest_energy_j
    outbound_total_u_squared = np.maximum(outbound_gamma**2 - 1.0, 0.0)
    outbound_perpendicular_u_squared = (
        initial_perpendicular_u_squared
        * outbound_field_magnitude
        / field_magnitude[0]
    )
    outbound_parallel_u_squared = np.maximum(
        outbound_total_u_squared - outbound_perpendicular_u_squared,
        0.0,
    )

    finite_parallel = np.sqrt(
        np.maximum(
            outbound_parallel_u_squared[:-1]
            if outcome == "reflected"
            else outbound_parallel_u_squared,
            np.finfo(float).tiny,
        )
    )
    finite_gamma = (
        outbound_gamma[:-1] if outcome == "reflected" else outbound_gamma
    )
    finite_path = (
        outbound_path_length[:-1]
        if outcome == "reflected"
        else outbound_path_length
    )
    inverse_parallel_speed = finite_gamma / (c * finite_parallel)
    outbound_time_finite = cumulative_trapezoid(
        inverse_parallel_speed,
        finite_path,
        initial=0.0,
    )

    if outcome == "reflected":
        lower_s = float(outbound_path_length[-2])
        turning_s = float(outbound_path_length[-1])
        interval = turning_s - lower_s
        lower_field = float(outbound_field_magnitude[-2])
        turning_field_magnitude = float(outbound_field_magnitude[-1])
        lower_potential = float(outbound_potential[-2])
        turning_potential = float(outbound_potential[-1])

        lower_gamma = (
            conserved_total_energy_j
            - particle.charge_c * lower_potential
        ) / rest_energy_j
        turning_gamma = (
            conserved_total_energy_j
            - particle.charge_c * turning_potential
        ) / rest_energy_j
        gamma_slope = turning_gamma - lower_gamma
        lower_perpendicular = (
            initial_perpendicular_u_squared
            * lower_field
            / field_magnitude[0]
        )
        parallel_polynomial_constant = (
            lower_gamma**2 - 1.0 - lower_perpendicular
        )
        parallel_polynomial_quadratic = gamma_slope**2
        # With linear B and Phi on this final path segment,
        # p_parallel**2 is quadratic in x=(s-s0)/(s_turn-s0) and has
        # its root at x=1:
        # P(x)=(1-x)*(c0-c2*x).  x=1-y**2 removes the square-root
        # singularity exactly.
        nodes, weights = np.polynomial.legendre.leggauss(64)
        root_coordinate = 0.5 * (nodes + 1.0)
        root_weights = 0.5 * weights
        fraction = 1.0 - root_coordinate**2
        local_gamma = lower_gamma + gamma_slope * fraction
        regular_denominator = np.sqrt(
            np.maximum(
                parallel_polynomial_constant
                - parallel_polynomial_quadratic * fraction,
                np.finfo(float).tiny,
            )
        )
        final_interval_time = float(
            interval
            / c
            * np.sum(
                root_weights
                * 2.0
                * local_gamma
                / regular_denominator
            )
        )
        outbound_time = np.concatenate(
            (
                outbound_time_finite,
                np.array([outbound_time_finite[-1] + final_interval_time]),
            )
        )
        path_position = np.vstack(
            (outbound_position, outbound_position[-2::-1])
        )
        path_field = np.vstack(
            (outbound_field, outbound_field[-2::-1])
        )
        path_potential = np.concatenate(
            (outbound_potential, outbound_potential[-2::-1])
        )
        path_length = np.concatenate(
            (
                outbound_path_length,
                outbound_path_length[-1]
                + (
                    outbound_path_length[-1]
                    - outbound_path_length[-2::-1]
                ),
            )
        )
        time = np.concatenate(
            (
                outbound_time,
                outbound_time[-1]
                + (outbound_time[-1] - outbound_time[-2::-1]),
            )
        )
        direction_sign = np.concatenate(
            (
                np.ones(outbound_position.shape[0]),
                -np.ones(outbound_position.shape[0] - 1),
            )
        )
        direction_sign[outbound_position.shape[0] - 1] = 0.0
    else:
        path_position = outbound_position
        path_field = outbound_field
        path_potential = outbound_potential
        path_length = outbound_path_length
        time = outbound_time_finite
        direction_sign = np.ones(path_position.shape[0])

    path_field_magnitude = np.linalg.norm(path_field, axis=1)
    path_gamma = (
        conserved_total_energy_j - particle.charge_c * path_potential
    ) / rest_energy_j
    path_total_u_squared = np.maximum(path_gamma**2 - 1.0, 0.0)
    path_perpendicular_u_squared = (
        initial_perpendicular_u_squared
        * path_field_magnitude
        / field_magnitude[0]
    )
    path_parallel_u_squared = np.maximum(
        path_total_u_squared - path_perpendicular_u_squared,
        0.0,
    )
    perpendicular_u = np.sqrt(path_perpendicular_u_squared)
    parallel_u = direction_sign * np.sqrt(path_parallel_u_squared)
    parallel_speed = c * parallel_u / path_gamma
    perpendicular_speed = c * perpendicular_u / path_gamma
    pitch_angle = np.rad2deg(
        np.arctan2(perpendicular_u, np.abs(parallel_u))
    )
    kinetic_energy = (
        (path_gamma - 1.0) * rest_energy_j / elementary_charge
    )
    total_energy = (
        path_gamma * rest_energy_j
        + particle.charge_c * path_potential
    ) / elementary_charge
    physical_perpendicular_momentum = particle.mass_kg * c * perpendicular_u
    larmor_radius = physical_perpendicular_momentum / (
        abs(particle.charge_c) * path_field_magnitude
    )
    magnetic_moment = physical_perpendicular_momentum**2 / (
        2.0 * particle.mass_kg * path_field_magnitude
    )
    gyrofrequency = (
        abs(particle.charge_c)
        * path_field_magnitude
        / (path_gamma * particle.mass_kg)
    )
    gyro_phase = cumulative_trapezoid(
        gyrofrequency,
        time,
        initial=0.0,
    )

    return AdiabaticElectrostaticTransportSolution(
        path_position_m=path_position,
        path_length_m=path_length,
        time_s=time,
        magnetic_field_t=path_field,
        magnetic_field_magnitude_t=path_field_magnitude,
        electric_potential_v=path_potential,
        kinetic_energy_ev=kinetic_energy,
        total_energy_ev=total_energy,
        pitch_angle_deg=pitch_angle,
        parallel_speed_m_s=parallel_speed,
        perpendicular_speed_m_s=perpendicular_speed,
        parallel_normalized_momentum=parallel_u,
        perpendicular_normalized_momentum=perpendicular_u,
        larmor_radius_m=larmor_radius,
        gyro_phase_rad=gyro_phase,
        cumulative_turns=gyro_phase / (2.0 * np.pi),
        magnetic_moment_j_per_t=magnetic_moment,
        outcome=outcome,
        turning_position_m=turning_position,
    )


def _field_cartesian(
    field: AxisymmetricMagneticField,
    position_m: FloatArray,
) -> FloatArray:
    x, y, z = position_m
    rho = float(np.hypot(x, y))
    b_rho, b_z = field.field_cylindrical(rho, z)
    if rho == 0.0:
        return np.array([0.0, 0.0, float(b_z)])
    return np.array(
        [
            float(b_rho) * x / rho,
            float(b_rho) * y / rho,
            float(b_z),
        ]
    )


def _electric_field_cartesian(
    electric_potential: Any,
    position_m: FloatArray,
) -> FloatArray:
    """Evaluate an axisymmetric electrostatic model in Cartesian coordinates."""

    x, y, z = position_m
    rho = float(np.hypot(x, y))
    e_rho, e_z = electric_potential.electric_field_cylindrical(rho, z)
    if rho == 0.0:
        return np.array([0.0, 0.0, float(e_z)])
    return np.array(
        [
            float(e_rho) * x / rho,
            float(e_rho) * y / rho,
            float(e_z),
        ]
    )


def trajectory_diagnostics(
    field: AxisymmetricMagneticField,
    particle: Particle,
    position_m: FloatArray,
    normalized_momentum: FloatArray,
) -> TrajectoryDiagnostics:
    """Evaluate energy, pitch, and first adiabatic invariant diagnostics."""

    x, y, z = position_m.T
    rho = np.hypot(x, y)
    b_rho, b_z = field.field_cylindrical(rho, z)
    b_x = np.zeros_like(rho)
    b_y = np.zeros_like(rho)
    off_axis = rho > 0.0
    b_x[off_axis] = b_rho[off_axis] * x[off_axis] / rho[off_axis]
    b_y[off_axis] = b_rho[off_axis] * y[off_axis] / rho[off_axis]
    magnetic_field = np.column_stack((b_x, b_y, b_z))
    field_magnitude = np.linalg.norm(magnetic_field, axis=1)
    if np.any(field_magnitude == 0.0):
        raise ValueError("Pitch and magnetic moment are undefined at B=0")

    u_squared = np.einsum(
        "ij,ij->i", normalized_momentum, normalized_momentum
    )
    gamma = np.sqrt(1.0 + u_squared)
    velocity = c * normalized_momentum / gamma[:, None]
    kinetic_energy_ev = (
        (gamma - 1.0) * particle.mass_kg * c**2 / elementary_charge
    )

    field_unit = magnetic_field / field_magnitude[:, None]
    physical_momentum = particle.mass_kg * c * normalized_momentum
    parallel_momentum = np.einsum(
        "ij,ij->i", physical_momentum, field_unit
    )
    momentum_squared = np.einsum(
        "ij,ij->i", physical_momentum, physical_momentum
    )
    perpendicular_momentum_squared = np.maximum(
        momentum_squared - parallel_momentum**2, 0.0
    )
    pitch_angle_deg = np.rad2deg(
        np.arctan2(
            np.sqrt(perpendicular_momentum_squared),
            np.abs(parallel_momentum),
        )
    )
    magnetic_moment = perpendicular_momentum_squared / (
        2.0 * particle.mass_kg * field_magnitude
    )
    gyrofrequency = (
        abs(particle.charge_c)
        * field_magnitude
        / (gamma * particle.mass_kg)
    )

    return TrajectoryDiagnostics(
        velocity_m_s=velocity,
        magnetic_field_t=magnetic_field,
        magnetic_field_magnitude_t=field_magnitude,
        gamma=gamma,
        kinetic_energy_ev=kinetic_energy_ev,
        pitch_angle_deg=pitch_angle_deg,
        magnetic_moment_j_per_t=magnetic_moment,
        local_gyrofrequency_rad_s=gyrofrequency,
    )


def integrate_relativistic_magnetic_trajectory(
    field: AxisymmetricMagneticField,
    initial_position_m: ArrayLike,
    initial_normalized_momentum: ArrayLike,
    *,
    stop_z_m: float,
    maximum_time_s: float,
    particle: Particle = ELECTRON,
    settings: IntegrationSettings = IntegrationSettings(),
) -> TrajectorySolution:
    """Integrate the full relativistic Lorentz equation in a static B field.

    The evolved momentum variable is ``u = p/(m c)``:

    ``dx/dt = c u/gamma`` and
    ``du/dt = q (u x B)/(m gamma)``.
    """

    initial_position = np.asarray(initial_position_m, dtype=float)
    initial_u = np.asarray(initial_normalized_momentum, dtype=float)
    if initial_position.shape != (3,) or initial_u.shape != (3,):
        raise ValueError("Initial position and momentum must be 3-vectors")
    if maximum_time_s <= 0.0:
        raise ValueError("maximum_time_s must be positive")
    direction = np.sign(stop_z_m - initial_position[2])
    if direction == 0.0:
        raise ValueError("Initial z already equals stop_z_m")
    initial_parallel_velocity_sign = np.sign(initial_u[2])
    if initial_parallel_velocity_sign != direction:
        raise ValueError("Initial longitudinal momentum points away from stop_z_m")

    initial_state = np.concatenate((initial_position, initial_u))

    def right_hand_side(_time_s: float, state: FloatArray) -> FloatArray:
        position = state[:3]
        normalized_momentum = state[3:]
        gamma = np.sqrt(1.0 + np.dot(normalized_momentum, normalized_momentum))
        magnetic_field = _field_cartesian(field, position)
        velocity = c * normalized_momentum / gamma
        du_dt = (
            particle.charge_c
            / (particle.mass_kg * gamma)
            * np.cross(normalized_momentum, magnetic_field)
        )
        return np.concatenate((velocity, du_dt))

    def stop_event(_time_s: float, state: FloatArray) -> float:
        return float(state[2] - stop_z_m)

    stop_event.terminal = True  # type: ignore[attr-defined]
    stop_event.direction = direction  # type: ignore[attr-defined]

    solution = solve_ivp(
        right_hand_side,
        (0.0, maximum_time_s),
        initial_state,
        method="DOP853",
        rtol=settings.relative_tolerance,
        atol=np.array(
            [settings.position_atol_m] * 3
            + [settings.normalized_momentum_atol] * 3
        ),
        max_step=settings.max_step_s(particle),
        dense_output=True,
        events=stop_event,
    )
    if not solution.success:
        raise RuntimeError(f"Trajectory integration failed: {solution.message}")
    if len(solution.t_events[0]) != 1:
        raise RuntimeError(
            f"Electron did not reach z={stop_z_m:g} m within "
            f"{maximum_time_s:g} s"
        )

    stop_time_s = float(solution.t_events[0][0])
    output_time = np.linspace(0.0, stop_time_s, settings.output_samples)
    output_state = solution.sol(output_time).T
    position = output_state[:, :3]
    normalized_momentum = output_state[:, 3:]
    diagnostics = trajectory_diagnostics(
        field, particle, position, normalized_momentum
    )
    observed_max_field = float(diagnostics.magnetic_field_magnitude_t.max())
    if observed_max_field > settings.field_bound_t * (1.0 + 1.0e-12):
        raise RuntimeError(
            f"Observed |B|={observed_max_field:g} T exceeds the declared "
            f"field_bound_t={settings.field_bound_t:g} T used for max_step"
        )

    return TrajectorySolution(
        time_s=output_time,
        position_m=position,
        normalized_momentum=normalized_momentum,
        diagnostics=diagnostics,
        solver_statistics={
            "method": "DOP853",
            "nfev": int(solution.nfev),
            "accepted_time_nodes": int(solution.t.size),
            "relative_tolerance": settings.relative_tolerance,
            "position_atol_m": settings.position_atol_m,
            "normalized_momentum_atol": settings.normalized_momentum_atol,
            "max_gyro_phase_rad": settings.max_gyro_phase_rad,
            "field_bound_t": settings.field_bound_t,
            "max_step_s": settings.max_step_s(particle),
        },
    )


def integrate_relativistic_electromagnetic_trajectory(
    magnetic_field: AxisymmetricMagneticField,
    electric_potential: Any,
    initial_position_m: ArrayLike,
    initial_normalized_momentum: ArrayLike,
    *,
    lower_stop_z_m: float,
    upper_stop_z_m: float,
    maximum_time_s: float,
    particle: Particle = ELECTRON,
    settings: IntegrationSettings = IntegrationSettings(),
) -> TrajectorySolution:
    """Integrate the full relativistic Lorentz equation in static E and B.

    The initial point must lie strictly between two axial stop planes.  The
    integration ends at whichever plane is reached first, allowing the same
    routine to classify transmission through the upper plane or reflection
    back through the lower plane.
    """

    initial_position = np.asarray(initial_position_m, dtype=float)
    initial_u = np.asarray(initial_normalized_momentum, dtype=float)
    if initial_position.shape != (3,) or initial_u.shape != (3,):
        raise ValueError("Initial position and momentum must be 3-vectors")
    if not lower_stop_z_m < initial_position[2] < upper_stop_z_m:
        raise ValueError("Initial z must lie strictly between the stop planes")
    if maximum_time_s <= 0.0:
        raise ValueError("maximum_time_s must be positive")

    initial_state = np.concatenate((initial_position, initial_u))

    def right_hand_side(_time_s: float, state: FloatArray) -> FloatArray:
        position = state[:3]
        normalized_momentum = state[3:]
        gamma = np.sqrt(1.0 + np.dot(normalized_momentum, normalized_momentum))
        local_b = _field_cartesian(magnetic_field, position)
        local_e = _electric_field_cartesian(electric_potential, position)
        velocity = c * normalized_momentum / gamma
        du_dt = (
            particle.charge_c / (particle.mass_kg * c) * local_e
            + particle.charge_c
            / (particle.mass_kg * gamma)
            * np.cross(normalized_momentum, local_b)
        )
        return np.concatenate((velocity, du_dt))

    def lower_event(_time_s: float, state: FloatArray) -> float:
        return float(state[2] - lower_stop_z_m)

    def upper_event(_time_s: float, state: FloatArray) -> float:
        return float(state[2] - upper_stop_z_m)

    lower_event.terminal = True  # type: ignore[attr-defined]
    lower_event.direction = -1.0  # type: ignore[attr-defined]
    upper_event.terminal = True  # type: ignore[attr-defined]
    upper_event.direction = +1.0  # type: ignore[attr-defined]

    solution = solve_ivp(
        right_hand_side,
        (0.0, maximum_time_s),
        initial_state,
        method="DOP853",
        rtol=settings.relative_tolerance,
        atol=np.array(
            [settings.position_atol_m] * 3
            + [settings.normalized_momentum_atol] * 3
        ),
        max_step=settings.max_step_s(particle),
        dense_output=True,
        events=(lower_event, upper_event),
    )
    if not solution.success:
        raise RuntimeError(f"Trajectory integration failed: {solution.message}")
    event_counts = [len(events) for events in solution.t_events]
    if sum(event_counts) != 1:
        raise RuntimeError(
            "Particle reached neither exactly one lower nor one upper stop "
            f"plane within {maximum_time_s:g} s"
        )
    exit_kind = "reflected" if event_counts[0] else "transmitted"
    exit_time_s = float(
        solution.t_events[0][0]
        if event_counts[0]
        else solution.t_events[1][0]
    )
    output_time = np.linspace(0.0, exit_time_s, settings.output_samples)
    output_state = solution.sol(output_time).T
    position = output_state[:, :3]
    normalized_momentum = output_state[:, 3:]
    diagnostics = trajectory_diagnostics(
        magnetic_field,
        particle,
        position,
        normalized_momentum,
    )
    observed_max_field = float(diagnostics.magnetic_field_magnitude_t.max())
    if observed_max_field > settings.field_bound_t * (1.0 + 1.0e-12):
        raise RuntimeError(
            f"Observed |B|={observed_max_field:g} T exceeds the declared "
            f"field_bound_t={settings.field_bound_t:g} T used for max_step"
        )

    return TrajectorySolution(
        time_s=output_time,
        position_m=position,
        normalized_momentum=normalized_momentum,
        diagnostics=diagnostics,
        solver_statistics={
            "method": "DOP853",
            "exit_kind": exit_kind,
            "nfev": int(solution.nfev),
            "accepted_time_nodes": int(solution.t.size),
            "relative_tolerance": settings.relative_tolerance,
            "position_atol_m": settings.position_atol_m,
            "normalized_momentum_atol": settings.normalized_momentum_atol,
            "max_gyro_phase_rad": settings.max_gyro_phase_rad,
            "field_bound_t": settings.field_bound_t,
            "max_step_s": settings.max_step_s(particle),
        },
    )


def uniform_field_exact_solution(
    initial_position_m: ArrayLike,
    initial_normalized_momentum: ArrayLike,
    time_s: ArrayLike,
    field_z_t: float,
    particle: Particle = ELECTRON,
) -> tuple[FloatArray, FloatArray]:
    """Exact relativistic helix in a uniform field parallel to z."""

    initial_position = np.asarray(initial_position_m, dtype=float)
    initial_u = np.asarray(initial_normalized_momentum, dtype=float)
    time = np.asarray(time_s, dtype=float)
    gamma = float(np.sqrt(1.0 + np.dot(initial_u, initial_u)))
    omega = particle.charge_c * field_z_t / (particle.mass_kg * gamma)
    if omega == 0.0:
        position = initial_position + (
            c * initial_u / gamma
        ) * time[..., None]
        momentum = np.broadcast_to(initial_u, position.shape).copy()
        return position, momentum

    phase = omega * time
    cosine = np.cos(phase)
    sine = np.sin(phase)
    ux0, uy0, uz0 = initial_u
    ux = ux0 * cosine + uy0 * sine
    uy = uy0 * cosine - ux0 * sine
    uz = np.full_like(time, uz0)
    momentum = np.stack((ux, uy, uz), axis=-1)

    velocity_factor = c / gamma
    x = initial_position[0] + velocity_factor / omega * (
        ux0 * sine + uy0 * (1.0 - cosine)
    )
    y = initial_position[1] + velocity_factor / omega * (
        uy0 * sine - ux0 * (1.0 - cosine)
    )
    z = initial_position[2] + velocity_factor * uz0 * time
    position = np.stack((x, y, z), axis=-1)
    return position, momentum
