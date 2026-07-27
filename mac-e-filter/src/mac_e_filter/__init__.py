"""Reusable numerical models for MAC-E filter animations."""

from .fields import (
    CircularLoop,
    CompositeMagneticField,
    RectangularCoilPack,
    UniformAxialField,
)
from .dynamics import (
    ELECTRON,
    AdiabaticElectrostaticTransportSolution,
    IntegrationSettings,
    Particle,
    TrajectorySolution,
    adiabatic_electrostatic_transport_along_path,
    adiabatic_pitch_angle_deg,
    decompose_velocity,
    integrate_relativistic_magnetic_trajectory,
    normalized_momentum_from_angles,
)
from .electrostatics import FiniteCylindricalElectrodePotential
from .katrin_nominal import (
    KATRIN_2013_REFERENCE,
    KatrinNominalField,
    build_katrin_2013_field,
)

__all__ = [
    "CircularLoop",
    "CompositeMagneticField",
    "ELECTRON",
    "AdiabaticElectrostaticTransportSolution",
    "FiniteCylindricalElectrodePotential",
    "IntegrationSettings",
    "KATRIN_2013_REFERENCE",
    "KatrinNominalField",
    "Particle",
    "RectangularCoilPack",
    "UniformAxialField",
    "adiabatic_pitch_angle_deg",
    "adiabatic_electrostatic_transport_along_path",
    "build_katrin_2013_field",
    "decompose_velocity",
    "integrate_relativistic_magnetic_trajectory",
    "normalized_momentum_from_angles",
    "TrajectorySolution",
]
