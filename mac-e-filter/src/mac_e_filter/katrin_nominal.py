"""Published 2013 nominal magnetic-field configurations for KATRIN.

The LFCS geometry and currents are reproduced directly from Tables 1 and 2 of
F. Glück et al., *Electromagnetic design of the KATRIN large-volume air coil
system*, New J. Phys. 15 (2013) 083025, arXiv:1304.6569.

The same paper gives only three scalar quantities for each remote
superconducting source: its axial location, a typical maximum field near that
location, and its field contribution at the main-spectrometer centre.  A
unique off-axis field cannot be inferred from those data.  Here each such
source is therefore represented by one explicitly labelled equivalent loop
whose radius and current are fixed by the two published field constraints.
This is a traceable central-spectrometer surrogate, not an as-built field map.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy.constants import mu_0

from .fields import (
    CircularLoop,
    CompositeMagneticField,
    RectangularCoilPack,
    UniformAxialField,
)

KATRIN_2013_REFERENCE = {
    "title": "Electromagnetic design of the KATRIN large-volume air coil system",
    "authors": "F. Glück et al.",
    "journal": "New Journal of Physics 15 (2013) 083025",
    "arxiv": "https://arxiv.org/abs/1304.6569",
    "doi": "https://doi.org/10.1088/1367-2630/15/8/083025",
    "tables": ("Table 1", "Table 2"),
}

ConfigurationName = Literal["one_minimum", "two_minima"]


@dataclass(frozen=True)
class PublishedSuperconductingSource:
    """Scalar source data quoted in Table 1 of the 2013 paper."""

    name: str
    center_z_m: float
    typical_max_field_t: float
    field_at_spectrometer_center_t: float

    def equivalent_loop(self) -> CircularLoop:
        """Construct the unique loop satisfying both published field values.

        For a loop of radius ``a`` centred at ``z_c``,

        ``B(0) / B(z_c) = a^3 / (a^2 + z_c^2)^(3/2)``.

        Solving this equation fixes ``a``; the field at the loop centre then
        fixes ``N I``.  No fitted or hidden free parameter remains.
        """

        ratio = abs(
            self.field_at_spectrometer_center_t / self.typical_max_field_t
        )
        if not 0.0 < ratio < 1.0:
            raise ValueError("Expected 0 < |B(0)/B_c| < 1")
        ratio_power = ratio ** (2.0 / 3.0)
        radius = abs(self.center_z_m) * np.sqrt(
            ratio_power / (1.0 - ratio_power)
        )
        signed_central_field = np.copysign(
            self.typical_max_field_t,
            self.field_at_spectrometer_center_t,
        )
        ampere_turns = 2.0 * radius * signed_central_field / mu_0
        return CircularLoop(
            name=f"{self.name} (published-scalar equivalent loop)",
            radius_m=float(radius),
            center_z_m=self.center_z_m,
            ampere_turns=float(ampere_turns),
        )


SUPERCONDUCTING_SOURCES_2013 = (
    PublishedSuperconductingSource("WGTS coil system", -38.87, 3.6, -9.7e-6),
    PublishedSuperconductingSource("DPS coil system", -27.25, 5.0, -16.3e-6),
    PublishedSuperconductingSource("CPS coil system", -20.58, 5.6, -38.2e-6),
    PublishedSuperconductingSource("PS1 coil", -16.46, 4.5, -18.5e-6),
    PublishedSuperconductingSource("PS2 coil", -12.10, 4.5, -46.5e-6),
    PublishedSuperconductingSource("PCH coil", 12.18, 6.0, -65.2e-6),
    PublishedSuperconductingSource("DET coil", 13.78, 3.6, -48.4e-6),
)


@dataclass(frozen=True)
class PublishedLfcsRow:
    """One row of Table 2; coil 14 is expanded to two packs."""

    index: int
    centers_z_m: tuple[float, ...]
    turns_each: int
    current_one_minimum_a: float
    current_two_minima_a: float

    def current(self, configuration: ConfigurationName) -> float:
        if configuration == "one_minimum":
            return self.current_one_minimum_a
        if configuration == "two_minima":
            return self.current_two_minima_a
        raise ValueError(f"Unknown LFCS configuration: {configuration}")


LFCS_ROWS_2013 = (
    PublishedLfcsRow(1, (-6.79,), 14, -11.2, -0.5),
    PublishedLfcsRow(2, (-4.94,), 14, -15.3, 0.0),
    PublishedLfcsRow(3, (-4.04,), 8, -7.9, -4.8),
    PublishedLfcsRow(4, (-3.14,), 8, -13.4, -7.1),
    PublishedLfcsRow(5, (-2.24,), 8, -12.2, -6.6),
    PublishedLfcsRow(6, (-1.34,), 8, -24.2, -19.4),
    PublishedLfcsRow(7, (-0.44,), 8, -17.1, -57.2),
    PublishedLfcsRow(8, (0.46,), 8, -20.3, -51.2),
    PublishedLfcsRow(9, (1.35,), 8, -18.5, -22.7),
    PublishedLfcsRow(10, (2.26,), 8, -23.1, -12.5),
    PublishedLfcsRow(11, (3.16,), 8, -21.9, -7.7),
    PublishedLfcsRow(12, (4.06,), 14, -18.1, -16.8),
    PublishedLfcsRow(13, (4.95,), 14, -13.3, -15.9),
    PublishedLfcsRow(14, (6.6, 6.9), 14, 27.3, 42.1),
)

LFCS_INNER_RADIUS_M = 6.3
LFCS_RADIAL_THICKNESS_M = 0.02
LFCS_AXIAL_LENGTH_M = 0.19
EARTH_AXIAL_FIELD_T = 20.0e-6


@dataclass(frozen=True)
class KatrinNominalField:
    """Components and metadata of a reconstructed 2013 configuration."""

    configuration: ConfigurationName
    total: CompositeMagneticField
    lfcs: CompositeMagneticField
    superconducting_surrogate: CompositeMagneticField
    earth: UniformAxialField
    reference: dict[str, object]
    validity_note: str

    def center_breakdown_t(self) -> dict[str, float]:
        z0 = np.asarray(0.0)
        return {
            "lfcs": float(self.lfcs.axis_field(z0)),
            "superconducting_surrogate": float(
                self.superconducting_surrogate.axis_field(z0)
            ),
            "earth": float(self.earth.axis_field(z0)),
            "total": float(self.total.axis_field(z0)),
        }


def _build_lfcs(
    configuration: ConfigurationName,
    radial_order: int,
    axial_order: int,
) -> CompositeMagneticField:
    packs: list[RectangularCoilPack] = []
    for row in LFCS_ROWS_2013:
        for subcoil, center_z_m in enumerate(row.centers_z_m, start=1):
            suffix = f".{subcoil}" if len(row.centers_z_m) > 1 else ""
            packs.append(
                RectangularCoilPack(
                    name=f"LFCS L{row.index}{suffix}",
                    inner_radius_m=LFCS_INNER_RADIUS_M,
                    radial_thickness_m=LFCS_RADIAL_THICKNESS_M,
                    center_z_m=center_z_m,
                    axial_length_m=LFCS_AXIAL_LENGTH_M,
                    turns=row.turns_each,
                    current_a=row.current(configuration),
                    radial_order=radial_order,
                    axial_order=axial_order,
                )
            )
    return CompositeMagneticField(
        name=f"LFCS 2013, {configuration.replace('_', ' ')}",
        components=tuple(packs),
    )


def build_katrin_2013_field(
    configuration: ConfigurationName = "one_minimum",
    *,
    radial_order: int = 6,
    axial_order: int = 10,
) -> KatrinNominalField:
    """Build one of the two published 2013 nominal field configurations."""

    if configuration not in ("one_minimum", "two_minima"):
        raise ValueError(f"Unknown LFCS configuration: {configuration}")

    lfcs = _build_lfcs(configuration, radial_order, axial_order)
    superconducting_surrogate = CompositeMagneticField(
        name="2013 superconducting-source scalar surrogate",
        components=tuple(
            source.equivalent_loop() for source in SUPERCONDUCTING_SOURCES_2013
        ),
    )
    earth = UniformAxialField(
        name="Axial Earth field quoted in Table 1",
        field_z_t=EARTH_AXIAL_FIELD_T,
    )
    total = CompositeMagneticField(
        name=f"KATRIN published 2013 nominal field ({configuration})",
        components=(lfcs, superconducting_surrogate, earth),
    )
    return KatrinNominalField(
        configuration=configuration,
        total=total,
        lfcs=lfcs,
        superconducting_surrogate=superconducting_surrogate,
        earth=earth,
        reference=KATRIN_2013_REFERENCE.copy(),
        validity_note=(
            "LFCS winding packs reproduce the published geometry and currents. "
            "Remote superconducting systems are equivalent loops fixed by the "
            "two scalar constraints in Table 1; they are not an as-built map "
            "and must not be used for precision tracking near those magnets."
        ),
    )

