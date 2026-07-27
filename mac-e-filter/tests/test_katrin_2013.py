from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mac_e_filter.katrin_nominal import (  # noqa: E402
    EARTH_AXIAL_FIELD_T,
    SUPERCONDUCTING_SOURCES_2013,
    build_katrin_2013_field,
)


class Katrin2013FieldTests(unittest.TestCase):
    def test_equivalent_loops_match_both_published_constraints(self) -> None:
        for source in SUPERCONDUCTING_SOURCES_2013:
            with self.subTest(source=source.name):
                loop = source.equivalent_loop()
                self.assertAlmostEqual(
                    float(loop.axis_field(0.0)),
                    source.field_at_spectrometer_center_t,
                    delta=1e-18,
                )
                expected_signed_maximum = np.copysign(
                    source.typical_max_field_t,
                    source.field_at_spectrometer_center_t,
                )
                self.assertAlmostEqual(
                    float(loop.axis_field(source.center_z_m)),
                    expected_signed_maximum,
                    delta=2e-15,
                )

    def test_table_1_center_sum(self) -> None:
        expected_superconducting_t = -242.8e-6
        model = build_katrin_2013_field()
        center = model.center_breakdown_t()

        self.assertAlmostEqual(
            center["superconducting_surrogate"],
            expected_superconducting_t,
            delta=2e-18,
        )
        self.assertAlmostEqual(center["earth"], EARTH_AXIAL_FIELD_T, delta=0.0)

    def test_published_center_magnitude_is_reproduced(self) -> None:
        """The paper quotes 0.35 mT for both rounded current tables."""

        for configuration in ("one_minimum", "two_minima"):
            with self.subTest(configuration=configuration):
                center_t = build_katrin_2013_field(
                    configuration
                ).center_breakdown_t()["total"]
                self.assertAlmostEqual(abs(center_t), 0.35e-3, delta=3.5e-6)

    def test_finite_winding_pack_quadrature_converges(self) -> None:
        z = np.linspace(-7.0, 7.0, 101)
        baseline = build_katrin_2013_field(
            "one_minimum", radial_order=6, axial_order=10
        ).lfcs.axis_field(z)
        refined = build_katrin_2013_field(
            "one_minimum", radial_order=10, axial_order=16
        ).lfcs.axis_field(z)

        error = np.max(np.abs(baseline - refined))
        scale = np.max(np.abs(refined))
        self.assertLess(error / scale, 2e-12)

    def test_radial_field_vanishes_on_symmetry_axis(self) -> None:
        model = build_katrin_2013_field("two_minima")
        z = np.linspace(-7.0, 7.0, 31)
        b_rho, _ = model.total.field_cylindrical(np.zeros_like(z), z)
        np.testing.assert_array_equal(b_rho, 0.0)


if __name__ == "__main__":
    unittest.main()

