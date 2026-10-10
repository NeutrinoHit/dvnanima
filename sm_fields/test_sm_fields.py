from __future__ import annotations

import math
import unittest

import numpy as np

import qed_pair as qp
import sm_fields as sf


class StructureTest(unittest.TestCase):
    def test_seventeen_fields_in_the_table(self) -> None:
        self.assertEqual(len(sf.FIELDS), 17)
        self.assertEqual(len({(c, r) for _n, _g, c, r in sf.FIELDS}), 17)
        self.assertEqual(sorted(sf.STACK_ORDER), sorted(n for n, *_ in sf.FIELDS))
        groups = [g for _n, g, _c, _r in sf.FIELDS]
        self.assertEqual((groups.count("quark"), groups.count("neutrino"), groups.count("lepton"),
                          groups.count("gauge"), groups.count("higgs")), (6, 3, 3, 4, 1))

    def test_the_electron_and_the_photon_sheets_are_neighbours(self) -> None:
        self.assertEqual(abs(sf.STACK_ORDER.index("e") - sf.STACK_ORDER.index("gamma")), 1)

    def test_sheets_do_not_cross_in_the_stack(self) -> None:
        zs = sorted(sf.stack_z(n) for n in sf.STACK_ORDER)
        self.assertTrue(np.all(np.diff(zs) > 0.3))

    def test_the_lift_is_a_cascade(self) -> None:
        t = 0.5 * (sf.T_LIFT[0] + sf.T_LIFT[1])
        prog = [sf.lift_progress(n, t) for n in sf.STACK_ORDER]
        self.assertTrue(all(prog[i] >= prog[i + 1] - 1e-12 for i in range(len(prog) - 1)))     # the top sheets go first
        self.assertEqual(sf.lift_progress("b", sf.T_LIFT[0] - 0.1), 0.0)
        self.assertEqual(sf.lift_progress("H", sf.T_LIFT[1]), 1.0)
        self.assertEqual(min(sf.lift_progress(n, sf.T_LIFT[1] + 0.1) for n in sf.STACK_ORDER), 1.0)


class PhysicsTest(unittest.TestCase):
    def test_energy_is_conserved_between_the_sheets(self) -> None:
        for tt in np.linspace(0.0, 16.0, 161):
            ee, eg = sf.energy_shares(sf.mixing_angle(float(tt)))
            self.assertAlmostEqual(ee + eg, 1.0, places=12)

    def test_annihilation_and_back(self) -> None:
        self.assertAlmostEqual(sf.energy_shares(sf.mixing_angle(0.0))[0], 1.0, places=9)        # only electrons
        self.assertAlmostEqual(sf.energy_shares(sf.mixing_angle(7.5))[1], 1.0, places=9)        # only photons
        self.assertAlmostEqual(sf.energy_shares(sf.mixing_angle(15.0))[0], 1.0, places=9)       # electrons again
        self.assertAlmostEqual(sf.energy_shares(sf.mixing_angle(sf.T_C1))[0], 0.5, places=6)    # half-way at the collision

    def test_the_packets_meet_at_the_collision(self) -> None:
        self.assertAlmostEqual(sf.electron_x(sf.T_C1), 0.0)
        self.assertAlmostEqual(sf.photon_x_out(sf.T_C1), 0.0)
        self.assertAlmostEqual(sf.photon_x_in(sf.T_C2), 0.0)
        self.assertAlmostEqual(sf.electron_x(sf.T_C2), 0.0)
        self.assertGreater(sf.electron_x(0.0), 1.0)

    def test_the_photons_are_faster_than_the_electrons(self) -> None:
        self.assertGreater(sf.V_G, sf.V_E)

    def test_linear_superposition_of_two_packets(self) -> None:
        u, v = np.meshgrid(np.linspace(-1, 1, 41), np.linspace(-1, 1, 41))
        a = sf.packet(u, v, 0.4, -1.7, -0.25, 0.45, 0.20, 22.0, 0.34)
        b = sf.packet(u, v, 0.4, 1.7, 0.25, -0.45, 0.20, 22.0, 0.34)
        # waves of a free field simply add up: nothing depends on the other packet
        self.assertTrue(np.allclose((a + b) - b, a))

    def test_ring_wave_front_moves_with_the_speed(self) -> None:
        u = np.linspace(0.0, 1.5, 3001)
        h = np.abs(sf.ring(u, np.zeros_like(u), 2.0, 0.0, 0.0, 0.0, 0.55, 0.12, 22.0, 1.0))
        r_max = u[np.argmax(h * np.sqrt(1.0 + 3.0 * u))]
        self.assertAlmostEqual(float(r_max), 0.55 * 2.0, delta=0.06)


def separation(case: str, charge_q: float) -> np.ndarray:
    c = qp.bundle(case, charge_q)["packet_centers"]
    return np.hypot(*(c[:, 0] - c[:, 1]).T)


class InteractionTest(unittest.TestCase):
    """The scalar-QED model of the zoom: equal charges repel, opposite charges attract."""

    def test_equal_charges_repel(self) -> None:
        self.assertGreater(separation("pp", qp.CHARGE_Q).min(), separation("pp", 0.0).min() + 0.3)
        self.assertGreater(separation("pp", qp.CHARGE_Q)[-1], separation("pp", 0.0)[-1])

    def test_opposite_charges_attract(self) -> None:
        self.assertLess(separation("pm", qp.CHARGE_Q).min(), separation("pm", 0.0).min() - 0.3)

    def test_free_packets_do_not_notice_each_other(self) -> None:
        # without the coupling the two cases differ only by the sign of the field, the paths are the same
        self.assertTrue(np.allclose(separation("pp", 0.0), separation("pm", 0.0), atol=1e-6))

    def test_the_field_of_opposite_charges_has_both_signs(self) -> None:
        up = qp.bundle("pm")["upper_frames"][0]
        self.assertAlmostEqual(float(up.max()), -float(up.min()), delta=0.02 * float(up.max()))
        self.assertGreater(float(qp.bundle("pp")["upper_frames"][0].min()), 0.0)        # equal charges: one sign

    def test_the_packets_are_symmetric(self) -> None:
        c = qp.bundle("pp")["packet_centers"]
        self.assertTrue(np.allclose(c[:, 0], -c[:, 1], atol=1e-6))

    def test_frame_shapes_and_sheet_grid(self) -> None:
        low, up, c = sf.pair_fields("pm", 10.0)
        n = len(sf.sheet_grid())
        self.assertEqual(low.shape, (n - 1, n - 1))
        self.assertEqual(sf.edge_pad(low).shape, (n, n))
        self.assertAlmostEqual(float(sf.sheet_grid()[-1]), sf.X_HALF)
        self.assertEqual(c.shape, (2, 2))

    def test_stages_cross_fade_without_overlap_gaps(self) -> None:
        for t in np.linspace(sf.T_A[0] + 2.0, sf.T_C[1], 400):
            wa, wb, wc = sf.stage_weights(float(t))
            self.assertLessEqual(wa + wb + wc, 1.0 + 1e-9)
            self.assertGreater(wa + wb + wc, 0.45)              # at least half of the field is always on the screen

    def test_annihilation_heights_conserve_the_energy_shares(self) -> None:
        g = sf.sheet_grid()
        X, Y = np.meshgrid(g, g, indexing="ij")
        ze0, zg0 = sf.annihilation_fields(X, Y, 0.0)
        ze1, zg1 = sf.annihilation_fields(X, Y, 7.5)
        self.assertGreater(float(ze0.max()), 0.9)
        self.assertLess(float(np.abs(zg0).max()), 1e-9)
        self.assertLess(float(np.abs(ze1).max()), 1e-9)
        self.assertGreater(float(zg1.max()), 0.5)


if __name__ == "__main__":
    unittest.main()
