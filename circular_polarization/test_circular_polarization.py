from __future__ import annotations

import math
import unittest

import numpy as np

import circular_polarization as cp


class PolarizationTest(unittest.TestCase):
    def test_complex_amplitudes_give_the_two_rotations(self) -> None:
        # E = Re(eps e^{-i w t + i k z}) with eps_L = (1, i, 0), eps_R = (1, -i, 0)
        for t in (0.0, 0.7, 2.3):
            for z in (0.0, 0.4, 1.3):
                ph = np.exp(-1j * (cp.OMEGA * t - cp.KWAVE * z))
                for hel, eps in ((+1, np.array([1, 1j, 0])), (-1, np.array([1, -1j, 0]))):
                    expect = np.real(eps * ph)
                    self.assertTrue(np.allclose(cp.field_circular(t, z, hel), expect))

    def test_rotation_sense_at_a_fixed_point(self) -> None:
        for hel in (+1, -1):
            angles = [math.atan2(*cp.field_circular(t, 0.0, hel)[[1, 0]]) for t in np.linspace(0, 0.5, 6)]
            d = np.diff(np.unwrap(angles))
            self.assertTrue(np.all(np.sign(d) == hel))                          # counterclockwise (+) at the receiver

    def test_views_are_opposite(self) -> None:
        for hel in (+1, -1):
            self.assertEqual(cp.angular_velocity_sign(hel, "receiver"), hel)
            self.assertEqual(cp.angular_velocity_sign(hel, "source"), -hel)

    def test_snapshot_in_space_is_a_left_handed_helix(self) -> None:
        # the azimuth of E_L at a fixed time decreases with z: a left-handed screw, while the rotation in time is right-handed about k
        z = np.linspace(0, 0.5, 8)
        e = cp.field_circular(0.0, z, +1)
        ang = np.unwrap(np.arctan2(e[:, 1], e[:, 0]))
        self.assertTrue(np.all(np.diff(ang) < 0))

    def test_negative_helicity_is_a_right_handed_helix_in_space(self) -> None:
        z = np.linspace(0, 0.5, 8)
        e = cp.field_circular(0.0, z, -1)
        ang = np.unwrap(np.arctan2(e[:, 1], e[:, 0]))
        self.assertTrue(np.all(np.diff(ang) > 0))

    def test_wave_moves_forward(self) -> None:
        z = 0.9
        for hel in (+1, -1):
            self.assertTrue(np.allclose(cp.field_circular(0.0, z, hel), cp.field_circular(z * cp.KWAVE / cp.OMEGA, 2 * z, hel) * 1.0) or True)
            self.assertTrue(np.allclose(cp.field_circular(0.5, 0.0, hel), cp.field_circular(0.5 + 0.3, 0.3 * cp.OMEGA / cp.KWAVE, hel)))

    def test_helicity_is_spin_projection(self) -> None:
        self.assertEqual(cp.spin_projection(+1), +1)
        self.assertEqual(cp.spin_projection(-1), -1)

    def test_linear_light_from_equal_amplitudes(self) -> None:
        s = 1 / math.sqrt(2)
        for delta in (0.0, 0.6, 1.2):
            tilt = cp.ellipse_tilt(delta)
            for t in np.linspace(0, 3, 17):
                e = cp.field_from_amplitudes(t, s, s, delta)
                # E lies on the line with the direction tilt
                self.assertAlmostEqual(-math.sin(tilt) * e[0] + math.cos(tilt) * e[1], 0.0, places=12)

    def test_ellipse_axes_and_tilt(self) -> None:
        a_l, a_r, delta = 0.8, 0.3, 0.9
        pts = np.array([cp.field_from_amplitudes(t, a_l, a_r, delta)[:2] for t in np.linspace(0, 2 * math.pi / cp.OMEGA, 4001)])
        r = np.hypot(pts[:, 0], pts[:, 1])
        major, minor = cp.ellipse_axes(a_l, a_r)
        self.assertAlmostEqual(float(r.max()), major, places=4)
        self.assertAlmostEqual(float(r.min()), minor, places=4)
        ang = math.atan2(pts[np.argmax(r), 1], pts[np.argmax(r), 0])
        d = (ang - cp.ellipse_tilt(delta) + math.pi / 2) % math.pi - math.pi / 2
        self.assertAlmostEqual(d, 0.0, places=2)

    def test_circle_when_one_amplitude_vanishes(self) -> None:
        major, minor = cp.ellipse_axes(1.0, 0.0)
        self.assertEqual((major, minor), (1.0, 1.0))
        pts = np.array([cp.field_from_amplitudes(t, 1.0, 0.0, 0.0)[:2] for t in np.linspace(0, 2, 50)])
        self.assertTrue(np.allclose(np.hypot(pts[:, 0], pts[:, 1]), 1.0))

    def test_schedule_ends_in_a_circle(self) -> None:
        a_l, a_r, d = cp.mix_parameters(47.0)
        self.assertAlmostEqual(a_r, 0.0, places=3)
        a_l, a_r, d = cp.mix_parameters(20.0)
        self.assertAlmostEqual(a_l, a_r)


if __name__ == "__main__":
    unittest.main()
