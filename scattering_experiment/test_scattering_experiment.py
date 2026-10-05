from __future__ import annotations

import unittest

import numpy as np

import scattering_experiment as se


class AngularDistributionTest(unittest.TestCase):
    def test_density_is_normalized(self) -> None:
        c = np.linspace(-1, 1, 20001)
        self.assertAlmostEqual(float(np.trapezoid(se.angular_density(c), c)), 1.0, places=6)

    def test_sampling_follows_one_plus_cos_squared(self) -> None:
        c = se.sample_cos_theta(np.random.default_rng(0), 200_000)
        self.assertAlmostEqual(float(np.mean(c * c)), 0.4, delta=0.005)      # <cos^2> for (3/8)(1+c^2)
        self.assertAlmostEqual(float(np.mean(c)), 0.0, delta=0.005)          # no forward-backward asymmetry
        hist, edges = np.histogram(c, bins=10, range=(-1, 1), density=True)
        mid = 0.5 * (edges[1:] + edges[:-1])
        np.testing.assert_allclose(hist, se.angular_density(mid), atol=0.02)


class EventsTest(unittest.TestCase):
    def test_particles_meet_where_the_event_happens(self) -> None:
        crossings = se.make_events(1, 28.0)
        v = 560.0
        for cr in crossings[1:]:
            for ev in cr["events"]:
                t, xm, _ = se.event_time_and_point(ev, cr, v)
                xe = se.xc_of(t, cr, v) + cr["dx_e"][ev["ie"]]
                xp = -se.xc_of(t, cr, v) + cr["dx_p"][ev["ip"]]
                self.assertAlmostEqual(float(xe), float(xp), places=6)
                self.assertAlmostEqual(float(xe), float(xm), places=6)

    def test_first_event_is_unique_and_readable(self) -> None:
        cr = se.make_events(1, 28.0)[0]
        self.assertTrue(cr["first"])
        self.assertEqual(len(cr["events"]), 1)
        self.assertAlmostEqual(cr["events"][0]["c"], 0.62)

    def test_distinct_particles_per_crossing(self) -> None:
        for cr in se.make_events(3, 28.0):
            es = [ev["ie"] for ev in cr["events"]]
            ps = [ev["ip"] for ev in cr["events"]]
            self.assertEqual(len(set(es)), len(es))
            self.assertEqual(len(set(ps)), len(ps))


if __name__ == "__main__":
    unittest.main()
