from __future__ import annotations

import unittest

import numpy as np

import pmt_multiplication as pm


class MultiplicationTest(unittest.TestCase):
    def test_mean_growth_is_delta_to_the_j(self) -> None:
        rng = np.random.default_rng(1)
        runs = np.array([pm.multiply(1, rng) for _ in range(4000)], dtype=float)
        mean = runs.mean(axis=0)
        for j in (1, 3, 5):
            self.assertAlmostEqual(mean[j] / pm.DELTA ** j, 1.0, delta=0.08)

    def test_gain_is_delta_to_the_n(self) -> None:
        rng = np.random.default_rng(2)
        gains = [pm.multiply(10, rng)[-1] / 10 for _ in range(300)]
        self.assertAlmostEqual(np.mean(gains) / pm.DELTA ** pm.N_DYN, 1.0, delta=0.05)

    def test_pulse_height_is_proportional_to_photoelectrons(self) -> None:
        rng = np.random.default_rng(3)
        one = np.mean([pm.multiply(1, rng)[-1] for _ in range(3000)])
        twelve = np.mean([pm.multiply(12, rng)[-1] for _ in range(300)])
        self.assertAlmostEqual(twelve / one, 12.0, delta=0.9)

    def test_film_events_do_not_die_out(self) -> None:
        for _, n in pm.EVENTS_1 + pm.EVENTS_2:
            ev = pm.make_event(0.0, n, 1.0)
            self.assertGreater(ev.counts[-1], 1e4)
            self.assertEqual(len(ev.flights), pm.N_DYN + 1)

    def test_geometry_is_inside_the_tube(self) -> None:
        for j in range(1, pm.N_DYN + 1):
            for p in pm.plate_endpoints(j):
                self.assertTrue(0.6 < p[0] < 15.6 and 0.55 < p[1] < 9.35)
        self.assertLess(pm.dynode_center(pm.N_DYN)[0], pm.ANODE_X)

    def test_drawn_flights_connect_the_right_nodes(self) -> None:
        ev = pm.make_event(0.0, 6, 1.0)
        for j, fl in enumerate(ev.flights):
            self.assertEqual(len(fl.a), min(ev.counts[j], pm.MAX_DRAWN))
            self.assertTrue(np.all(fl.arrive > fl.depart))
        self.assertTrue(np.all(np.abs(ev.flights[-1].b[:, 0] - pm.ANODE_X) < 1e-9))


if __name__ == "__main__":
    unittest.main()
