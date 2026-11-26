import unittest

import numpy as np

from training.search_allocation import FEATURES, divergence, fit_rule, observations, select_roots


class SearchAllocationTest(unittest.TestCase):
    def test_fit_recovers_difficulty_and_respects_partial_batches(self):
        rng = np.random.default_rng(42)
        x = rng.normal(size=(512, len(FEATURES)))
        rule = fit_rule(x, x[:, 0] * .1)
        probe = np.zeros((73, len(FEATURES)))
        probe[:, 0] = np.arange(len(probe))
        expected = list(range(63, 47, -1)) + [72, 71]
        self.assertEqual(select_roots(probe, rule).tolist(), expected)
        self.assertEqual(select_roots(probe[:3], rule).size, 0)

    def test_ties_follow_input_order_and_reject_an_invalid_scale(self):
        rule = fit_rule(np.ones((6, len(FEATURES))), np.ones(6))
        x = np.zeros((8, len(FEATURES)))
        self.assertEqual(select_roots(x, rule).tolist(), [0, 1])
        rule['featureScale'][0] = 0.
        with self.assertRaises(ValueError):
            select_roots(x, rule)

    def test_shallow_observations_ignore_illegal_actions_and_require_correct_work(self):
        x = np.zeros((1, 280), np.uint8)
        x[0, 162:164] = 1
        p = np.zeros((1, 81)); p[0, :2] = [.25, .75]
        n = np.zeros((1, 81), np.int32); n[0, :2] = [127, 0]
        q = np.zeros((1, 81)); v = np.zeros(1)
        y = observations(x, n, q, v, p, q.copy())[0]
        self.assertTrue(np.isfinite(y).all())
        self.assertEqual(y[FEATURES.index('unvisitedPriorMass')], .75)
        self.assertEqual(y[FEATURES.index('visitedFraction')], .5)
        np.testing.assert_allclose(divergence(p, p), 0.)
        n[0, 2] = 1
        with self.assertRaises(ValueError):
            observations(x, n, q, v, p, q.copy())
