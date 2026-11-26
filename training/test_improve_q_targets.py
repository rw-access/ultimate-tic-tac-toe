import unittest

import numpy as np

from training.improve_q_targets import improved_targets, reverse_kl_policy, sample_children


class ActionImprovementTests(unittest.TestCase):
    def test_reverse_kl_optimum_and_tied_outcomes(self):
        prior = np.array([[.2, .8, 0.], [.6, .4, 0.]])
        values = np.array([[.85, -.4, 500.], [-.25, -.25, 8.]])
        legal = prior > 0
        policy = reverse_kl_policy(prior, values, legal, .18)
        np.testing.assert_allclose(policy.sum(1), 1.)
        np.testing.assert_allclose(policy[:, 2], 0.)
        np.testing.assert_allclose(policy[1], prior[1])
        self.assertGreater(policy[0, 0], prior[0, 0])
        stationary = values[0, :2] + .18 * prior[0, :2] / policy[0, :2]
        self.assertAlmostEqual(*stationary, places=6)
        np.testing.assert_allclose(reverse_kl_policy(prior, values + 3., legal, .18), policy)

    def test_underexplored_and_illegal_moves(self):
        prior = np.array([[.6, .4, 0]], dtype=np.float32)
        legal = np.array([[True, True, False]])
        q = np.array([[.9, -.6, 1]], dtype=np.float32)
        visits = np.array([[2048, 2048, 0]], dtype=np.uint16)
        p, v = improved_targets(prior, prior, visits, q, np.zeros(1), legal, 6, 64, .7)
        self.assertGreater(p[0, 0], prior[0, 0])
        self.assertEqual(p[0, 2], 0)
        self.assertAlmostEqual(p.sum(), 1.)
        self.assertGreater(v.item(), 0)
        neutral, _ = improved_targets(prior, prior, visits * 0, q, np.zeros(1), legal, 6, 64, .7)
        np.testing.assert_allclose(neutral, prior, atol=1.e-6)

    def test_child_value_changes_player(self):
        x = np.zeros((2, 280), dtype=np.uint8)
        x[:, 162:243] = 1
        x[:, 279] = 1
        visits = np.zeros((2, 81), dtype=np.uint16)
        visits[0, 12] = 75
        q = np.zeros((2, 81), dtype=np.float32)
        q[0, 12] = .64
        next_state, value, count = sample_children(x, visits, q, np.random.default_rng(42), 64)
        self.assertEqual(len(next_state), 1)
        self.assertEqual(next_state[0, 81 + 12], 1)
        self.assertAlmostEqual(value[0], -.64, places=6)
        self.assertEqual(count[0], 75)


if __name__ == '__main__':
    unittest.main()
