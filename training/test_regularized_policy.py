import unittest
import numpy as np
from training.regularized_policy import posterior, relabel_batch, shrink_action_values


class RegularizedPolicyTest(unittest.TestCase):
    def test_stationarity_mass_and_illegal_action(self):
        prior = np.array([[.6, .4, 0.], [.01, .49, .5]])
        q = np.array([[.3, .8, 20.], [1., .2, -.2]])
        policy = posterior(prior, q, prior > 0, .12)
        np.testing.assert_allclose(policy.sum(1), 1.)
        self.assertEqual(policy[0, 2], 0.)
        for p, new, value in zip(prior, policy, q):
            legal = p > 0
            marginal = value[legal] + .12 * p[legal] / new[legal]
            np.testing.assert_allclose(marginal, marginal[0], rtol=1e-12)
        self.assertTrue((policy.argmax(1) == [1, 0]).all())

    def test_equal_values_and_forced_move(self):
        p = np.array([[0., 1., 0.], [.4, 0., .6]])
        np.testing.assert_allclose(posterior(p, np.ones_like(p), p > 0, .07), p)
        with self.assertRaises(ValueError):
            posterior(p, p, np.ones_like(p, dtype=bool), .07)

    def test_tiny_prior_on_highest_value_action(self):
        p = np.array([[1e-120, 1.]])
        result = posterior(p, np.array([[1., 0.]]), p > 0, .1)
        np.testing.assert_allclose(result, [[.9, .1]], rtol=1e-12)

    def test_unvisited_and_supported_values(self):
        n = np.array([[0, 4, 12]])
        leaf = np.ones_like(n) * -.5
        np.testing.assert_allclose(shrink_action_values(n, np.full(n.shape,.5), leaf, 4), [[-.5, 0., .25]])

    def test_training_batch_uses_child_perspective_and_legal_mask(self):
        from training.policy_arena import initial_state
        x = initial_state()[None]
        n = np.ones((1, 81), dtype=np.int32)
        q = np.zeros_like(n, dtype=float)
        def network(states, values):
            if not values:
                return np.zeros((len(states), 81))
            # After move 7 the opponent's position is bad, so that move should gain prior mass.
            return np.where(states[:, 81 + 7], -.8, .2)
        target = relabel_batch(network, x, n, q, mix=1.)
        self.assertEqual(target.argmax(1)[0], 7)
        np.testing.assert_allclose(target.sum(1), 1.)
        np.testing.assert_allclose(relabel_batch(network, x, n, q, mix=0.), np.full((1, 81), 1/81))
        with self.assertRaises(ValueError):
            relabel_batch(network, x, n * 0, q)
