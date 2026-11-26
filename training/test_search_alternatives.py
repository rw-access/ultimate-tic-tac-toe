import unittest

import numpy as np

from training.search_alternatives import pairs, updated_policy


class AlternativeSearchTests(unittest.TestCase):
    def test_each_comparison_uses_two_distinct_legal_actions(self):
        a = np.array([[.7,.2,.1,0.], [.65,.35,0.,0.]])
        b = np.array([[.2,.3,.5,0.], [.6,.4,0.,0.]])
        selected, disagreement = pairs(a,b,a>0)
        np.testing.assert_array_equal(selected, [[0,2],[0,1]])
        np.testing.assert_array_equal(disagreement, [True,False])

    def test_deeper_search_updates_only_relative_odds(self):
        p = np.array([[.2,.3,0.,.5]])
        m = np.array([[3,0]])
        q = np.array([[.4,.1]])
        np.testing.assert_allclose(updated_policy(p,m,q,q),p)
        result = updated_policy(p,m,q,np.array([[-.4,.5]]))
        self.assertGreater(result[0,0], result[0,3])
        self.assertEqual(result[0,1], p[0,1])
        self.assertEqual(result[0,2], 0.)
        self.assertAlmostEqual(result.sum(),1.)
