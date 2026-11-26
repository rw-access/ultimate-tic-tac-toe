import unittest

import numpy as np

from training.search_errors import action_error


class SearchErrorTests(unittest.TestCase):
    def test_unsupported_or_unvisited_actions_cannot_create_hard_examples(self):
        counts=np.array([[120,120,0],[10,100,0],[0,0,0]])
        q=np.array([[-.5,.5,.99],[-.9,.9,.8],[1.,.8,.5]])
        error, supported=action_error(counts,q,np.zeros(3),np.zeros(3,dtype=int))
        np.testing.assert_array_equal(supported,[True,False,False])
        self.assertAlmostEqual(error[0],120/168)
        np.testing.assert_array_equal(error[1:],0.)
