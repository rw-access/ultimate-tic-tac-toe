import unittest

import numpy as np

from training.distill_oracle import children
from training.policy_arena import choose, initial_state, openings, original_input, play, summarize


class PolicyArenaTests(unittest.TestCase):
    def test_greedy_masks_and_one_ply_changes_value_perspective(self):
        root = initial_state()[None]
        state, _, _ = children(root, np.array([0]), np.array([0]))
        def evaluate(x, value_only):
            if value_only:
                # The just-played mark is in the opponent's plane at the child.
                return -.75 * x[:, 81+7].astype(np.float32)
            score = np.zeros((len(x),81))
            score[:,0] = 100
            score[:,4] = 2
            return score
        self.assertEqual(choose(state,evaluate,False)[0].tolist(),[4])
        moves, calls = choose(state,evaluate,True)
        self.assertEqual(moves.tolist(),[7])
        self.assertEqual(calls,8)
        np.testing.assert_array_equal(original_input(state)[:,2],-np.ones((1,9,9)))

    def test_terminal_children_are_scored_without_network_calls(self):
        state = initial_state()
        state[243:245] = 1
        state[18:20] = 1
        state[162:243] = 0
        state[162+20] = 1
        def no_evaluation(*args):
            self.fail("Terminal wins must be scored by rules")
        chosen, calls = choose(state[None],no_evaluation,True)
        self.assertEqual((chosen.item(),calls),(20,0))

    def test_swapping_identical_players_is_even(self):
        starts=openings(2,1,7)
        self.assertEqual(len(starts),720)
        self.assertEqual(len(set(openings(6,12,9))),12)
        def first_legal(x, value_only):
            return np.zeros(len(x)) if value_only else np.zeros((len(x),81))
        games,counts=play(starts[:3],[first_legal,first_legal],False,2)
        self.assertEqual(summarize(games)['score'],.5)
        self.assertEqual(counts[0],counts[1])
        self.assertTrue(all(r['firstAsX']==r['firstAsO'] for r in games))
