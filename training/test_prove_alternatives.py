import unittest

import numpy as np

from training.prove_alternatives import alternative_moves, remaining


class ProvedAlternativesTests(unittest.TestCase):
    def test_selects_likely_inferior_and_optimal_actions_from_parent_perspective(self):
        classes = np.array([list(b'WDWL' + b'.' * 77), list(b'LL' + b'.' * 79)], dtype=np.uint8)
        prior = np.zeros((2, 81), np.float16)
        prior[0, :4] = [.15, .65, .17, .03]
        prior[1, :2] = [.55, .45]
        parents, moves, values = alternative_moves(classes, prior, np.random.default_rng(43))
        self.assertEqual(list(zip(parents, moves, values)), [(0, 2, 1), (1, 0, -1), (0, 1, 0)])

    def test_remaining_counts_only_playable_spots_on_open_local_boards(self):
        x = np.zeros((1, 280), np.uint8)
        x[0, :3] = 1
        x[0, 243] = 1
        x[0, 81 + 12] = 1
        self.assertEqual(int(remaining(x)[0]), 71)
