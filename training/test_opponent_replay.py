import unittest

import numpy as np

from training.distill_oracle import children
from training.opponent_replay import trajectory


class OpponentTrajectoryTests(unittest.TestCase):
    def test_reconstructs_complete_game_and_rejects_wrong_result(self):
        rng = np.random.default_rng(23)
        state = np.zeros((1,280), np.uint8)
        state[0,162:243] = 1
        state[0,279] = 1
        moves = []
        while True:
            move = int(rng.choice(np.flatnonzero(state[0,162:243])))
            moves.append(move)
            state,terminal,won = children(state,np.array([0]),np.array([move]))
            if terminal[0]:
                break
        winner = ('X' if len(moves)%2 else 'O') if won[0] else 'draw'
        game = dict(moves=moves,seat='X',winner=winner,score=.5 if winner=='draw' else float(winner=='X'))
        picks,value = trajectory(game,moves[:2],12)
        self.assertTrue(picks)
        self.assertTrue(all(state[162+move] and state[:162].sum() >= 12 for state,move in picks))
        self.assertEqual(value,2*game['score']-1)
        with self.assertRaises(ValueError):
            trajectory({**game,'winner':'incorrect'},moves[:2],12)
        with self.assertRaises(ValueError):
            trajectory({**game,'moves':moves[:-1]},moves[:2],12)
