import json
import unittest
from pathlib import Path

import torch

from training.line_student import LineStudent
from training.pipeline import _symmetry_maps


class LineStudentTests(unittest.TestCase):
    @torch.inference_mode()
    def test_full_network_commutes_with_all_board_symmetries(self):
        data = json.loads((Path(__file__).parent / "parity.json").read_text())
        states = torch.tensor([item["encoded"] for item in data["cases"]], dtype=torch.float32)
        torch.manual_seed(75)
        net = LineStudent().eval()
        logits, values = net(states)
        for feature_map, action_map in _symmetry_maps():
            moved = torch.empty_like(states)
            moved[:, feature_map] = states
            moved_logits, moved_values = net(moved)
            torch.testing.assert_close(moved_logits[:, action_map], logits, atol=2e-6, rtol=2e-5)
            torch.testing.assert_close(moved_values, values, atol=2e-6, rtol=2e-5)

    def test_forcing_a_win_and_redirecting_to_a_free_board(self):
        state = torch.zeros(1, 280)
        mine, other = state[:, :81].view(9, 9), state[:, 81:162].view(9, 9)
        other[:2, :3] = 1  # Opponent owns global boards 0 and 1.
        state[0, 252:254] = 1
        other[2, :2] = 1   # Opponent can win global board 2 at cell 2.
        mine[4, [3, 5]] = 1
        mine[5, :2] = 1
        state[0, 279] = 1
        state[:, 162:243] = (1 - mine - other).reshape(1, 81)
        state[:, 162:180] = 0
        _, _, _, routing, _ = LineStudent().encode(state)
        loses = routing[0, :, :, 7]
        self.assertEqual(loses[4, 2].item(), 1)  # Force the opponent to the winning board.
        self.assertEqual(loses[4, 3].item(), 0)  # Send them to a harmless board.
        self.assertEqual(loses[4, 0].item(), 1)  # Closed destination gives free choice.
        self.assertEqual(loses[2, 2].item(), 0)  # Occupy their winning cell before routing.
        self.assertEqual(loses[4, 4].item(), 1)  # Newly won destination also gives free choice.

        # Completing our own global row ends the game before the opponent can reply.
        state[0, 246:248] = 1
        mine[3:5, 6:9] = 1
        state[:, 162 + 3 * 9:162 + 5 * 9] = 0
        _, _, _, routing, _ = LineStudent().encode(state)
        self.assertEqual(routing[0, 5, 2, 0].item(), 1)
        self.assertEqual(routing[0, 5, 2, 7].item(), 0)


if __name__ == "__main__":
    unittest.main()
