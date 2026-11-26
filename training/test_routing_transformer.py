import json
from pathlib import Path
import unittest

import torch

from training.routing_transformer import RoutingTransformer


class RoutingTransformerTests(unittest.TestCase):
    @torch.inference_mode()
    def test_independent_board_tokens_for_each_position(self):
        cases = json.loads((Path(__file__).parent / "parity.json").read_text())["cases"]
        states = torch.tensor([c["encoded"] for c in cases], dtype=torch.float32)
        model = RoutingTransformer().eval()
        logits, values = model(states)
        self.assertEqual(logits.shape, (len(states), 81))
        self.assertEqual(values.shape, (len(states), 1))
        self.assertTrue(torch.isfinite(logits).all())
        self.assertTrue((values.abs() <= 1).all())
        single = model(states[3:4])
        torch.testing.assert_close(single[0], logits[3:4], atol=1e-5, rtol=1e-4)
        torch.testing.assert_close(single[1], values[3:4], atol=1e-5, rtol=1e-4)

    def test_attention_edge_matches_its_destination(self):
        state = torch.zeros(1, 280)
        mine = state[0, :81].reshape(9, 9)
        other = state[0, 81:162].reshape(9, 9)
        other[:2, :3] = 1
        other[2, :2] = 1
        mine[4, [3, 5]] = 1
        state[0, 252:254] = 1
        state[0, 279] = 1
        state[0, 162:243] = (1 - mine - other).flatten()
        state[0, 162:180] = 0
        _, _, moves = RoutingTransformer().encode(state)
        self.assertEqual(moves[0, 4, 2, 7].item(), 1)  # Sending to board 2 allows a global win.
        self.assertEqual(moves[0, 2, 2, 7].item(), 0)  # Same-board move occupies the winning cell.
        self.assertEqual(moves[0, 4, 0, 5].item(), 1)  # Board 0 was already closed.
        self.assertEqual(moves[0, 4, 4, 5].item(), 1)  # This move itself closes the destination.


if __name__ == "__main__":
    unittest.main()
