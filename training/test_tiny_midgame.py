import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from training.line_student import LineStudent
from training.pipeline import _symmetry_maps
from training.tiny_midgame import MidgameLine, best_moves, plausible
from training.tiny_routing_panel import episode_clusters


class MidgameTests(unittest.TestCase):
    def test_episode_membership_checks_exact_states(self):
        states = np.zeros((3, 280), np.uint8)
        states[np.arange(3), [5, 8, 12]] = 1
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "episodes.jsonl"
            games = [{"game": i, "samples": [[state.tolist(), [], 0]]} for i, state in enumerate(states)]
            path.write_text("".join(json.dumps(row) + "\n" for row in games))
            unit, group = episode_clusters(path, states[[2, 0]])
            self.assertEqual(unit, "self-play game")
            np.testing.assert_array_equal(group, [2, 0])
            with path.open("a") as stream:
                stream.write(json.dumps(dict(games[0], game=3)) + "\n")
            with self.assertRaisesRegex(ValueError, "more than one episode"):
                episode_clusters(path, states)

    def test_plausibility_and_confidence_shrink(self):
        x = np.zeros((4, 280), np.uint8)
        n = np.zeros((4, 81), np.int32)
        q = np.zeros_like(n, np.float32)
        for i, ply in enumerate([11, 12, 35, 36]):
            x[i, :ply] = 1
        n[:, :3] = [130, 120, 1]
        q[:, :3] = [0.4, 0.3, 1.0]
        np.testing.assert_array_equal(plausible(x, n), [False, True, True, False])
        best, backed, supported = best_moves(n, q, np.zeros(4, np.float32))
        np.testing.assert_array_equal(best, 0)
        self.assertTrue((backed[:, 2] < 0.025).all())
        self.assertFalse(supported[:, 2].any())

    @torch.inference_mode()
    def test_zero_start_inactive_gates_and_all_d4_transforms(self):
        torch.manual_seed(37)
        base = LineStudent().eval()
        m = MidgameLine().eval()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "base.pt"
            torch.save(base.state_dict(), path)
            m.warm_start(path)
        states = json.loads((Path(__file__).parent / "parity.json").read_text())["cases"]
        x = torch.tensor([row["encoded"] for row in states], dtype=torch.float32)
        early = torch.zeros(1, 280)
        early[:, 162:243] = 1
        early[:, 279] = 1
        middle = early.clone()
        middle[:, :16] = 1
        middle[:, 162:178] = 0
        x = torch.cat((early, middle, x))
        first = base(x)
        for aa, bb in zip(first, m(x)):
            torch.testing.assert_close(aa, bb, atol=1e-6, rtol=1e-6)
        m.late_policy.move[-1].weight.normal_()
        m.late_value[-1].weight.normal_()
        logits, v = m(x)
        for actual, old in zip((logits, v), first):
            torch.testing.assert_close(actual[0], old[0], atol=1e-6, rtol=0)
        self.assertNotEqual(v[1].item(), first[1][1].item())
        for xm, pm in _symmetry_maps():
            rotated = torch.empty_like(x)
            rotated[:, xm] = x
            rl, rv = m(rotated)
            torch.testing.assert_close(rl[:, pm], logits, atol=6e-6, rtol=2e-5)
            torch.testing.assert_close(rv, v, atol=6e-6, rtol=2e-5)


if __name__ == "__main__":
    unittest.main()
