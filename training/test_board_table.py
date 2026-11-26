import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch

from training.board_table import BoardTable, LEARNED, PATTERNS, PatternLookup
from training.fit_search_student import export


class BoardTableTests(unittest.TestCase):
    @torch.inference_mode()
    def test_all_trit_indexes_and_local_win_block_codes(self):
        ids = torch.arange(PATTERNS)
        trits = (ids[:, None] // (3 ** torch.arange(9))) % 3
        own, other = (trits == 1).float(), (trits == 2).float()
        identity = PatternLookup(ids.float().unsqueeze(1), torch.ones(1))
        torch.testing.assert_close(identity(own, other)[:, 0], ids.float(), rtol=0, atol=0)

        # Check the local tactical channels against direct placement into each winning line.
        expected = np.zeros((PATTERNS, 9), dtype=np.int64)
        marks = trits.numpy()
        lines = ((0, 1, 2), (3, 4, 5), (6, 7, 8), (0, 3, 6),
                 (1, 4, 7), (2, 5, 8), (0, 4, 8), (2, 4, 6))
        for line in lines:
            for move in line:
                a, b = [x for x in line if x != move]
                for player in (1, 2):
                    wins = (marks[:, move] == 0) & (marks[:, a] == player) & (marks[:, b] == player)
                    expected[wins, move] |= player
        encoder = BoardTable().eval().encoder
        codes = torch.cat([encoder(a, b)[:, LEARNED:] for a, b in zip(own.split(1024), other.split(1024))])
        np.testing.assert_array_equal(codes.numpy(), expected)

    def test_export_compiles_local_encoder_and_saves_trainable_checkpoint(self):
        cases = json.loads((Path(__file__).parent / "parity.json").read_text())["cases"]
        states = torch.tensor([c["encoded"] for c in cases], dtype=torch.float32)
        torch.manual_seed(1024)
        model = BoardTable().eval()
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "cached.onnx"
            export(model, states, path, "board_table")
            self.assertLessEqual(path.stat().st_size, 1_492_760)
            resumed = BoardTable().eval()
            resumed.load_state_dict(torch.load(path.with_suffix(".pt"), weights_only=True), strict=True)
            with torch.inference_mode():
                for a, b in zip(model(states), resumed(states)):
                    torch.testing.assert_close(a, b, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
