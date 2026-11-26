from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import torch

from training.average_checkpoints import average


class CheckpointAverageTests(unittest.TestCase):
    def test_numeric_average_and_checks_on_structural_state(self):
        def model():
            m = torch.nn.Linear(3, 1)
            m.register_buffer("indices", torch.tensor([2, 0, 1]))
            return m

        with TemporaryDirectory() as tmp:
            paths = [Path(tmp) / f"{i}.pt" for i in range(3)]
            for i, p in enumerate(paths):
                m = model()
                with torch.no_grad():
                    m.weight.fill_([1e8, 1, -1e8][i])
                    m.bias.fill_(i)
                torch.save(m.state_dict(), p)
            result = average(model(), paths)
            torch.testing.assert_close(result.weight, torch.full_like(result.weight, 1 / 3))
            self.assertEqual(result.bias.item(), 1)
            self.assertEqual(result.indices.tolist(), [2, 0, 1])
            for key, value in (("indices", torch.tensor([0, 1, 2])),
                               ("indices", torch.tensor([2., 0., 1.])),
                               ("weight", torch.full((1, 3), float("nan")))):
                with self.subTest(key=key):
                    state = m.state_dict()
                    state[key] = value
                    torch.save(state, paths[2])
                    with self.assertRaises(ValueError):
                        average(model(), paths)


if __name__ == "__main__":
    unittest.main()
