import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch

from training import fit_search_student as trainer


class SmallModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = torch.nn.Sequential(torch.nn.Linear(280, 8), torch.nn.Dropout(0.2))
        self.policy = torch.nn.Linear(8, 81)
        self.value = torch.nn.Linear(8, 1)

    def forward(self, x):
        h = self.hidden(x)
        return self.policy(h), self.value(h).tanh()


class ResumeTest(unittest.TestCase):
    def assert_same(self, left, right):
        if isinstance(left, torch.Tensor):
            self.assertTrue(torch.equal(left, right))
        elif isinstance(left, dict):
            self.assertEqual(left.keys(), right.keys())
            for key in left:
                self.assert_same(left[key], right[key])
        elif isinstance(left, (tuple, list)):
            self.assertEqual(len(left), len(right))
            for a, b in zip(left, right):
                self.assert_same(a, b)
        else:
            self.assertEqual(left, right)

    def check_restart(self, device):
        with tempfile.TemporaryDirectory() as folder:
            base = Path(folder)
            x = np.zeros((21, 280), dtype=np.uint8)
            x[:, 162:243] = 1
            p = np.zeros((len(x), 81), dtype=np.float16)
            p[np.arange(len(x)), np.arange(len(x))] = 1
            v = np.linspace(-0.9, 0.9, len(x), dtype=np.float32)
            for corpus in ("primary", "replay"):
                directory = base / corpus
                directory.mkdir()
                (directory / "manifest.json").write_text("{}")
                np.savez(directory / "part.npz", x=x, p=p, v=v)
            args = ["trainer", "--data", str(base / "primary"), "--replay", str(base / "replay"),
                    "--max-positions", "13", "--replay-fraction", ".25", "--device", device,
                    "--steps", "5", "--batch-size", "4", "--save-every", "2", "--output",
                    str(base / "out.onnx")]

            def train(path, *extra):
                with (mock.patch.object(sys, "argv", args + ["--resume-state", str(path), *extra]),
                      mock.patch.object(trainer, "build_harnet", return_value=None) as build,
                      mock.patch.object(trainer, "export")):
                    build.side_effect = lambda _tier: SmallModel()
                    trainer.main()

            reference, resumed = (base / name for name in ("reference.pt", "restarted.pt"))
            train(reference)
            save = trainer.save_checkpoint

            def interrupt(path, state):
                save(path, state)
                self.assertEqual(state["step"], 2)
                raise InterruptedError("simulated shutdown")

            with mock.patch.object(trainer, "save_checkpoint", side_effect=interrupt):
                with self.assertRaises(InterruptedError):
                    train(resumed)
            train(resumed)
            self.assert_same(torch.load(reference, map_location="cpu", weights_only=True),
                             torch.load(resumed, map_location="cpu", weights_only=True))
            evidence = trainer.file_hash(resumed)
            with self.assertRaisesRegex(ValueError, "disagrees"):
                train(resumed, "--lr", "0.02")
            # The source manifest stayed the same, but its actual training rows changed.
            np.savez(base / "primary" / "part.npz", x=x, p=p, v=-v)
            with self.assertRaisesRegex(ValueError, "disagrees"):
                train(resumed)
            self.assertEqual(evidence, trainer.file_hash(resumed))

    def test_cpu_restart(self):
        self.check_restart("cpu")

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
    def test_cuda_restart(self):
        self.check_restart("cuda")

    def test_partial_write_preserves_previous_checkpoint(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "state.pt"
            trainer.save_checkpoint(path, {"step": 2})
            previous = path.read_bytes()

            def disk_error(_state, output):
                output.write(b"partial")
                raise OSError("disk full")

            with mock.patch.object(torch, "save", side_effect=disk_error):
                with self.assertRaises(OSError):
                    trainer.save_checkpoint(path, {"step": 3})
            self.assertEqual(path.read_bytes(), previous)
            self.assertEqual(list(path.parent.iterdir()), [path])


if __name__ == "__main__":
    unittest.main()
