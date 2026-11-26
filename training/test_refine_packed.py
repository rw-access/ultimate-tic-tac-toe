import copy
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import onnx
from onnx import numpy_helper
import torch
from torch import nn

from training.pack_weights import eligible, make_graph, quantize
from training.refine_packed import bind_codebooks, check_parity, layout_hash, update_centroids


class SmallPolicyValue(nn.Module):
    def __init__(self):
        super().__init__()
        self.kernel = nn.Conv2d(1, 4, 3, padding=1)

    def forward(self, state):
        grid = state[:, :81].reshape(-1, 1, 9, 9)
        logits = self.kernel(grid).sum(1).flatten(1)
        return logits, logits.mean(1, keepdim=True).tanh()


class RefinePackedTests(unittest.TestCase):
    def test_only_centroids_change_and_the_packed_runtime_still_agrees(self):
        torch.manual_seed(145)
        base = SmallPolicyValue().eval()
        states = torch.randn(12, 280)
        with TemporaryDirectory() as temp:
            path = Path(temp) / "tiny.onnx"
            torch.onnx.export(base, (states[:1],), str(path), input_names=["state"],
                              output_names=["policy", "value"], dynamo=False, opset_version=14,
                              dynamic_axes={n: {0: "batch"} for n in ("state", "policy", "value")})
            source = onnx.load(path)
        tensors = eligible(source)
        self.assertEqual(len(tensors), 1)
        for bits in (2, 4):
            with self.subTest(bits=bits):
                # 36 kernel entries exercise a partially populated final group.
                tensor = tensors[0]
                codes, centers, _ = quantize(numpy_helper.to_array(tensor), bits, 32, 2)
                graph = make_graph(source, tensors, bits, 32, {tensor.name: (codes, centers)})
                student = copy.deepcopy(base)
                books = bind_codebooks(graph, student)
                before = graph.SerializeToString()
                layout = layout_hash(graph, books)
                check_parity(before, student, states, 1)
                book = next(iter(books.values()))
                self.assertEqual([id(p) for p in student.parameters() if p.requires_grad],
                                 [id(book.centers)])
                optimizer = torch.optim.Adam([book.centers], lr=0.01)
                logits, value = student(states)
                (logits.square().mean() + value.square().mean()).backward()
                self.assertGreater(book.centers.grad.abs().sum().item(), 0)
                optimizer.step()
                self.assertGreater(update_centroids(graph, books), 0)
                after = graph.SerializeToString()
                self.assertNotEqual(before, after)
                self.assertEqual(len(before), len(after))
                self.assertEqual(layout_hash(graph, books), layout)
                check_parity(after, student, states, 1)


if __name__ == "__main__":
    unittest.main()
