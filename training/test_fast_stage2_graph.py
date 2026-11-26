import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
import onnx
from onnx import numpy_helper
import torch

from training.fast_stage2_graph import TokenGraph, verify
from training.pack_weights import make_graph, quantize
from training.stage2_student import FactoredLift, Stage2Net


class FastStage2GraphTests(unittest.TestCase):
    def test_layout_parity_and_packed_codes_in_all_three_stage2_models(self):
        cases = json.loads(Path(__file__).with_name("parity.json").read_text())["cases"]
        positions = np.asarray([c["encoded"] for c in cases], dtype=np.float32)
        torch.manual_seed(27)
        for name, compact, moderate in (("dense", False, False), ("compact", True, False),
                                        ("moderate", True, True)):
            with self.subTest(name=name), TemporaryDirectory() as directory:
                model = Stage2Net(compact=compact, moderate=moderate).eval()
                with torch.no_grad():
                    for layer in model.modules():
                        if isinstance(layer, FactoredLift):
                            layer.offset.normal_(std=0.1)  # Detect tile or mini-board permutations.
                path = Path(directory) / "source.onnx"
                torch.onnx.export(model, (torch.from_numpy(positions[:1]),), str(path),
                                  input_names=["state"], output_names=["policy", "value"],
                                  dynamic_axes={n: {0: "batch"} for n in ("state", "policy", "value")},
                                  opset_version=14, dynamo=False)
                source = onnx.load(path)
                stem = next(t for t in source.graph.initializer if t.name == "layers.stem.weight")
                codes, centers, _ = quantize(numpy_helper.to_array(stem), 4, 32, 1)
                packed = make_graph(source, [stem], 4, 32, {stem.name: (codes, centers)})
                rewritten = TokenGraph(packed).build()
                self.assertFalse(any(n.op_type in ("Conv", "ConvTranspose") for n in rewritten.graph.node))
                stored = {t.name: t.raw_data for t in rewritten.graph.initializer}
                self.assertEqual(stored["wpack/0/bytes"], codes.tobytes())
                self.assertEqual(stored["wpack/0/centers"], centers.tobytes())
                metrics = verify(packed.SerializeToString(), rewritten.SerializeToString(), positions, threads=1)
                self.assertGreater(metrics["legalBestProbesIncludingSingleton"], 1)


if __name__ == "__main__":
    unittest.main()
