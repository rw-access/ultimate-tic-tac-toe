import unittest

import numpy as np
import onnx
from onnx import numpy_helper
import torch
from torch import nn

from training.pack_weights import quantize_int8
from training.refit_tiny_scales import bind_scales


class TinyQuantizationTest(unittest.TestCase):
    def test_transposed_matrix_accepts_scales_from_a_previous_release(self):
        torch.manual_seed(12)
        model = nn.Sequential(nn.Linear(12, 4, bias=False))
        weight = model[0].weight.detach().numpy().T.copy()
        codes, scale, _ = quantize_int8(weight, 8)
        scale *= np.float32(.93)
        graph = onnx.ModelProto()
        for name, array in [('i8', codes), ('scale', scale), ('shape', np.asarray(weight.shape, np.int64))]:
            graph.graph.initializer.append(numpy_helper.from_array(array, name='group/' + name))
        b = bind_scales(graph, model)['group/scale']
        np.testing.assert_array_equal(model[0].weight.detach().numpy(), (codes * scale).reshape(weight.shape).T)
        with torch.no_grad():
            b.gain.fill_(1.)
        np.testing.assert_allclose(b.scales().detach().numpy(), scale * np.exp(.22), rtol=1.e-6)
        np.testing.assert_array_equal(b.codes.numpy(), codes)

    def test_missing_or_ambiguous_weight_mapping_is_rejected(self):
        with self.assertRaises(ValueError):
            bind_scales(onnx.ModelProto(), nn.Sequential(nn.Linear(3, 2)))
        model = nn.Sequential(nn.Linear(8, 4, bias=False), nn.Linear(8, 4, bias=False))
        with torch.no_grad():
            model[1].weight.copy_(model[0].weight)
        with self.assertRaises(ValueError):
            bind_scales(onnx.ModelProto(), model)
