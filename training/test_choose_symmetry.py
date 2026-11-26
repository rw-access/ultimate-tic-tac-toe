import unittest

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper

from training.choose_symmetry import export, fit, view_maps


class SymmetryRouterTests(unittest.TestCase):
    def test_centered_fit_recovers_preferred_view(self):
        rng = np.random.default_rng(41)
        states = rng.integers(0, 2, (180, 280), dtype=np.uint8)
        transforms, _ = view_maps()
        w = rng.normal(size=280)
        scores = states[:, transforms] @ w + rng.normal(0, 80, size=(len(states), 1))
        fitted = fit(states, scores, transforms, .001)
        np.testing.assert_array_equal(np.argmax(states[:, transforms] @ fitted.astype(np.float32), 1),
                                      np.argmax(scores, 1))

    def test_every_orientation_returns_policy_to_original_board(self):
        transforms, moves = view_maps()
        # The base uses coordinates directly, so a missing inverse changes many logits.
        def tensor(name, dims):
            return helper.make_tensor_value_info(name, TensorProto.FLOAT, dims)
        table = np.arange(81, dtype=np.float32)[None]
        selector = np.eye(280, 81, dtype=np.float32)
        value = np.eye(280, 1, dtype=np.float32)
        g = helper.make_graph([helper.make_node('MatMul', ['state', 's'], ['bits']),
                               helper.make_node('Add', ['bits', 't'], ['policy']),
                               helper.make_node('MatMul', ['state', 'v'], ['value'])],
                              'coordinates', [tensor('state', ['N', 280])],
                              [tensor('policy', ['N', 81]), tensor('value', ['N', 1])],
                              [numpy_helper.from_array(a, n) for n, a in [('s', selector), ('t', table), ('v', value)]])
        src = helper.make_model(g, opset_imports=[helper.make_opsetid('', 14)], ir_version=8)
        weights = np.random.default_rng(22).normal(size=280).astype(np.float16)
        states = np.random.default_rng(48).integers(0, 2, (240, 280)).astype(np.float32)
        best = (states[:, transforms] @ weights.astype(np.float32)).argmax(1)
        self.assertEqual(len(np.unique(best)), 8)
        model = export(src, weights)
        opt = ort.SessionOptions()
        opt.intra_op_num_threads = opt.inter_op_num_threads = 1
        session = ort.InferenceSession(model.SerializeToString(), opt, providers=['CPUExecutionProvider'])
        p, v = session.run(None, {'state': states})
        x = np.take_along_axis(states, transforms[best], axis=1)
        np.testing.assert_array_equal(p, np.take_along_axis(x[:, :81] + table, moves[best], axis=1))
        np.testing.assert_array_equal(v, x[:, :1])


if __name__ == '__main__':
    unittest.main()
