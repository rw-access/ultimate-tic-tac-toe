import json
from pathlib import Path
import unittest

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

from training.sparse_stage2_graph import SparseDecoder, verify


def sample_graph():
    # Exercise the exporter's tensor layout with small, position-dependent
    # learned weights. Test coverage is independent of the next web release.
    rng = np.random.default_rng(4589)
    nodes, tensors = [], []

    def parameter(values, name=None):
        name = name or f"t{len(tensors)}"
        tensors.append(numpy_helper.from_array(np.asarray(values), name))
        return name

    def weights(shape):
        return parameter(rng.normal(0, 0.1, shape).astype(np.float32))

    def op(kind, *inputs, output=None, **attrs):
        output = output or f"x{len(nodes)}"
        nodes.append(helper.make_node(kind, list(inputs), [output], **attrs))
        return output

    def view(name, shape, output=None):
        return op('Reshape', name, parameter(np.array(shape, np.int64)), output=output)

    planes = [op('Slice', 'state', parameter([i]), parameter([i + 81]), parameter([1])) for i in (0, 81, 162)]
    cells = op('Concat', *(view(t, [-1, 81, 1]) for t in [planes[0], planes[1], planes[0], planes[2]]), axis=2)
    boards = view(cells, [-1, 9, 36])
    context = op('Concat', *(op('MatMul', boards, weights((36, width))) for width in (128, 256, 256)), axis=2)
    projected = op('Add', op('MatMul', context, weights((640, 2))), parameter(np.zeros(2, np.float32), 'layers.policy.bias'))
    ranks = view(op('MatMul', projected, weights((2, 18))), [-1, 81, 2])
    expanded = view(op('MatMul', ranks, weights((2, 256))), [-1, 9, 9, 256])
    offset = op('Add', expanded, weights((9, 256)))
    action = op('Elu', view(offset, [-1, 81, 256]))
    logits = op('Concat', cells, action, axis=2)
    scores = op('Add', op('MatMul', logits, weights((260, 1))), parameter(np.zeros(1, np.float32), 'layers.logits.bias'))
    view(scores, [-1, 81], 'policy')
    op('ReduceMean', planes[0], output='value', axes=[1], keepdims=1)
    def info(name, width):
        return helper.make_tensor_value_info(name, TensorProto.FLOAT, ['batch', width])

    graph = helper.make_graph(nodes, 'test', [info('state', 280)], [info('policy', 81), info('value', 1)], tensors)
    return helper.make_model(graph, opset_imports=[helper.make_opsetid('', 14)], ir_version=8)


class SparsePolicyTests(unittest.TestCase):
    def test_distinct_boards_cells_and_terminal_in_a_batch(self):
        cases = json.loads(Path(__file__).with_name('parity.json').read_text())['cases']
        states = np.array([item['encoded'] for item in cases], np.float32)
        source = sample_graph().SerializeToString()
        for mode in ('board', 'cell'):
            with self.subTest(mode=mode):
                candidate = SparseDecoder(onnx.load_from_string(source)).convert(mode).SerializeToString()
                result = verify(source, candidate, states, mode)
                self.assertLess(result['maxRetainedLogitError'], 1e-5)
                self.assertEqual(result['maxValueError'], 0)


if __name__ == '__main__':
    unittest.main()
