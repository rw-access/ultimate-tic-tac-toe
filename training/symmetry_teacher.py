"""Build a training oracle that averages one model across rotations and reflections.

The oracle shares weights between all views. Each state nevertheless costs one network
forward per view. Use it for offline labelling or report the extra work explicitly.
"""

import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
import onnxruntime as ort

from training.pipeline import _symmetry_maps


SUBGROUPS = {2: (0, 2), 4: (0, 1, 2, 3), 8: tuple(range(8))}


def wrap(source, count):
    model = onnx.load(source)
    graph = model.graph
    if ([field.name for field in graph.input] != ['state']
            or [field.name for field in graph.output] != ['policy', 'value']):
        raise ValueError('Expected project policy/value model')
    opset = next(item.version for item in model.opset_import if item.domain == '')
    if opset < 13:
        raise ValueError('Expected ONNX opset 13 or newer')
    model = onnx.compose.add_prefix(model, 'teacher/')
    graph = model.graph
    views = SUBGROUPS[count]
    maps = _symmetry_maps()
    node = helper.make_node

    def constant(name, array):
        graph.initializer.append(numpy_helper.from_array(np.asarray(array), name=name))
        return name

    states = ['state']
    before = []
    for view in views[1:]:
        indices = constant(f'inverse_features_{view}', np.argsort(maps[view][0]).astype(np.int64))
        states.append(f'state_{view}')
        before.append(node('Gather', ['state', indices], [states[-1]], axis=1))
    before.append(node('Concat', states, ['teacher/state'], axis=0))
    after = []
    split_options = dict(axis=0, **({'num_outputs': count} if opset >= 18 else {}))
    for head in ('policy', 'value'):
        after.append(node('Split', ['teacher/' + head], [f'{head}_{i}' for i in views], **split_options))
    legal_indices = constant('legal_indices', np.arange(162, 243, dtype=np.int64))
    cutoff = constant('cutoff', np.float32(.5))
    masked_logit = constant('masked_logit', np.float32(-100))
    floor = constant('floor', np.float32(1e-30))
    after.extend((node('Gather', ['state', legal_indices], ['legal_plane'], axis=1),
                  node('Greater', ['legal_plane', cutoff], ['legal'])))
    probabilities = []
    for view in views:
        aligned = f'policy_{view}'
        if view:
            indices = constant(f'forward_moves_{view}', maps[view][1].astype(np.int64))
            aligned = f'aligned_{view}'
            after.append(node('Gather', [f'policy_{view}', indices], [aligned], axis=1))
        probabilities.append(f'probability_{view}')
        after.extend((node('Where', ['legal', aligned, masked_logit], [f'masked_{view}']),
                      node('Softmax', [f'masked_{view}'], [probabilities[-1]], axis=1)))
    after.extend((node('Mean', probabilities, ['mean_probability']),
                  node('Max', ['mean_probability', floor], ['positive_probability']),
                  node('Log', ['positive_probability'], ['policy']),
                  node('Mean', [f'value_{view}' for view in views], ['value'])))
    nodes = before + list(graph.node) + after
    del graph.node[:]
    graph.node.extend(nodes)
    del graph.input[:]
    del graph.output[:]
    for fields, name, width in [(graph.input, 'state', 280), (graph.output, 'policy', 81), (graph.output, 'value', 1)]:
        fields.append(helper.make_tensor_value_info(name, TensorProto.FLOAT, ['batch', width]))
    onnx.checker.check_model(model)
    return model


def verify(source, candidate, states, count):
    options = ort.SessionOptions()
    options.inter_op_num_threads = options.intra_op_num_threads = 1
    base, oracle = [ort.InferenceSession(str(path), options, providers=['CPUExecutionProvider'])
                    for path in (source, candidate)]
    maps = _symmetry_maps()
    for size in (1, len(states)):
        x = states[:size].astype(np.float32)
        probabilities, values = [], []
        for view in SUBGROUPS[count]:
            feature, move = maps[view]
            transformed = np.empty_like(x)
            transformed[:, feature] = x
            logits, value = base.run(['policy', 'value'], {'state': transformed})
            scores = np.where(x[:, 162:243] > .5, logits[:, move], -100.)
            mass = np.exp(scores - scores.max(axis=1, keepdims=True))
            probabilities.append(mass / mass.sum(axis=1, keepdims=True))
            values.append(value)
        log_policy, averaged = oracle.run(['policy', 'value'], {'state': x})
        np.testing.assert_allclose(np.exp(log_policy), np.mean(probabilities, axis=0), atol=1e-5, rtol=2e-4)
        np.testing.assert_allclose(averaged, np.mean(values, axis=0), atol=2e-5, rtol=2e-4)
        if not np.isfinite(log_policy).all() or not np.isfinite(averaged).all():
            raise ValueError('Oracle returned nonfinite predictions')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--views', choices=tuple(SUBGROUPS), type=int, default=4)
    parser.add_argument('--fixtures', type=Path, default=Path('training/parity.json'))
    parser.add_argument('--sample', type=Path, help='Additional compact position shard for parity')
    args = parser.parse_args()
    cases = json.loads(args.fixtures.read_text())['cases']
    states = np.asarray([case['encoded'] for case in cases], dtype=np.float32)
    if args.sample:
        with np.load(args.sample, allow_pickle=False) as shard:
            states = np.concatenate((states, shard['x'][:96]))
    if args.output.exists():
        parser.error('Select a fresh output')
    model = wrap(args.input, args.views)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=args.output.parent) as directory:
        output = Path(directory) / 'oracle.onnx'
        onnx.save(model, output)
        verify(args.input, output, states, args.views)
        args.output.hardlink_to(output)
    print(f'Checked {len(states)} positions for {args.views} views at batch sizes 1 and {len(states)}')


if __name__ == '__main__':
    main()
