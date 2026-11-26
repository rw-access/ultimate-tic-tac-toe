"""Learn a single board orientation per position using searched training targets.

The deployed network evaluates the chosen orientation once. Policy logits are
aligned back to the caller's board. Training compares all eight orientations
of each calibration state and estimates within-position differences in error.
"""

import argparse
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper

from training.calibrate_outputs import sample, split_files, unique_rows
from training.pipeline import _symmetry_maps


def view_maps():
    raw = _symmetry_maps()
    state = np.stack([np.argsort(features) for features, _ in raw])
    moves = np.stack([actions for _, actions in raw])
    return state, moves


def quality(session, dataset, state_map, move_map):
    """Legal-move NLL and squared value error for each view of each original board."""
    ce = np.empty((len(dataset['x']), len(state_map)), dtype=np.float32)
    mse = np.empty_like(ce)
    for first in range(0, len(ce), 16):
        rows = slice(first, first + 16)
        x = dataset['x'][rows]
        viewed = x[:, state_map].reshape(-1, 280).astype(np.float32)
        policy, v = session.run(['policy', 'value'], {'state': viewed})
        logits = np.take_along_axis(policy.reshape(len(x), -1, 81), move_map[None], axis=2)
        legal = x[:, None, 162:243].astype(bool)
        z = np.where(legal, logits, -np.inf).max(2, keepdims=True)
        shift = logits - z
        log_norm = np.log(np.where(legal, np.exp(np.where(legal, shift, -np.inf)), 0.).sum(2))
        target = dataset['p'][rows].astype(np.float32)
        target /= target.sum(1, keepdims=True)
        ce[rows] = log_norm - (target[:, None] * shift).sum(2)
        mse[rows] = (v.reshape(len(x), -1) - dataset['v'][rows, None]) ** 2
    return ce, mse


def fit(states, quality_target, state_map, ridge):
    """Fit centered within-board scores, removing the mean over all orientations."""
    gram = np.eye(280, dtype=np.float64) * ridge
    rhs = np.zeros(280, dtype=np.float64)
    for first in range(0, len(states), 512):
        features = states[first:first + 512, state_map].astype(np.float32)
        centered = (features - features.mean(1, keepdims=True)).reshape(-1, 280)
        target = quality_target[first:first + len(features)]
        gram += centered.T @ centered
        rhs += centered.T @ (target - target.mean(1, keepdims=True)).ravel()
    return np.linalg.solve(gram, rhs).astype(np.float16)


def export(source, weights):
    model = onnx.ModelProto()
    model.CopyFrom(source)
    g = model.graph
    if [i.name for i in g.input] != ['state'] or [i.name for i in g.output] != ['policy', 'value']:
        raise ValueError('Expected the production state, policy and value interface')
    state_map, move_map = view_maps()
    base = list(g.node)
    if any(x.startswith('sym/') for n in base for x in (*n.input, *n.output)):
        raise ValueError('Input graph already contains symmetry routing')
    for n in base:
        for i, name in enumerate(n.input):
            if name == 'state':
                n.input[i] = 'sym/state'
            elif name == 'policy':
                raise ValueError('Unexpected internal use of the policy output')
        for i, name in enumerate(n.output):
            if name == 'policy':
                n.output[i] = 'sym/policy'
    del g.node[:]
    def constant(name, a):
        g.initializer.append(numpy_helper.from_array(np.asarray(a), name))
    def op(kind, inputs, output, **attrs):
        g.node.append(helper.make_node(kind, inputs, [output], **attrs))
    # Eight small cell and board permutations expand into the 280 input indices.
    constant('sym/i', state_map[:, :81].astype(np.uint8))
    constant('sym/b', (state_map[:, 243:252] - 243).astype(np.uint8))
    constant('sym/f', move_map.astype(np.uint8))
    constant('sym/e', np.full((8, 1), 279, dtype=np.int64))
    constant('sym/w', weights.astype(np.float16).reshape(280, 1))
    op('Cast', ['sym/i'], 'sym/i0', to=TensorProto.INT64)
    op('Cast', ['sym/b'], 'sym/b0', to=TensorProto.INT64)
    parts = ['sym/i0']
    for off in (81, 162, 243, 252, 261, 270):
        c = 'sym/c' + str(off)
        n = 'sym/m' + str(off)
        constant(c, np.int64(off))
        op('Add', ['sym/i0' if off < 243 else 'sym/b0', c], n)
        parts.append(n)
    op('Concat', [*parts, 'sym/e'], 'sym/maps', axis=1)
    op('Gather', ['state', 'sym/maps'], 'sym/views', axis=1)
    op('Cast', ['sym/w'], 'sym/weights', to=TensorProto.FLOAT)
    op('MatMul', ['sym/views', 'sym/weights'], 'sym/raw')
    constant('sym/axes', np.array([2], dtype=np.int64))
    op('Squeeze', ['sym/raw', 'sym/axes'], 'sym/scores')
    op('ArgMax', ['sym/scores'], 'sym/best', axis=1, keepdims=0)
    op('Gather', ['sym/maps', 'sym/best'], 'sym/in', axis=0)
    op('GatherElements', ['state', 'sym/in'], 'sym/state', axis=1)
    g.node.extend(base)
    op('Gather', ['sym/f', 'sym/best'], 'sym/out8', axis=0)
    op('Cast', ['sym/out8'], 'sym/out', to=TensorProto.INT64)
    op('GatherElements', ['sym/policy', 'sym/out'], 'policy', axis=1)
    onnx.checker.check_model(model)
    return model


def run(a):
    if a.output.exists() or a.report.exists() or a.samples < 1024 or a.ridge <= 0:
        raise ValueError('Choose fresh outputs, positive regularization and at least 1024 rows')
    state_map, move_map = view_maps()
    files = sorted(a.data.glob('*.npz'))
    train_files, valid_files = split_files(files, a.seed)
    sizes = {}
    for f in files:
        with np.load(f, allow_pickle=False) as s:
            sizes[f] = len(s['v'])
    seen = set()
    datasets = [unique_rows(sample(paths, sizes, n, seed), state_map, seen) for paths, n, seed in
                ((train_files, a.samples, a.seed + 1), (valid_files, a.samples // 2, a.seed + 2))]
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = opts.inter_op_num_threads = 1
    session = ort.InferenceSession(str(a.input), opts, providers=['CPUExecutionProvider'])
    metrics = [quality(session, data, state_map, move_map) for data in datasets]
    ce, mse = metrics[0]
    weights = fit(datasets[0]['x'], -ce - a.value_weight * mse, state_map, a.ridge)
    valid = datasets[1]
    x = valid['x']
    scores = x[:, state_map].astype(np.float32) @ weights.astype(np.float32)
    choice = scores.argmax(1)
    rows = np.arange(len(x))
    ce, mse = metrics[1]
    def evaluate(loss):
        return {'unmodified': float(loss[:, 0].mean()), 'selected': float(loss[rows, choice].mean()),
                'meanView': float(loss.mean()), 'bestViewBound': float(loss.min(1).mean())}
    diff = (ce[:, 0] - ce[rows, choice]) + a.value_weight * (mse[:, 0] - mse[rows, choice])
    clusters = np.unique(valid['file'])
    cluster_stats = np.asarray([[diff[valid['file'] == i].sum(), (valid['file'] == i).sum()] for i in clusters])
    rng = np.random.default_rng(a.seed + 3)
    boots = cluster_stats[rng.integers(0, len(clusters), (5000, len(clusters)))].sum(1)
    interval = np.quantile(boots[:, 0] / boots[:, 1], [.025, .975])
    model = export(onnx.load(a.input), weights)
    payload = model.SerializeToString()
    if len(payload) > a.max_bytes:
        raise ValueError(f'Symmetry-routed export uses {len(payload)} bytes')
    deployed = ort.InferenceSession(payload, opts, providers=['CPUExecutionProvider'])
    for n in (1, 12, 127):
        probe = x[:n]
        selected = choice[:n]
        source = np.take_along_axis(probe, state_map[selected], axis=1).astype(np.float32)
        pol, val = session.run(None, {'state': source})
        expected = np.take_along_axis(pol, move_map[selected], axis=1)
        actual, v = deployed.run(None, {'state': probe.astype(np.float32)})
        np.testing.assert_allclose(actual, expected, atol=3e-5, rtol=1e-5)
        np.testing.assert_allclose(v, val, atol=1e-5, rtol=1e-5)
    receipt = dict(seed=a.seed, sourceSha256=hashlib.sha256(a.input.read_bytes()).hexdigest(),
                   sha256=hashlib.sha256(payload).hexdigest(), bytes=len(payload), trainingRows=len(datasets[0]['x']),
                   validationRows=len(x), policyNll=evaluate(ce), valueMse=evaluate(mse),
                   choiceCounts=np.bincount(choice, minlength=8).tolist(), meanImprovement=float(diff.mean()),
                   clusterInterval95=interval.tolist(), ridge=a.ridge, valueWeight=a.value_weight)
    for path, content in ((a.output, payload), (a.report, (json.dumps(receipt, indent=2) + '\n').encode())):
        path.parent.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(dir=path.parent) as temp:
            t = Path(temp) / 'artifact'
            t.write_bytes(content)
            path.hardlink_to(t)
    print(json.dumps(receipt), flush=True)


if __name__ == '__main__':
    q = argparse.ArgumentParser(description=__doc__)
    for name in ('data', 'input', 'output', 'report'):
        q.add_argument('--' + name, type=Path, required=True)
    q.add_argument('--samples', type=int, default=16000)
    q.add_argument('--seed', type=int, default=26315588)
    q.add_argument('--ridge', type=float, default=45.)
    q.add_argument('--value-weight', type=float, default=1.8)
    q.add_argument('--max-bytes', type=int, default=1480935)
    run(q.parse_args())
