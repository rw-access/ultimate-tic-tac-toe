"""Refresh compact position targets with independent, fixed-work, native ONNX PUCT searches.

Example: python -m training.reanalyse_positions --data old-shards --model teacher.onnx
    --output fresh-shards --simulations 1024 --max-positions 8192 --parallel 32
For disjoint, restartable jobs, keep the same source, seed and max-positions and
assign each job a non-overlapping --offset/--count interval and fresh output.
"""
import argparse
import base64
import hashlib
import json
import os
import subprocess
import tempfile
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
WORKER = ROOT / 'scripts' / 'reanalyse.ts'


def sample_states(data, maximum, offset, count, seed):
    paths = sorted(data.glob('*.npz')) if data.is_dir() else [data]
    sources = []
    for path in paths:
        with np.load(path, allow_pickle=False) as shard:
            values = shard['v']
            if values.ndim != 1:
                raise ValueError(f'{path}: expected a vector of values')
            sources.append({'file': path.name, 'rows': len(values)})
    total = sum(source['rows'] for source in sources)
    limit = min(total, maximum)
    if offset >= limit:
        raise ValueError(f'Sample offset {offset} outside {limit} selected rows ({total} available)')
    # Slice before sorting. Adjacent jobs then form a partition of the same draw.
    draw = np.random.default_rng(seed).choice(total, size=limit, replace=False)
    ids = np.sort(draw[offset:offset + count if count is not None else None])
    states = np.empty((len(ids), 280), dtype=np.uint8)
    first = base = 0
    for source, path in zip(sources, paths):
        stop = int(np.searchsorted(ids, base + source['rows']))
        source['selected'] = stop - first
        if stop > first:
            with np.load(path, allow_pickle=False) as shard:
                x = shard['x']
                if x.shape != (source['rows'], 280) or x.dtype != np.uint8:
                    raise ValueError(f'{path}: expected uint8 [N,280] features')
                states[first:stop] = x[ids[first:stop] - base]
        first = stop
        base += source['rows']
    return ids, states, sources


def response(worker, expected):
    line = worker.stdout.readline()
    if not line:
        raise RuntimeError(f'Reanalysis worker closed its output (exit {worker.poll()}); see stderr')
    reply = json.loads(line)
    if reply.get('type') != expected:
        raise ValueError(f'Unexpected worker response; expected {expected}')
    return reply


def targets(rows, features, simulations):
    n = len(features)
    policy = np.array([row['p'] for row in rows], dtype=np.float32)
    value = np.array([row['v'] for row in rows], dtype=np.float32)
    visits = np.array([row['visits'] for row in rows])
    edges = np.array([row['edgeVisits'] for row in rows])
    if (policy.shape != (n, 81) or value.shape != (n,) or not np.isfinite(policy).all()
            or not np.isfinite(value).all() or np.any(np.abs(value) > 1) or np.any(policy < 0)
            or np.any((features[:, 162:243] == 0) & (policy > 0))
            or not np.allclose(policy.sum(axis=1), 1, atol=1e-6, rtol=0)):
        raise ValueError('Invalid reanalysis policy/value output')
    direct = visits == 0
    if (not np.issubdtype(visits.dtype, np.integer) or not np.issubdtype(edges.dtype, np.integer)
            or np.any(visits[~direct] != simulations) or np.any(edges[~direct] < 1)
            or np.any(edges[~direct] >= simulations) or np.any(edges[direct] != 0)
            or np.any(value[direct] != 1) or np.any(policy[direct].max(axis=1) != 1)):
        raise ValueError('Unmatched visit budget or invalid direct win')
    return policy.astype(np.float16), value, visits.astype(np.uint16), edges.astype(np.uint16)


def run(args):
    if args.output.exists():
        raise ValueError('Choose a fresh output directory')
    ids, states, sources = sample_states(args.data, args.max_positions, args.offset, args.count, args.seed)
    print(f'Selected {len(ids):,} independent roots from {sum(s["rows"] for s in sources):,} rows', flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    command = [args.node, '--experimental-strip-types', str(WORKER), '--model', str(args.model.resolve()),
               '--simulations', str(args.simulations), '--parallel', str(args.parallel), '--seed', str(args.seed)]
    report = {'format': 'uttt-independent-root-reanalysis-v1', 'model': args.model.name,
              'seed': args.seed, 'simulations': args.simulations, 'parallel': args.parallel,
              'requestedCuda': os.environ.get('ORT_CUDA') == '1', 'sources': sources,
              'maxPositions': args.max_positions, 'offset': args.offset, 'count': len(ids),
              'positions': 0, 'immediateWins': 0, 'rootVisits': 0, 'rootEdgeVisits': 0,
              'tacticalProof': True, 'rootNoise': False, 'shards': []}
    with tempfile.TemporaryDirectory(dir=args.output.parent, prefix='.reanalysis-') as temporary:
        staging = Path(temporary) / 'positions'
        staging.mkdir()
        with subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True) as worker:
            try:
                hello = response(worker, 'ready')
                if (hello['simulations'], hello['seed'], hello['parallel']) != (args.simulations, args.seed, args.parallel):
                    raise ValueError('Worker configuration mismatch')
                report['modelSha256'] = hello['modelSha256']
                print(f'Model SHA-256 {hello["modelSha256"]}; cap {args.simulations} simulations', flush=True)
                for start in range(0, len(ids), args.shard_size):
                    features = states[start:start + args.shard_size]
                    source_rows = ids[start:start + len(features)]
                    request = {'ids': source_rows.tolist(), 'x': base64.b64encode(features).decode('ascii')}
                    worker.stdin.write(json.dumps(request, separators=(',', ':')) + '\n')
                    worker.stdin.flush()
                    reply = response(worker, 'targets')
                    p, v, visits, edges = targets(reply['targets'], features, args.simulations)
                    selected = np.asarray([row['move'] for row in reply['targets']])
                    if (selected.shape != (len(features),) or selected.dtype.kind not in 'iu'
                            or np.any((selected < 0) | (selected >= 81))
                            or not features[np.arange(len(features)), 162 + selected].all()):
                        raise ValueError('Worker selected an illegal or invalid action')
                    name = f'shard-{len(report["shards"]):05d}.npz'
                    path = staging / name
                    np.savez_compressed(path, x=features, p=p, v=v, visits=visits, edge_visits=edges,
                                        source_row=source_rows, selected_move=selected.astype(np.uint8))
                    report['shards'].append({'file': name, 'positions': len(features),
                                             'sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
                    report['positions'] += len(features)
                    report['immediateWins'] += int(np.count_nonzero(visits == 0))
                    report['rootVisits'] += int(visits.sum())
                    report['rootEdgeVisits'] += int(edges.sum())
                    report['inference'] = reply['inference']
                    print(f'{report["positions"]:,}/{len(ids):,} roots, {report["rootVisits"]:,} visits, '
                          f'{report["immediateWins"]:,} immediate wins', flush=True)
                worker.stdin.close()
                if worker.wait():
                    raise RuntimeError('Reanalysis worker exited with an error')
            except BaseException:
                worker.terminate()
                raise
        (staging / 'manifest.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
        staging.rename(args.output)
    print(f'Published {report["positions"]:,} reanalysed positions to {args.output}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', required=True, type=Path, help='Compact NPZ shard or directory of shards')
    parser.add_argument('--model', required=True, type=Path, help='Project-format ONNX evaluator')
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--max-positions', type=int, default=8192, help='Global uniform sample size')
    parser.add_argument('--offset', type=int, default=0)
    parser.add_argument('--count', type=int, help='Maximum rows from the selected offset')
    parser.add_argument('--simulations', type=int, default=1024)
    parser.add_argument('--parallel', type=int, default=32)
    parser.add_argument('--seed', type=int, default=20261126)
    parser.add_argument('--shard-size', type=int, default=512)
    parser.add_argument('--node', default='node')
    args = parser.parse_args()
    if not 2 <= args.simulations <= 10000 or not 1 <= args.parallel <= 256 or not 0 <= args.seed <= 0xffffffff:
        parser.error('Choose 2..10000 simulations, 1..256 parallel roots, and an unsigned 32-bit seed')
    if min(args.max_positions, args.shard_size) < 1 or args.offset < 0 or (args.count is not None and args.count < 1):
        parser.error('Sample and shard sizes must be positive; offset must be nonnegative')
    run(args)


if __name__ == '__main__':
    main()
