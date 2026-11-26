"""Solve late middlegames and label plausible alternative moves from both player perspectives.

Only positions with every legal action proved enter the output. For a solved root,
explore its most plausible inferior move and one optimal move. Prove those resulting
positions in their own right, checking that their minimax values reverse sign.
"""

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from training.distill_oracle import children, distribution, ort_evaluator, validate
from training.exact_targets import classify, proved_targets
from training.suite_exclusion import Exclusion


def remaining(states):
    open_boards = ~states[:, 243:270].reshape(-1, 3, 9).any(1)
    empty = ~(states[:, :81] | states[:, 81:162]).astype(bool)
    return (empty.reshape(-1, 9, 9).sum(2) * open_boards).sum(1)


def alternative_moves(classes, prior, rng):
    ranks = np.full(classes.shape, -2, dtype=np.int8)
    for token, value in zip(b'LDW', (-1, 0, 1)):
        ranks[classes == token] = value
    best = ranks.max(1, keepdims=True)
    if (best < -1).any():
        raise ValueError('Solved root has no legal moves')
    choices = []
    for allowed in (ranks == best, (ranks >= -1) & (ranks < best)):
        rows = np.flatnonzero(allowed.any(1))
        # The first preference is the incumbent policy, with seeded tie-breaking.
        # Adding the noise after log prevents it from overwhelming small probabilities.
        weights = np.log(np.maximum(prior[rows].astype(np.float64), 1e-12))
        moves = np.where(allowed[rows], weights + 1e-6 * rng.random(weights.shape), -np.inf).argmax(1)
        choices.append((rows, moves, ranks[rows, moves]))
    return tuple(np.concatenate(fields) for fields in zip(*choices))


def generate(args):
    exclusion = Exclusion(args.excluded_suite)
    paths = sorted(args.data.glob('shard-*.npz'))
    if args.output.exists() or not paths:
        raise ValueError('Expected compact training shards and a fresh output directory')
    manifest = json.loads((args.data / 'manifest.json').read_bytes())
    hashes = {entry['file']: entry['sha256'] for entry in manifest['shards']}
    if len(hashes) != len(manifest['shards']) or set(hashes) != {path.name for path in paths}:
        raise ValueError('Input shards differ from the source manifest')
    rng = np.random.default_rng(args.seed)
    eligible = []
    for path in paths:
        if hashlib.sha256(path.read_bytes()).hexdigest() != hashes[path.name]:
            raise ValueError('Source shard digest mismatch')
        with np.load(path, allow_pickle=False) as shard:
            states = shard['x']
            left = remaining(states)
            eligible.append(np.flatnonzero((left >= args.min_remaining) & (left <= args.max_remaining)
                                           & (states[:, 162:243].sum(1) > 1) & exclusion.keep(states)))
    sizes = np.array([len(picks) for picks in eligible])
    count = min(args.roots, sizes.sum())
    if not count:
        raise ValueError('No eligible late middlegames')
    indices = np.sort(rng.choice(sizes.sum(), count, replace=False))
    evaluate = ort_evaluator(args.teacher, 'cpu', args.teacher_threads)
    log = dict(sourceManifestSha256=hashlib.sha256((args.data / 'manifest.json').read_bytes()).hexdigest(),
               sourceTeacherSha256=hashlib.sha256(args.teacher.read_bytes()).hexdigest(), seed=args.seed,
               solverSha256=hashlib.sha256(args.solver.read_bytes()).hexdigest(),
               excludedSuiteSha256=hashlib.sha256(args.excluded_suite.read_bytes()).hexdigest(),
               minRemaining=args.min_remaining, maxRemaining=args.max_remaining, nodesPerRoot=args.nodes,
               eligible=int(sizes.sum()), attemptedRoots=int(count), solvedRoots=0,
               attemptedChildren=0, solvedChildren=0, terminalChildren=0, nodes=0, shards=[])
    args.output.parent.mkdir(exist_ok=True, parents=True)
    with TemporaryDirectory(dir=args.output.parent) as temporary:
        staging = Path(temporary) / 'positions'
        staging.mkdir()
        with subprocess.Popen([str(args.solver.resolve()), str(args.nodes), str(args.threads)],
                              stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True) as worker:
            try:
                base = 0
                for path, picks in zip(paths, eligible):
                    selected = picks[indices[(indices >= base) & (indices < base + len(picks))] - base]
                    base += len(picks)
                    if not len(selected):
                        continue
                    with np.load(path, allow_pickle=False) as shard:
                        x, p, v = [shard[key][selected] for key in ('x', 'p', 'v')]
                    validate(x, p, v)
                    rows = []
                    for start in range(0, len(x), args.batch):
                        states, prior = x[start:start + args.batch], p[start:start + args.batch]
                        solved, classes, nodes = classify(worker, states)
                        log['nodes'] += nodes
                        log['solvedRoots'] += len(solved)
                        if not len(solved):
                            continue
                        states, prior = states[solved], prior[solved]
                        policy, value = proved_targets(classes, prior, .15)
                        rows.append((states, policy, value))
                        parents, moves, q = alternative_moves(classes, prior, rng)
                        child, terminal, win = children(states, parents, moves)
                        if not np.array_equal(q[terminal], win[terminal].astype(np.int8)):
                            raise ValueError('Terminal move disagrees with proved parent value')
                        log['terminalChildren'] += int(terminal.sum())
                        keep = ~terminal & exclusion.keep(child)
                        child, known = child[keep], -q[keep]
                        if not len(child):
                            continue
                        log['attemptedChildren'] += len(child)
                        done, labels, nodes = classify(worker, child)
                        log['nodes'] += nodes
                        if not len(done):
                            continue
                        child = child[done]
                        child_logits, _ = evaluate(child)
                        child_prior = distribution(child_logits, child[:, 162:243].astype(bool))
                        cp, cv = proved_targets(labels, child_prior, .15)
                        if not np.array_equal(cv, known[done]):
                            raise ValueError('Full child proof disagrees with parent action outcome')
                        rows.append((child, cp, cv))
                        log['solvedChildren'] += len(done)
                    if rows:
                        sx, sp, sv = [np.concatenate(arrays) for arrays in zip(*rows)]
                        validate(sx, sp, sv)
                        out = staging / f'shard-{len(log["shards"]):05d}.npz'
                        np.savez_compressed(out, x=sx, p=sp, v=sv)
                        log['shards'].append(dict(file=out.name, rows=len(sx),
                                                  sha256=hashlib.sha256(out.read_bytes()).hexdigest()))
                    print(f"{log['solvedRoots']:,} roots and {log['solvedChildren']:,} children proved", flush=True)
                worker.stdin.close()
                if worker.wait():
                    raise RuntimeError('Proof process failed')
            finally:
                if worker.poll() is None:
                    worker.terminate()
        if not log['shards']:
            raise ValueError('No positions proved at this work limit')
        (staging / 'manifest.json').write_text(json.dumps(log, indent=2) + '\n')
        staging.rename(args.output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for field in ('data', 'teacher', 'solver', 'excluded-suite', 'output'):
        parser.add_argument('--' + field, type=Path, required=True)
    parser.add_argument('--roots', type=int, default=20000)
    parser.add_argument('--min-remaining', type=int, default=15)
    parser.add_argument('--max-remaining', type=int, default=26)
    parser.add_argument('--nodes', type=int, default=400000)
    parser.add_argument('--batch', type=int, default=64)
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--teacher-threads', type=int, default=1)
    parser.add_argument('--seed', type=int, default=20261126)
    args = parser.parse_args()
    if (not 1 <= args.min_remaining <= args.max_remaining <= 28 or not 1 <= args.batch <= 2048
            or not 1 <= args.threads <= 64 or min(args.roots, args.nodes, args.teacher_threads) < 1
            or args.seed < 0):
        parser.error('Invalid proof limits')
    generate(args)


if __name__ == '__main__':
    main()
