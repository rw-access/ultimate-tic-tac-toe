"""Create policy-improvement and child-value targets from Stage 2 per-action search values.

The public archive reports each action's value from the parent player's perspective.
A visit-dependent shrinkage term gives poorly explored actions less influence.
"""

import argparse
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from training.distill_oracle import children, distribution, validate


def reverse_kl_policy(prior, values, legal, regularization):
    """Maximize expected value minus regularization times KL(prior || policy)."""
    if not np.isfinite(regularization) or regularization <= 0 or not legal.any(1).all():
        raise ValueError('Expected positive regularization and legal moves on every row')
    weights = np.where(legal, np.maximum(prior.astype(np.float64), 1e-10), 0.)
    weights /= weights.sum(1, keepdims=True)
    # Subtract the best value before bisection to avoid cancellation near the pole.
    advantage = values.astype(np.float64) - np.where(legal, values, -np.inf).max(1, keepdims=True)
    advantage[~legal] = -np.inf
    low = np.zeros((len(prior), 1), dtype=np.float64)
    high = np.full_like(low, regularization)
    for _ in range(48):
        middle = (high + low) / 2
        mass = (regularization * weights / (middle - advantage)).sum(1, keepdims=True)
        low, high = np.where(mass > 1, middle, low), np.where(mass > 1, high, middle)
    result = regularization * weights / (high - advantage)
    return (result / result.sum(1, keepdims=True)).astype(np.float32)


def improved_targets(prior, backup, visits, q, root_value, legal, beta, shrink, mix,
                     improvement='exponential', regularization=.18):
    """Improve the incumbent's policy using supported action values and search visits."""
    counts = visits.astype(np.float32)
    value = root_value.reshape(-1, 1)
    confidence = counts / (counts + shrink)
    completed = value + confidence * (q - value)
    if improvement == 'reverse-kl':
        tilted = reverse_kl_policy(prior, completed, legal, regularization)
    elif improvement == 'exponential':
        log_prior = np.log(np.maximum(prior, np.finfo(np.float32).tiny))
        tilted = distribution(log_prior + beta * (completed - value), legal)
    else:
        raise ValueError(f'Unknown policy-improvement algorithm {improvement}')
    backup = backup.astype(np.float32) / backup.sum(axis=1, keepdims=True, dtype=np.float32)
    policy = (1 - mix) * backup + mix * tilted
    target = .5 * root_value + .5 * (policy * completed).sum(1)
    return policy, target


def sample_children(states, visits, q, rng, min_visits):
    """Select one well-explored alternative per root without favoring the most visited move."""
    supported = visits >= min_visits
    rows = np.flatnonzero(supported.any(axis=1))
    if not len(rows):
        return states[:0], q.reshape(-1)[:0], rows
    # Gumbel-max samples with mass proportional to the fourth root of visit count.
    weights = np.where(supported[rows], .25 * np.log(np.maximum(visits[rows].astype(np.float32), 1)), -np.inf)
    moves = np.argmax(weights + rng.gumbel(size=weights.shape), axis=1)
    child, terminal, _ = children(states, rows, moves)
    keep = ~terminal
    return child[keep], -q[rows[keep], moves[keep]], visits[rows[keep], moves[keep]]


def transform(args):
    import torch

    from training.fit_search_student import load_weights
    from training.stage2_student import Stage2Net

    if args.output.exists():
        raise ValueError('Output directory must be fresh')
    source = json.loads((args.data / 'manifest.json').read_text())
    paths = sorted(args.data.glob('shard-*.npz'))
    expected = {s['file']: s for s in source['shards']}
    if not paths or len(expected) != len(source['shards']) or {f.name for f in paths} != set(expected):
        raise ValueError('Action-value shards differ from their input manifest')
    exclusion = None
    if args.excluded_suite:
        from training.suite_exclusion import Exclusion
        exclusion = Exclusion(args.excluded_suite)
    torch.set_num_threads(2)
    model = Stage2Net(compact=False)
    load_weights(model, args.teacher)
    model.eval().to(args.device)

    @torch.inference_mode()
    def evaluate(x):
        probs, values = [], []
        for start in range(0, len(x), args.batch):
            batch = x[start:start + args.batch]
            logits, value = model(torch.from_numpy(batch).to(args.device, dtype=torch.float32))
            probs.append(distribution(logits.cpu().numpy(), batch[:, 162:243].astype(bool)))
            values.append(value.flatten().cpu().numpy())
        return np.concatenate(probs), np.concatenate(values)

    rng = np.random.default_rng(args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    receipt = {'format': 'uttt-action-value-targets-v1', 'seed': args.seed, 'beta': args.beta,
               'minVisits': args.min_visits, 'shrink': args.shrink, 'priorMix': args.mix,
               'rootRows': 0, 'childRows': 0, 'excludedRoots': 0, 'excludedChildren': 0, 'shards': [],
               'sourceManifestSha256': hashlib.sha256((args.data / 'manifest.json').read_bytes()).hexdigest(),
               'teacherCheckpointSha256': hashlib.sha256(args.teacher.read_bytes()).hexdigest()}
    receipt.update(improvement=args.improvement, regularization=args.regularization)
    if args.excluded_suite:
        receipt['excludedSuiteSha256'] = hashlib.sha256(args.excluded_suite.read_bytes()).hexdigest()
    with TemporaryDirectory(dir=args.output.parent, prefix='.q-targets-') as tmp:
        root = Path(tmp) / 'data'
        root.mkdir()
        for index, path in enumerate(paths):
            if hashlib.sha256(path.read_bytes()).hexdigest() != expected[path.name]['sha256']:
                raise ValueError('Action-value shard digest differs from manifest')
            with np.load(path, allow_pickle=False) as s:
                x, p, v, n, q = (s[k] for k in ('x', 'p', 'v', 'n', 'q'))
            if len(x) != expected[path.name]['rows']:
                raise ValueError('Action-value row count differs from manifest')
            if exclusion is not None:
                keep = exclusion.keep(x)
                receipt['excludedRoots'] += int((~keep).sum())
                x, p, v, n, q = (arr[keep] for arr in (x, p, v, n, q))
            if not len(x):
                continue
            validate(x, p, v)
            legal = x[:, 162:243].astype(bool)
            if (n.shape != p.shape or q.shape != p.shape or n.dtype != np.uint16
                    or (n[~legal] != 0).any() or not np.isfinite(q).all() or (np.abs(q) > 1).any()):
                raise ValueError('Invalid per-action search labels')
            prior, _ = evaluate(x)
            targets, values = improved_targets(prior, p, n, q, v, legal, args.beta, args.shrink, args.mix,
                                               args.improvement, args.regularization)
            cx, cv, cn = sample_children(x, n, q, rng, args.min_visits)
            if exclusion is not None and len(cx):
                keep = exclusion.keep(cx)
                receipt['excludedChildren'] += int((~keep).sum())
                cx, cv, cn = cx[keep], cv[keep], cn[keep]
            if len(cx):
                cp, incumbent_v = evaluate(cx)
                weight = cn.astype(np.float32) / (cn.astype(np.float32) + args.shrink)
                cv = weight * cv + (1 - weight) * incumbent_v
                combined = (np.concatenate((x, cx)), np.concatenate((targets, cp)), np.concatenate((values, cv)))
            else:
                combined = (x, targets, values)
            states, policies, results = combined
            validate(states, policies, results)
            output = root / f'shard-{index:05d}.npz'
            np.savez_compressed(output, x=states, p=policies.astype(np.float16), v=results.astype(np.float32))
            receipt['rootRows'] += len(x)
            receipt['childRows'] += len(cx)
            receipt['shards'].append({'file': output.name, 'rows': len(states),
                                      'sha256': hashlib.sha256(output.read_bytes()).hexdigest()})
            print(index + 1, '/', len(paths), 'shards;', receipt['rootRows'], 'roots', flush=True)
        if not receipt['shards']:
            raise ValueError('No training rows remain after exclusions')
        (root / 'manifest.json').write_text(json.dumps(receipt, indent=2) + '\n')
        root.rename(args.output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('data', 'teacher', 'output'):
        parser.add_argument('--' + key, type=Path, required=True)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    parser.add_argument('--batch', type=int, default=256)
    parser.add_argument('--seed', type=int, default=271126)
    parser.add_argument('--beta', type=float, default=7.)
    parser.add_argument('--improvement', choices=('exponential', 'reverse-kl'), default='exponential')
    parser.add_argument('--regularization', type=float, default=.18)
    parser.add_argument('--shrink', type=float, default=64.)
    parser.add_argument('--mix', type=float, default=.6)
    parser.add_argument('--min-visits', type=int, default=64)
    parser.add_argument('--excluded-suite', type=Path, help='Pinned published tournament positions to exclude in all D4 views')
    args = parser.parse_args()
    if (args.batch < 1 or args.min_visits < 1 or args.shrink <= 0 or not np.isfinite(args.shrink)
            or not np.isfinite(args.regularization) or args.regularization <= 0
            or not 0 <= args.mix <= 1 or not 0 <= args.beta <= 100 or args.seed < 0):
        parser.error('Invalid sampling or improvement parameters')
    transform(args)


if __name__ == '__main__':
    main()
