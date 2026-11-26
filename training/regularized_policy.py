"""Fit a move distribution to action values while retaining a supplied legal prior.

The reverse-KL policy update follows Grill et al., ICML 2020, section 3.
All arrays use the same legal-action order. A terminal outcome or a network's
one-ply value can supply a leaf estimate when search has few visits to a move.

Run python -m training.regularized_policy --search SEARCH --model ORIGINAL.onnx
--output TARGETS to convert checked original-engine reanalysis shards.
"""
import numpy as np


def shrink_action_values(counts, searched, leaf, pseudo_visits):
    """Average searched and leaf estimates using the visit count of each move."""
    n = np.asarray(counts)
    searched = np.asarray(searched, dtype=np.float64)
    leaf = np.asarray(leaf, dtype=np.float64)
    if (n.shape != searched.shape or searched.shape != leaf.shape or n.ndim != 2
            or n.dtype.kind not in 'ui' or (n < 0).any()
            or not np.isfinite(pseudo_visits) or pseudo_visits <= 0):
        raise ValueError('Expected aligned action estimates and positive pseudo-visits')
    if not np.isfinite(searched).all() or not np.isfinite(leaf).all():
        raise ValueError('Non-finite action value')
    share = n / (n.astype(np.float64) + pseudo_visits)
    return leaf + share * (searched - leaf)


def posterior(prior, action_values, legal, regularization):
    """Maximize expected action value minus regularization times KL(prior || policy)."""
    prior = np.asarray(prior, dtype=np.float64)
    values = np.asarray(action_values, dtype=np.float64)
    legal = np.asarray(legal)
    if (prior.ndim != 2 or prior.shape != values.shape or legal.shape != values.shape
            or legal.dtype != np.bool_ or not legal.any(1).all()):
        raise ValueError('Expected two-dimensional arrays with at least one legal action per row')
    if (not np.isfinite(regularization) or regularization <= 0 or not np.isfinite(values).all()
            or not np.isfinite(prior).all() or (prior[legal] <= 0).any() or (prior[~legal] != 0).any()
            or not np.allclose(prior.sum(1), 1, atol=1e-6, rtol=1e-6)):
        raise ValueError('Expected a positive regularizer and a normalized prior on legal moves')

    prior = prior / prior.sum(1, keepdims=True)
    best = np.where(legal, values, -np.inf).max(1, keepdims=True)
    gap = np.where(legal, best - values, 0.)
    # Solve for the excess of the Lagrange multiplier over the largest Q.
    # Using the excess avoids cancellation for a low-prior best action.
    # Work in units of regularization. A log-space interval resolves a
    # best action with a tiny prior without depending on its absolute scale.
    lo = np.log(np.where(legal & (gap == 0), prior, 0.).sum(1, keepdims=True))
    hi = np.zeros_like(lo)
    gap = gap / regularization
    for _ in range(52):
        middle = (lo + hi) * .5
        total = (prior / (np.exp(middle) + gap)).sum(1, keepdims=True)
        lo = np.where(total > 1., middle, lo)
        hi = np.where(total > 1., hi, middle)
    answer = prior / (np.exp((lo + hi) * .5) + gap)
    return answer / answer.sum(1, keepdims=True)


def prior_and_leaves(network, states):
    """Return a legal policy prior and one-ply values in the root player's perspective."""
    from .distill_oracle import children

    if states.ndim != 2 or states.shape[1] != 280:
        raise ValueError('Expected encoded game positions')
    legal = states[:, 162:243].astype(bool)
    if not legal.any(1).all():
        raise ValueError('Expected at least one legal move in each position')
    logits = np.asarray(network(states, False), dtype=np.float64)
    if logits.shape != legal.shape or not np.isfinite(logits).all():
        raise ValueError('Invalid root policy logits')
    masked = np.where(legal, logits, -np.inf)
    prior = np.exp(masked - masked.max(1, keepdims=True))
    prior /= prior.sum(1, keepdims=True)
    prior = np.where(legal, np.maximum(prior, 1e-12), 0.)
    prior /= prior.sum(1, keepdims=True)

    rows, moves = np.nonzero(legal)
    child, terminal, won = children(states, rows, moves)
    estimate = won.astype(np.float64)
    if (~terminal).any():
        estimate[~terminal] = -network(child[~terminal], True)
    leaf = np.zeros(legal.shape)
    leaf[rows, moves] = estimate
    return prior, leaf


def relabel_batch(network, states, counts, searched, pseudo_visits=2., regularization=.04, mix=.55):
    """Combine empirical search visits and a regularized policy on each root's legal moves."""
    if not 0. <= mix <= 1. or states.ndim != 2 or states.shape[1] != 280:
        raise ValueError('Expected encoded game positions and a mixing fraction in [0, 1]')
    legal = states[:, 162:243].astype(bool)
    if (counts.shape != legal.shape or searched.shape != legal.shape or (counts[~legal] != 0).any()
            or (counts.sum(1) <= 0).any() or not legal.any(1).all()):
        raise ValueError('Search counts must cover the legal moves of each position')
    prior, leaf = prior_and_leaves(network, states)
    values = shrink_action_values(counts, searched, leaf, pseudo_visits)
    improved = posterior(prior, values, legal, regularization)
    empirical = counts / counts.sum(1, keepdims=True)
    return (1. - mix) * empirical + mix * improved


def main():
    import argparse
    import json
    from pathlib import Path
    from tempfile import TemporaryDirectory

    from .distill_oracle import validate
    from .policy_arena import Network
    from .search_alternatives import digest

    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('search', 'model', 'output'):
        parser.add_argument('--' + name, required=True, type=Path)
    parser.add_argument('--pseudo-visits', type=float, default=2.)
    parser.add_argument('--regularization', type=float, default=.04)
    parser.add_argument('--mix', type=float, default=.55)
    parser.add_argument('--batch', type=int, default=96)
    args = parser.parse_args()
    if (args.batch < 1 or not 0. <= args.mix <= 1. or not np.isfinite(args.pseudo_visits)
            or args.pseudo_visits <= 0 or not np.isfinite(args.regularization) or args.regularization <= 0
            or args.output.exists()):
        parser.error('Use a fresh output directory, positive parameters, and a mixing fraction in [0, 1]')

    manifest_path = args.search / 'manifest.json'
    source = json.loads(manifest_path.read_bytes())
    model = Network(args.model, 1, args.batch)
    if (source.get('format') != 'uttt-published-search-targets-v1' or source['modelSha256'] != model.sha256
            or not source['shards'] or source['visitsPerRoot'] < 2):
        raise ValueError('Expected an original-engine search and its matching prior network')
    receipt = dict(format='uttt-regularized-training-targets-v1', searchManifestSha256=digest(manifest_path),
                   modelSha256=model.sha256, visitsPerRoot=source['visitsPerRoot'],
                   pseudoVisits=args.pseudo_visits, regularization=args.regularization, mix=args.mix,
                   roots=0, shards=[])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=args.output.parent) as temp:
        folder = Path(temp) / 'completed'
        folder.mkdir()
        for idx, shard in enumerate(source['shards']):
            file = args.search / shard['file']
            if file.parent != args.search or digest(file) != shard['sha256']:
                raise ValueError('Search shard does not match the source manifest')
            with np.load(file, allow_pickle=False) as arrays:
                x, p, v, n, q = [arrays[k] for k in ('x', 'p', 'v', 'n', 'q')]
            validate(x, p, v)
            edges = source['visitsPerRoot'] - 1
            if (len(x) != shard['rows'] or len(x) == 0 or n.shape != p.shape or (n.sum(1) != edges).any()
                    or not np.array_equal((n / edges).astype(np.float16), p)):
                raise ValueError('Root visits or move frequencies do not match the source manifest')
            targets = []
            for i in range(0, len(x), args.batch):
                sl = slice(i, i + args.batch)
                targets.append(relabel_batch(model, x[sl], n[sl], q[sl], args.pseudo_visits,
                                             args.regularization, args.mix))
            target = np.concatenate(targets).astype(np.float16)
            validate(x, target, v)
            out = folder / f'shard-{idx:05d}.npz'
            # Values stay in the root player's perspective used by the original search.
            np.savez_compressed(out, x=x, p=target, v=v.astype(np.float32))
            receipt['shards'].append(dict(file=out.name, rows=len(x), sha256=digest(out)))
            receipt['roots'] += len(x)
        if receipt['roots'] != source['roots'] or source['rootVisits'] != source['roots'] * source['visitsPerRoot']:
            raise ValueError('Total root count or search work differs from the manifest')
        (folder / 'manifest.json').write_text(json.dumps(receipt, indent=2, allow_nan=False) + '\n')
        folder.rename(args.output)
    print(f"Relabeled {receipt['roots']:,} roots from {len(receipt['shards'])} verified search shards.")


if __name__ == '__main__':
    main()
