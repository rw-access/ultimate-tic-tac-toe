"""Select roots for extra training search using only shallow-search observations.

Run python -m training.search_allocation --search SHALLOW --model ORIGINAL.onnx
--rule RULE.json --output SELECTION.npz. The archive records indices into the
source search in shard order. Continue selected roots from 128 to 512 visits.
"""
import argparse
import json
from pathlib import Path
from tempfile import NamedTemporaryFile
import os

import numpy as np

from .regularized_policy import posterior, prior_and_leaves, shrink_action_values


FEATURES = (
    'legalCount', 'ply', 'absRootValue', 'priorEntropy', 'searchEntropy', 'priorGap',
    'searchGap', 'posteriorGap', 'priorSearchJS', 'posteriorSearchJS', 'unvisitedPriorMass',
    'visitedFraction', 'searchedValueSD', 'leafSearchedDisagreement', 'priorSearchDisagree',
    'posteriorSearchDisagree', 'topLeafDisagrees', 'topTwoValueGap',
)


def divergence(first, second):
    """Jensen–Shannon divergence per board, using natural logarithms."""
    mid = (first + second) * .5
    log_mid = np.log(np.maximum(mid, 1e-30))
    def kl(prob):
        return (prob * (np.log(np.maximum(prob, 1e-30)) - log_mid)).sum(1)
    return (kl(first) + kl(second)) * .5


def observations(states, counts, searched, values, prior, leaf):
    """Compute the features available immediately after a shallow search."""
    legal = states[:, 162:243].astype(bool)
    if (states.ndim != 2 or states.shape[1] != 280 or counts.shape != legal.shape
            or values.shape != (len(states),) or (counts[~legal] != 0).any()
            or (counts.sum(1) != 127).any() or not np.isfinite(values).all()):
        raise ValueError('Expected 128-visit search results aligned with legal moves')
    visits = counts / counts.sum(1, keepdims=True)
    q = searched.astype(np.float64)
    adjusted = shrink_action_values(counts, q, leaf, 2.)
    improved = .55 * posterior(prior, adjusted, legal, .04) + .45 * visits
    def entropy(prob):
        return -(prob * np.log(np.maximum(prob, 1e-30))).sum(1)
    def margin(prob):
        return np.diff(np.sort(prob, axis=1)[:, -2:], axis=1)[:, 0]
    ranked = np.sort(np.where(legal, adjusted, -10.), axis=1)
    deviation = np.sqrt(np.maximum(0., (visits * q * q).sum(1) - (visits * q).sum(1) ** 2))
    selection = visits.argmax(1)
    return np.column_stack((
        np.log1p(legal.sum(1)), states[:, :162].sum(1) / 81., np.abs(values),
        entropy(prior), entropy(visits), margin(prior), margin(visits), margin(improved),
        divergence(prior, visits), divergence(improved, visits), (prior * (counts == 0)).sum(1),
        (counts > 0).sum(1) / legal.sum(1), deviation, (visits * np.abs(leaf - q)).sum(1),
        prior.argmax(1) != selection, improved.argmax(1) != selection,
        np.where(legal, leaf, -10.).argmax(1) != selection, ranked[:, -1] - ranked[:, -2],
    ))


def fit_rule(features, improvement):
    """Fit a ridge regression to the reduction in divergence from additional visits."""
    if (features.ndim != 2 or features.shape[1] != len(FEATURES) or len(features) < 2
            or improvement.shape != (len(features),) or not np.isfinite(features).all()
            or not np.isfinite(improvement).all()):
        raise ValueError('Expected aligned, finite calibration features and improvements')
    mean = features.mean(0)
    scale = np.maximum(features.std(0), 1e-8)
    z = (features - mean) / scale
    regularizer = .03 * len(z) * np.eye(z.shape[1])
    weights = np.linalg.solve(z.T @ z + regularizer, z.T @ (improvement - improvement.mean()))
    return dict(featureNames=list(FEATURES), featureMean=mean.tolist(), featureScale=scale.tolist(),
                coefficients=weights.tolist(), extraVisits=384, selectedFraction=.25)


def select_roots(features, rule):
    """Choose up to a quarter of each consecutive 64-root batch, stably breaking ties."""
    if (features.ndim != 2 or features.shape[1] != len(FEATURES) or not np.isfinite(features).all()
            or rule['featureNames'] != list(FEATURES) or rule['selectedFraction'] != .25 or rule['extraVisits'] != 384):
        raise ValueError('Features or search allocation settings differ from the fitted rule')
    mean, scale, coefficients = (np.asarray(rule[k], dtype=np.float64)
                                 for k in ('featureMean', 'featureScale', 'coefficients'))
    if (any(p.shape != (len(FEATURES),) or not np.isfinite(p).all() for p in (mean, scale, coefficients))
            or (scale <= 0).any()):
        raise ValueError('Invalid allocation coefficients or feature scales')
    selected = []
    for start in range(0, len(features), 64):
        batch = features[start:start + 64]
        score = (batch - mean) / scale @ coefficients
        selected.extend(start + np.argsort(-score, kind='stable')[:len(batch) // 4])
    return np.asarray(selected, dtype=np.int64)


def main():
    from .distill_oracle import validate
    from .policy_arena import Network
    from .search_alternatives import digest, load_verified

    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('search', 'model', 'rule', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    a = parser.parse_args()
    if a.output.exists():
        parser.error('Use a fresh output path')
    manifest = a.search / 'manifest.json'
    spec = json.loads(manifest.read_bytes())
    params = json.loads(a.rule.read_bytes())
    rule = params.get('trainingAllocationRule', params)
    network = Network(a.model, 1, 96)
    if (spec.get('format') != 'uttt-published-search-targets-v1' or spec.get('visitsPerRoot') != 128
            or spec.get('modelSha256') != network.sha256 or spec.get('rootVisits') != spec['roots'] * 128
            or not spec.get('shards')):
        raise ValueError('Expected a 128-visit original-engine search and its prior network')
    source = load_verified(a.search, ('x', 'p', 'v', 'n', 'q'))
    x = source['x']
    validate(x, source['p'], source['v'])
    if len(x) != spec['roots'] or not np.array_equal((source['n'] / 127).astype(np.float16), source['p']):
        raise ValueError('Source root count or policy differs from the search counts')
    feature_rows = []
    for start in range(0, len(x), 96):
        sl = slice(start, start + 96)
        prior, leaf = prior_and_leaves(network, x[sl])
        feature_rows.append(observations(x[sl], source['n'][sl], source['q'][sl], source['v'][sl], prior, leaf))
    selected = select_roots(np.concatenate(feature_rows), rule)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=a.output.parent) as tmp:
        np.savez_compressed(tmp, selected=selected, roots=np.int64(len(x)), sourceSha256=digest(manifest),
                            ruleSha256=digest(a.rule), totalVisits=np.int64(128 * len(x) + 384 * len(selected)))
        tmp.flush()
        os.link(tmp.name, a.output)
    print(f'Selected {len(selected):,} of {len(x):,} roots for 384 additional training visits.')


if __name__ == '__main__':
    main()
