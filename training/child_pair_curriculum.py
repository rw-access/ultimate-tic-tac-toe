r"""Prepare and diagnose a compact policy trained on independently searched sibling moves.

The pair margin uses the difference between the two fresh, parent-perspective
child values. The archived root estimates are used only to count reversals.
Families with a shared root or child under D4 are held out together.

The 500-step comparison started from the 1,200-step packed student documented
in training.regret_replay. SOURCE contains the completed original-engine
root and child shards, the search manifest identifies its 8,192-visit children,
and BACKGROUND is the clean mixture of deep search, Stage 2, proof, improved
action-value and replay targets. The fit schedule ends at step 4,000.

    python -m training.child_pair_curriculum prepare \
        --background BACKGROUND --targets SOURCE --search-manifest CHILDREN/manifest.json \
        --suite evaluation_uttt_states.py --output sibling-data \
        --min-margin .12 --visits 8192 --seed 26315812 --holdout-fraction .2
    python -m training.regret_replay fit --input parent-1200-raw.onnx --base moderate.pt \
        --data sibling-data/replay --output child-checkpoints --name child-student \
        --mix pairs .075 --mix siblings .075 --mix deep .28 --mix paper .24 \
        --mix proof .12 --mix action .15 --mix replay .06 --hard-fraction .25 \
        --margin .65 --teacher-scale .9 --anchor .35 --biases --lr .00003 \
        --value-weight .9 --batch-size 192 --seed 2026112941 --orbit-seed 5469684 \
        --guard --save-raw --at 500 1500 4000
    python -m training.child_pair_curriculum audit-history --data sibling-data \
        --replay PARENT_TRAINING --output history.json
    python -m training.child_pair_curriculum evaluate --data sibling-data \
        --model child-checkpoints/child-student-500.onnx --anchor parent-1200.onnx \
        --history-audit history.json

Pass --estimate leaf-value to score the two backed-up child values. The
default scores the root policy's preferences for the same two moves.
"""
import argparse
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import numpy as np

from .distill_oracle import children, validate
from .pipeline import _symmetry_maps
from .regret_replay import excluded_rows, filter_mined, immutable, stratified_hardness
from .search_alternatives import digest, keys, load_verified
from .suite_exclusion import Exclusion, PACKED_ORBITS_SHA256

MODELS = {
    '81161d2e262c9904dbb9a14961a2fcd043009fab4fd4f76b41451587f89c60d2',
    'b4f92e809e1eaa2053b202202d3853e1ed10be6d4c6980c2c46339f6e14f70b7',
}
COMMIT = '622dae66b18198f6f12aef0a0bccd5bd40de2d8e'


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def variants(x):
    return [keys(x[:, perm]) for perm, _ in _symmetry_maps()]


def family_components(roots, successors):
    """Union rows connected by any parent or child symmetry orbit."""
    n = len(roots)
    if roots.shape != (n, 280) or successors.shape != (2*n, 280) or n < 1:
        raise ValueError('Invalid family dimensions')
    states = np.column_stack((roots, successors.reshape(n, 560))).reshape(-1, 280)
    canonical = [min(orbit) for orbit in zip(*variants(states))]
    parent = list(range(n))

    def leader(i):
        while i != parent[i]:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    first = {}
    for j, signature in enumerate(canonical):
        row = j // 3
        other = first.setdefault(signature, row)
        parent[leader(row)] = leader(other)
    group = [leader(i) for i in range(n)]
    minimum = {}
    for j, signature in enumerate(canonical):
        component = group[j // 3]
        minimum[component] = min(minimum.get(component, signature), signature)
    return states, group, minimum


def family_split(roots, successors, seed, fraction):
    if not 0 < fraction < 1 or not 0 <= seed < 2**32:
        raise ValueError('Invalid family split fraction or seed')
    states, group, minimum = family_components(roots, successors)
    if len(minimum) < 2:
        raise ValueError('All sibling pairs belong to one symmetry component')
    salt = b'uttt-child-family-v1\0' + seed.to_bytes(4, 'big')
    scores = {c: int.from_bytes(hashlib.sha256(salt + k).digest()[:8], 'big') for c, k in minimum.items()}
    selected = {c for c, score in scores.items() if score < fraction * 2**64}
    if not selected or len(selected) == len(scores):
        raise ValueError('Seed gave an empty train or holdout partition')
    held = np.array([g in selected for g in group], bool)
    protected = sorted({k for orbit in variants(states[np.repeat(held, 3)]) for k in orbit})
    return held, np.frombuffer(b''.join(protected), np.uint8).reshape(-1, 35), len(minimum)


def rank_targets(moves, q, min_margin):
    gap = q[:, 0] - q[:, 1]
    cols = (gap < 0).astype(np.intp)
    rows = np.arange(len(moves))
    gain = np.where(np.abs(gap) >= min_margin, np.minimum(1., np.abs(gap) / .5), 0)
    return moves[rows, 1-cols].astype(np.uint8), moves[rows, cols].astype(np.uint8), gain.astype(np.float32)


def align_children(root_x, moves, cx, cv, q):
    if (moves.shape != (len(root_x), 2) or moves.dtype.kind not in 'ui' or
            (moves < 0).any() or (moves >= 81).any() or (moves[:, 0] == moves[:, 1]).any()):
        raise ValueError('Each root requires two distinct legal actions')
    successors, terminal, won = children(root_x, np.repeat(np.arange(len(moves)), 2), moves.ravel().astype(np.intp))
    index = {k: i for i, k in enumerate(keys(cx))}
    ids = np.array([index.get(k, -1) for k in keys(successors)])
    if (len(index) != len(cx) or set(ids[~terminal]) != set(range(len(cx))) or (ids[terminal] != -1).any()):
        raise ValueError('Child search did not cover exactly the requested live positions')
    expected = won.astype(np.float32)
    expected[~terminal] = -cv[ids[~terminal]]
    if not np.array_equal(q, expected.reshape(-1, 2)):
        raise ValueError('Pair values disagree with child-to-parent sign or terminal outcome')
    return successors, ids.reshape(-1, 2)


def read_source(args):
    manifest = json.loads((args.targets/'manifest.json').read_bytes())
    upstream = json.loads(args.search_manifest.read_bytes())
    if (manifest.get('format') != 'uttt-counterfactual-search-targets-v1' or
            manifest['childSearchSha256'] != digest(args.search_manifest) or
            upstream.get('format') != 'uttt-published-search-targets-v1' or
            upstream['modelSha256'] not in MODELS or upstream['upstreamCommit'] != COMMIT or
            upstream['visitsPerRoot'] != args.visits or upstream['roots'] != manifest['children'] or
            upstream['rootVisits'] != args.visits * upstream['roots']):
        raise ValueError('Search identity or budget differs from requested child supervision')
    parts = []
    specs = manifest['shards']
    if len(specs) != 2:
        raise ValueError('Expected root and child shards')
    for i, (key, spec) in enumerate(zip(('roots', 'children'), specs)):
        path = args.targets/f'shard-{i:05d}.npz'
        if path.name != spec['file'] or digest(path) != spec['sha256']:
            raise ValueError('Changed root or child shard')
        with np.load(path, allow_pickle=False) as data:
            x, p, v = (data[k] for k in ('x', 'p', 'v'))
        validate(x, p, v)
        if len(x) != spec['rows'] or len(x) != manifest[key]:
            raise ValueError('Root or child count mismatch')
        parts.append((x, p, v))
    # Match the finished labels against the original per-action visit receipts.
    searched = load_verified(args.search_manifest.parent, ('x', 'p', 'v', 'n'))
    index = {k: i for i, k in enumerate(keys(searched['x']))}
    child_x, child_p, child_v = parts[1]
    target_keys = keys(child_x)
    if len(index) != len(searched['x']) or set(index) != set(target_keys):
        raise ValueError('Independent search rows do not match the sibling shard')
    order = [index[k] for k in target_keys]
    n = searched['n'][order]
    if (n.shape != child_p.shape or n.dtype.kind not in 'iu' or (n < 0).any() or
            (n.sum(1) != args.visits-1).any() or (n[child_x[:, 162:243] == 0] != 0).any() or
            not np.array_equal((n/(args.visits-1)).astype(np.float16), child_p) or
            not np.array_equal(searched['p'][order], child_p) or not np.array_equal(searched['v'][order], child_v)):
        raise ValueError('Sibling supervision disagrees with original fixed-visit child search')
    path = args.targets/'diagnostics'/'pair-evidence.npz'
    if digest(path) != manifest['pairEvidenceSha256']:
        raise ValueError('Changed pair diagnostics')
    with np.load(path, allow_pickle=False) as data:
        x, m, q, old, prior, disagree = (data[k] for k in
            ('x', 'moves', 'childValuesFromParent', 'rootActionValues', 'initialRootPolicy', 'engineDisagreement'))
    if not np.array_equal(x, parts[0][0]) or len(set(keys(x))) != len(x):
        raise ValueError('Pair diagnostics did not uniquely align with the root shard')
    validate(x, prior, parts[0][2])
    if (old.shape != (len(x), 2) or q.shape != old.shape or not (np.abs(old) <= 1).all() or
            not (np.abs(q) <= 1).all() or disagree.shape != (len(x),) or disagree.dtype != bool or
            int(disagree.sum()) != manifest['disagreementRoots']):
        raise ValueError('Malformed pair evidence')
    successors, ids = align_children(x, m, parts[1][0], parts[1][2], q)
    return parts, (m, q, old, disagree), successors, ids


def source_blob(x, p, v, chosen, better, gain):
    return dict(x=x, p=p, v=v, teacher=p, teacher_v=v, chosen=chosen, better=better, gain=gain,
                hard=stratified_hardness(x, gain))


def prepare(args, folder):
    parts, evidence, successors, child_ids = read_source(args)
    (x, p, v), (cx, cp, cv) = parts
    moves, q, old, disagree = evidence
    background = json.loads((args.background/'manifest.json').read_bytes())
    suite = Exclusion(args.suite)
    if background.get('excludedPackedSha256') != PACKED_ORBITS_SHA256 or not (
            suite.keep(x).all() and suite.keep(successors).all()):
        raise ValueError('Training inputs must exclude the published tournament suite')
    held, protected, components = family_split(x, successors, args.seed, args.holdout_fraction)
    # Live child policies and blended root search policies also include other moves.
    # Keep direct predecessors of held-out states out of the supplemental sources.
    suite_keys = np.frombuffer(b''.join(sorted(suite.features)), np.uint8).reshape(-1, 35)
    all_excluded = np.unique(np.concatenate((protected, suite_keys)), axis=0)
    blocked = lambda states: np.logical_or(*excluded_rows(states, all_excluded))
    child_blocked = blocked(cx)
    child_mask = (child_ids >= 0) & child_blocked[np.maximum(0, child_ids)]
    train = ~(held | blocked(x) | child_mask.any(1))
    child_indices = np.unique(child_ids[train])
    use_child = child_indices[child_indices >= 0]
    if min(train.sum(), held.sum(), len(use_child)) < 2:
        raise ValueError('Too few usable pairs or children after complete exclusions')

    excluded = folder/'held-orbits.npy'
    np.save(excluded, protected)
    replay = folder/'replay'
    filter_mined(SimpleNamespace(data=args.background, output=replay, published_suite=None, excluded=excluded))
    manifest = json.loads((replay/'manifest.json').read_bytes())
    # Retain the original suite declaration and record the additional disjoint split.
    manifest.update({k: background[k] for k in ('excludedSetSha256', 'excludedPackedSha256', 'excludedStates')})
    manifest['extraHeldOrbitsSha256'] = digest(excluded)
    if {'pairs', 'siblings'} & {s['name'] for s in manifest['sources']}:
        raise ValueError('New source names already exist in background replay')
    chosen, better, gain = rank_targets(moves[train], q[train], args.min_margin)
    root_blob = source_blob(x[train], p[train], v[train], chosen, better, gain)
    greedy = cp[use_child].argmax(1).astype(np.uint8)
    child_blob = source_blob(cx[use_child], cp[use_child], cv[use_child], greedy, greedy, np.zeros(len(greedy), np.float32))
    targets_sha = digest(args.targets/'manifest.json')
    for name, blob in [('pairs', root_blob), ('siblings', child_blob)]:
        path = replay/(name+'.npz')
        np.savez_compressed(path, **blob)
        manifest['sources'].append(dict(name=name, rows=len(blob['x']), sha256=digest(path),
                                        reference='independent-child-search', targetsSha256=targets_sha))
    write_json(replay/'manifest.json', manifest)
    held_path = folder/'held-pairs.npz'
    np.savez_compressed(held_path, x=x[held], moves=moves[held], q=q[held], old=old[held], disagree=disagree[held])
    return dict(format='uttt-child-pair-curriculum-v1', targetsSha256=targets_sha,
                childSearchSha256=digest(args.search_manifest), backgroundSha256=digest(args.background/'manifest.json'),
                heldSha256=digest(held_path), heldOrbitsSha256=digest(excluded), replaySha256=digest(replay/'manifest.json'),
                visitsPerChild=args.visits, minMargin=args.min_margin, seed=args.seed, holdoutFraction=args.holdout_fraction,
                families=components, rootPairs=len(x), usedTrainPairs=int(train.sum()), heldPairs=int(held.sum()),
                droppedTrainPairs=int((~held & ~train).sum()), usedTrainChildren=len(greedy),
                trainAboveMargin=int((gain > 0).sum()), heldAboveMargin=int((np.abs(np.diff(q[held], axis=1)) >= args.min_margin).sum()),
                rankReversals=int(((q[:, 0]-q[:, 1])*(old[:, 0]-old[:, 1]) < 0).sum()))


def audit_history(args):
    """Trim a frozen holdout when the starting checkpoint already saw related states."""
    receipt = json.loads((args.data/'manifest.json').read_bytes())
    file = args.data/'held-pairs.npz'
    held_sha = digest(file)
    if receipt['format'] != 'uttt-child-pair-curriculum-v1' or held_sha != receipt['heldSha256']:
        raise ValueError('Held-out data changed')
    with np.load(file, allow_pickle=False) as data:
        x, moves = data['x'], data['moves']
    successors = children(x, np.repeat(np.arange(len(x)), 2), moves.ravel().astype(np.intp))[0]
    states, groups, _ = family_components(x, successors)
    lookup = {key: groups[i//3] for view in variants(states) for i, key in enumerate(view)}
    low, high = states[:,:162].sum(1).min(), states[:,:162].sum(1).max()
    blocked = set()
    history = []
    def check(batch):
        for row in np.packbits(batch, axis=1):
            group = lookup.get(row.tobytes())
            if group is not None:
                blocked.add(group)
    for replay in args.replay:
        path = replay/'manifest.json'
        prior = json.loads(path.read_bytes())
        if prior.get('format') != 'uttt-regret-replay-v1':
            raise ValueError('Expected a mined training replay history')
        history.append(digest(path))
        for source in prior['sources']:
            shard = replay/(source['name']+'.npz')
            if not source['name'].isalnum() or digest(shard) != source['sha256']:
                raise ValueError('Historical training replay changed')
            with np.load(shard, allow_pickle=False) as blob:
                rows = blob['x']
            ply = rows[:,:162].sum(1)
            eligible = np.flatnonzero((ply >= int(low)-1) & (ply <= high))
            for i in range(0, len(eligible), 256):
                batch = rows[eligible[i:i+256]]
                check(batch)
                r, m = np.nonzero(batch[:,162:243])
                check(children(batch, r, m)[0])
    keep = np.array([g not in blocked for g in groups], bool)
    if len(set(history)) != len(history) or not keep.any():
        raise ValueError('Repeated training history or no independent held families')
    record = dict(format='uttt-child-pair-history-audit-v1', heldSha256=held_sha, historyManifests=history,
                  roots=len(groups), kept=int(keep.sum()), historicalOverlap=int((~keep).sum()),
                  excludedIndices=np.flatnonzero(~keep).tolist())
    immutable(args.output, lambda path: write_json(path, record))
    return {k: v for k, v in record.items() if k != 'excludedIndices'} | {'auditSha256': digest(args.output)}


def preference_metrics(q, first, anchor):
    """Scores are conditional on the supplied pair, with ties worth half."""
    advantage = q[:, 0] - q[:, 1]
    def scores(logit_difference):
        return .5 + .5 * np.sign(advantage) * np.sign(logit_difference)
    a, b = scores(first), scores(anchor)
    weights = np.abs(advantage)
    return dict(rows=len(a), correct=float(a.sum()), accuracy=float(a.mean()), anchorAccuracy=float(b.mean()),
                weightedAccuracy=float(np.average(a, weights=weights)), improved=int((a > b).sum()),
                worsened=int((a < b).sum()), meanPairRegret=float(np.average((1-a)*weights)),
                anchorMeanPairRegret=float(np.average((1-b)*weights)))


def leaf_preferences(session, x, moves):
    """Back up each successor once, including an exact terminal result."""
    states, terminal, won = children(x, np.repeat(np.arange(len(x)), 2), moves.ravel().astype(np.intp))
    parent = won.astype(np.float32)
    if (~terminal).any():
        values = session.run(['value'], {'state': states[~terminal].astype(np.float32)})[0]
        if values.shape != (int((~terminal).sum()),1) or not (np.abs(values) <= 1.00001).all():
            raise ValueError('Invalid candidate child values')
        parent[~terminal] = -values[:, 0]
    return parent.reshape(-1, 2)


def evaluate(args):
    import onnxruntime as ort
    receipt = json.loads((args.data/'manifest.json').read_bytes())
    path = args.data/'held-pairs.npz'
    if receipt.get('format') != 'uttt-child-pair-curriculum-v1' or digest(path) != receipt['heldSha256']:
        raise ValueError('Held-out data changed')
    with np.load(path, allow_pickle=False) as raw:
        x, moves, q, old, disagree = (raw[k] for k in ('x', 'moves', 'q', 'old', 'disagree'))
    def differences(model):
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = opts.inter_op_num_threads = 1
        sess = ort.InferenceSession(str(model), sess_options=opts, providers=['CPUExecutionProvider'])
        diffs = []
        for i in range(0, len(x), 128):
            batch = x[i:i+128].astype(np.float32)
            pair = moves[i:i+len(batch)]
            if getattr(args, 'estimate', 'policy') == 'leaf-value':
                selected = leaf_preferences(sess, batch.astype(np.uint8), pair)
            else:
                logits = sess.run(['policy'], {sess.get_inputs()[0].name: batch})[0]
                selected = np.take_along_axis(logits, pair, axis=1)
            if selected.shape != (len(batch), 2) or not np.isfinite(selected).all():
                raise ValueError('Non-finite candidate policy')
            diffs.append(selected[:, 0] - selected[:, 1])
        return np.concatenate(diffs)
    a, b = differences(args.model), differences(args.anchor)
    keep = np.ones(len(x), bool)
    if args.history_audit:
        audit = json.loads(args.history_audit.read_bytes())
        dropped = audit['excludedIndices']
        if (audit.get('format') != 'uttt-child-pair-history-audit-v1' or audit['heldSha256'] != digest(path) or
                audit['roots'] != len(x) or audit['historicalOverlap'] != len(dropped) or
                any(type(i) != int or not 0 <= i < len(x) for i in dropped) or len(set(dropped)) != len(dropped) or
                audit['kept'] != len(x)-len(dropped) or audit['kept'] < 1):
            raise ValueError('History audit did not match held pairs')
        keep[dropped] = False
    eligible = keep & (np.abs(q[:, 0]-q[:, 1]) >= receipt['minMargin'])
    report = dict(heldSha256=digest(path), modelSha256=digest(args.model), anchorSha256=digest(args.anchor),
                  minMargin=receipt['minMargin'], totalHeld=len(x), afterHistoryExclusion=int(keep.sum()))
    if getattr(args, 'estimate', 'policy') == 'leaf-value':
        report['estimate'] = 'backed-up-child-value'
    if args.history_audit:
        report['historyAuditSha256'] = digest(args.history_audit)
    reversal = ((q[:, 0]-q[:, 1])*(old[:, 0]-old[:, 1])) < 0
    for name, mask in [('aboveMargin', eligible), ('engineDisagreement', eligible & disagree),
                       ('reversedPrior', eligible & reversal)]:
        report[name] = preference_metrics(q[mask], a[mask], b[mask]) if mask.any() else None
        if mask.any() and getattr(args, 'estimate', 'policy') == 'leaf-value':
            target_difference = q[mask, 0] - q[mask, 1]
            report[name].update(gapMSE=float(np.mean((a[mask]-target_difference)**2)),
                                anchorGapMSE=float(np.mean((b[mask]-target_difference)**2)))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest='command', required=True)
    build = modes.add_parser('prepare')
    for key in ('background', 'targets', 'search-manifest', 'suite', 'output'):
        build.add_argument('--'+key, type=Path, required=True)
    build.add_argument('--visits', type=int, default=8192)
    build.add_argument('--min-margin', type=float, default=.12)
    build.add_argument('--holdout-fraction', type=float, default=.2)
    build.add_argument('--seed', type=int, default=26315812)
    check = modes.add_parser('evaluate')
    for key in ('data', 'model', 'anchor'):
        check.add_argument('--'+key, type=Path, required=True)
    check.add_argument('--history-audit', type=Path)
    check.add_argument('--estimate', choices=['policy','leaf-value'], default='policy')
    prior = modes.add_parser('audit-history', help='Exclude all families the warm start saw in specified earlier replay')
    prior.add_argument('--data', required=True, type=Path)
    prior.add_argument('--output', required=True, type=Path)
    prior.add_argument('--replay', required=True, type=Path, action='append')
    args = parser.parse_args()
    if args.command == 'audit-history':
        result = audit_history(args)
    elif args.command == 'evaluate':
        result = evaluate(args)
    else:
        if args.output.exists() or args.visits < 2 or not 0 < args.min_margin <= 2:
            parser.error('Choose a fresh directory, positive action margin, and at least two visits')
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(dir=args.output.parent) as tmp:
            folder = Path(tmp)/'result'
            folder.mkdir()
            result = prepare(args, folder)
            write_json(folder/'manifest.json', result)
            folder.rename(args.output)
    print(json.dumps(result, allow_nan=False), flush=True)


if __name__ == '__main__':
    main()
