r"""Transfer the eight-view predictions of a compact policy into a one-view student.

Prepare a mined replay directory for training.regret_replay fit. The immutable
labels average legal probabilities in original move order over every D4 view.
Independent root supervision qualifies supplemental action-ranking targets.
The trained checkpoint retains the original single-evaluation inference graph.

The 6,000-step single-view comparison started from the 1,200-step packed
student in training.regret_replay and its original five-source protected
replay. Both training stages use the same starting checkpoint.

    python -m training.consensus_replay --data protected --input parent-1200-raw.onnx \
        --base moderate.pt --output consensus --weight .85 --batch-size 256 \
        --suite evaluation_uttt_states.py
    python -m training.regret_replay fit --input parent-1200-raw.onnx --base moderate.pt \
        --data consensus --output symmetry-checkpoints --name symmetry-student \
        --mix deep .39 --mix exact .10 --mix paper .27 --mix oracle .18 --mix replay .06 \
        --hard-fraction .30 --margin .24 --teacher-scale 1.55 --anchor .20 --biases \
        --lr .000045 --value-weight .9 --batch-size 192 --seed 2026112961 \
        --orbit-seed 5469694 --guard --save-raw --at 600 2200 6000
"""
import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch

from .distill_oracle import validate
from .fit_search_student import file_hash
from .pipeline import _symmetry_maps
from .refine_packed import check_parity
from .regret_replay import excluded_rows, stratified_hardness, student_from
from .suite_exclusion import Exclusion, PACKED_ORBITS_SHA256, PUBLISHED_SUITE_SHA256


def view_maps(device):
    return [(torch.as_tensor(np.argsort(features), device=device), torch.as_tensor(moves, device=device))
            for features, moves in _symmetry_maps()]


@torch.inference_mode()
def consensus(model, x, maps):
    """Align each distribution before averaging; return every view's preferred move."""
    if len(maps) != 8:
        raise ValueError('Expected all eight symmetry views')
    ps, vs, preferred = [], [], []
    for features, moves in maps:
        state = x.index_select(1, features)
        logits, value = model(state)
        p = logits.float().masked_fill(state[:,162:243] == 0, -1e9).softmax(1).index_select(1, moves)
        ps.append(p)
        vs.append(value.float().flatten())
        preferred.append(p.argmax(1))
    prob = torch.stack(ps)
    values = torch.stack(vs)
    average = prob.mean(0)
    js = (prob*(prob.clamp_min(1e-30).log() - average.clamp_min(1e-30).log())).sum(-1).mean(0)
    return [t.cpu().numpy() for t in (average, values.mean(0), js, values.std(0, unbiased=False),
                                      torch.stack(preferred, dim=1))]


def supported_pairs(target, ensemble, preferred, exact):
    """Prefer a consensus move only where the independent root target also supports it."""
    n = len(target)
    rows = np.arange(n)[:,None]
    best = (target if exact else ensemble).argmax(1)
    source = target[rows, best[:,None]] - target[rows, preferred]
    gap = ensemble[rows, best[:,None]] - ensemble[rows, preferred]
    if exact:
        # Every positive-mass proof action is optimal, including low-prior ties.
        signal = (target[rows, preferred] == 0).astype(np.float32)
    else:
        signal = np.where((source >= .04) & (gap >= .03), np.minimum(1., 2*np.minimum(source,gap)), 0)
    col = signal.argmax(1)
    actual = np.arange(n)
    gain = signal[actual, col].astype(np.float32)
    worse = np.where(gain > 0, preferred[actual, col], best)
    return worse.astype(np.uint8), best.astype(np.uint8), gain


def build(args, folder):
    source = json.loads((args.data/'manifest.json').read_bytes())
    if source.get('format') != 'uttt-regret-replay-v1' or source.get('base') != file_hash(args.base):
        raise ValueError('Expected mined replay with the matching base model')
    # Earlier mining receipts recorded the serialized exclusion file digest
    # instead of the packed bytes digest. Verify the actual rows in either format.
    suite = Exclusion(args.suite)
    excluded = np.frombuffer(b''.join(sorted(suite.features)), np.uint8).reshape(-1, 35)
    device = torch.device(args.device)
    packed, model, _ = student_from(args.input, args.base, device)
    fixtures = json.loads(args.fixtures.read_bytes())['cases'][:12]
    states = torch.tensor([c['encoded'] for c in fixtures], dtype=torch.float32)
    parity = check_parity(packed.SerializeToString(), model, states, 1)
    model.to(device)
    maps = view_maps(device)
    ref_sha = file_hash(args.input)
    report = {k:v for k,v in source.items() if k != 'sources'}
    report.update(student=ref_sha, originalMiningStudent=source['student'], frozenConsensusSha256=ref_sha,
                  referenceMixture=args.weight, consensusViews=8, sourceManifestSha256=file_hash(args.data/'manifest.json'),
                  consensusParity=parity, excludedSetSha256=PUBLISHED_SUITE_SHA256,
                  excludedPackedSha256=PACKED_ORBITS_SHA256, excludedStates=len(excluded), sources=[])
    for spec in source['sources']:
        name = spec['name']
        path = args.data/(name+'.npz')
        if not name.isalnum() or file_hash(path) != spec['sha256']:
            raise ValueError('Changed source replay')
        with np.load(path, allow_pickle=False) as data:
            d = {k:data[k] for k in data.files}
        validate(d['x'], d['p'], d['v'])
        x = d['x']
        root_hit, child_hit = excluded_rows(x, excluded)
        if root_hit.any() or child_hit.any():
            raise ValueError('Replay contains a published suite position or one-move predecessor')
        js, sd = np.empty(len(x), np.float32), np.empty(len(x), np.float32)
        changed = 0
        for start in range(0, len(x), args.batch_size):
            ix = slice(start, start+args.batch_size)
            p, v, js[ix], sd[ix], views = consensus(model, torch.from_numpy(x[ix]).to(device).float(), maps)
            original = d['p'][ix].astype(np.float32)
            original /= original.sum(1, keepdims=True)
            chosen, better, gain = supported_pairs(original, p, views, name in ('exact', 'proof'))
            ref = d['teacher'][ix].astype(np.float32)
            ref /= ref.sum(1, keepdims=True)
            d['teacher'][ix] = (1-args.weight)*ref + args.weight*p
            d['teacher_v'][ix] = (1-args.weight)*d['teacher_v'][ix] + args.weight*v
            d['chosen'][ix], d['better'][ix], d['gain'][ix] = chosen, better, gain
            changed += int((views != p.argmax(1)[:, None]).sum())
        difficulty = stratified_hardness(x, js+.15*sd)
        d['hard'] = stratified_hardness(x, .6*difficulty + .25*d['hard'] + .15*d['gain'])
        if not all(np.isfinite(arr).all() for arr in d.values()):
            raise ValueError('Non-finite consensus labels')
        out = folder/(name+'.npz')
        np.savez_compressed(out, **d)
        info = dict(name=name, rows=len(x), sourceSha256=spec['sha256'], sha256=file_hash(out),
                    meanJS=float(js.mean()), meanValueSpread=float(sd.mean()), meanGreedyDisagreement=changed/(8*len(x)),
                    supportedPairFraction=float((d['gain'] > 0).mean()))
        print(json.dumps(info), flush=True)
        report['sources'].append(info)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('data','input','base','suite','output'):
        parser.add_argument('--'+key, required=True, type=Path)
    parser.add_argument('--device', choices=['cpu','cuda'], default='cuda')
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--weight', type=float, default=.85, help='Fraction of the reference teacher replaced by D4 consensus')
    parser.add_argument('--fixtures', type=Path, default=Path(__file__).with_name('parity.json'))
    args = parser.parse_args()
    if args.output.exists() or not 0 < args.weight <= 1 or args.batch_size < 1:
        parser.error('Use a fresh output, positive batch size, and consensus weight in (0,1]')
    torch.set_num_threads(1)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=args.output.parent) as tmp:
        folder = Path(tmp)/'data'
        folder.mkdir()
        result = build(args, folder)
        (folder/'manifest.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
        folder.rename(args.output)
    print(json.dumps({k:v for k,v in result.items() if k != 'sources'}), flush=True)


if __name__ == '__main__':
    main()
