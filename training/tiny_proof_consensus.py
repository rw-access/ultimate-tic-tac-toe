"""Fit a phase-gated tiny specialist from complete endgame proofs and two independent searches."""

import argparse
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from training.distill_oracle import validate
from training.fit_search_student import export, load_weights, save_checkpoint
from training.line_student import LineStudent
from training.pipeline import _symmetry_maps
from training.suite_exclusion import Exclusion, PUBLISHED_SUITE_SHA256


def digest(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


class ProofLine(LineStudent):
    """Frozen incumbent with bounded residual heads for the late middlegame and endgame."""

    def __init__(self):
        super().__init__()
        self.requires_grad_(False)
        self.late_policy = nn.Sequential(nn.Linear(88, 32), nn.ReLU(), nn.Linear(32, 1))
        self.late_value = nn.Sequential(nn.Linear(320, 48), nn.ReLU(), nn.Linear(48, 1))
        for head in (self.late_policy, self.late_value):
            nn.init.zeros_(head[-1].weight)
            nn.init.zeros_(head[-1].bias)

    def warm_start(self, weights):
        base = LineStudent()
        load_weights(base, weights)
        missing, extra = self.load_state_dict(base.state_dict(), strict=False)
        if extra or any(not name.startswith(('late_policy.', 'late_value.')) for name in missing):
            raise ValueError('Unexpected incumbent parameters')

    def predictions(self, state):
        action, summary, routing, available = self.representations(state)
        incumbent_policy = self.policy_head(action).reshape(-1, 81)
        incumbent_value = self.value_head[:-1](summary)
        # Remaining playable empties includes every open board regardless of the forced board.
        remaining = available.sum((1, 2)).unsqueeze(1)
        gate = ((48 - remaining) / 18).clamp(0, 1)
        correction = self.late_policy(torch.cat((action, routing), dim=-1)).reshape(-1, 81)
        policy = incumbent_policy + gate * (3.0 * (correction / 3.0).tanh())
        correction = self.late_value(summary)
        value = (incumbent_value + gate * (2.2 * (correction / 2.2).tanh())).tanh()
        return policy, value, incumbent_policy, incumbent_value.tanh(), gate

    def forward(self, state):
        return self.predictions(state)[:2]


def read_source(path, limit, rng):
    manifest = path / 'manifest.json'
    info = json.loads(manifest.read_bytes())
    paths = sorted(path.glob('*.npz'))
    entries = {e['file']: e for e in info.get('shards', [])}
    if (not paths or len(entries) != len(info.get('shards', []))
            or (entries and set(entries) != {p.name for p in paths})):
        raise ValueError('Shard inventory disagrees with manifest')
    sizes = []
    fingerprints = []
    for p in paths:
        h = digest(p)
        if entries and h != entries[p.name]['sha256']:
            raise ValueError(f'Shard digest mismatch in {p.name}')
        with np.load(p, allow_pickle=False) as data:
            length = len(data['v'])
        if entries:
            entry = entries[p.name]
            if length != entry.get('rows', entry.get('positions')):
                raise ValueError('Shard count mismatch')
        sizes.append(length)
        fingerprints.append((p.name, h))
    offsets = np.cumsum([0, *sizes])
    selected = np.sort(rng.choice(offsets[-1], min(offsets[-1], limit), replace=False))
    columns = [[] for _ in range(4)]
    for index, p in enumerate(paths):
        rows = selected[selected.searchsorted(offsets[index]):selected.searchsorted(offsets[index + 1])] - offsets[index]
        with np.load(p, allow_pickle=False) as data:
            x, policy, value = (data[key][rows] for key in ('x', 'p', 'v'))
        validate(x, policy, value)
        for col, array in zip(columns, (x, policy.astype(np.float16), value.astype(np.float32),
                                        np.full(len(x), index, np.int32))):
            col.append(array)
    source_id = hashlib.sha256(json.dumps(fingerprints).encode()).hexdigest()
    return tuple(np.concatenate(col) for col in columns), info, (digest(manifest), source_id)


def packed(x):
    bits = np.ascontiguousarray(np.packbits(x, axis=1, bitorder='big'))
    return bits.view('V35').reshape(-1)


def consensus(first, second):
    """Align by complete state, blend overlapping policies, and downweight disagreements."""
    x, p1, v1, file_ids = first
    y, p2, v2, _ = second
    a, b = packed(x), packed(y)
    if len(a) != len(b) or len(np.unique(a)) != len(a) or len(np.unique(b)) != len(b):
        raise ValueError('Independent searches must contain exactly the same unique roots')
    order = np.argsort(b)
    locations = np.searchsorted(b[order], a)
    if (locations >= len(b)).any() or not np.array_equal(a, b[order[locations]]):
        raise ValueError('Independent search states differ')
    rows = order[locations]
    p1, p2 = p1.astype(np.float32), p2[rows].astype(np.float32)
    p1 /= p1.sum(1, keepdims=True)
    p2 /= p2.sum(1, keepdims=True)
    intersection = np.sqrt(p1 * p2)
    overlap = intersection.sum(1, keepdims=True)
    mean = .5 * (p1 + p2)
    joint = np.where(overlap > 1e-7, intersection / np.maximum(overlap, 1e-7), mean)
    target = .65 * joint + .35 * mean
    v2 = v2[rows]
    policy_confidence = (.20 + .80 * overlap[:, 0] ** 2).astype(np.float32)
    value_confidence = np.clip(1 - .6 * np.abs(v1 - v2), .10, 1).astype(np.float32)
    return (x, target.astype(np.float16), (v1 + v2) * .5, file_ids,
            policy_confidence, value_confidence)


def normalized_loss(logits, value, incumbent, base_value, x, target, outcome, source, policy_conf, value_conf):
    logp = logits.float().masked_fill(x[:, 162:243] == 0, -1e9).log_softmax(1)
    reference = incumbent.float().masked_fill(x[:, 162:243] == 0, -1e9).softmax(1)
    p = target.float() / target.float().sum(1, keepdim=True)
    # Solver marks every optimal move, including all legal moves in a forced loss.
    mass_nll = -torch.logsumexp(logp.masked_fill(p <= 0, -1e9), 1)
    proof = source == 0
    paired = source == 1
    pc = torch.where(paired, .85 * policy_conf, .26)
    vc = torch.where(paired, .92 * value_conf, .30)
    soft = pc[:, None] * p + (1 - pc[:, None]) * reference
    soft_nll = -(soft * logp).sum(1)
    exact_nll = .90 * mass_nll - .10 * (p * logp).sum(1)
    policy_loss = torch.where(proof, exact_nll, soft_nll)
    desired = torch.where(proof, outcome, vc * outcome + (1 - vc) * base_value.float().flatten())
    value_loss = (value.float().flatten() - desired).square()
    return policy_loss, value_loss, mass_nll


def split_sets(args, rng):
    proof, pm, proof_id = read_source(args.proof, 120000, rng)
    first, sm, first_id = read_source(args.official, 100000, rng)
    second, fm, second_id = read_source(args.four_view, 100000, rng)
    replay, _, replay_id = read_source(args.replay, 48000, rng)
    if (pm.get('minRemaining') != 15 or pm.get('maxRemaining', 0) < 25
            or not pm.get('solvedChildren') or pm['solvedRoots'] + pm['solvedChildren'] != len(proof[0])
            or pm.get('excludedSuiteSha256') != PUBLISHED_SUITE_SHA256
            or not np.isin(proof[2], (-1., 0., 1.)).all()):
        raise ValueError('Expected fully solved extended parent and child outcomes')
    if (sm.get('format') != 'uttt-published-search-targets-v1' or sm.get('visitsPerRoot') != 3072
            or sm.get('rootVisits') != 3072 * len(first[0])
            or sm.get('modelSha256') != 'b4f92e809e1eaa2053b202202d3853e1ed10be6d4c6980c2c46339f6e14f70b7'
            or sm.get('upstreamCommit') != '622dae66b18198f6f12aef0a0bccd5bd40de2d8e'
            or fm.get('format') != 'uttt-independent-root-reanalysis-v1' or fm.get('simulations') != 1536
            or fm.get('modelSha256') != args.four_view_sha or fm.get('rootNoise') is not False):
        raise ValueError('Unexpected budget or model in independent search sources')
    paired = consensus(first, second)
    ex = Exclusion(args.excluded_suite)
    def unit(data):
        return (*data, np.ones(len(data[0]),np.float32), np.ones(len(data[0]),np.float32))
    train, held, stats = [], [], []
    for name, data in [('proof', unit(proof)), ('consensus', paired), ('replay', unit(replay))]:
        x, p, v, f, pc, vc = data
        keep = ex.keep(x)
        # Grouped proof shards keep parent/child families together. All fresh consensus
        # rows share only 16 physical shards. Reserve two whole source shards for testing.
        val = f % (8 if name == 'consensus' else 17) == 3
        arrays = x, p, v, pc, vc
        train.append(tuple(a[keep & ~val] for a in arrays))
        held.append(tuple(a[keep & val] for a in arrays))
        stats.append({'name': name, 'kept': int(keep.sum()), 'excludedSuite': int((~keep).sum())})
    # Guard all three training streams against D4 copies of externally reserved dev rows.
    held_x = np.concatenate([s[0] for s in held])
    forbidden = np.unique(np.concatenate([packed(held_x[:, m]) for m, _ in _symmetry_maps()]))
    for i, corpus in enumerate(train):
        keep = ~np.isin(packed(corpus[0]), forbidden)
        train[i] = tuple(arr[keep] for arr in corpus)
        if not len(train[i][0]) or not len(held[i][0]):
            raise ValueError('Insufficient records for train/validation split')
        stats[i].update(train=int(keep.sum()), validation=len(held[i][0]), removedDevelopmentOrbits=int((~keep).sum()))
    return train, held, {'dataSha256': [proof_id, first_id, second_id, replay_id], 'counts': stats}


def on_device(data, device):
    return tuple(torch.as_tensor(x, device=device) for x in data)


@torch.no_grad()
def validate_model(model, corpora):
    metric = []
    for data in corpora:
        logs = []
        for start in range(0, len(data[0]), 128):
            x, p, v = (entry[start:start + 128] for entry in data[:3])
            l, y = model(x.float())
            logp = l.float().masked_fill(x[:,162:243] == 0, -1e9).log_softmax(1)
            target = p.float() / p.float().sum(1, keepdim=True)
            correct = target.gather(1, logp.argmax(1, keepdim=True)).flatten() > 0
            mass = (logp.exp() * (target > 0)).sum(1)
            logs.append(torch.stack((-(target * logp).sum(1), (v-y.flatten()).square(), correct, mass), dim=1))
        metric.append(dict(zip(('policyNLL','valueMSE','top1InSupport','massOnSupport'),
                               torch.cat(logs).mean(0).cpu().tolist())))
    return metric


def fit(args):
    torch.set_num_threads(2)
    device = torch.device(args.device)
    if device.type == 'cuda':
        torch.cuda.set_per_process_memory_fraction(.15)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    if sorted(set(args.steps)) != args.steps or min(args.steps) < 1 or args.lr <= 0:
        raise ValueError('Expected positive learning rate and increasing snapshot steps')
    train, held, identity = split_sets(args, rng)
    identity.update(initialSha256=digest(args.initial), seed=args.seed, steps=args.steps, lr=args.lr,
                    stepCounts=[64, 16, 48], version='phase-gated-proof-consensus-v1')
    print('Frozen corpus',identity,flush=True)
    train, held = ([on_device(data, device) for data in group] for group in (train, held))
    model = ProofLine().to(device).eval()
    model.warm_start(args.initial)
    ema = copy.deepcopy(model).eval().requires_grad_(False)
    trainable = [p for p in model.parameters() if p.requires_grad]
    averaged = [p for name, p in ema.named_parameters() if name.startswith(('late_policy.', 'late_value.'))]
    opt = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps[-1], eta_min=.12*args.lr)
    with torch.no_grad():
        priorities = []
        for start in range(0, len(train[0][0]), 192):
            x, p, v = (arr[start:start+192] for arr in train[0][:3])
            lp, y = model(x.float())
            logp = lp.float().masked_fill(x[:,162:243] == 0, -1e9).log_softmax(1)
            priorities.append(-torch.logsumexp(logp.masked_fill(p <= 0, -1e9), 1) + .65 * (v-y.flatten()).abs())
        difficulty = torch.cat(priorities)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.state.parent.mkdir(parents=True, exist_ok=True)
    if not args.state.exists() and any(args.output.with_name(f'{args.output.name}-{s}.onnx').exists()
                                       for s in args.steps):
        raise ValueError('Existing model outputs require the matching resume state')
    baseline = validate_model(ema, held)
    print('Initial metrics',baseline,flush=True)
    finished = 0
    if args.state.exists():
        ckpt = torch.load(args.state, map_location='cpu', weights_only=True)
        if ckpt['identity'] != identity:
            raise ValueError('Resume identifies a different corpus or fit')
        model.load_state_dict(ckpt['model'])
        ema.load_state_dict(ckpt['ema'])
        opt.load_state_dict(ckpt['optimizer'])
        sch.load_state_dict(ckpt['scheduler'])
        torch.set_rng_state(ckpt['rng'])
        if device.type == 'cuda':
            torch.cuda.set_rng_state(ckpt['cudaRng'], device)
        finished = ckpt['step']
    def publish(step):
        path = args.output.with_name(f'{args.output.name}-{step}.onnx')
        report = path.with_suffix('.metrics.json')
        if not path.exists():
            export(copy.deepcopy(ema).cpu(), held[0][0][:24].cpu(), path, 'line_student')
        if not path.with_suffix('.pt').exists():
            raise ValueError('Incomplete model snapshot')
        if not report.exists():
            metrics = validate_model(ema, held)
            record = dict(identity=identity, step=step, initial=baseline, held=metrics, sha256=digest(path))
            with report.open('x') as f:
                json.dump(record,f,indent=2)
            print('Snapshot',step,metrics,flush=True)
    if finished in args.steps:
        publish(finished)
    counts = (64, 16, 48)
    sources = torch.repeat_interleave(torch.arange(3, device=device), torch.tensor(counts, device=device))
    for step in range(finished + 1, args.steps[-1] + 1):
        batches = []
        for source_index, (data, count) in enumerate(zip(train, counts)):
            indices = torch.randint(len(data[0]), (count,), device=device)
            if source_index == 0:
                hard_count = count // 2
                choices = torch.randint(len(data[0]), (hard_count, 4), device=device)
                hardest = difficulty[choices].argmax(1)
                indices[:hard_count] = choices[torch.arange(hard_count, device=device), hardest]
            batches.append(tuple(column[indices] for column in data))
        x, policy, outcome, policy_conf, value_conf = [torch.cat(items) for items in zip(*batches)]
        logits, value, base_logits, base_value, gate = model.predictions(x.float())
        policy_loss, value_loss, _ = normalized_loss(logits, value, base_logits, base_value, x, policy,
                                                     outcome, sources, policy_conf, value_conf)
        active = (gate.flatten() > 0).float()
        loss = ((policy_loss + 1.15 * value_loss) * active).sum() / active.sum().clamp_min(1)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1., error_if_nonfinite=True)
        opt.step()
        sch.step()
        with torch.no_grad():
            torch._foreach_lerp_(averaged, trainable, .008)
        if step == 1 or step % 250 == 0:
            print(f'Step {step}, loss {loss.item():.5f}', flush=True)
        if step % 500 == 0 or step in args.steps:
            save_checkpoint(args.state, {
                'identity': identity, 'step': step, 'model': model.state_dict(), 'ema': ema.state_dict(),
                'optimizer': opt.state_dict(), 'scheduler': sch.state_dict(), 'rng': torch.get_rng_state(),
                'cudaRng': torch.cuda.get_rng_state(device) if device.type == 'cuda' else None,
            })
        if step in args.steps:
            publish(step)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('proof', 'official', 'four-view', 'replay', 'initial', 'output', 'state', 'excluded-suite'):
        parser.add_argument('--' + key, type=Path, required=True)
    parser.add_argument('--four-view-sha', required=True)
    parser.add_argument('--steps', type=int, nargs='+', default=[1400, 5000])
    parser.add_argument('--seed', type=int, default=26316507)
    parser.add_argument('--lr', type=float, default=.0014)
    parser.add_argument('--device', choices=('cuda', 'cpu'), default='cuda')
    fit(parser.parse_args())


if __name__ == '__main__':
    main()
