"""Fine-tune the block scales in a Tiny INT8 download while preserving its signed weights.

Each --source provides a directory, a maximum number of sampled rows, and its
batch contribution. The reported validation set is excluded, with all eight
symmetries and its immediate predecessors, before fitting any source.
"""
import argparse
import copy
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import onnx
from onnx import numpy_helper
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils import parametrize

from .fit_search_student import load_positions, load_weights
from .line_student import LineStudent
from .pack_weights import load_states, quantize_int8
from .pipeline import _symmetry_maps
from .refine_packed import check_parity, layout_hash
from .regret_replay import excluded_rows
from .search_alternatives import digest
from .stage2_student import Stage2Net
from .suite_exclusion import Exclusion


class BlockScale(nn.Module):
    def __init__(self, scales, codes, shape, transpose):
        super().__init__()
        self.register_buffer('base', torch.tensor(scales, dtype=torch.float32))
        self.register_buffer('codes', torch.tensor(codes, dtype=torch.int8))
        self.gain = nn.Parameter(torch.zeros_like(self.base))
        self.shape = shape
        self.transpose = transpose

    def scales(self):
        return self.base * self.gain.clamp(-.22, .22).exp()

    def forward(self, original):
        weight = (self.codes * self.scales()).flatten()[:original.numel()].reshape(self.shape)
        return weight.T if self.transpose else weight


def bind_scales(graph, model):
    initializers = {t.name: numpy_helper.to_array(t) for t in graph.graph.initializer}
    sources = {}
    for name, layer in model.named_modules():
        if isinstance(layer, nn.Linear):
            w = layer.weight.detach().cpu().numpy()
            for transposed, matrix in ((False, w), (True, w.T)):
                codes, _, _ = quantize_int8(np.ascontiguousarray(matrix), 8)
                key = (matrix.shape, codes.tobytes())
                if key in sources:
                    raise ValueError('Two source matrices have identical signed weights')
                sources[key] = (name, layer, transposed)

    bindings = {}
    used = set()
    for key, codes in initializers.items():
        if not key.endswith('/i8'):
            continue
        prefix = key.rsplit('/', 1)[0]
        shape = tuple(initializers[prefix + '/shape'].tolist())
        found = sources.get((shape, codes.tobytes()))
        if found is None:
            # The fixed local-line incidence matrices appear in the inference graph too.
            if shape not in ((9, 8), (8, 9), (9, 9)):
                raise ValueError(f'Unknown signed matrix {prefix} with shape {shape}')
            continue
        name, layer, transpose = found
        scale_name = prefix + '/scale'
        scales = initializers[scale_name]
        if (name in used or codes.dtype != np.int8 or codes.ndim != 2 or codes.shape[1] != 8
                or scales.shape != (len(codes), 1) or scales.dtype != np.float32
                or not np.isfinite(scales).all() or (scales <= 0).any()):
            raise ValueError(f'Invalid weight groups or scales for {name}')
        used.add(name)
        block = BlockScale(scales, codes, shape, transpose)
        parametrize.register_parametrization(layer, 'weight', block)
        bindings[scale_name] = block
    if not bindings or used != {name for name, layer in model.named_modules() if isinstance(layer, nn.Linear)}:
        raise ValueError('The packed graph and the Tiny checkpoint contain different learned matrices')
    return bindings


def predict(model, x):
    logits, value = model(x.float())
    return logits.float().masked_fill(x[:, 162:243] == 0, -1.e9).log_softmax(1), value.float().flatten()


@torch.inference_mode()
def validation(model, dev):
    ce, mse = [], []
    x, p, v = dev
    for start in range(0, len(x), 128):
        sl = slice(start, start + 128)
        lp, value = predict(model, x[sl])
        ce.extend((-(p[sl] * lp).sum(1)).cpu().tolist())
        mse.extend((v[sl] - value).square().cpu().tolist())
    return dict(policyCrossEntropy=float(np.mean(ce)), valueMSE=float(np.mean(mse)))


def run(args):
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    source = args.packed.read_bytes()
    graph = onnx.load_from_string(source)
    student = LineStudent().eval().requires_grad_(False)
    load_weights(student, args.checkpoint)
    bindings = bind_scales(graph, student)
    layout = layout_hash(graph, bindings)
    fixtures = torch.from_numpy(load_states(args.parity, args.parity_data, 128, args.seed))
    check_parity(source, student, fixtures, 1)
    fixed = copy.deepcopy(student).requires_grad_(False).to(args.device)
    teacher = Stage2Net(compact=False).eval().requires_grad_(False)
    load_weights(teacher, args.teacher)
    teacher.to(args.device)
    dx, dp, dv = load_positions(args.development, 8192, rng, 'cpu')
    symmetries = _symmetry_maps()
    protected = [np.frombuffer(b''.join(sorted(Exclusion(args.suite).features)), np.uint8).reshape(-1, 35)]
    protected.extend(np.packbits(dx.numpy()[:, perm], axis=1) for perm, _ in symmetries)
    guard = np.unique(np.concatenate(protected), axis=0)
    sources, pools = [], []
    for directory, limit, batch in args.source:
        identity = {}
        data = load_positions(Path(directory), int(limit), rng, 'cpu', identity)
        blocked, reaches = excluded_rows(data[0].numpy(), guard)
        keep = ~(blocked | reaches)
        if not keep.any():
            raise ValueError('A source has no eligible boards after exclusions')
        pools.append((tuple(a[keep].to(args.device) for a in data), int(batch)))
        sources.append(dict(rows=int(keep.sum()), batch=int(batch), manifest=identity.get('manifest_sha256'),
                            sample=identity['sample_sha256'], excludedRoots=int(blocked.sum()),
                            excludedParents=int(reaches.sum())))
    dp = dp.to(args.device).float()
    dev = (dx.to(args.device), dp / dp.sum(1, keepdim=True), dv.to(args.device))
    features, moves = [torch.tensor(np.argsort(np.stack(indices), axis=1), device=args.device)
                       for indices in zip(*symmetries)]
    baseline = validation(fixed, dev)
    student.to(args.device)
    gains = [b.gain for b in bindings.values()]
    opt = torch.optim.Adam(gains, lr=.0015)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.schedule_steps, eta_min=.00015)
    for step in range(args.steps):
        samples = []
        for data, count in pools:
            index = torch.randint(len(data[0]), (count,), device=args.device)
            samples.append([x[index] for x in data])
        x, p, v = [torch.cat(t) for t in zip(*samples)]
        poses = torch.randint(8, (len(x),), device=args.device)
        x = x.gather(1, features[poses])
        p = p.gather(1, moves[poses]).float()
        p /= p.sum(1, keepdim=True)
        with torch.no_grad():
            tp, tv = predict(teacher, x)
            bp, bv = predict(fixed, x)
            target = .45 * p + .4 * tp.exp() + .15 * bp.exp()
            target_value = .48 * v + .37 * tv + .15 * bv
        lp, value = predict(student, x)
        loss = -(target * lp).sum(1).mean() + 1.2 * F.mse_loss(value, target_value)
        loss += .003 * torch.stack([g.square().mean() for g in gains]).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        schedule.step()
        if (step + 1) % 500 == 0:
            print(f'{step + 1:,} updates, training loss {loss.detach().item():.5f}', flush=True)
    result = validation(student, dev)
    for t in graph.graph.initializer:
        if t.name in bindings:
            t.raw_data = bindings[t.name].scales().detach().cpu().numpy().astype('<f4').tobytes()
    payload = graph.SerializeToString()
    if len(payload) != len(source) or layout_hash(graph, bindings) != layout:
        raise ValueError('Packed size or bytes outside the selected scales changed')
    parity = check_parity(payload, student, fixtures, 1)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=args.output.parent) as tmp:
        folder = Path(tmp) / 'ready'
        folder.mkdir()
        model = folder / 'model.onnx'
        model.write_bytes(payload)
        info = dict(seed=args.seed, steps=args.steps, scheduleSteps=args.schedule_steps, sources=sources,
                    heldRows=len(dx), trainableScales=sum(w.numel() for w in gains), layoutSha256=layout,
                    baseSha256=digest(args.packed), modelSha256=digest(model), bytes=len(payload),
                    baseline=baseline, trained=result, parity=parity)
        (folder / 'report.json').write_text(json.dumps(info, indent=2, allow_nan=False) + '\n')
        folder.rename(args.output)
    return info


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('packed', 'checkpoint', 'teacher', 'development', 'suite', 'parity-data', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--parity', type=Path, default=Path(__file__).with_name('parity.json'))
    parser.add_argument('--source', action='append', nargs=3, required=True, metavar=('DIRECTORY', 'LIMIT', 'BATCH'))
    parser.add_argument('--steps', type=int, default=1200)
    parser.add_argument('--schedule-steps', type=int, default=5400)
    parser.add_argument('--seed', type=int, default=26316751)
    parser.add_argument('--device', choices=['cuda', 'cpu'], default='cuda')
    args = parser.parse_args()
    try:
        valid = all(min(int(limit), int(batch)) > 0 for _, limit, batch in args.source)
    except ValueError:
        valid = False
    if not valid or not 0 < args.steps <= args.schedule_steps or args.output.exists():
        parser.error('Expected positive sample and batch sizes, a valid endpoint and a fresh output directory')
    print(json.dumps(run(args), allow_nan=False), flush=True)


if __name__ == '__main__':
    main()
