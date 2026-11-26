r"""Fit the packed student's final value readout to independently searched sibling moves.

The fit adjusts the existing readout centroids and its one bias. Every hidden
activation and policy parameter is frozen. The sibling objective compares both
parent-perspective action values. Exact terminal results bypass the network.
Use the family-filtered replay from training.child_pair_curriculum prepare.

The readout experiment used 533 verified sibling pairs, all eight training
orientations and a separate reference panel for generalization.

    python -m training.sibling_value_readout --input parent-1200-raw.onnx \
        --base moderate.pt --data sibling-data --output value-checkpoints \
        --background deep 14500 --background action 6500 --background proof 7000 \
        --background paper 8500 --fractions .25 .6 1 --pair-weight .18 \
        --iterations 80 --batch-size 160 --seed 26315971
"""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import onnxruntime as ort
import torch
from torch.nn import functional as F

from .distill_oracle import children
from .fast_stage2_graph import TokenGraph, verify
from .fit_search_student import file_hash
from .pipeline import _symmetry_maps
from .refine_packed import check_parity, layout_hash, update_centroids
from .regret_replay import student_from
from .search_alternatives import keys


def read_replay(folder, name, fields, manifest):
    spec = next(row for row in manifest['sources'] if row['name'] == name)
    path = folder/(name+'.npz')
    if file_hash(path) != spec['sha256']:
        raise ValueError(f'Changed replay source {name}')
    with np.load(path, allow_pickle=False) as z:
        arrays = [z[k] for k in fields]
    if any(len(a) != spec['rows'] for a in arrays):
        raise ValueError(f'Incorrect replay count for {name}')
    return arrays


def sibling_targets(roots, moves, sx, sv, min_margin):
    states, terminal, won = children(roots, np.repeat(np.arange(len(roots)), 2), moves.flatten().astype(np.intp))
    look = {key: i for i, key in enumerate(keys(sx))}
    if len(look) != len(sx):
        raise ValueError('Repeated search child')
    ix = np.asarray([look.get(k, -1) for k in keys(states)])
    if not np.array_equal(ix < 0, terminal):
        raise ValueError('Expected one independent value per live child')
    target = won.astype(np.float32)
    target[~terminal] = -sv[ix[~terminal]]
    q = target.reshape(-1, 2)
    if not np.isfinite(q).all() or (np.abs(q)>1).any() or (q[:, 1]-q[:, 0] < min_margin-1e-6).any():
        raise ValueError('Child values conflict with the promised parent ranking')
    return states.reshape(-1,2,280), terminal.reshape(-1,2), q


def hidden_projection(codes, width, count):
    """Group final-layer activations by the shared coefficient they multiply."""
    if codes.ndim != 2 or count <= 0 or not (codes.shape[0]-1)*codes.shape[1] < count <= codes.numel():
        raise ValueError('Unexpected final-layer coefficient layout')
    if not (0 <= codes).all() or not (codes < width).all():
        raise ValueError('Invalid readout assignment')
    ids = (codes + torch.arange(len(codes), device=codes.device)[:,None]*width).flatten()[:count]
    return F.one_hot(ids, num_classes=len(codes)*width).float()


@torch.inference_mode()
def observations(model, book, x, device, batch_size):
    last = model.layers['value_out']
    projection = hidden_projection(book.codes, book.centers.shape[1], last.weight.numel())
    dest = []
    def observe(_, arguments):
        hidden = arguments[0].flatten(1)
        dest.append((hidden @ projection).cpu())
    handle = last.register_forward_pre_hook(observe)
    try:
        for i in range(0,len(x),batch_size):
            _, y = model(torch.as_tensor(x[i:i+batch_size], device=device, dtype=torch.float32))
            estimate = (dest[-1].to(device) @ book.centers.flatten().half().float() + last.bias).tanh()
            torch.testing.assert_close(estimate, y.flatten(), atol=1e-5, rtol=1e-5)
    finally:
        handle.remove()
    return torch.cat(dest).double()


def predict(z, w):
    return (z @ w[:-1] + w[-1]).tanh()


def parent_values(z, w, terminal, truth):
    return torch.where(terminal, truth, -predict(z,w))


def fit_readout(root, truth, siblings, terminal, q, initial, weight, iterations):
    w = initial.detach().clone().requires_grad_()
    opt = torch.optim.LBFGS([w], lr=.7, max_iter=iterations, line_search_fn='strong_wolfe',
                            tolerance_grad=1e-8, tolerance_change=1e-11)
    root0 = predict(root,initial)
    pair0 = parent_values(siblings,initial,terminal,q)
    def loss():
        opt.zero_grad()
        value = predict(root,w)
        pair = parent_values(siblings,w,terminal,q)
        absolute = F.smooth_l1_loss(value,truth,beta=.30)
        differential = F.smooth_l1_loss(pair[:,1]-pair[:,0],q[:,1]-q[:,0],beta=.30)
        leaf = F.smooth_l1_loss(pair,q,beta=.30)
        # Penalize output movement on both the broad replay and mined siblings.
        penalty = F.mse_loss(value,root0) + .25*F.mse_loss(pair,pair0)
        result = absolute + weight*(differential+.35*leaf) + 2.5*penalty + .08*(w-initial).square().mean()
        result.backward()
        return result
    opt.step(loss)
    if not torch.isfinite(w).all():
        raise ValueError('Nonfinite final readout')
    return w.detach()


def unchanged_policy(source, candidate, x):
    options = ort.SessionOptions()
    options.intra_op_num_threads = options.inter_op_num_threads = 1
    digest = []
    for payload in (source,candidate):
        session = ort.InferenceSession(payload,options,providers=['CPUExecutionProvider'])
        h = hashlib.sha256()
        for z in [x[:1], x[:12], *np.array_split(x, max(1,(len(x)+47)//48))]:
            y = session.run(['policy'],{'state':z.astype(np.float32)})[0]
            if y.shape != (len(z),81) or not np.isfinite(y).all():
                raise ValueError('Invalid policy output')
            h.update(y.astype('<f4').tobytes())
        digest.append(h.hexdigest())
    if digest[0] != digest[1]:
        raise ValueError('A policy output changed during value-only fitting')
    return digest[0]


def train(a, output):
    rng = np.random.default_rng(a.seed)
    device = torch.device(a.device)
    torch.set_num_threads(a.threads)
    torch.manual_seed(a.seed)
    manifest = json.loads((a.data/'manifest.json').read_bytes())
    replay = a.data/'replay'
    sub = json.loads((replay/'manifest.json').read_bytes())
    if (manifest.get('format') != 'uttt-child-pair-curriculum-v1' or
            manifest['replaySha256'] != file_hash(replay/'manifest.json') or
            sub['base'] != file_hash(a.base)):
        raise ValueError('Expected the verified family-filtered sibling training split')
    packed, model, books = student_from(a.input,a.base,device)
    model.requires_grad_(False)
    last = model.layers['value_out']
    book = last.parametrizations.weight[0]
    name = next(k for k,b in books.items() if b is book)
    bias_name = 'layers.value_out.bias'
    fingerprint = layout_hash(packed,{name,bias_name})
    initial = torch.cat([book.centers.detach().flatten().half().float(),last.bias.detach()]).double().cpu()
    fixtures = np.asarray([r['encoded'] for r in json.loads(a.fixtures.read_bytes())['cases'][:12]],np.float32)
    check_parity(packed.SerializeToString(),model,torch.from_numpy(fixtures),a.threads)
    model.to(device)
    transforms = np.stack([np.argsort(perm) for perm,_ in _symmetry_maps()])
    x, bad, good, gain = read_replay(replay,'pairs',('x','chosen','better','gain'),sub)
    sx, sv = read_replay(replay,'siblings',('x','v'),sub)
    use = gain>0
    states, terminal, q = sibling_targets(x[use],np.column_stack([bad[use],good[use]]),sx,sv,manifest['minMargin'])
    del sx,sv
    pair_features = []
    for perm in transforms:
        z=observations(model,book,states[...,perm].reshape(-1,280),device,a.batch_size)
        pair_features.append(z.reshape(len(states),2,-1))
    pair_features = torch.cat(pair_features)
    q = torch.from_numpy(np.tile(q,(8,1))).double()
    terminal = torch.from_numpy(np.tile(terminal,(8,1)))
    features, values = [], []
    counts = {}
    for source, limit in a.background:
        xx, v = read_replay(replay,source,('x','v'),sub)
        take = rng.choice(len(xx),min(int(limit),len(xx)),replace=False)
        counts[source] = len(take)
        view = rng.integers(8,size=len(take))
        batch = np.take_along_axis(xx[take],transforms[view],1)
        features.append(observations(model,book,batch,device,a.batch_size))
        values.append(torch.from_numpy(v[take]).double().flatten())
    z, v = torch.cat(features), torch.cat(values)
    fitted = fit_readout(z,v,pair_features,terminal,q,initial,a.pair_weight,a.iterations)
    native0 = TokenGraph(packed).build().SerializeToString()
    probing = np.concatenate((fixtures,x[rng.choice(len(x),min(100,len(x)),replace=False)]))
    report = dict(format='uttt-packed-sibling-value-readout-v1',inputSha256=file_hash(a.input),
                  baseSha256=file_hash(a.base),dataSha256=file_hash(a.data/'manifest.json'),
                  heldSha256=manifest['heldSha256'],seed=a.seed,background=counts,pairWeight=a.pair_weight,
                  iterations=a.iterations,trainPairs=int(use.sum()),views=len(transforms),
                  readoutCenters=book.centers.numel(),layoutSha256=fingerprint,exports=[])
    for fraction in a.fractions:
        w = torch.lerp(initial,fitted,fraction)
        with torch.no_grad():
            book.centers.copy_(w[:-1].reshape_as(book.centers))
            last.bias.copy_(w[-1:])
        graph = copy.deepcopy(packed)
        update_centroids(graph,{name:book})
        t = next(t for t in graph.graph.initializer if t.name==bias_name)
        if list(t.dims)!=[1] or len(t.raw_data)!=4:
            raise ValueError('Expected one directly stored float32 value bias')
        t.raw_data = last.bias.detach().cpu().numpy().astype('<f4').tobytes()
        if layout_hash(graph,{name,bias_name}) != fingerprint:
            raise ValueError('Another parameter changed')
        raw = graph.SerializeToString()
        native = TokenGraph(graph).build().SerializeToString()
        if len(native)>a.max_bytes or len(native)!=len(native0):
            raise ValueError('Export changed the size or exceeded the cap')
        parity = check_parity(raw,model,torch.from_numpy(fixtures),a.threads)
        graph_parity = verify(raw,native,probing.astype(np.float32),a.threads)
        identity = unchanged_policy(native0,native,probing)
        base = a.name+'-'+str(round(1000*fraction))
        (output/(base+'-raw.onnx')).write_bytes(raw)
        (output/(base+'.onnx')).write_bytes(native)
        report['exports'].append(dict(name=base,fraction=fraction,bytes=len(native),sha256=hashlib.sha256(native).hexdigest(),
             rawSha256=hashlib.sha256(raw).hexdigest(),policySha256=identity,torchParity=parity,graphParity=graph_parity))
    (output/'manifest.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('data','base','input','output'):
        p.add_argument('--'+key,required=True,type=Path)
    p.add_argument('--name',default='sibling-value')
    p.add_argument('--background',nargs=2,action='append',required=True,metavar=('SOURCE','ROWS'))
    p.add_argument('--fractions',nargs='+',type=float,default=[.25,.60,1.])
    p.add_argument('--pair-weight',type=float,default=.18)
    p.add_argument('--iterations',type=int,default=80)
    p.add_argument('--batch-size',type=int,default=192)
    p.add_argument('--seed',type=int,default=26315971)
    p.add_argument('--max-bytes',type=int,default=1480935)
    p.add_argument('--threads',type=int,default=1)
    p.add_argument('--device',choices=['cpu','cuda'],default='cuda')
    p.add_argument('--fixtures',type=Path,default=Path(__file__).with_name('parity.json'))
    a=p.parse_args()
    if (not a.name.replace('-','').isalnum() or len({k for k,_ in a.background})!=len(a.background) or
            any(k in ('pairs','siblings') or int(n)<1 for k,n in a.background) or
            any(not 0<f<=1 for f in a.fractions) or len({round(1000*f) for f in a.fractions})!=len(a.fractions) or
            not 0<a.pair_weight<=5 or min(a.iterations,a.batch_size,a.max_bytes,a.threads)<1):
        p.error('Invalid training configuration')
    if a.output.exists():
        p.error('Use a fresh output directory')
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with TemporaryDirectory(dir=a.output.parent) as tmp:
        out=Path(tmp)/'output';out.mkdir()
        result=train(a,out)
        os.rename(out,a.output)
    print(json.dumps(result,allow_nan=False))


if __name__=='__main__':
    main()
