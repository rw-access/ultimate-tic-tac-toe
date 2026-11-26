"""Fit a same-size Stage 2 teacher to solved alternatives, fresh search and replay.

Exact examples supervise total probability on the set of optimal moves. Search
examples retain part of the incumbent's policy/value distribution. Export an
exponential average of the updated weights at the selected training steps.
"""

import argparse
import copy
import json
import math
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from training.fit_search_student import export, file_hash, load_positions, load_weights, save_checkpoint
from training.pipeline import _symmetry_maps
from training.stage2_student import Stage2Net


def objective(logits, value, reference, reference_value, x, p, v, exact, anchor, value_weight):
    legal = x[:, 162:243] > .5
    logp = logits.float().masked_fill(~legal, -1e9).log_softmax(1)
    incumbent = reference.float().masked_fill(~legal, -1e9).softmax(1)
    target = p.float() / p.float().sum(1, keepdim=True)
    backed = torch.lerp(target, incumbent, anchor[:, None])
    cross_entropy = -(backed * logp).sum(1)
    optimal_mass = torch.logsumexp(logp.masked_fill(target <= 0, -1e9), dim=1)
    proof_loss = -.85 * optimal_mass - .15 * (target * logp).sum(1)
    policy_loss = torch.where(exact, proof_loss, cross_entropy).mean()
    outcome = torch.lerp(v.float(), reference_value.float().flatten(), anchor)
    value_loss = F.mse_loss(value.float().flatten(), outcome)
    return policy_loss + value_weight * value_loss, policy_loss, value_loss


def choose_heads(model, selected):
    if selected == 'all':
        return
    policy = {'policy', 'upsample', 'logits', 'logit_out'}
    value = {'v1', 'vpool1', 'v2', 'vpool2', 'value', 'value_out'}
    active = {'policy': policy, 'value': value, 'heads': policy | value}[selected]
    model.requires_grad_(False)
    for name in active:
        model.layers[name].requires_grad_(True)


def train(args):
    device = torch.device(args.device)
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    snapshots = sorted(set(args.steps))
    if snapshots != args.steps or snapshots[0] < 1 or len(args.counts) != 4 or min(args.counts) < 1:
        raise ValueError('Use increasing snapshot steps and four positive sample counts')
    if any(args.output.with_name(f'{args.output.name}-{step}.onnx').exists() for step in snapshots) and not args.state.exists():
        raise ValueError('Use fresh snapshot paths or resume the matching training state')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    model = Stage2Net(compact=False)
    load_weights(model, args.initial)
    model.to(device)
    anchor_model = copy.deepcopy(model).eval().requires_grad_(False)
    ema = copy.deepcopy(model).eval().requires_grad_(False)
    choose_heads(model, args.train_heads)
    paths = (args.proof, args.fresh, args.searched, args.replay)
    proof_info = json.loads((args.proof / 'manifest.json').read_bytes())
    if proof_info.get('minRemaining', 0) < 15 or not proof_info.get('solvedChildren'):
        raise ValueError('Expected complete parent and child proofs beyond the endgame solver horizon')
    dataset_ids = []
    sets = []
    protected = None
    if args.exclude_orbits:
        from training.regret_replay import excluded_rows
        with np.load(args.exclude_orbits, allow_pickle=False) as held:
            roots = held['x']
        protected = np.unique(np.concatenate([np.packbits(roots[:, perm], axis=1)
                                              for perm, _ in _symmetry_maps()]), axis=0)
        if not len(protected):
            raise ValueError('The development orbit file is empty')
    for directory in paths:
        identity = {}
        data = load_positions(directory, args.max_positions, rng, device, identity)
        if protected is not None:
            direct, predecessor = excluded_rows(data[0].cpu().numpy(), protected)
            keep = torch.as_tensor(~(direct | predecessor), device=device)
            if not keep.any():
                raise ValueError('No training rows remain after development exclusions')
            data = tuple(t[keep] for t in data)
            identity.update(heldOutRoots=int(direct.sum()), heldOutPredecessors=int(predecessor.sum()))
        sets.append(data)
        dataset_ids.append(identity)
    if not torch.isin(sets[0][2],torch.tensor([-1.,0.,1.], device=device)).all():
        raise ValueError('Proof targets must be exact win, draw or loss values')
    identity = {'format':'uttt-proof-teacher-v1','seed':args.seed,'initialSha256':file_hash(args.initial),
                'steps':snapshots,'sources':dataset_ids,'counts':args.counts,'maxPositions':args.max_positions,
                'lr':args.lr,'emaDecay':args.ema_decay,'weightDecay':1e-4,'valueWeight':args.value_weight,
                'device':device.type,'torchVersion':str(torch.__version__)}
    if args.train_heads != 'all':
        identity['trainHeads'] = args.train_heads
    if args.exclude_orbits:
        identity['heldOutSha256'] = file_hash(args.exclude_orbits)
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=snapshots[-1], eta_min=args.lr * .12)
    completed = 0
    if args.state.is_file():
        saved = torch.load(args.state,map_location='cpu',weights_only=True)
        if saved['identity'] != identity or saved['step'] not in snapshots:
            raise ValueError('Resume state does not match the frozen corpora or optimizer schedule')
        completed = saved['step']
        model.load_state_dict(saved['model'])
        ema.load_state_dict(saved['ema'])
        optimizer.load_state_dict(saved['optimizer'])
        scheduler.load_state_dict(saved['scheduler'])
        torch.set_rng_state(saved['rng'])
        if device.type=='cuda':
            torch.cuda.set_rng_state(saved['cudaRng'],device)
    maps = _symmetry_maps()
    xmap, pmap = [torch.tensor(np.argsort(np.stack(axis),axis=1),device=device) for axis in zip(*maps)]
    policy_tags = torch.repeat_interleave(torch.tensor([0.,.15,.22,.45],device=device),
                                         torch.tensor(args.counts,device=device))
    exact = torch.arange(sum(args.counts),device=device) < args.counts[0]
    live, averaged = list(model.parameters()), list(ema.parameters())
    use_bf16 = device.type=='cuda' and torch.cuda.is_bf16_supported()
    def write_model(step):
        path=args.output.with_name(f'{args.output.name}-{step}.onnx')
        if not path.exists():
            # Export's CPU conversion must leave the EMA on the training device.
            export(copy.deepcopy(ema).cpu(),sets[1][0][:16].cpu(),path,'stage3_teacher')
        if not path.with_suffix('.pt').is_file():
            raise ValueError('Incomplete snapshot')
    if completed:
        write_model(completed)
    for step in range(completed + 1, snapshots[-1] + 1):
        sampled = []
        for data, count in zip(sets,args.counts):
            ids=torch.randint(len(data[0]),(count,),device=device)
            sampled.append(tuple(arr[ids] for arr in data))
        x,p,v=[torch.cat(items) for items in zip(*sampled)]
        symmetry=torch.randint(8,(len(x),),device=device)
        x=x.gather(1,xmap[symmetry]).float()
        p=p.gather(1,pmap[symmetry])
        with torch.no_grad(),torch.autocast(device_type=device.type,dtype=torch.bfloat16,enabled=use_bf16):
            rp,rv=anchor_model(x)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type,dtype=torch.bfloat16,enabled=use_bf16):
            lp,value=model(x)
        loss,ce,mse=objective(lp,value,rp,rv,x,p,v,exact,policy_tags,args.value_weight)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable,1.,error_if_nonfinite=True)
        optimizer.step()
        scheduler.step()
        with torch.no_grad():
            torch._foreach_lerp_(averaged,live,1.-args.ema_decay)
        if step==1 or step%200==0:
            print(f'{step}/{snapshots[-1]} policy {ce.item():.5f}, value {mse.item():.5f}',flush=True)
        if step in snapshots:
            save_checkpoint(args.state,{'identity':identity,'step':step,'model':model.state_dict(),
                'ema':ema.state_dict(),'optimizer':optimizer.state_dict(),'scheduler':scheduler.state_dict(),
                'rng':torch.get_rng_state(),'cudaRng':torch.cuda.get_rng_state(device) if device.type=='cuda' else None})
            write_model(step)
            print('Saved averaged teacher at step',step,flush=True)


def main():
    q=argparse.ArgumentParser(description=__doc__)
    for name in ('proof','fresh','searched','replay','initial','output','state'):
        q.add_argument('--'+name,type=Path,required=True)
    q.add_argument('--steps',nargs='+',type=int,default=[1800,6500])
    q.add_argument('--counts',nargs=4,type=int,default=[32,32,32,32])
    q.add_argument('--max-positions',type=int,default=100000)
    q.add_argument('--lr',type=float,default=3e-5)
    q.add_argument('--ema-decay',type=float,default=.998)
    q.add_argument('--value-weight',type=float,default=1.35)
    q.add_argument('--seed',type=int,default=26315473)
    q.add_argument('--device',choices=('cuda','cpu'),default='cuda')
    q.add_argument('--train-heads',choices=('all','heads','policy','value'),default='all',
                   help='Optionally keep the shared representation and the other output head frozen')
    q.add_argument('--exclude-orbits',type=Path,help='Development NPZ whose roots and direct predecessors are excluded in all D4 views')
    args=q.parse_args()
    if (not 0<=args.ema_decay<1 or not math.isfinite(args.lr) or args.lr<=0 or
            not math.isfinite(args.value_weight) or args.value_weight<=0 or args.max_positions<16 or args.seed<0):
        q.error('Invalid training settings')
    train(args)


if __name__=='__main__':
    main()
