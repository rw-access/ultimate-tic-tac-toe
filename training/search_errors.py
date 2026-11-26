"""Select midgame training targets where a realistic search budget leaves useful errors.

Join fixed-budget decisions to independently searched higher-budget action values.
Rank only supported action comparisons. Reserve whole D4 orbits for validation
before selecting hard positions, value errors and a uniform midgame control.
"""

import argparse
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from training.distill_oracle import validate
from training.search_alternatives import load_verified
from training.suite_exclusion import Exclusion


def action_error(counts, q, v, move, minimum=24, shrink=48.):
    rows = np.arange(len(v))
    n = counts.astype(np.float32)
    support = n >= minimum
    estimate = v[:, None] + (q-v[:, None])*n/(n+shrink)
    best = np.max(np.where(support, estimate, -np.inf), axis=1)
    measured = support[rows, move] & np.isfinite(best)
    return np.where(measured, np.maximum(0.,best-estimate[rows,move]), 0.), measured


def run(a, folder):
    from training.pipeline import _symmetry_maps
    original = load_verified(a.reference, ('x','p','v','q','n'))
    decisions = load_verified(a.search, ('x','v','visits','selected_move','source_row'))
    decision_info = json.loads((a.search/'manifest.json').read_bytes())
    if decision_info['simulations'] != a.visits or decision_info['modelSha256'] != a.model_sha256:
        raise ValueError('Fixed-budget search identity differs from configuration')
    ref_info = json.loads((a.reference/'manifest.json').read_bytes())
    if ref_info['visitsPerRoot'] < 4*a.visits:
        raise ValueError('Reference search is too shallow for this diagnostic')
    x = original['x']
    order = np.argsort(decisions['source_row'])
    if (not np.array_equal(decisions['source_row'][order], np.arange(len(x)))
            or not np.array_equal(decisions['x'][order], x)):
        raise ValueError('Expected a one-to-one match between searched roots')
    move = decisions['selected_move'][order]
    potential = ((x[:, :162].sum(1)>=12) & (x[:, :162].sum(1)<=42) & (x[:, 162:243].sum(1)>=2)
                 & (decisions['visits'][order]==a.visits) & Exclusion(a.suite).keep(x))
    error, supported = action_error(original['n'],original['q'],original['v'],move)
    value_error=np.abs(original['v']-decisions['v'][order])
    maps=np.stack([perm for perm, _ in _symmetry_maps()])
    groups=[min(row.tobytes() for row in np.packbits(board[maps],axis=1)) for board in x]
    positions={}
    for i in np.flatnonzero(potential):
        positions.setdefault(groups[i],i)
    development, training = [], []
    for orbit,i in positions.items():
        target=development if hashlib.sha256(orbit).digest()[0]<40 else training
        target.append(i)
    if not training or not development:
        raise ValueError('Empty midgame training or development split')
    rng=np.random.default_rng(a.seed)
    available=rng.permutation(training)
    rank=available[np.argsort(-error[available],kind='stable')]
    hard=rank[(error[rank]>.025) & supported[rank]][:len(training)//3]
    remaining=available[~np.isin(available,hard)]
    value=remaining[np.argsort(-value_error[remaining],kind='stable')[:max(1,len(hard)//2)]]
    control=remaining[~np.isin(remaining,value)][:max(256,len(hard))]
    result={'format':'uttt-search-error-replay-v1','modelSha256':a.model_sha256,'fixedVisits':a.visits,
            'referenceVisits':ref_info['visitsPerRoot'],'candidateRows':len(x),'uniqueMidgameRows':len(positions),
            'policyErrorRows':len(hard),'valueErrorRows':len(value),'uniformRows':len(control),'seed':a.seed,
            'referenceManifestSha256':hashlib.sha256((a.reference/'manifest.json').read_bytes()).hexdigest(),
            'searchManifestSha256':hashlib.sha256((a.search/'manifest.json').read_bytes()).hexdigest(),'splits':{}}
    for split,ids in (('train',np.concatenate((hard,value,control))),('development',np.array(development))):
        ids=np.asarray(ids,dtype=np.int64)
        states,p,v=(original[k][ids] for k in ('x','p','v'))
        validate(states,p,v)
        out=folder/split
        out.mkdir()
        f=out/'shard-00000.npz'
        np.savez_compressed(f,x=states,p=p.astype(np.float16),v=v.astype(np.float32),selected_move=move[ids],
                            action_regret=error[ids],supported=supported[ids])
        supported_errors = error[ids][supported[ids]]
        result['splits'][split]=dict(rows=len(ids),supported=len(supported_errors),
             meanSupportedRegret=float(supported_errors.mean()) if len(supported_errors) else None,
             sha256=hashlib.sha256(f.read_bytes()).hexdigest())
    return result


if __name__=='__main__':
    q=argparse.ArgumentParser(description=__doc__)
    for name in ('reference','search','output','suite'):
        q.add_argument('--'+name,type=Path,required=True)
    q.add_argument('--model-sha256',required=True)
    q.add_argument('--visits',type=int,default=256)
    q.add_argument('--seed',type=int,default=26315621)
    a=q.parse_args()
    if a.output.exists() or a.visits<2:
        q.error('Use fresh output and a valid positive search budget')
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with TemporaryDirectory(dir=a.output.parent) as tmp:
        folder=Path(tmp)/'data'
        folder.mkdir()
        result=run(a,folder)
        (folder/'manifest.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
        folder.rename(a.output)
    print(json.dumps(result),flush=True)
