"""Compare two plausible moves using an independent fixed search after each move.

Prepare reads searched roots from two engines. Finish aligns the deeper child
searches by exact state, reverses their values to the parent seat and updates
only the relative probability of the two selected moves at each parent.
"""

import argparse
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from training.distill_oracle import children, validate
from training.suite_exclusion import Exclusion


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_verified(folder, fields):
    info = json.loads((folder / 'manifest.json').read_bytes())
    pieces = {name: [] for name in fields}
    for shard in info['shards']:
        path = folder / shard['file']
        if path.parent != folder or digest(path) != shard['sha256']:
            raise ValueError('Mismatch between search shard and manifest')
        with np.load(path, allow_pickle=False) as data:
            for name in fields:
                pieces[name].append(data[name])
    return {k: np.concatenate(parts) for k, parts in pieces.items()}


def keys(x):
    return [row.tobytes() for row in np.packbits(x, axis=1)]


def pairs(first, second, legal):
    a = first.argmax(1)
    b = second.argmax(1)
    equal = a == b
    ranked = np.where(legal, first + second, -np.inf)
    ranked[np.arange(len(a)), a] = -np.inf
    return np.stack([a, np.where(equal, ranked.argmax(1), b)], axis=1), ~equal


def updated_policy(p, moves, old_q, new_q):
    """Keep the pair's combined policy mass while correcting its relative odds."""
    rows = np.arange(len(p))[:, None]
    chosen = p[rows, moves]
    difference = (new_q[:, 0] - new_q[:, 1]) - (old_q[:, 0] - old_q[:, 1])
    delta = np.clip(5. * difference, -4., 4.)
    odds = np.log(np.maximum(chosen[:, 0], 1.e-7)) - np.log(np.maximum(chosen[:, 1], 1.e-7)) + delta
    first = 1. / (1. + np.exp(-np.clip(odds, -20., 20.)))
    result = p.copy()
    result[rows, moves] = chosen.sum(1, keepdims=True) * np.column_stack((first, 1. - first))
    return result


def prepare(a, folder):
    official = load_verified(a.official, ('x', 'p', 'v', 'n', 'q'))
    other = load_verified(a.project, ('x', 'p', 'v'))
    index = {k: i for i, k in enumerate(keys(other['x']))}
    official_keys = keys(official['x'])
    if len(index) != len(other['x']) or len(set(official_keys)) != len(official_keys):
        raise ValueError('Expected unique source roots')
    exclude = Exclusion(a.suite)
    keep = exclude.keep(official['x']) & (official['x'][:, 162:243].sum(1) >= 2)
    chosen = [i for i, k in enumerate(official_keys) if keep[i] and k in index]
    other_rows = [index[official_keys[i]] for i in chosen]
    x, p0, v0, n, q = [official[k][chosen] for k in ('x','p','v','n','q')]
    p1, v1 = other['p'][other_rows].astype(np.float32), other['v'][other_rows]
    p0 = p0.astype(np.float32)
    p0 /= p0.sum(1, keepdims=True)
    p1 /= p1.sum(1, keepdims=True)
    moves, disagree = pairs(p0, p1, x[:, 162:243].astype(bool))
    rng = np.random.default_rng(a.seed)
    ids = rng.permutation(len(x))
    priority = ids[disagree[ids]][:int(a.roots * .75)]
    selected = np.concatenate((priority, ids[~np.isin(ids, priority)][:a.roots - len(priority)]))
    if len(selected) < a.roots:
        raise ValueError('Too few eligible paired roots')
    x, p0, p1, v0, v1, n, q, moves, disagree = [z[selected] for z in (x,p0,p1,v0,v1,n,q,moves,disagree)]
    rows = np.arange(len(x))
    cx, terminal, won = children(x, np.repeat(rows, 2), moves.ravel())
    keep = (terminal | exclude.keep(cx)).reshape(-1, 2).all(1)
    accepted = np.repeat(keep, 2)
    cx, terminal, won = cx[accepted], terminal[accepted], won[accepted]
    x, p0, p1, v0, v1, n, q, moves, disagree = [z[keep] for z in (x,p0,p1,v0,v1,n,q,moves,disagree)]
    idx = np.arange(len(x))[:, None]
    confidence = n[idx, moves].astype(np.float32) / (n[idx, moves].astype(np.float32) + 48.)
    original_q = v0[:, None] + confidence * (q[idx, moves] - v0[:, None])
    live = ~terminal
    child_ids = np.full(len(cx), -1, dtype=np.int32)
    unique = {}
    states = []
    for row, k in zip(np.flatnonzero(live), keys(cx[live])):
        if k not in unique:
            unique[k] = len(states)
            states.append(cx[row])
        child_ids[row] = unique[k]
    states = np.stack(states)
    child_folder = folder / 'children'
    child_folder.mkdir()
    # Only x and v are read by the root sampler. These zeros are placeholders for indexing.
    np.savez_compressed(child_folder/'positions.npz', x=states, v=np.zeros(len(states), np.float32))
    rootfile = folder / 'roots.npz'
    np.savez_compressed(rootfile, x=x, p=(p0+p1)/2, v=(v0+v1)/2, moves=moves, originalQ=original_q,
                        child=child_ids.reshape(-1,2), terminalWon=won.reshape(-1,2), disagreed=disagree)
    return dict(format='uttt-alternative-search-requests-v1', roots=len(x), disagreements=int(disagree.sum()),
                uniqueChildren=len(states), directTerminal=int(terminal.sum()), excludedParents=int((~keep).sum()),
                officialManifest=digest(a.official/'manifest.json'), projectManifest=digest(a.project/'manifest.json'),
                seed=a.seed, rootSha256=digest(rootfile), childrenSha256=digest(child_folder/'positions.npz'))


def finish(a, folder):
    req = json.loads((a.requests/'manifest.json').read_bytes())
    roots_file, children_file = a.requests/'roots.npz', a.requests/'children/positions.npz'
    if digest(roots_file) != req['rootSha256'] or digest(children_file) != req['childrenSha256']:
        raise ValueError('Alternative requests changed')
    targets = load_verified(a.searched, ('x','p','v'))
    indices = {k:i for i,k in enumerate(keys(targets['x']))}
    with np.load(children_file, allow_pickle=False) as child:
        needed = keys(child['x'])
    if len(targets['x']) != len(indices) or len(indices) != len(needed) or set(indices) != set(needed):
        raise ValueError('Deeper search does not cover each requested child exactly once')
    order = [indices[k] for k in needed]
    cx, cp, cv = (targets[k][order] for k in ('x','p','v'))
    with np.load(roots_file, allow_pickle=False) as roots:
        x,p,v,m,old,ids,won,disagree = [roots[k] for k in ('x','p','v','moves','originalQ','child','terminalWon','disagreed')]
    q = np.where(ids >= 0, -cv[np.maximum(ids, 0)], won.astype(np.float32))
    policy = updated_policy(p,m,old,q)
    selected = np.arange(len(p))[:,None]
    # Replace the pair's contribution while retaining the other branches' search estimate.
    value = np.clip(v + (policy[selected,m]*q - p[selected,m]*old).sum(1), -1., 1.)
    shards = []
    for i,(states,pol,val) in enumerate(((x,policy,value),(cx,cp,cv))):
        validate(states,pol,val)
        path=folder/f'shard-{i:05d}.npz'
        np.savez_compressed(path,x=states,p=pol.astype(np.float16),v=val.astype(np.float32))
        shards.append({'file':path.name,'rows':len(states),'sha256':digest(path)})
    audit = folder / 'diagnostics'
    audit.mkdir()
    np.savez_compressed(audit/'pair-evidence.npz', x=x, moves=m, childValuesFromParent=q, rootActionValues=old,
                        initialRootPolicy=p, engineDisagreement=disagree)
    return dict(format='uttt-counterfactual-search-targets-v1', roots=len(x), children=len(cx), shards=shards,
                disagreementRoots=int(disagree.sum()), meanAbsoluteMargin=float(np.abs(q[:,0]-q[:,1]).mean()),
                reversals=int(((q[:,0]-q[:,1])*(old[:,0]-old[:,1])<0).sum()),
                requestsSha256=digest(a.requests/'manifest.json'), childSearchSha256=digest(a.searched/'manifest.json'),
                pairEvidenceSha256=digest(audit/'pair-evidence.npz'))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    modes=parser.add_subparsers(dest='mode',required=True)
    prepare_args=modes.add_parser('prepare')
    for name in ('official','project','suite','output'):
        prepare_args.add_argument('--'+name,type=Path,required=True)
    prepare_args.add_argument('--roots',type=int,default=1024)
    prepare_args.add_argument('--seed',type=int,default=26315572)
    finish_args=modes.add_parser('finish')
    for name in ('requests','searched','output'):
        finish_args.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists() or (args.mode=='prepare' and args.roots<1):
        parser.error('Select a fresh directory and a positive number of roots')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with TemporaryDirectory(dir=args.output.parent) as tmp:
        folder=Path(tmp)/'result'
        folder.mkdir()
        result=prepare(args,folder) if args.mode=='prepare' else finish(args,folder)
        (folder/'manifest.json').write_text(json.dumps(result,indent=2)+'\n')
        folder.rename(args.output)
    print(json.dumps(result),flush=True)


if __name__=='__main__':
    main()
