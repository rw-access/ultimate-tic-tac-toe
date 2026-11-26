"""Collect independently searched action values from the original uttt.ai Stage 2 engine."""

import argparse
import base64
import hashlib
import json
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory

import numpy as np

from training.distill_oracle import validate
from training.reanalyse_positions import response, sample_states
from training.suite_exclusion import Exclusion


def run(args):
    if args.output.exists():
        raise ValueError('Select a fresh output directory')
    ids, states, sources = sample_states(args.data, args.roots, 0, args.roots, args.seed)
    exclusion = Exclusion(args.upstream / 'utttpy/selfplay/evaluation_uttt_states.py')
    keep = exclusion.keep(states)
    ids, states = ids[keep], states[keep]
    if not len(ids):
        raise ValueError('No eligible training positions')
    worker = Path(__file__).resolve().parents[1] / 'scripts/reanalyse-official.ts'
    cmd = [args.node, '--experimental-strip-types', str(worker), '--model', str(args.model.resolve()),
           '--upstream', str(args.upstream.resolve()), '--provider', args.provider, '--visits', str(args.visits),
           '--parallel', str(args.parallel), '--seed', str(args.seed)]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=args.output.parent) as tmp:
        folder = Path(tmp) / 'data'
        folder.mkdir()
        with subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True) as process:
            try:
                header = response(process, 'ready')
                if (header['visits'],header['parallel'],header['seed']) != (args.visits,args.parallel,args.seed):
                    raise ValueError('Upstream engine configuration mismatch')
                report = {'format':'uttt-published-search-targets-v1', 'modelSha256':header['modelSha256'],
                          'upstreamCommit':header['upstreamCommit'], 'provider':args.provider, 'seed':args.seed,
                          'roots':len(states), 'excluded':int((~keep).sum()), 'visitsPerRoot':args.visits,
                          'rootVisits':len(states)*args.visits, 'sources':sources, 'shards':[]}
                for first in range(0,len(ids),args.shard_size):
                    x=states[first:first+args.shard_size]
                    chosen=ids[first:first+len(x)]
                    request={'ids':chosen.tolist(),'x':base64.b64encode(x).decode()}
                    process.stdin.write(json.dumps(request,separators=(',',':'))+'\n')
                    process.stdin.flush()
                    rows=response(process,'targets')['targets']
                    n=np.asarray([r['n'] for r in rows],dtype=np.int32)
                    q=np.asarray([r['q'] for r in rows],dtype=np.float32)
                    v=np.asarray([r['v'] for r in rows],dtype=np.float32)
                    if (n.shape!=(len(x),81) or q.shape!=n.shape or (n<0).any() or
                            (n.sum(1)!=args.visits-1).any() or (n[x[:,162:243]==0]!=0).any() or
                            not np.isfinite(q).all() or (np.abs(q)>1.00001).any() or (q[n==0]!=0).any()):
                        raise ValueError('Invalid searched counts or action values')
                    p=(n/(args.visits-1)).astype(np.float16)
                    validate(x,p,v)
                    path=folder/f'shard-{len(report["shards"]):05d}.npz'
                    np.savez_compressed(path,x=x,p=p,v=v,n=n.astype(np.uint16),q=q,source_row=chosen)
                    report['shards'].append({'file':path.name,'rows':len(x),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
                    print(f'{first+len(x):,}/{len(ids):,} complete at {args.visits:,} published NMCTS visits',flush=True)
                process.stdin.close()
                if process.wait():
                    raise RuntimeError('Official search worker exited unsuccessfully')
            finally:
                if process.poll() is None:
                    process.terminate()
        (folder/'manifest.json').write_text(json.dumps(report,indent=2)+'\n')
        folder.rename(args.output)


def main():
    q=argparse.ArgumentParser(description=__doc__)
    for k in ('data','output','model','upstream'):
        q.add_argument('--'+k,type=Path,required=True)
    q.add_argument('--node',default='node')
    q.add_argument('--provider',choices=('cpu','cuda'),default='cuda')
    q.add_argument('--visits',type=int,default=3072)
    q.add_argument('--parallel',type=int,default=64)
    q.add_argument('--roots',type=int,default=2048)
    q.add_argument('--shard-size',type=int,default=256)
    q.add_argument('--seed',type=int,default=26315520)
    a=q.parse_args()
    if not 2<=a.visits<=10000 or not 1<=a.parallel<=128 or min(a.roots,a.shard_size)<1 or not 0<=a.seed<=0xffffffff:
        q.error('Invalid root-search budget or seed')
    run(a)


if __name__=='__main__':
    main()
