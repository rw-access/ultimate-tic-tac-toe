"""Sample fresh positions from dedicated training games against the original NMCTS.

Reconstruct each game from the empty board and verify the reported terminal
outcome. Keep complete opening families together in development. Remove their
symmetry-equivalent sampled positions from the training split before relabeling.
"""

import argparse
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from training.distill_oracle import children, validate


def trajectory(game, opening, minimum_ply):
    moves = game['moves']
    if moves[:len(opening)] != opening or game['seat'] not in ('X','O'):
        raise ValueError('Training game has an inconsistent opening or seat')
    state = np.zeros((1, 280), dtype=np.uint8)
    state[0,162:243] = 1
    state[0,279] = 1
    choices = []
    for ply, move in enumerate(moves):
        if type(move) is not int or not 0 <= move < 81:
            raise ValueError('Invalid recorded move')
        active = 'O' if ply % 2 else 'X'
        if ply >= minimum_ply and game['seat'] == active and state[0,162:243].sum() >= 2:
            choices.append((state[0].copy(),move))
        state, terminal, won = children(state,np.array([0]),np.array([move]))
        if terminal[0] != (ply==len(moves)-1):
            raise ValueError('Terminal ply differs from complete game record')
    if not moves:
        raise ValueError('Empty game')
    winner = active if won[0] else 'draw'
    score = .5 if winner=='draw' else float(winner==game['seat'])
    if winner != game['winner'] or score != game['score']:
        raise ValueError('Recorded winner or score differs from reconstructed board')
    return choices, 2*score-1


def run(a, out):
    from training.pipeline import _symmetry_maps
    from training.suite_exclusion import Exclusion

    raw=a.games.read_bytes()
    report=json.loads(raw)
    if (report['format']!='official-uttt-arena-v1' or report['status']!='complete'
            or report['openingSuite']!='generated' or report['openingSchedule']!='independent'
            or report['seed']!=a.training_seed or report['candidate']['sha256']!=a.model_sha256
            or report['official']['sha256'] not in ('b4f92e809e1eaa2053b202202d3853e1ed10be6d4c6980c2c46339f6e14f70b7',
                                                   '81161d2e262c9904dbb9a14961a2fcd043009fab4fd4f76b41451587f89c60d2')
            or report['pairs']!=len(report['results'])):
        raise ValueError('Expected a complete, independently seeded opponent training match')
    rng=np.random.default_rng(a.training_seed+1)
    maps=_symmetry_maps()
    features=np.stack([i for i,_ in maps])
    actions=np.stack([p for _,p in maps])
    excluded=Exclusion(a.suite)
    pools={'train':[], 'development':[]}
    for i,pair in enumerate(report['results']):
        opening=pair['opening']
        if pair['pair']!=i or {g['seat'] for g in pair['games']}!={'X','O'} or len(pair['games'])!=2:
            raise ValueError('Incomplete or unordered training opening')
        orbit=min(bytes(row) for row in actions[:,opening].astype(np.uint8))
        opening_digest = hashlib.sha256(orbit).digest()
        split='development' if int.from_bytes(opening_digest[:4],'big')%5==0 else 'train'
        for game in pair['games']:
            moves,value=trajectory(game,opening,a.minimum_ply)
            subsample=rng.choice(len(moves),min(len(moves),a.per_game),replace=False)
            for row in sorted(subsample):
                state,move=moves[row]
                key=min(r.tobytes() for r in np.packbits(state[features],axis=1))
                pools[split].append((state,move,value,key,i,game['seat'],opening_digest))
    seen=set()
    receipt=dict(format='uttt-opponent-training-positions-v1',gameReportSha256=hashlib.sha256(raw).hexdigest(),
                 candidateSha256=a.model_sha256,officialSha256=report['official']['sha256'],trainingSeed=a.training_seed,
                 games=2*report['pairs'],projectVisits=report['maxRootVisits'],officialVisits=report['officialMaxRootVisits'],
                 minimumPly=a.minimum_ply,maximumPerGame=a.per_game,splits={})
    for split in ('development','train'):
        selected=[]
        for row in pools[split]:
            if row[3] not in seen:
                seen.add(row[3])
                selected.append(row)
        if not selected:
            raise ValueError('Both splits need unique positions')
        x=np.stack([r[0] for r in selected])
        if not excluded.keep(x).all():
            raise ValueError('Training positions overlap the protected tournament')
        p=np.zeros((len(x),81),dtype=np.float16)
        p[np.arange(len(x)),[r[1] for r in selected]]=1
        v=np.asarray([r[2] for r in selected],np.float32)
        validate(x,p,v)
        folder=out/split
        folder.mkdir()
        file=folder/'positions.npz'
        # Keep the two seats and repeated equivalent openings linked for game-level uncertainty.
        np.savez_compressed(file,x=x,p=p,v=v,
                            opening_pair=np.asarray([r[4] for r in selected], dtype=np.int32),
                            candidate_as_x=np.asarray([r[5]=='X' for r in selected], dtype=np.bool_),
                            opening_orbit_sha256=np.asarray([list(r[6]) for r in selected], dtype=np.uint8))
        receipt['splits'][split]=dict(rows=len(x),pairs=len(set(r[4] for r in selected)),
                                      sha256=hashlib.sha256(file.read_bytes()).hexdigest())
    return receipt


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('games','suite','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--model-sha256',required=True)
    parser.add_argument('--training-seed',type=int,required=True)
    parser.add_argument('--minimum-ply',type=int,default=12)
    parser.add_argument('--per-game',type=int,default=10)
    args=parser.parse_args()
    if args.output.exists() or args.minimum_ply<8 or args.per_game<1:
        parser.error('Choose fresh output, at least eight opening plies and positive sampling')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with TemporaryDirectory(dir=args.output.parent) as tmp:
        result=Path(tmp)/'data'
        result.mkdir()
        info=run(args,result)
        (result/'manifest.json').write_text(json.dumps(info,indent=2)+'\n')
        result.rename(args.output)
    print(json.dumps(info),flush=True)
