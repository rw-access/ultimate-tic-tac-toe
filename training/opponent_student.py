"""Train/evaluate split preparation for a compact student facing original-engine opponents.

The prepare command reads board identities and opening groups from development
games. Its search targets come exclusively from the training partitions. Freeze
the two deployment-form graphs before running evaluate on the development search
shards. All comparisons resample complete symmetry-equivalent opening families.
"""
import argparse
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from .child_pair_curriculum import COMMIT, MODELS, source_blob, variants
from .distill_oracle import children, validate
from .pipeline import _symmetry_maps
from .regret_replay import stratified_hardness
from .search_alternatives import digest, keys, load_verified
from .search_errors import action_error
from .suite_exclusion import Exclusion, PACKED_ORBITS_SHA256


def read(path):
    return json.loads(path.read_bytes())


def write(path, obj):
    with path.open('x') as out:
        json.dump(obj, out, allow_nan=False, indent=2, sort_keys=True)
        out.write('\n')


def orbit_keys(x):
    return {k for view in variants(x) for k in view}


def canonical(x):
    return [min(view) for view in zip(*variants(x))]


def name_allowed(label):
    return label.isascii() and label.isalnum()


def search_labels(directory, visits, seed):
    path = directory/'manifest.json'
    meta = read(path)
    if (meta['format'] != 'uttt-published-search-targets-v1' or meta['modelSha256'] not in MODELS
            or meta['upstreamCommit'] != COMMIT or meta['seed'] != seed or meta['visitsPerRoot'] != visits
            or meta['rootVisits'] != visits * meta['roots']):
        raise ValueError('Unexpected original-engine search identity or budget')
    data = load_verified(directory, ('x','p','v','q','n'))
    x, p, v, q, n = [data[k] for k in ('x','p','v','q','n')]
    validate(x,p,v)
    if (len(x) != meta['roots'] or len(set(keys(x))) != len(x) or
            n.shape != p.shape or n.dtype.kind not in 'ui' or not (n.sum(1)==visits-1).all() or
            not np.array_equal(p,(n/(visits-1)).astype(np.float16)) or q.shape != p.shape or
            not (np.abs(q)<=1).all()):
        raise ValueError('Search visits, action values, or root count disagree')
    return data


def aligned_order(x, reference):
    order = {k:i for i,k in enumerate(keys(reference))}
    indices = np.asarray([order.get(k,-1) for k in keys(x)], dtype=np.intp)
    if (len(order) != len(reference) or len(x) != len(reference) or
            not np.array_equal(np.sort(indices), np.arange(len(reference)))):
        raise ValueError('Search roots do not match the complete designated partition')
    return indices


def source_metadata(spec, baseline):
    label, games_name, sample_name, search_name, seed, pairs, cap, opponent, analysis_seed = spec
    if not name_allowed(label):
        raise ValueError('Use an alphanumeric source name')
    path, sample = Path(games_name), Path(sample_name)
    seed, pairs, cap, opponent, analysis_seed = map(int,(seed,pairs,cap,opponent,analysis_seed))
    series = read(path)
    manifest = read(sample/'manifest.json')
    if (series['format'] != 'official-uttt-arena-v1' or series['status'] != 'complete'
            or series['openingSuite'] != 'generated' or series['openingSchedule'] != 'independent'
            or series['seed'] != seed or series['pairs'] != pairs or len(series['results']) != pairs
            or series['maxRootVisits'] != cap or series['officialMaxRootVisits'] != opponent
            or series['candidate']['sha256'] != baseline or series['official']['sha256'] not in MODELS
            or manifest['format'] != 'uttt-opponent-training-positions-v1'
            or manifest['gameReportSha256'] != digest(path) or manifest['trainingSeed'] != seed
            or manifest['candidateSha256'] != baseline or manifest['officialSha256'] != series['official']['sha256']
            or manifest['games'] != 2*pairs or manifest['projectVisits'] != cap or manifest['officialVisits'] != opponent):
        raise ValueError('Opponent training game provenance or budgets differ')
    maps = np.stack([moves for _, moves in _symmetry_maps()])
    groups = []
    for i,item in enumerate(series['results']):
        opening = item['opening']
        if (item['pair'] != i or {g['seat'] for g in item['games']} != {'X','O'} or len(item['games']) != 2
                or not all(type(m) is int and 0<=m<81 for m in opening)):
            raise ValueError('Invalid two-seat opening family')
        representative = min(row.tobytes() for row in maps[:,opening].astype(np.uint8))
        groups.append(hashlib.sha256(representative).digest())
    result = {}
    for split in ('train','development'):
        path = sample/split/'positions.npz'
        receipt = manifest['splits'][split]
        if digest(path) != receipt['sha256']:
            raise ValueError('Sampled opponent positions changed')
        # The sampler also records moves and game outcomes. Only features and
        # family metadata are needed to designate the held positions.
        with np.load(path,allow_pickle=False) as z:
            x, pair, orbit = [z[k] for k in ('x','opening_pair','opening_orbit_sha256')]
        if (x.shape != (receipt['rows'],280) or x.dtype != np.uint8 or (x>1).any()
                or orbit.shape != (len(x),32) or orbit.dtype != np.uint8 or pair.shape != (len(x),)
                or pair.dtype.kind not in 'iu' or (pair<0).any() or (pair>=pairs).any()
                or len(set(pair.tolist())) != receipt['pairs'] or len(set(keys(x))) != len(x)):
            raise ValueError('Bad sampled board identities or opening indices')
        expected = [groups[i] for i in pair]
        if [row.tobytes() for row in orbit] != expected or any(
                (int.from_bytes(k[:4],'big')%5==0) != (split=='development') for k in expected):
            raise ValueError('Opening family assigned to the wrong train/development partition')
        result[split] = (x,orbit)
    info = dict(trainingSeed=seed,analysisSeed=analysis_seed,games=2*pairs,candidateVisits=cap,
                opponentVisits=opponent,gameReportSha256=manifest['gameReportSha256'],
                sampleManifestSha256=digest(sample/'manifest.json'))
    return label,result,Path(search_name),info


def reachable_blocks(x, protected, low=0, high=81):
    """Find rows equal or immediately prior to a protected board in any supplied orientation."""
    root = np.zeros(len(x),bool)
    child = np.zeros(len(x),bool)
    k = x[:,:162].sum(1)
    eligible = np.flatnonzero((k>=max(0,low-1)) & (k<=high))
    for i in range(0,len(eligible),192):
        rows = eligible[i:i+192]
        batch = x[rows]
        root[rows] = [key in protected for key in keys(batch)]
        ii,jj = np.nonzero(batch[:,162:243])
        if len(ii):
            succ = children(batch,ii,jj)[0]
            hit = np.fromiter((key in protected for key in keys(succ)),bool,len(ii))
            child[rows[ii[hit]]] = True
    return root,child & ~root


def historical_overlap(x, history):
    identity = canonical(x)
    indexed = {key:identity[i] for view in variants(x) for i,key in enumerate(view)}
    seen = set()
    scanned = 0
    done = set()
    lo,hi = int(x[:,:162].sum(1).min()),int(x[:,:162].sum(1).max())
    def mark(states):
        for key in keys(states):
            group = indexed.get(key)
            if group is not None:
                seen.add(group)
    for folder in history:
        m = read(folder/'manifest.json')
        if m['format'] != 'uttt-regret-replay-v1':
            raise ValueError('Expected a recorded starting-model training history')
        for source in m['sources']:
            if source['sha256'] in done:
                continue
            path = folder/(source['name']+'.npz')
            if digest(path) != source['sha256']:
                raise ValueError('Historical replay changed')
            done.add(source['sha256'])
            with np.load(path,allow_pickle=False) as z:
                xx = z['x']
            scanned += len(xx)
            ply = xx[:,:162].sum(1)
            eligible = np.flatnonzero((ply>=max(lo-1,0)) & (ply<=hi))
            for i in range(0,len(eligible),192):
                batch = xx[eligible[i:i+192]]
                mark(batch)
                ii,jj = np.nonzero(batch[:,162:243])
                if len(ii):
                    mark(children(batch,ii,jj)[0])
    return np.asarray([k in seen for k in identity]),scanned


def prepare(a, output):
    parsed = [source_metadata(spec,a.baseline_sha256) for spec in a.source]
    names = [n for n,*_ in parsed]
    base_names = [label for _,label in a.background]
    if len(set(names+base_names)) != len(names+base_names):
        raise ValueError('Duplicated replay name')
    dev_groups = set()
    train_groups = set()
    dev_keys = set()
    for _,part,_,_ in parsed:
        for split,groups in [('train',train_groups),('development',dev_groups)]:
            groups.update(k.tobytes() for k in part[split][1])
        dev_keys.update(orbit_keys(part['development'][0]))
    if train_groups & dev_groups:
        raise ValueError('A development opening family appears in training')
    all_development = np.concatenate([part['development'][0] for _,part,_,_ in parsed])
    overlap,scanned = historical_overlap(all_development,a.history)
    protected = set(Exclusion(a.suite).features) | dev_keys
    for path in a.protected_orbits:
        z = np.load(path,allow_pickle=False)
        if z.ndim!=2 or z.shape[1]!=35 or z.dtype!=np.uint8:
            raise ValueError('Expected packed D4 board keys')
        protected.update(row.tobytes() for row in z)
    replay = output/'replay'
    replay.mkdir()
    out = dict(format='uttt-opponent-student-replay-v1',baselineSha256=a.baseline_sha256,
               visits=a.visits,protectedStates=len(protected),
               protectedSha256=hashlib.sha256(b''.join(sorted(protected))).hexdigest(),
               historicalManifestSha256=[digest(h/'manifest.json') for h in a.history],
               historicalRowsScanned=scanned,background={},sources={})
    reference = None
    entries = []
    for location,name in a.background:
        folder = Path(location)
        m = read(folder/'manifest.json')
        if (not name_allowed(name) or m.get('format')!='uttt-regret-replay-v1' or
                m.get('excludedPackedSha256')!=PACKED_ORBITS_SHA256):
            raise ValueError('Background needs a verified published-suite exclusion')
        traits = {key:m[key] for key in ('student','teacher','base','excludedSetSha256',
                                         'excludedPackedSha256','excludedStates')}
        if reference is not None and traits != reference:
            raise ValueError('Background sources have different original model and suite identities')
        reference = traits
        item = next(k for k in m['sources'] if k['name']==name)
        shard = folder/(name+'.npz')
        if digest(shard)!=item['sha256']:
            raise ValueError('Background source digest changed')
        with np.load(shard,allow_pickle=False) as z:
            arrays = {k:z[k] for k in z.files}
        root,child = reachable_blocks(arrays['x'],protected)
        keep = ~(root|child)
        if keep.sum()<2:
            raise ValueError('Too few background rows after exclusions')
        clean = {k:v[keep] for k,v in arrays.items()}
        clean['hard'] = stratified_hardness(clean['x'],clean['hard'])
        dest = replay/(name+'.npz')
        np.savez_compressed(dest,**clean)
        entries.append(dict(name=name,rows=int(keep.sum()),sha256=digest(dest)))
        out['background'][name] = dict(input=len(keep),retained=int(keep.sum()),roots=int(root.sum()),
                                       parents=int(child.sum()),originalSha256=item['sha256'])
    offset = 0
    train_orbits = set()
    for name,part,source,info in parsed:
        sampled = part['train'][0]
        blob = search_labels(source,a.visits,info['analysisSeed'])
        order = aligned_order(sampled,blob['x'])
        x,p,v = (blob[k][order] for k in ('x','p','v'))
        root,child = reachable_blocks(x,protected)
        indices = []
        for i,signature in enumerate(canonical(x)):
            if not root[i] and not child[i] and signature not in train_orbits:
                indices.append(i)
                train_orbits.add(signature)
        if len(indices)<2:
            raise ValueError('Too few new original-engine roots after exclusions')
        x,p,v = x[indices],p[indices],v[indices]
        best = p.argmax(1).astype(np.uint8)
        dest = replay/(name+'.npz')
        np.savez_compressed(dest,**source_blob(x,p,v,best,best,np.zeros(len(x),np.float32)))
        entries.append(dict(name=name,rows=len(x),sha256=digest(dest)))
        held,groups = part['development']
        lineage = ~overlap[offset:offset+len(held)]
        offset += len(held)
        if not lineage.any():
            raise ValueError('All development roots overlap the known starting history')
        devfile = output/(name+'-development-features.npz')
        np.savez_compressed(devfile,x=held,opening=groups,lineage_clean=lineage)
        info.update(searchManifestSha256=digest(source/'manifest.json'),input=len(sampled),retained=len(indices),
                    directOverlap=int(root.sum()),parentOverlap=int(child.sum()),held=len(held),
                    cleanHeld=int(lineage.sum()),heldOpeningFamilies=len(set(map(bytes,groups))),
                    developmentFeaturesSha256=digest(devfile))
        out['sources'][name] = info
    write(replay/'manifest.json',dict(format='uttt-regret-replay-v1',**reference,sources=entries))
    out['replaySha256'] = digest(replay/'manifest.json')
    write(output/'manifest.json',out)
    return out


def freeze(a):
    manifest = read(a.data/'manifest.json')
    baseline,candidate = digest(a.baseline),digest(a.candidate)
    if (manifest['format']!='uttt-opponent-student-replay-v1' or baseline!=manifest['baselineSha256'] or
            candidate==baseline or a.candidate.stat().st_size>a.max_bytes):
        raise ValueError('Final graphs differ from the designated release baseline or budget')
    result = dict(format='uttt-opponent-student-frozen-v1',preparationSha256=digest(a.data/'manifest.json'),
                  baselineSha256=baseline,candidateSha256=candidate,bytes=a.candidate.stat().st_size)
    write(a.output,result)
    return result


def boot_delta(candidate,baseline,available,opening,seed):
    mask = np.asarray(available,bool)
    if not mask.any():
        return None
    delta = np.asarray(candidate)[mask]-np.asarray(baseline)[mask]
    groups,ix = np.unique(opening[mask],axis=0,return_inverse=True)
    counts = np.bincount(ix)
    totals = np.bincount(ix,weights=delta)
    draws = np.random.default_rng(seed).integers(len(groups),size=(10000,len(groups)))
    sampled = totals[draws].sum(1)/counts[draws].sum(1)
    return dict(rows=len(delta),openingFamilies=len(groups),meanDifference=float(delta.mean()),
                interval95=np.quantile(sampled,[.025,.975]).tolist() if len(groups)>1 else None)


def predictions(model, x, target, v, q, n):
    import onnxruntime as ort
    opt = ort.SessionOptions()
    opt.intra_op_num_threads = opt.inter_op_num_threads = 1
    sess = ort.InferenceSession(str(model),opt,providers=['CPUExecutionProvider'])
    logits,values = [],[]
    for start in range(0,len(x),96):
        batch = x[start:start+96]
        z,w = sess.run(['policy','value'],{'state':batch.astype(np.float32)})
        if (z.shape!=(len(batch),81) or w.shape!=(len(batch),1) or not np.isfinite(z).all()
                or not (np.abs(w)<=1.00001).all()):
            raise ValueError('Invalid sealed model output')
        masked = np.where(batch[:,162:243],z.astype(np.float64),-1e9)
        masked -= masked.max(1,keepdims=True)
        logits.append(masked-np.log(np.exp(masked).sum(1,keepdims=True)))
        values.extend(w[:,0])
    lp = np.concatenate(logits)
    y = np.asarray(values,dtype=float)
    chosen = lp.argmax(1)
    target = target.astype(float)
    target /= target.sum(1,keepdims=True)
    raw = target[np.arange(len(x)),chosen]
    error,supported = action_error(n,q,v,chosen)
    metrics = dict(policyCE=-(target*lp).sum(1),valueMSE=(y-v)**2,valueBias=y-v,
                   topSearchAction=(raw==target.max(1)).astype(float),chosenSearchMass=raw,
                   supportedActionRegret=error)
    return metrics,supported


def evaluate(a):
    m = read(a.data/'manifest.json')
    frozen = read(a.frozen)
    if (frozen['format']!='uttt-opponent-student-frozen-v1' or frozen['preparationSha256']!=digest(a.data/'manifest.json') or
            frozen['candidateSha256']!=digest(a.candidate) or frozen['baselineSha256']!=digest(a.baseline) or
            m['baselineSha256']!=frozen['baselineSha256']):
        raise ValueError('Candidate, baseline or prepared family splits changed after freezing')
    dirs = dict(a.development)
    if len(dirs)!=len(a.development) or set(dirs)!=set(m['sources']):
        raise ValueError('Provide exactly one development search per registered opponent')
    out = dict(format='uttt-frozen-opponent-student-development-v1',freezeSha256=digest(a.frozen),
               candidateSha256=frozen['candidateSha256'],baselineSha256=frozen['baselineSha256'],sources={})
    for name,where in dirs.items():
        meta = m['sources'][name]
        ref = Path(where)
        blob = search_labels(ref,m['visits'],meta['analysisSeed'])
        path = a.data/(name+'-development-features.npz')
        if digest(path)!=meta['developmentFeaturesSha256']:
            raise ValueError('Changed held-board identities or family assignments')
        with np.load(path,allow_pickle=False) as f:
            x,opening,clean = (f[k] for k in ('x','opening','lineage_clean'))
        order = aligned_order(x,blob['x'])
        args = [blob[k][order] for k in ('p','v','q','n')]
        candidate,ca = predictions(a.candidate,x,*args)
        baseline,ba = predictions(a.baseline,x,*args)
        source = dict(held=len(x),cleanHeld=int(clean.sum()),searchSha256=digest(ref/'manifest.json'),
                      allOpeningFamilies=len(np.unique(opening,axis=0)),metrics={})
        for k in baseline:
            mask = clean & (ca&ba if k=='supportedActionRegret' else True)
            delta = boot_delta(candidate[k],baseline[k],mask,opening,a.seed)
            if delta is not None:
                delta.update(candidate=float(candidate[k][mask].mean()),baseline=float(baseline[k][mask].mean()))
            source['metrics'][k] = delta
        out['sources'][name] = source
    write(a.output,out)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command',required=True)
    p = commands.add_parser('prepare')
    p.add_argument('--source',nargs=9,action='append',required=True,
                   metavar=('NAME','GAMES','SAMPLES','TRAIN_SEARCH','TRAIN_SEED','PAIRS','PROJECT_CAP','OPPONENT_CAP','SEARCH_SEED'))
    p.add_argument('--background',nargs=2,action='append',required=True,metavar=('REPLAY','SOURCE'))
    p.add_argument('--history',type=Path,action='append',required=True)
    p.add_argument('--protected-orbits',type=Path,action='append',default=[])
    p.add_argument('--suite',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--visits',type=int,default=8192)
    p.add_argument('--baseline-sha256',required=True)
    final = commands.add_parser('freeze')
    scan = commands.add_parser('evaluate')
    for sub in (final,scan):
        for key in ('data','candidate','baseline','output'):
            sub.add_argument('--'+key,required=True,type=Path)
    final.add_argument('--max-bytes',type=int,required=True)
    scan.add_argument('--frozen',type=Path,required=True)
    scan.add_argument('--development',nargs=2,action='append',required=True,metavar=('SOURCE','SEARCH'))
    scan.add_argument('--seed',type=int,default=26316593)
    a = parser.parse_args()
    if a.output.exists():
        parser.error('Use a fresh output')
    a.output.parent.mkdir(parents=True,exist_ok=True)
    if a.command=='prepare':
        if a.visits<2:
            parser.error('Expected a multi-visit search')
        with TemporaryDirectory(dir=a.output.parent) as tmp:
            folder=Path(tmp)/'replay';folder.mkdir()
            result=prepare(a,folder)
            folder.rename(a.output)
    else:
        result=(freeze if a.command=='freeze' else evaluate)(a)
    print(json.dumps(result,allow_nan=False,sort_keys=True),flush=True)


if __name__=='__main__':
    main()
