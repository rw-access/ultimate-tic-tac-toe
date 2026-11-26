import hashlib
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

from training.child_pair_curriculum import COMMIT, MODELS, source_blob
from training.distill_oracle import children
from training.opponent_student import (aligned_order, boot_delta, evaluate, freeze, prepare, reachable_blocks,
                                      source_metadata)
from training.pipeline import _symmetry_maps
from training.search_alternatives import digest, keys
from training.suite_exclusion import PACKED_ORBITS_SHA256
from training.test_child_pair_curriculum import roots
from training.test_suite_exclusion import fixture_bytes


def write(path, data):
    path.write_text(json.dumps(data))


def model(path, bias):
    init = [numpy_helper.from_array(np.array([i],np.int64),n) for n,i in
            [('start',0),('end',81),('axis',1),('vstart',279),('vend',280)]]
    init += [numpy_helper.from_array(np.array([bias],np.float32),'constant')]
    nodes = [helper.make_node('Slice',['state','start','end','axis'],['policy']),
             helper.make_node('Slice',['state','vstart','vend','axis'],['z']),
             helper.make_node('Mul',['z','constant'],['value'])]
    vi = lambda n,w: helper.make_tensor_value_info(n,TensorProto.FLOAT,['batch',w])
    graph = helper.make_graph(nodes,'tiny',[vi('state',280)],[vi('policy',81),vi('value',1)],init)
    m = helper.make_model(graph,opset_imports=[helper.make_opsetid('',17)])
    m.ir_version = 9
    onnx.save(m,path)


def searched(folder, states, seed=31):
    folder.mkdir()
    legal = states[:,162:243]
    best = legal.argmax(1)
    n = np.zeros(legal.shape,np.uint16)
    n[np.arange(len(n)),best] = 63
    p = (n/63).astype(np.float16)
    q = .3*p
    shard = folder/'shard-00000.npz'
    np.savez_compressed(shard,x=states,p=p,v=np.linspace(-.5,.5,len(n),dtype=np.float32),q=q,n=n)
    write(folder/'manifest.json',dict(format='uttt-published-search-targets-v1',roots=len(n),
             rootVisits=64*len(n),visitsPerRoot=64,seed=seed,upstreamCommit=COMMIT,modelSha256=sorted(MODELS)[0],
             shards=[dict(file=shard.name,rows=len(n),sha256=digest(shard))]))


def fixture(folder):
    first,second = folder/'baseline.onnx',folder/'candidate.onnx'
    model(first,.15)
    model(second,-.15)
    b = digest(first)
    choices = {}
    transforms = np.stack([move for _,move in _symmetry_maps()])
    for move in range(81):
        h = hashlib.sha256(min(bytes(row) for row in transforms[:,[move]].astype(np.uint8))).digest()
        split = 'development' if int.from_bytes(h[:4],'big')%5==0 else 'train'
        choices.setdefault(split,([move],h))
    game = folder/'games.json'
    sample = folder/'samples'
    sample.mkdir()
    games = []
    gm = dict(format='uttt-opponent-training-positions-v1',trainingSeed=19,games=4,candidateSha256=b,
              officialSha256=sorted(MODELS)[0],projectVisits=64,officialVisits=256,splits={})
    boards = roots(64)
    for i,(split,which) in enumerate([('train',boards[:12]),('development',boards[12:18])]):
        opening,h = choices[split]
        games.append(dict(pair=i,opening=opening,games=[{'seat':'X'},{'seat':'O'}]))
        sub = sample/split;sub.mkdir()
        f = sub/'positions.npz'
        # No targets, outcomes or played actions are present in the raw development fixture.
        np.savez_compressed(f,x=which,opening_pair=np.full(len(which),i,np.int32),
                            opening_orbit_sha256=np.tile(np.frombuffer(h,np.uint8),(len(which),1)))
        gm['splits'][split] = dict(rows=len(which),pairs=1,sha256=digest(f))
    write(game,dict(format='official-uttt-arena-v1',status='complete',openingSuite='generated',
          openingSchedule='independent',seed=19,pairs=2,maxRootVisits=64,officialMaxRootVisits=256,
          candidate=dict(sha256=b),official=dict(sha256=sorted(MODELS)[0]),results=games))
    gm['gameReportSha256'] = digest(game)
    write(sample/'manifest.json',gm)
    train = folder/'searched-train'
    searched(train,boards[:12][::-1])
    spec = ['training',game,sample,train,19,2,64,256,31]
    bg = folder/'background';bg.mkdir()
    x = np.concatenate((boards[12:13],boards[24:48]))
    p = x[:,162:243].astype(np.float32)
    p = (p/p.sum(1,keepdims=True)).astype(np.float16)
    top = p.argmax(1).astype(np.uint8)
    f = bg/'deep.npz'
    np.savez_compressed(f,**source_blob(x,p,np.zeros(len(x),np.float32),top,top,np.zeros(len(x),np.float32)))
    write(bg/'manifest.json',dict(format='uttt-regret-replay-v1',base='b',teacher='t',student='s',
           excludedStates=353,excludedSetSha256='x',excludedPackedSha256=PACKED_ORBITS_SHA256,
           sources=[dict(name='deep',rows=len(x),sha256=digest(f))]))
    suite = folder/'suite.py';suite.write_bytes(fixture_bytes())
    args=SimpleNamespace(source=[spec],background=[(bg,'deep')],history=[bg],suite=suite,protected_orbits=[],
                         visits=64,baseline_sha256=b)
    return args,first,second,boards


class OpponentStudentTest(unittest.TestCase):
    def test_blind_preparation_freeze_and_sealed_evaluation(self):
        with TemporaryDirectory() as tmp:
            root=Path(tmp)
            args,first,second,boards=fixture(root)
            data=root/'prepared';data.mkdir()
            trained=prepare(args,data)
            self.assertEqual(trained['sources']['training']['held'],6)
            self.assertEqual(trained['sources']['training']['cleanHeld'],5)
            self.assertEqual(trained['background']['deep']['roots'],1)
            self.assertFalse((root/'searched-development').exists())
            frozen=root/'frozen.json'
            freeze(SimpleNamespace(data=data,baseline=first,candidate=second,max_bytes=100_000,output=frozen))
            sealed=root/'searched-development'
            searched(sealed,boards[12:18][::-1])
            run=SimpleNamespace(data=data,frozen=frozen,baseline=first,candidate=second,
                                development=[('training',sealed)],output=root/'score.json',seed=47)
            result=evaluate(run)['sources']['training']
            self.assertEqual(result['cleanHeld'],5)
            self.assertEqual(result['metrics']['policyCE']['meanDifference'],0.)
            self.assertIsNone(result['metrics']['valueMSE']['interval95'])
            run.candidate=first
            with self.assertRaisesRegex(ValueError,'changed after freezing'):
                evaluate(run)

    def test_budget_identity_and_alignment_reject_wrong_sets(self):
        with TemporaryDirectory() as tmp:
            args,_,_,boards=fixture(Path(tmp))
            wrong=args.source[0].copy();wrong[-2]=512
            with self.assertRaisesRegex(ValueError,'provenance or budgets'):
                source_metadata(wrong,args.baseline_sha256)
            with self.assertRaisesRegex(ValueError,'complete designated partition'):
                aligned_order(boards[:3],boards[[0,0,2]])
            z=boards[:3]
            next_state=children(z[:1],np.array([0]),np.array([z[0,162:243].argmax()]))[0]
            root,child=reachable_blocks(z,{keys(next_state)[0],keys(z[1:2])[0]})
            self.assertEqual(root.tolist(),[False,True,False])
            self.assertEqual(child.tolist(),[True,False,False])

    def test_bootstrap_keeps_correlated_openings_together(self):
        delta=np.array([1.,1.,-1.,-1.])
        family=np.array([[0],[0],[1],[1]])
        stat=boot_delta(delta,np.zeros(4),np.ones(4,bool),family,77)
        self.assertEqual((stat['openingFamilies'],stat['meanDifference']),(2,0.))
        self.assertEqual(stat['interval95'],[-1.,1.])
        self.assertIsNone(boot_delta(delta,delta,np.zeros(4,bool),family,77))


if __name__=='__main__':
    unittest.main()
