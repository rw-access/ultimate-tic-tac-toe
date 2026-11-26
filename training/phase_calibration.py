"""Fit and validate game-phase calibration on independent searched training shards."""

import argparse
import copy
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
import onnxruntime as ort

from training.calibrate_outputs import Scores, accepted, fit_scales, sample, split_files, summary, unique_rows
from training.pipeline import _symmetry_maps


def phases(states):
    moves = states[:, :162].sum(1)
    closed = states[:, 243:270].sum(1)
    return np.minimum((moves + 2 * closed) // 16, 4).astype(np.int64)


def before_tanh(source):
    model = copy.deepcopy(source)
    heads = [n for n in model.graph.node if n.op_type == 'Tanh']
    if len(heads) != 1:
        raise ValueError('Expected one final value tanh')
    head = heads[0]
    uses = [n for n in model.graph.node if head.output[0] in n.input]
    if 'value' not in head.output and (len(uses) != 1 or uses[0].op_type not in ('Reshape', 'Flatten')
                                        or list(uses[0].output) != ['value']):
        raise ValueError('Unrecognized layout after the value tanh')
    head.op_type = 'Identity'
    return model


def calibrate_graph(source, coefficients):
    model = before_tanh(source)
    g = model.graph
    inputs = {name for n in g.node for name in (*n.input, *n.output)}
    if any(k.startswith('phase/') for k in inputs) or set(v.name for v in g.output) != {'policy', 'value'}:
        raise ValueError('Expected a standalone, uncalibrated policy/value graph')
    for n in g.node:
        for i, name in enumerate(n.output):
            if name in ('policy', 'value'):
                if any(name in other.input for other in g.node):
                    raise ValueError('Output is reused internally')
                n.output[i] = 'phase/raw_' + name
    weight = np.zeros((280, 1), dtype=np.float32)
    weight[:162] = 1
    weight[243:270] = 2
    values = {'weight': weight, 'width': np.float32(16), 'max': np.int64(4)}
    for i, name in enumerate(('policy_gain','value_gain','value_bias')):
        values[name] = coefficients[:, i].astype(np.float32)
    for name, array in values.items():
        g.initializer.append(numpy_helper.from_array(np.asarray(array), 'phase/' + name))
    def node(op, ins, out, **attrs):
        names = [('phase/' + k) if k != 'state' else k for k in ins]
        g.node.append(helper.make_node(op, names, [out if out in ('policy','value') else 'phase/' + out], **attrs))
    node('MatMul', ['state','weight'], 'counter')
    node('Div', ['counter','width'], 'bucket')
    node('Cast', ['bucket'], 'integer', to=TensorProto.INT64)
    node('Min', ['integer','max'], 'index')
    for name in ('policy_gain','value_gain','value_bias'):
        node('Gather',[name,'index'],name+'_rows', axis=0)
    node('Mul',['raw_policy','policy_gain_rows'],'policy')
    node('Mul',['raw_value','value_gain_rows'],'scaled')
    node('Add',['scaled','value_bias_rows'],'shifted')
    node('Tanh',['shifted'],'value')
    onnx.checker.check_model(model)
    return model


def predict(model, states):
    opt = ort.SessionOptions()
    opt.intra_op_num_threads = opt.inter_op_num_threads = 1
    session = ort.InferenceSession(model.SerializeToString(), opt, providers=['CPUExecutionProvider'])
    batches = []
    for first in range(0, len(states), 64):
        x = states[first:first + 64].astype(np.float32)
        batches.append(session.run(['policy', 'value'], {'state':x}))
    p, v = (np.concatenate(head) for head in zip(*batches))
    if p.shape != (len(states),81) or v.shape != (len(states),1) or not np.isfinite(p).all() or not np.isfinite(v).all():
        raise ValueError('Unexpected model output')
    return p, v[:,0]


def main():
    q=argparse.ArgumentParser(description=__doc__)
    for name in ('input','data','output','report'):
        q.add_argument('--'+name, type=Path, required=True)
    q.add_argument('--seed',type=int,default=26315431)
    q.add_argument('--samples',type=int,default=18000)
    q.add_argument('--max-bytes',type=int,default=1480935)
    a=q.parse_args()
    if a.output.exists() or a.report.exists() or a.samples<512 or a.max_bytes<1:
        q.error('Use fresh outputs and at least 512 calibration rows')
    files=sorted(a.data.glob('*.npz'))
    train_paths, valid_paths=split_files(files,a.seed)
    sizes={}
    for path in files:
        with np.load(path,allow_pickle=False) as shard:
            sizes[path]=len(shard['v'])
    maps=np.stack([np.argsort(x) for x,_ in _symmetry_maps()])
    seen=set()
    training,validation=(unique_rows(sample(paths,sizes,n,a.seed+i),maps,seen)
                         for i,(paths,n) in enumerate(((train_paths,a.samples),(valid_paths,a.samples//2))))
    source=onnx.load(a.input);raw=before_tanh(source)
    factors=np.tile([1.,1.,0.],(5,1))
    p,z=predict(raw,training['x'])
    groups=phases(training['x'])
    for i in range(5):
        use=groups==i
        if use.sum()<128:
            continue
        scores=Scores(p[use],z[use],{k:v[use] for k,v in training.items()},False)
        fit=np.asarray(fit_scales(scores))
        factors[i]+=np.clip(fit-factors[i],[-.45,-.4,-.2],[.65,.6,.2])*.8
    p,z=predict(raw,validation['x'])
    g=phases(validation['x']);s=Scores(p,z,validation,False)
    before=[s.nll(1.),s.mse(1.,0.)]
    after=[s.nll(factors[g,:1]),s.mse(factors[g,1],factors[g,2])]
    reports=[summary(old,new,validation['file'],a.seed,valid_paths,a.data) for old,new in zip(before,after)]
    approved=[accepted(report) for report in reports]
    if not approved[0]:factors[:,0]=1.
    if not approved[1]:factors[:,1:]=[1.,0.]
    info=dict(sourceSha256=hashlib.sha256(a.input.read_bytes()).hexdigest(),trainRows=len(training['x']),
              heldoutRows=len(g),seed=a.seed,fitByPhase=factors.tolist(),policy=reports[0],value=reports[1],approved=approved)
    if any(approved):
        model=calibrate_graph(source,factors)
        payload=model.SerializeToString()
        if len(payload)>a.max_bytes:
            raise ValueError('Calibrated model exceeds the fixed byte limit')
        probe=validation['x'][:192]
        actual_p, actual_v=predict(model,probe)
        gain=factors[g[:len(probe)]]
        np.testing.assert_allclose(actual_p,p[:len(probe)]*gain[:,:1],atol=5e-5,rtol=2e-5)
        np.testing.assert_allclose(actual_v,np.tanh(z[:len(probe)]*gain[:,1]+gain[:,2]),atol=2e-5,rtol=2e-5)
        a.output.parent.mkdir(exist_ok=True,parents=True)
        with TemporaryDirectory(dir=a.output.parent) as temp:
            tmp=Path(temp)/'model';tmp.write_bytes(payload);a.output.hardlink_to(tmp)
        info.update(bytes=len(payload),sha256=hashlib.sha256(payload).hexdigest())
    a.report.parent.mkdir(exist_ok=True,parents=True)
    with a.report.open('x') as f:json.dump(info,f,indent=2);f.write('\n')
    print('Calibration holdout:',len(g),'rows;',[(reported['reduction'],ok) for reported,ok in zip(reports,approved)],flush=True)


if __name__=='__main__':
    main()
