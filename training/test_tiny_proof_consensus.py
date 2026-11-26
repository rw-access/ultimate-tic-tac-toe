import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from training.line_student import LineStudent
from training.tiny_proof_consensus import ProofLine, consensus, normalized_loss


class TinyProofConsensusTests(unittest.TestCase):
    def test_pairs_by_exact_state_and_rejects_missing_or_duplicate_roots(self):
        x = np.zeros((3, 280), np.uint8)
        x[np.arange(3), np.arange(3)] = 1
        p = np.zeros((3,81), np.float16)
        p[:, :2] = [[1,0],[.5,.5],[0,1]]
        v = np.array([1,.3,-1],np.float32)
        idx = np.arange(3)
        a = x,p,v,idx
        out = consensus(a, (x[::-1],p[::-1],v[::-1],idx))
        np.testing.assert_allclose(out[1], p)
        np.testing.assert_allclose(out[2], v)
        np.testing.assert_allclose(out[4], 1, atol=1e-6)
        changed = x.copy()
        changed[1,77] = 1
        for bad in (changed, x[[0,0,2]]):
            with self.assertRaisesRegex(ValueError, 'states differ|unique roots'):
                consensus(a, (bad,p,v,idx))

    def test_disjoint_searches_have_low_confidence_and_finite_legal_targets(self):
        x = np.zeros((1,280),np.uint8)
        p = np.zeros((1,81),np.float16)
        p[0,1]=1
        q = np.zeros_like(p)
        q[0,4]=1
        v=np.ones(1,np.float32)
        f=np.zeros(1,int)
        _,policy,value,_,pc,vc=consensus((x,p,v,f),(x,q,-v,f))
        np.testing.assert_allclose(policy,(p+q)*.5)
        np.testing.assert_allclose(value,0)
        np.testing.assert_allclose(pc,.2)
        np.testing.assert_allclose(vc,.1)

    def test_loss_does_not_penalize_different_equally_proved_moves(self):
        x=torch.zeros(2,280)
        x[:,162:165]=1
        p=torch.zeros(2,81)
        p[:,:2]=.5
        logit=torch.zeros(2,81)
        logit[0,0]=7
        logit[1,1]=7
        loss,vl,mass=normalized_loss(logit,torch.ones(2,1),logit*0,torch.zeros(2,1),
            x,p,torch.ones(2),torch.zeros(2),torch.ones(2),torch.ones(2))
        torch.testing.assert_close(loss[0],loss[1])
        self.assertLess(mass.max().item(),.002)
        self.assertEqual(vl.sum().item(),0)

    def test_pack_keeps_kernel_groups_and_public_tensor_names(self):
        from onnx import TensorProto, helper, numpy_helper
        from training.pack_tiny_proof import mixed_weights, shorten_internal_names

        rng = np.random.default_rng(118)
        labels = ('base', 'late_policy/first', 'late_policy/last', 'late_value/first', 'late_value/last')
        nodes, initializers = [], []
        previous = 'state'
        for i, label in enumerate(labels):
            name = f'weight-{i}'
            out = 'policy' if i == len(labels) - 1 else f'hidden-{i}'
            initializers.append(numpy_helper.from_array(rng.normal(size=(8, 8)).astype(np.float32), name))
            nodes.append(helper.make_node('MatMul', [previous, name], [out], name=label))
            previous = out
        graph = helper.make_graph(nodes, 'synthetic',
                                  [helper.make_tensor_value_info('state', TensorProto.FLOAT, ['batch', 8])],
                                  [helper.make_tensor_value_info('policy', TensorProto.FLOAT, ['batch', 8])],
                                  initializer=initializers)
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid('', 14)], ir_version=8)
        candidate, groups = mixed_weights(model)
        self.assertEqual(groups, {f'weight-{i}': 64 for i in range(1, 5)})
        old_size = candidate.ByteSize()
        shorten_internal_names(candidate)
        self.assertEqual([vi.name for vi in candidate.graph.input], ['state'])
        self.assertEqual([vi.name for vi in candidate.graph.output], ['policy'])
        self.assertLess(candidate.ByteSize(), old_size)

    @torch.inference_mode()
    def test_warm_start_and_early_positions_stay_at_incumbent(self):
        torch.manual_seed(123)
        base=LineStudent().eval()
        model=ProofLine().eval()
        with tempfile.TemporaryDirectory() as td:
            path=Path(td)/'base.pt'
            torch.save(base.state_dict(),path)
            model.warm_start(path)
        states=json.loads((Path(__file__).parent/'parity.json').read_bytes())['cases']
        x=torch.tensor([s['encoded'] for s in states],dtype=torch.float32)
        expected=base(x)
        for actual,original in zip(model(x),expected):
            torch.testing.assert_close(actual,original,atol=1e-7,rtol=1e-6)
        model.late_policy[-1].bias.fill_(1.2)
        model.late_value[-1].bias.fill_(.7)
        lp,v,_,_,gate=model.predictions(x)
        before=gate.flatten()==0
        self.assertTrue(before.any() and (~before).any())
        torch.testing.assert_close(lp[before],expected[0][before],atol=1e-7,rtol=1e-6)
        torch.testing.assert_close(v[before],expected[1][before],atol=1e-7,rtol=1e-6)
        self.assertTrue((v[~before]!=expected[1][~before]).all())
        self.assertTrue(all(name.startswith('late_') for name,p in model.named_parameters() if p.requires_grad))


if __name__=='__main__':
    unittest.main()
