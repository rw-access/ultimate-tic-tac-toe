import unittest

import numpy as np
import torch

from training.consensus_replay import consensus, supported_pairs, view_maps
from training.pipeline import _symmetry_maps


class ConsensusReplayTests(unittest.TestCase):
    def test_consensus_is_equivariant_even_for_an_orientation_biased_evaluator(self):
        state = torch.zeros(2,280)
        state[:,162:243] = 1
        state[0,162+7] = 0
        state[1,162+21] = 0
        state[0,2] = state[1,10] = 1
        def predictor(x):
            return torch.arange(81).float().repeat(len(x),1)/16 + x[:,:81], (.8*x[:,2:3]-.6*x[:,10:11])
        maps = view_maps(torch.device('cpu'))
        p,v,js,sd,moves = consensus(predictor,state,maps)
        self.assertTrue((js>0).all() and (sd>0).all())
        self.assertEqual(moves.shape,(2,8))
        np.testing.assert_allclose(p.sum(1),1,rtol=1e-6)
        self.assertTrue((p[state[:,162:243]==0]==0).all())
        for features, actions in _symmetry_maps():
            transformed = state[:,np.argsort(features)]
            pp,vv,jj,ss,_ = consensus(predictor,transformed,maps)
            np.testing.assert_allclose(pp[:,actions],p,rtol=2e-5,atol=1e-7)
            np.testing.assert_allclose(vv,v,atol=1e-6)
            np.testing.assert_allclose(jj,js,atol=1e-6)
            np.testing.assert_allclose(ss,sd,atol=1e-6)

    def test_legacy_exclusion_receipt_is_accepted_only_for_actually_clean_rows(self):
        import json
        from pathlib import Path
        from tempfile import TemporaryDirectory
        from types import SimpleNamespace
        from unittest.mock import Mock, patch
        from training.child_pair_curriculum import source_blob
        from training.consensus_replay import build
        from training.fit_search_student import file_hash
        from training.suite_exclusion import PACKED_ORBITS_SHA256
        from training.test_child_pair_curriculum import roots
        from training.test_suite_exclusion import fixture_bytes, feature_row

        class SmallModel(torch.nn.Module):
            def forward(self, state):
                return state[:,:81], state[:,:1] * .2

        with TemporaryDirectory() as tmp:
            d = Path(tmp)
            out = d/'output'
            out.mkdir()
            args = SimpleNamespace(data=d, input=d/'input', base=d/'base', suite=d/'suite', output=out,
                                   batch_size=2, fixtures=Path(__file__).with_name('parity.json'), weight=.85,
                                   device='cpu')
            args.base.write_bytes(b'base')
            args.input.write_bytes(b'frozen')
            args.suite.write_bytes(fixture_bytes())
            def write(x):
                p = x[:,162:243].astype(np.float32)
                p /= p.sum(1, keepdims=True)
                g = p.argmax(1).astype(np.uint8)
                v = np.zeros(len(x), np.float32)
                shard = d/'deep.npz'
                np.savez_compressed(shard, **source_blob(x,p,v,g,g,v))
                info = dict(format='uttt-regret-replay-v1',base=file_hash(args.base),student='original',
                            excludedSetSha256='legacy-serialized-npy-digest',
                            sources=[dict(name='deep',sha256=file_hash(shard))])
                (d/'manifest.json').write_text(json.dumps(info))
            x = roots(3)
            with patch('training.consensus_replay.student_from',return_value=(Mock(), SmallModel(), {})), \
                    patch('training.consensus_replay.check_parity',return_value={}):
                write(x)
                result = build(args, out)
                self.assertEqual(result['excludedPackedSha256'], PACKED_ORBITS_SHA256)
                with np.load(out/'deep.npz') as checked:
                    self.assertTrue(np.isfinite(checked['teacher']).all())
                x[0] = feature_row([0])
                write(x)
                with self.assertRaisesRegex(ValueError, 'published suite'):
                    build(args, out)

    def test_ranking_requires_actual_source_and_consensus_support(self):
        p = np.array([[.48,.39,.13],[.2,.6,.2],[.25,.75,0.]])
        ensemble = np.array([[.60,.35,.05],[.60,.30,.10],[.1,.7,.2]])
        views = np.array([[0,1,0],[0,1,0],[0,2,1]])
        a,b,g = supported_pairs(p[:2],ensemble[:2],views[:2],False)
        np.testing.assert_array_equal(a,[1,0])
        np.testing.assert_array_equal(b,[0,0])
        np.testing.assert_allclose(g,[.18,0])
        a,b,g = supported_pairs(p[2:],ensemble[2:],views[2:],True)
        np.testing.assert_array_equal([a[0],b[0],g[0]],[2,1,1])
