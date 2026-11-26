import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

import numpy as np

from training.child_pair_curriculum import (COMMIT, MODELS, align_children, audit_history, digest, evaluate, family_split, keys,
                                            preference_metrics, prepare, rank_targets, source_blob, variants, write_json)
from training.distill_oracle import children
from training.regret_replay import excluded_rows
from training.suite_exclusion import PACKED_ORBITS_SHA256, PUBLISHED_SUITE_SHA256
from training.test_suite_exclusion import fixture_bytes


def roots(n, previous=False):
    rng = np.random.default_rng(2628)
    x = np.zeros((n, 280), np.uint8)
    x[:, 162:243] = x[:, 279:280] = 1
    for _ in range(16):
        before = x
        moves = [rng.choice(np.flatnonzero(z[162:243])) for z in x]
        x, terminal, _ = children(x, np.arange(n), np.array(moves))
        assert not terminal.any()
    return before if previous else x


def first_two(x):
    return np.array([np.flatnonzero(z[162:243])[:2] for z in x])


class ChildPairTests(unittest.TestCase):
    def test_direct_margin_and_half_credit(self):
        m = np.array([[3, 5], [12, 17], [5, 4]])
        q = np.array([[-.1, .4], [.5, .45], [.6, -.2]])
        a, b, gain = rank_targets(m, q, .12)
        np.testing.assert_array_equal(a, [3, 17, 4])
        np.testing.assert_array_equal(b, [5, 12, 5])
        np.testing.assert_allclose(gain, [1, 0, 1])
        result = preference_metrics(q[[0, 2]], np.array([-1, 0]), np.array([1, 1]))
        self.assertEqual((result['accuracy'], result['improved'], result['worsened']), (.75, 1, 1))
        self.assertAlmostEqual(result['meanPairRegret'], .2)

    def test_grouped_split_transitive_and_stable_under_reordering(self):
        x = roots(40)
        m = first_two(x)
        s = children(x, np.repeat(np.arange(len(x)), 2), m.ravel())[0]
        # Link two families through a reflected common child, then a third through a parent.
        from training.pipeline import _symmetry_maps
        perm = _symmetry_maps()[3][0]
        s[3] = s[0, perm]
        s[5] = x[1, perm]
        held, protected, count = family_split(x, s, 2294, .3)
        self.assertEqual(count, len(x)-2)
        self.assertTrue(held[0] == held[1] == held[2])
        protected_keys = {row.tobytes() for row in protected}
        for batch in (x[~held], s[np.repeat(~held, 2)]):
            self.assertFalse(any(k in protected_keys for orbit in variants(batch) for k in orbit))
        order = np.arange(len(x))[::-1]
        again, encoded, _ = family_split(x[order], s.reshape(-1, 2, 280)[order].reshape(-1, 280), 2294, .3)
        np.testing.assert_array_equal(again[::-1], held)
        np.testing.assert_array_equal(encoded, protected)

    def test_child_sign_and_terminal_utility(self):
        x = np.zeros((1, 280), np.uint8)
        x[0, [0,1,2,9,10,11,18,19]] = 1
        x[0, 81+np.array([27,29,31,36,40,45,49,54])] = 1
        x[0, 243:245] = 1
        x[0, 272] = 1
        x[0, 162+20:162+27] = 1
        m = np.array([[20, 21]])
        cx = children(x, [0, 0], m.ravel())[0][1:]
        q = np.array([[1., -.375]])
        _, ids = align_children(x, m, cx, np.array([.375]), q)
        np.testing.assert_array_equal(ids, [[-1, 0]])
        with self.assertRaisesRegex(ValueError, 'sign'):
            align_children(x, m, cx, np.array([.375]), -q)

    def test_end_to_end_with_budget_and_family_leakage_checks(self):
        with TemporaryDirectory() as t:
            root = Path(t)
            target, searched, bg, out = (root/key for key in ('targets','search','background','output'))
            for path in (target, searched, bg, out, target/'diagnostics'):
                path.mkdir()
            suite = root/'suite.py'
            suite.write_bytes(fixture_bytes())
            x = roots(48)
            m = first_two(x)
            cx = children(x, np.repeat(np.arange(len(x)), 2), m.ravel())[0]
            cv = np.linspace(-.85, .85, len(cx)).astype(np.float32)
            cv[1::2] *= -1
            q = -cv.reshape(-1, 2)
            p = x[:, 162:243].astype(np.float32)
            p /= p.sum(1, keepdims=True)
            n = np.zeros((len(cx),81), np.uint16)
            n[np.arange(len(cx)), cx[:,162:243].argmax(1)] = 8191
            cp = (n/8191).astype(np.float16)
            v = np.zeros(len(x), np.float32)
            path = searched/'shard-00000.npz'
            np.savez(path, x=cx, p=cp, v=cv, n=n)
            upstream = dict(format='uttt-published-search-targets-v1', modelSha256=sorted(MODELS)[0],
                            upstreamCommit=COMMIT, visitsPerRoot=8192, roots=len(cx), rootVisits=len(cx)*8192,
                            shards=[dict(file=path.name, rows=len(cx), sha256=digest(path))])
            write_json(searched/'manifest.json', upstream)
            specs = []
            for i,(a,b,c) in enumerate(((x,p,v),(cx,cp,cv))):
                path = target/f'shard-{i:05d}.npz'
                np.savez(path, x=a, p=b, v=c)
                specs.append(dict(file=path.name, rows=len(a), sha256=digest(path)))
            path = target/'diagnostics'/'pair-evidence.npz'
            np.savez(path, x=x, moves=m, childValuesFromParent=q, rootActionValues=-q, initialRootPolicy=p,
                     engineDisagreement=np.ones(len(x), bool))
            write_json(target/'manifest.json', dict(format='uttt-counterfactual-search-targets-v1',
                       roots=len(x), children=len(cx), shards=specs, disagreementRoots=len(x),
                       childSearchSha256=digest(searched/'manifest.json'), pairEvidenceSha256=digest(path)))
            g = m[:,0].astype(np.uint8)
            np.savez(bg/'deep.npz', **source_blob(x, p, v, g, g, v))
            write_json(bg/'manifest.json', dict(format='uttt-regret-replay-v1', teacher='a', base='b', student='c',
                       excludedPackedSha256=PACKED_ORBITS_SHA256, excludedSetSha256=PUBLISHED_SUITE_SHA256,
                       excludedStates=353, sources=[dict(name='deep', sha256=digest(bg/'deep.npz'))]))
            args = SimpleNamespace(targets=target, search_manifest=searched/'manifest.json', suite=suite,
                                   background=bg, min_margin=.12, visits=8192, seed=310, holdout_fraction=.25)
            receipt = prepare(args, out)
            self.assertEqual(receipt['rankReversals'], len(x))
            self.assertGreater(receipt['heldAboveMargin'], 0)
            manifest = json.loads((out/'replay/manifest.json').read_bytes())
            protected = np.load(out/'held-orbits.npy')
            self.assertEqual(manifest['excludedPackedSha256'], PACKED_ORBITS_SHA256)
            for src in manifest['sources']:
                with np.load(out/'replay'/(src['name']+'.npz')) as rows:
                    self.assertFalse(any(z.any() for z in excluded_rows(rows['x'], protected)))
            write_json(out/'manifest.json', receipt)
            with np.load(out/'held-pairs.npz') as bh:
                held_x = bh['x']
            history = root/'history'
            history.mkdir()
            which = {k:i for i,k in enumerate(keys(x))}
            parent = roots(len(x), previous=True)[which[keys(held_x)[1]]]
            from training.pipeline import _symmetry_maps
            rot = _symmetry_maps()[3][0]
            np.savez(history/'old.npz', x=np.stack((held_x[0,rot],parent[rot])))
            write_json(history/'manifest.json', dict(format='uttt-regret-replay-v1',
                       sources=[dict(name='old', sha256=digest(history/'old.npz'))]))
            record = root/'history.json'
            audit = audit_history(SimpleNamespace(data=out, replay=[history], output=record))
            self.assertEqual(audit['historicalOverlap'], 2)
            self.assertEqual(json.loads(record.read_bytes())['excludedIndices'], [0, 1])
            # Run the public CPU evaluator on a dynamic ONNX to check ties and
            # that the inherited exclusions apply to every reported subgroup.
            import onnx
            from onnx import TensorProto, helper, numpy_helper
            graph = helper.make_graph([helper.make_node('MatMul',['state','weights'],['policy'])], 'test',
                 [helper.make_tensor_value_info('state',TensorProto.FLOAT,['N',280])],
                 [helper.make_tensor_value_info('policy',TensorProto.FLOAT,['N',81])],
                 [numpy_helper.from_array(np.zeros((280,81),np.float32),'weights')])
            model = root/'tie.onnx'
            onnx.save(helper.make_model(graph,ir_version=9,opset_imports=[helper.make_opsetid('',17)]),model)
            checked = evaluate(SimpleNamespace(data=out,model=model,anchor=model,history_audit=record))
            self.assertEqual(checked['afterHistoryExclusion'],len(held_x)-2)
            self.assertEqual(checked['aboveMargin']['accuracy'],.5)
            self.assertEqual(checked['aboveMargin']['improved'],0)
            args.visits = 512
            with self.assertRaisesRegex(ValueError, 'budget'):
                prepare(args, root)
