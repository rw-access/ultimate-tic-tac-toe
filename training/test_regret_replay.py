import unittest

import numpy as np
import torch

from training.refine_packed import Codebook
from training.regret_replay import AdaptiveBook, board_choice_loss, child_scores, excluded_rows, objective, orbit_holdout, ranking_evidence, stored_affines, stratified_hardness, validate_alternative_manifest


class RegretTest(unittest.TestCase):
    def test_board_target_compares_total_legal_mass_and_skips_forced_and_exact(self):
        logits = torch.zeros(3, 81, requires_grad=True)
        legal = torch.zeros(3, 81, dtype=torch.bool)
        legal[0, [0, 1, 10]] = True
        legal[1, [2, 3]] = True
        legal[2, [0, 9]] = True
        target = torch.zeros_like(logits)
        target[0, 0], target[1, 2], target[2, 9] = 1, 1, 1
        exact = torch.tensor([False, False, True])
        loss = board_choice_loss(logits, legal, target, exact)
        self.assertAlmostEqual(loss.item(), -np.log(2 / 3), places=6)
        loss.backward()
        self.assertLess(logits.grad[0, 1].item(), 0)
        self.assertGreater(logits.grad[0, 10].item(), 0)
        self.assertEqual(logits.grad[~legal].abs().sum().item(), 0)
        self.assertEqual(logits.grad[1:].abs().sum().item(), 0)
        self.assertEqual(board_choice_loss(logits[1:], legal[1:], target[1:], exact[1:]).item(), 0)

    def test_best_cell_board_target_prefers_the_highest_individual_move(self):
        legal = torch.zeros(1, 81, dtype=torch.bool)
        legal[0, [0, 1, 9]] = True
        target = torch.zeros(1, 81)
        target[0, [0, 1, 9]] = torch.tensor([.3, .3, .4])
        logits = torch.zeros_like(target, requires_grad=True)
        loss = board_choice_loss(logits, legal, target, torch.tensor([False]), best_cell=True)
        loss.backward()
        self.assertAlmostEqual(loss.item(), np.log(2), places=6)
        self.assertLess(logits.grad[0, 9].item(), 0)
        self.assertGreater(logits.grad[0, :2].sum().item(), 0)
        self.assertEqual(logits.grad[~legal].abs().sum().item(), 0)

    def test_agreement_or_prior_support_and_terminal_proof(self):
        target = np.array([[.8,.15,.05],[.75,.20,.05],[.1,.75,.15],[.9,.1,0.]])
        choices = np.array([[0,1,0],[0,1,0],[0,2,1],[1,0,0]])
        q = np.array([[0,.7,0],[0,1,0],[0,.95,.6],[.7,.65,.65]])
        terminal = np.zeros_like(q, bool)
        terminal[1,1] = True
        better, gain = ranking_evidence(target, choices, q, terminal, False)
        np.testing.assert_array_equal(better, [0,1,1,1])
        np.testing.assert_allclose(gain, [0,1,.6,0])

    def test_child_regret_uses_root_perspective_and_exact_win(self):
        x = np.zeros((1, 280), np.uint8)
        x[0,[0,1,2,9,10,11,18,19]] = 1
        x[0,81+np.array([27,29,31,36,40,45,49,54])] = 1
        x[0,243:245] = 1
        x[0,272] = 1
        x[0,162+20:162+27] = 1
        calls = []
        def teacher(states):
            calls.append(len(states))
            return torch.zeros(len(states),81), torch.full((len(states),1),.375)
        q, terminal = child_scores(x, np.array([[20,21,20]]), teacher, torch.device("cpu"))
        np.testing.assert_array_equal(terminal, [[True,False,True]])
        np.testing.assert_allclose(q, [[1.,-.375,1.]])
        self.assertEqual(calls, [1])

    def test_exact_ties_and_inferior_incumbent(self):
        target = np.array([[.12,.88,0],[.12,.88,0]])
        choices = np.array([[0,1,1],[2,1,1]])
        q = np.zeros((2,3))
        better, gain = ranking_evidence(target, choices, q, q.astype(bool), True)
        np.testing.assert_array_equal(better, [0,1])
        np.testing.assert_array_equal(gain, [0,1])

    def test_ranks_track_ties_and_phase(self):
        x = np.zeros((6,280),np.uint8)
        x[3:,:24]=1
        r = stratified_hardness(x, np.array([0,0,1,150,150,160]))
        np.testing.assert_allclose(r[:3],r[3:])
        self.assertAlmostEqual(r[0],r[1])
        self.assertLess(r[0],r[2])

    def test_unversioned_proof_requires_pinned_complete_parent_and_child_shards(self):
        import hashlib
        import json
        from pathlib import Path
        from tempfile import TemporaryDirectory
        from training.suite_exclusion import PUBLISHED_SUITE_SHA256
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            shard = root/"shard-00000.npz"
            np.savez_compressed(shard,v=np.array([-1.,1.],np.float32))
            sha = lambda f: hashlib.sha256(f.read_bytes()).hexdigest()
            proof = {"sourceManifestSha256":"1"*64, "sourceTeacherSha256":"2"*64, "solverSha256":"3"*64,
                     "excludedSuiteSha256":PUBLISHED_SUITE_SHA256, "minRemaining":15, "maxRemaining":28,
                     "nodesPerRoot":4096, "nodes":4040, "eligible":3, "attemptedRoots":1, "solvedRoots":1,
                     "attemptedChildren":1, "solvedChildren":1, "terminalChildren":0,
                     "shards":[{"file":shard.name,"rows":2,"sha256":sha(shard)}]}
            manifest=root/"manifest.json"
            manifest.write_text(json.dumps(proof))
            validate_alternative_manifest(root,sha(manifest))
            with self.assertRaisesRegex(ValueError,"pinned digest"):
                validate_alternative_manifest(root,"f"*64)
            proof["shards"][0]["rows"] = 3
            manifest.write_text(json.dumps(proof))
            with self.assertRaisesRegex(ValueError,"rows disagree"):
                validate_alternative_manifest(root,sha(manifest))

    def test_policy_head_control_freezes_shared_and_value_parameters(self):
        from torch.nn.utils import parametrize
        from training.regret_replay import policy_head_parameters
        model = torch.nn.Module()
        model.layers = torch.nn.ModuleDict({n: torch.nn.Linear(2, 2) for n in ('trunk', 'v1', 'policy', 'logits')})
        model.requires_grad_(False)
        books, biases = {}, {}
        for name, layer in model.layers.items():
            book = Codebook(np.array([[.1,.3,.5,.7]], np.float32), np.array([[0,1,2,3]], np.int64))
            parametrize.register_parametrization(layer, 'weight', book)
            books[name] = book
            biases['layers.' + name + '.bias'] = layer.bias.requires_grad_(True)
        kept, direct = policy_head_parameters(model, books, biases)
        self.assertEqual(set(kept), {'policy', 'logits'})
        self.assertEqual(set(direct), {'layers.policy.bias', 'layers.logits.bias'})
        z = torch.tensor([[.5, -.3]])
        trunk = model.layers['trunk'](z)
        initial_value = model.layers['v1'](trunk).detach().clone()
        original_policy = model.layers['logits'](model.layers['policy'](trunk)).detach().clone()
        opt = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=.1)
        model.layers['logits'](model.layers['policy'](trunk)).sum().backward()
        opt.step()
        trunk = model.layers['trunk'](z)
        self.assertTrue(torch.equal(model.layers['v1'](trunk), initial_value))
        self.assertFalse(torch.equal(model.layers['logits'](model.layers['policy'](trunk)), original_policy))

    def test_warm_start_reads_trained_direct_bias_tables(self):
        import onnx
        from onnx import numpy_helper
        model = torch.nn.Sequential(torch.nn.Linear(2, 3))
        raw = onnx.ModelProto()
        values = np.array([.5, -.25, 1.], np.float32)
        raw.graph.initializer.add().CopyFrom(numpy_helper.from_array(values, "0.bias"))
        with torch.no_grad():
            for key, parameter, stored in stored_affines(raw, model):
                self.assertEqual(key, "0.bias")
                parameter.copy_(torch.from_numpy(stored.copy()))
        np.testing.assert_array_equal(model[0].bias.detach().numpy(), values)
        raw.graph.initializer[0].CopyFrom(numpy_helper.from_array(values[:1], "0.bias"))
        with self.assertRaisesRegex(ValueError, "Invalid stored bias"):
            list(stored_affines(raw, model))

    def test_recoding_moves_only_useful_entries_and_preserves_forward_values(self):
        centers = np.array([[0.,.25,.50,.75]],np.float32)
        codes = np.array([[0,1,2,3]],np.uint8)
        book = AdaptiveBook(Codebook(centers,codes),3)
        with torch.no_grad():
            book.residual[0] = torch.tensor([.24,0,-.24,-.5])
        self.assertEqual(book.packed_bytes(),bytes([228]))
        dense = book(torch.empty(3))
        torch.testing.assert_close(dense,torch.tensor([0,.25,.5]))
        dense.sum().backward()
        torch.testing.assert_close(book.residual.grad,torch.tensor([[1.,1.,1.,0.]]))
        self.assertEqual(book.reassign().item(),2)
        self.assertEqual(book.codes.tolist(),[[1,1,1,3]])
        self.assertEqual(book.packed_bytes(),bytes([213]))
        torch.testing.assert_close(book(torch.empty(3)),torch.tensor([.25,.25,.25]))

    def test_recoding_respects_per_step_and_total_change_limits(self):
        centers=np.array([[0.,.25,.5,.75]],np.float32)
        codes=np.zeros((1,16),np.uint8)
        book=AdaptiveBook(Codebook(centers,codes),16)
        with torch.no_grad():
            book.residual.fill_(.26)
        self.assertEqual(book.reassign(.125,.25).item(),2)
        self.assertEqual(book.reassign(.125,.25).item(),2)
        self.assertEqual(book.reassign(.125,.25).item(),0)
        self.assertEqual(torch.count_nonzero(book.codes).item(),4)

    def test_validation_orbits_never_split_a_state_and_its_augmented_copies(self):
        from training.pipeline import _symmetry_maps
        x=np.random.default_rng(77).integers(0,2,(128,280),dtype=np.uint8)
        fold=orbit_holdout(x,17,fraction=4)
        self.assertGreater(fold.sum(),0)
        self.assertLess(fold.sum(),len(x))
        for m,_ in _symmetry_maps():
            np.testing.assert_array_equal(orbit_holdout(x[:,m],17,fraction=4),fold)

    def test_conservative_exclusion_removes_parents_and_equal_roots(self):
        from training.distill_oracle import children
        x=np.zeros((1,280),np.uint8)
        x[:,162:243]=1
        x[:,279]=1
        child=children(x,np.array([0]),np.array([13]))[0]
        sibling=children(x,np.array([0]),np.array([14]))[0]
        root,parent=excluded_rows(np.concatenate([x,child,sibling]),np.packbits(child,axis=1))
        np.testing.assert_array_equal(root,[0,1,0])
        np.testing.assert_array_equal(parent,[1,0,0])

    def test_exact_value_ignores_teacher_and_mask_preserves_tied_optima(self):
        states = torch.zeros(2,280)
        states[:,162:165] = 1
        p = torch.zeros(2,81)
        p[0,:2] = .5
        p[1,2] = 1
        v = torch.tensor([[1.],[-1.]])
        ref_v = torch.tensor([[-1.],[1.]])
        model_v = torch.tensor([[1.],[0.]])
        logits = torch.zeros(2,81,requires_grad=True)
        source = torch.tensor([1,2])
        _, mse, rank = objective(logits, model_v, states, p, v, p, ref_v, source, source,
                                 torch.zeros(2), torch.full((2,1),.5),torch.tensor([True,False]),.5)
        self.assertEqual(float(mse),0)
        self.assertEqual(float(rank),0)


if __name__ == "__main__":
    unittest.main()
