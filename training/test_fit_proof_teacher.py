import unittest
import torch
from training.fit_proof_teacher import choose_heads, objective
from training.stage2_student import Stage2Net


class ProofTeacherTests(unittest.TestCase):
    def test_selecting_output_heads_freezes_the_trunk(self):
        for head, active in [('policy', {'policy','upsample','logits','logit_out'}),
                             ('value', {'v1','vpool1','v2','vpool2','value','value_out'})]:
            model=Stage2Net(compact=False)
            choose_heads(model,head)
            for name, layer in model.layers.items():
                self.assertTrue(all(p.requires_grad == (name in active) for p in layer.parameters()))

    def test_optimal_support_ignores_illegal_logits_and_handles_ties(self):
        x=torch.zeros((2,280))
        x[:,162:165]=1
        p=torch.zeros((2,81))
        p[:,0:2]=.5
        logits=torch.zeros((2,81))
        logits[:,3:]=1000
        def loss(s):
            return objective(s,torch.zeros(2),logits,torch.zeros(2),x,p,torch.tensor([1.,-1.]),
                             torch.ones(2,dtype=torch.bool),torch.zeros(2),1.)[0].item()
        self.assertAlmostEqual(loss(logits),loss(logits.masked_fill(x[:,162:243]==0,-500)),places=6)
        preferred=logits.clone()
        preferred[:,:2]+=3
        self.assertLess(loss(preferred),loss(logits))
