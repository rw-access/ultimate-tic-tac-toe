import unittest

import numpy as np
import torch

from training.child_pair_curriculum import leaf_preferences
from training.distill_oracle import children
from training.sibling_value_readout import fit_readout, hidden_projection, parent_values, sibling_targets


class SiblingValueTest(unittest.TestCase):
    def test_compressed_projection_equals_padded_dense_readout(self):
        codes=torch.tensor([[0,1,1,2],[2,0,0,0]])
        weights=torch.tensor([[.1,.2,-.5],[.3,.4,.5]])
        dense=weights.gather(1,codes).flatten()[:5]
        hidden=torch.tensor([[.3,-1.,2.,0.,.3],[.1,.4,.2,.1,1.]])
        grouped=hidden @ hidden_projection(codes,3,5)
        torch.testing.assert_close(grouped @ weights.ravel(),hidden @ dense)
        with self.assertRaisesRegex(ValueError,'assignment'):
            hidden_projection(codes+3,3,5)

    def test_terminal_and_perspective_match_fresh_search(self):
        x=np.zeros((1,280),np.uint8)
        x[0,[0,1,2,9,10,11,18,19]]=1
        x[0,81+np.array([27,29,31,36,40,45,49,54])]=1
        x[0,243:245]=1
        x[0,272]=1
        x[0,162+20:162+27]=1
        moves=np.array([[21,20]],np.uint8)
        states,terminal,_=children(x,np.array([0,0]),moves.ravel().astype(np.intp))
        _,t,q=sibling_targets(x,moves,states[~terminal],np.array([.25]),.12)
        np.testing.assert_array_equal(t,[[False,True]])
        np.testing.assert_allclose(q,[[-.25,1.]])
        class Echo:
            def run(self, fields, args):
                assert fields==['value'] and len(args['state'])==1
                return [np.array([[.25]],np.float32)]
        np.testing.assert_array_equal(leaf_preferences(Echo(),x,moves),q)
        with self.assertRaisesRegex(ValueError,'parent ranking'):
            sibling_targets(x,moves[:,::-1],states[~terminal],np.array([.25]),.12)

    def test_coupled_fit_improves_target_and_terminal_has_zero_gradient(self):
        z=torch.tensor([[1.],[-1.],[0.],[.3]],dtype=torch.float64)
        pair=torch.stack([z[:2],z[:2].flip(0)],1)
        w=torch.tensor([0.,0.],dtype=z.dtype)
        terminal=torch.tensor([[False,False],[False,True]])
        q=torch.tensor([[-.4,.4],[.4,1.]],dtype=z.dtype)
        fitted=fit_readout(z,torch.zeros(4,dtype=z.dtype),pair,terminal,q,w,1.,45)
        truth=q[:,1]-q[:,0]
        old=parent_values(pair,w,terminal,q)
        new=parent_values(pair,fitted,terminal,q)
        self.assertLess(((new[:,1]-new[:,0]-truth)**2).mean(),((old[:,1]-old[:,0]-truth)**2).mean())
        w=w.requires_grad_()
        parent_values(pair,w,terminal,q)[1,1].backward()
        torch.testing.assert_close(w.grad,torch.zeros_like(w))


if __name__=='__main__':
    unittest.main()
