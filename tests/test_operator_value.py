import unittest
import torch
from h65.paper.operator_value import depth_mask,spatial_mask,ff_natives_quota,apply_exchange,OperatorValueRouter


class OperatorValueTests(unittest.TestCase):
    def test_exact_packed_and_nested_native_counts(self):
        torch.manual_seed(3)
        valid=torch.ones(3,800,dtype=torch.bool);valid[1,600:]=False;valid[2,100:]=False
        score=torch.randn(3,800)
        for uniform in (True,False):
            d=depth_mask(score,valid,.75,uniform)
            s=spatial_mask(score,d,valid,.5,uniform)
            self.assertEqual(d.sum(-1).tolist(),[600,450,75])
            self.assertEqual(s.sum(-1).tolist(),[400,300,50])
            self.assertFalse(bool((s&~d).any()))
            self.assertTrue(torch.equal(s.reshape(3,8,100).sum(-1),ff_natives_quota(d,valid,.5)))

    def test_exchange_keeps_capacity(self):
        valid=torch.ones(1,16,dtype=torch.bool)
        d=depth_mask(torch.arange(16.)[None],valid,.75)
        new=apply_exchange(d,dict(axis='D',layer=4,pack=0,remove=15,insert=0),'D',4,valid)
        self.assertEqual(int((d!=new).sum()),2)
        self.assertEqual(int(d.sum()),int(new.sum()))

    def test_value_gradient_stays_in_controller(self):
        torch.manual_seed(42)
        state=torch.randn(2,16,8,requires_grad=True)
        valid=torch.ones(2,16,dtype=torch.bool)
        router=OperatorValueRouter(8,['D'])
        value,_=router('D',state,valid,torch.randn(2,16,3),4,12)
        (value[0,0]-value[0,1]).square().sum().backward()
        self.assertIsNone(state.grad)
        self.assertGreater(sum(float(p.grad.abs().sum()) for p in router.parameters()),0)


if __name__=='__main__':unittest.main()
