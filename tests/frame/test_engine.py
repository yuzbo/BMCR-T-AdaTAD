import unittest
from unittest.mock import patch
import torch
from torch.nn import functional as F
from h65.frame.engine import PackedStateEngine
from h65.frame.contracts import EnginePolicy


class EngineTests(unittest.TestCase):
    def setUp(self):
        from opentad.models.backbones.vit_adapter import VisionTransformerAdapter
        torch.manual_seed(8);torch.set_num_threads(2)
        self.vit=VisionTransformerAdapter(img_size=32,patch_size=16,embed_dims=32,depth=12,num_heads=4,
                    num_frames=16,total_frames=32,adapter_index=list(range(12)),return_feat_map=True).eval()
        for block in self.vit.blocks:torch.nn.init.normal_(block.adapter.up_proj.weight,std=.02)
        self.engine=PackedStateEngine(32,8).eval();self.clips=torch.randn(2,3,16,32,32);self.valid=torch.ones(1,16,dtype=torch.bool)

    def test_actual_full_gate_identity(self):
        with torch.no_grad():
            expected=self.vit(self.clips);tokens,h,w,trace=self.engine(self.vit,self.clips,EnginePolicy(),self.valid)
            actual=self.vit.norm(tokens).reshape(2,8,h,w,32).permute(0,4,1,2,3)
        self.assertTrue(torch.allclose(expected,actual,atol=1e-6))
        self.assertEqual(trace['q'],[64]*12)

    def test_real_packed_calls_and_dense_mask_parity(self):
        policy=EnginePolicy(depth_schedule='amod',depth_ratio=.5,spatial_ratio=.5,use_light=True)
        calls=[];original=F.scaled_dot_product_attention
        def record(q,k,v,*a,**kw):calls.append((q.shape[0]*q.shape[-2],k.shape[0]*k.shape[-2]));return original(q,k,v,*a,**kw)
        mlp=[0]*12;hooks=[]
        for i,block in enumerate(self.vit.blocks):
            def hook(m,a,i=i):mlp[i]+=a[0].numel()//a[0].shape[-1]
            hooks.append(block.mlp.register_forward_pre_hook(hook))
        try:
            with torch.no_grad(),patch.object(F,'scaled_dot_product_attention',side_effect=record):
                compact,_,_,trace=self.engine(self.vit,self.clips,policy,self.valid)
        finally:
            for h in hooks:h.remove()
        self.assertEqual(mlp,trace['heavy_mlp']);self.assertEqual(trace['q'][0],64);self.assertEqual(trace['q'][-1],64)
        self.assertEqual(trace['q'][1],32);self.assertEqual(trace['kv'][1],32)
        self.assertEqual(sum(x[0] for x in calls),sum(trace['q']))
        self.assertEqual(sum(x[1] for x in calls),sum(trace['kv']))
        policy.mode='dense_mask'
        with torch.no_grad():dense,_,_,_=self.engine(self.vit,self.clips,policy,self.valid)
        self.assertTrue(torch.allclose(compact,dense,atol=2e-5,rtol=2e-4))

    def test_static_drop_retains_first_last(self):
        with torch.no_grad():_,_,_,trace=self.engine(self.vit,self.clips,EnginePolicy(static_depth=8),self.valid)
        self.assertGreater(trace['q'][0],0);self.assertGreater(trace['q'][-1],0)
        self.assertEqual(sum(x>0 for x in trace['q']),8)


if __name__=='__main__':unittest.main()
