import copy
from types import SimpleNamespace
import unittest
import torch
from h65.frame.contracts import EnginePolicy
from h65.frame.measure import matrix_counter
from h65.paper.engine import PackedStateEngine
from h65.paper.geometry import candidate_mask,feature_target_data,interpolate_anchors
from h65.paper.profile import encoder_macs
from h65.paper.routing import BudgetRouter


class PaperContracts(unittest.TestCase):
    def test_anet_candidate_and_detector_axes(self):
        data=dict(inputs=torch.zeros(1,1,3,768,2,2),masks=torch.ones(1,192,dtype=torch.bool),
                  gt_segments=[torch.tensor([[24.,48.]])])
        self.assertEqual(candidate_mask(data).shape,(1,768))
        self.assertTrue(torch.equal(feature_target_data(data)['gt_segments'][0],torch.tensor([[96.,192.]])))

    def test_repeated_real_frame_times_and_gradients(self):
        features=torch.tensor([[[1.],[3.],[5.]]],requires_grad=True)
        anchors=SimpleNamespace(features=features,centers=torch.tensor([[0.,0.,2.]]),valid=torch.ones(1,3,dtype=torch.bool))
        queries=SimpleNamespace(centers=torch.tensor([[0.,1.,2.]]),valid=torch.ones(1,3,dtype=torch.bool))
        result=interpolate_anchors(anchors,queries)
        self.assertTrue(torch.equal(result[0,:,0],torch.tensor([2.,3.5,5.])))
        result.sum().backward()
        self.assertTrue(torch.allclose(features.grad[0,:,0],torch.tensor([.75,.75,1.5])))

    def test_budget_never_selects_unaffordable_plan(self):
        router=BudgetRouter();router.cost_gflops.copy_(torch.arange(len(router.menu)).float()+1)
        router.reference_gflops.fill_(10.)
        mean=torch.zeros(1,len(router.menu),2);mean[:,10]=100
        picked,_=router.choose(torch.zeros(1,196),.5,0.,(mean,torch.zeros_like(mean)))
        self.assertLessEqual(float(router.cost_gflops[picked]),5.)
        self.assertEqual(int(picked),4)

    def test_vectorized_frame_route_preserves_original_pairs_and_features(self):
        from h65.transport import sample_rates
        from h65.frame.router import candidate_pairs as original_pairs,ActionRouter
        from h65.paper.routing import candidate_pairs,FrameRouter
        torch.manual_seed(12);masks=torch.ones(2,128,dtype=torch.bool);masks[1,93:]=False
        output=dict(hidden=torch.randn(2,128,96,dtype=torch.bfloat16),action_logits=torch.randn(2,128,dtype=torch.bfloat16))
        selected=sample_rates(torch.randn(2,128),masks,64,alpha=1.)
        for scope in ('local','global'):
            expected=original_pairs(output,selected,masks,scope);actual=candidate_pairs(output,selected,masks,scope)
            self.assertEqual(expected,actual)
            self.assertTrue(torch.equal(ActionRouter().features(output,selected,masks,expected),FrameRouter().features(output,selected,masks,actual)))

    def test_24_layer_full_gate_and_real_cost_ledger(self):
        from opentad.models.backbones.vit_adapter import VisionTransformerAdapter
        torch.set_num_threads(2);torch.manual_seed(8)
        vit=VisionTransformerAdapter(img_size=32,patch_size=16,embed_dims=32,depth=24,num_heads=4,
                                     num_frames=16,total_frames=32,adapter_index=list(range(24)),return_feat_map=True).eval()
        for block in vit.blocks:torch.nn.init.normal_(block.adapter.up_proj.weight,std=.02)
        engine=PackedStateEngine(32,24).eval();clips=torch.randn(2,3,16,32,32);valid=torch.ones(1,16,dtype=torch.bool)
        policy=EnginePolicy(static_depth=24,mod_layers=tuple(range(1,23,2)))
        with torch.no_grad():
            expected=vit(clips);tokens,h,w,trace,taps=engine(vit,clips,policy,valid,(12,18,24))
        actual=vit.norm(tokens).reshape(2,8,h,w,32).permute(0,4,1,2,3)
        self.assertTrue(torch.allclose(expected,actual,atol=1e-6,rtol=1e-5))
        self.assertEqual(set(taps),{12,18,24})
        model=SimpleNamespace(encoder=SimpleNamespace(vit=vit,engine=engine))
        for partial in (False,True):
            use=valid.clone()
            if partial:use[:,11:]=False
            policy.depth_schedule='amod';policy.depth_ratio=.5;policy.spatial_ratio=.5
            with torch.no_grad(),matrix_counter() as counter:
                _,_,_,trace,_=engine(vit,clips,policy,use)
            self.assertEqual(encoder_macs(model,trace),sum(counter.macs.values()))
            self.assertGreater(trace['q'][0],trace['q'][1]);self.assertEqual(trace['q'][0],trace['q'][-1])

    def test_real_student_head_updates_without_changing_reference(self):
        from mmengine.config import Config
        from h65.paper.readout import TaskReadout
        from pathlib import Path
        root=Path(__file__).resolve().parents[2]
        cfg=Config.fromfile(str(root/'upstream/configs/adatad/thumos/e2e_thumos_videomae_s_768x1_160_adapter.py')).model
        student=TaskReadout(cfg,trainable=True);reference=TaskReadout(cfg,trainable=False)
        reference.detector.load_state_dict(student.detector.state_dict(),strict=True)
        before={k:v.clone() for k,v in reference.state_dict().items()}
        data=dict(masks=torch.ones(1,768,dtype=torch.bool),metas=[{}],gt_segments=[torch.tensor([[100.,180.]])],gt_labels=[torch.tensor([0])])
        native=torch.randn(1,384,384,requires_grad=True)
        normalizer=student.detector.rpn_head.loss_normalizer.clone()
        cls=student.detector.rpn_head.cls_head.weight.detach().clone()
        loss=student.loss(native,data)['cost']
        self.assertTrue(torch.equal(normalizer,student.detector.rpn_head.loss_normalizer))
        loss.backward();torch.optim.SGD(student.parameters(),lr=.01).step();student.commit_normalizer()
        self.assertTrue(torch.isfinite(native.grad).all());self.assertGreater(float(native.grad.abs().sum()),0.)
        self.assertFalse(torch.equal(cls,student.detector.rpn_head.cls_head.weight))
        for k,v in reference.state_dict().items():self.assertTrue(torch.equal(before[k],v),k)

    def test_mixed_budget_checkpoint_keeps_each_tia_axis(self):
        from opentad.models.backbones.vit_adapter import VisionTransformerAdapter
        torch.set_num_threads(2);torch.manual_seed(4)
        vit=VisionTransformerAdapter(img_size=32,patch_size=16,embed_dims=32,depth=4,num_heads=4,
            num_frames=16,total_frames=32,adapter_index=list(range(4)),return_feat_map=True).eval()
        for block in vit.blocks:torch.nn.init.normal_(block.adapter.up_proj.weight,std=.02)
        reference=copy.deepcopy(vit);engine=PackedStateEngine(32,4).train();plain=copy.deepcopy(engine).eval()
        first=torch.randn(2,3,16,32,32,requires_grad=True);second=torch.randn(4,3,16,32,32,requires_grad=True)
        a=first.detach().clone().requires_grad_();b=second.detach().clone().requires_grad_()
        policy=EnginePolicy(static_depth=4,mod_layers=(1,))
        def combined(network,runner,inputs):
            outputs=[]
            for value in inputs:
                length=len(value)*8
                for block in network.blocks:block.adapter.temporal_size=length
                outputs.append(runner(network,value,policy,torch.ones(1,length,dtype=torch.bool))[0].square().mean())
            sum(outputs).backward()
        combined(vit,engine,(first,second));combined(reference,plain,(a,b))
        self.assertTrue(torch.allclose(first.grad,a.grad,atol=1e-6,rtol=1e-4))
        self.assertTrue(torch.allclose(second.grad,b.grad,atol=1e-6,rtol=1e-4))


if __name__=='__main__':unittest.main()
