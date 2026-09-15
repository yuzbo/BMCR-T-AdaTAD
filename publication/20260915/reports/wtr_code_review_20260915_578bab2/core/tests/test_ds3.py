import copy
import unittest
import torch
from mmengine.config import Config
from pathlib import Path
from h65.ds3.model import DenseTeacher,ClipEngine
from h65.ds3.auxiliary import Auxiliaries
from h65.ds3.routes import Policy,route_depths,uniform_ids,interpolate_native,native_geometry
from h65.ds3.losses import boundary_weights
from h65.ds3.runtime import optimizer,scheduler

ROOT=Path(__file__).resolve().parents[1]


def tiny_teacher(scope='local'):
    cfg=Config.fromfile(str(ROOT/'upstream/configs/adatad/thumos/e2e_thumos_videomae_s_768x1_160_adapter.py'))
    cfg.model.backbone.backbone.update(embed_dims=32,num_heads=2,img_size=32,drop_path_rate=0.)
    cfg.model.projection.update(in_channels=32,out_channels=32)
    cfg.model.neck.update(in_channels=32,out_channels=32)
    cfg.model.rpn_head.update(in_channels=32,feat_channels=32)
    model=DenseTeacher(cfg.model,scope=scope)
    with torch.no_grad():
        for block in model.vit.blocks: block.adapter.up_proj.weight.normal_(0,.02)
    return model


class DS3Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)

    def test_80_epoch_scheduler_continues_through60(self):
        module=torch.nn.Linear(2,2);optim=optimizer(module);schedule=scheduler(optim)
        self.assertAlmostEqual(optim.param_groups[0]['lr'],1e-5)
        values={}
        for step in range(1,8001):
            optim.step();schedule.step()
            if step in (100,500,6000,8000):values[step]=optim.param_groups[0]['lr']
        self.assertGreater(values[100],1e-5);self.assertAlmostEqual(values[500],1e-4)
        self.assertGreater(values[6000],0);self.assertEqual(values[8000],0)

    def test_original_clip_uniform_and_nested_coverage(self):
        masks=torch.ones(2,768,dtype=torch.bool);masks[1,253:]=False
        logits=torch.randn(2,48,2)
        for policy in (Policy('t'),Policy('td',clips8=24,clips12=12),Policy('random',selection='random')):
            depths=route_depths(logits,masks,policy)
            for row,count in enumerate((48,16)):
                self.assertEqual(int((depths[row]>=8).sum()),min(count,policy.clips8))
                self.assertEqual(int((depths[row]==12).sum()),min(count,policy.clips12))
                ids=(depths[row]==12).nonzero().flatten()
                self.assertEqual(ids[0],0);self.assertEqual(ids[-1],count-1)
                self.assertLessEqual(int((ids[1:]-ids[:-1]).max()),24)
        self.assertEqual(uniform_ids(1,24).tolist(),[0])
        self.assertTrue(torch.equal(route_depths(logits,masks,Policy('r',selection='random')),
                                    route_depths(logits,masks,Policy('r',selection='random'))))

    def test_physical_native_interpolation_and_partial_pair(self):
        positions=torch.tensor([1.,5.,17.]);query=torch.tensor([-1.,1.,3.,5.,11.,17.,19.])
        values=torch.stack((2*positions+3,-positions),-1)
        actual=interpolate_native(values,positions,query)
        expected=torch.stack((2*query.clamp(1,17)+3,-query.clamp(1,17)),-1)
        torch.testing.assert_close(actual,expected)
        mask=torch.zeros(1,768,dtype=torch.bool);mask[:,:3]=True
        indices=torch.tensor([0,4,8]+[8]*765)
        valid,frames,centers=native_geometry(mask,[{'frame_inds':indices}])
        self.assertEqual(int(valid.sum()),2)
        self.assertEqual(centers[0,:2].tolist(),[2.,8.])

    def test_local_clip_equivalence_and_true_prefix_execution(self):
        teacher=tiny_teacher();engine=ClipEngine(teacher)
        clips=torch.randn(6,3,16,32,32)
        with torch.no_grad():
            dense=teacher.pool_maps(teacher.vit(clips))
            ids=torch.tensor([0,2,5])
            compact=teacher.pool_maps(teacher.vit(clips[ids]))
            torch.testing.assert_close(dense[ids],compact,atol=1e-5,rtol=1e-5)
            full=engine.execute(clips,torch.full((1,6),12))
            torch.testing.assert_close(full['at12'],dense,atol=1e-5,rtol=1e-5)
            seen=[[] for _ in range(12)];hooks=[]
            def capture(index):
                def hook(module,args):seen[index].append(args[0].shape[0])
                return hook
            for i,block in enumerate(teacher.vit.blocks):hooks.append(block.attn.register_forward_pre_hook(capture(i)))
            try:
                output=engine.execute(clips,torch.tensor([[12,0,8,0,12,8]]))
            finally:
                for hook in hooks:hook.remove()
            self.assertEqual(seen,[[4]]*8+[[2]]*4)
            self.assertEqual(output['trace']['attention_clips'],[4]*8+[2]*4)
            torch.testing.assert_close(output['at12'],dense[[0,4]],atol=1e-5,rtol=1e-5)
            poisoned=clips.clone();poisoned[[1,3]]=float('nan')
            poison_output=engine.execute(poisoned,torch.tensor([[12,0,8,0,12,8]]))
            torch.testing.assert_close(poison_output['at12'],output['at12'])

    def test_global_and_local_tia_are_not_silently_equated(self):
        teacher=tiny_teacher();clips=torch.randn(48,3,16,32,32)
        with torch.no_grad():
            local=teacher.vit(clips)
            for block in teacher.vit.blocks:block.adapter.temporal_size=384
            global_output=teacher.vit(clips)
        self.assertGreater(float((local-global_output).abs().max()),1e-5)
        teacher.scope='global'
        with self.assertRaises(ValueError):ClipEngine(teacher)

    def test_sparse_mlp_executes_selected_tokens_then_restores_raster(self):
        teacher=tiny_teacher();aux=Auxiliaries(32).eval();engine=ClipEngine(teacher)
        block=teacher.vit.blocks[8];x=torch.randn(2,32,32)
        with torch.no_grad():
            attended=x+block.attn(block.norm1(x));normalized=block.norm2(attended)
            scores=aux.spatial_scores['8'](normalized).squeeze(-1).reshape(2,8,4)
            ids=scores.argsort(dim=-1,descending=True,stable=True)[...,:2].sort(-1).values
            ids=(ids+torch.arange(8)[None,:,None]*4).flatten(1)
            expected_residual=aux.surrogates['8'](normalized)
            exact=block.mlp(normalized)
            expected_residual=expected_residual.scatter(1,ids[...,None].expand(-1,-1,32),exact.gather(1,ids[...,None].expand(-1,-1,32)))
            expected=block.adapter(attended+expected_residual,2,2)
            seen=[]
            def capture(module,args):seen.append(args[0].numel()//32)
            hook=block.mlp.register_forward_pre_hook(capture)
            try:
                trace=dict(attention_clips=[0]*12,attention_tokens=[0]*12,heavy_mlp_tokens=[0]*12)
                actual=engine.block(x,8,2,2,.5,aux,trace)
            finally:hook.remove()
            torch.testing.assert_close(actual,expected,atol=1e-5,rtol=1e-5)
            self.assertEqual(seen,[32]);self.assertEqual(trace['heavy_mlp_tokens'][8],32)

    def test_real_boundary_weights_and_frozen_detector_input_gradient(self):
        masks=torch.ones(1,768,dtype=torch.bool)
        weight,valid=boundary_weights(masks,[torch.tensor([[0.,40.]])],[torch.tensor([[False,True]])])
        self.assertEqual(float(weight[0,0]),1.)
        self.assertGreater(float(weight[0,20]),2.)
        teacher=tiny_teacher()
        native=torch.randn(1,32,384,requires_grad=True)
        normalizer=teacher.detector.rpn_head.loss_normalizer.clone()
        loss=teacher.loss(native,masks,[{}],[torch.tensor([[10.,40.]])],[torch.tensor([0])])['cost']
        gradient=torch.autograd.grad(loss,native)[0]
        self.assertTrue(torch.isfinite(gradient).all());self.assertGreater(float(gradient.abs().sum()),0)
        self.assertTrue(all(p.grad is None and not p.requires_grad for p in teacher.parameters()))
        torch.testing.assert_close(teacher.detector.rpn_head.loss_normalizer,normalizer)


if __name__=='__main__':unittest.main()
