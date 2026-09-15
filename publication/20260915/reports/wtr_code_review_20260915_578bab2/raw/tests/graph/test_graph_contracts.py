import copy
import math
import unittest
from types import SimpleNamespace
import torch
import torch.nn.functional as F
from h65.paper.edge_ops import coalesce_topk,initial_edges,indexed_attention,EdgeRouter,GraphKVAttention,GraphMessage,gather_nodes
from h65.paper.graph_recovery import deposit,gather_contributors,anchor_consistency,GraphRecovery
from h65.paper.graph_frames import protect_coverage,GraphFrameRouter
from h65.paper.routing import FrameRouter,candidate_pairs
from h65.frame.geometry import make_anchors,make_queries
from h65.transport import Selection


def selection(ids,total):
    ids=torch.tensor([ids]);valid=torch.ones_like(ids,dtype=torch.bool)
    rates=torch.full((1,total),len(ids[0])/total)
    return Selection(ids,ids.float(),valid,torch.full_like(rates,1/total),rates)


class GraphContracts(unittest.TestCase):
    def setUp(self):torch.manual_seed(42);torch.set_num_threads(2)

    def test_coalesce_value_and_retained_weight_gradient(self):
        ids=torch.tensor([[[0,3,0,2]]]);w=torch.tensor([[[.2,.1,.3,.4]]],requires_grad=True)
        out,mass=coalesce_topk(ids,w,2)
        self.assertEqual(out.tolist(),[[[0,2]]])
        reference=torch.stack((w[...,0]+w[...,2],w[...,3]),-1);reference=reference/reference.sum(-1,keepdim=True)
        torch.testing.assert_close(mass,reference)
        a=torch.autograd.grad(mass.square().sum(),w,retain_graph=True)[0]
        b=torch.autograd.grad(reference.square().sum(),w)[0]
        torch.testing.assert_close(a,b,rtol=1e-5,atol=1e-6)
        self.assertEqual(float(a[...,1]),0.)

    def test_indexed_attention_matches_dense_value_and_gradients(self):
        b,n,h,d=2,7,2,4
        q=torch.randn(b,n,h,d,requires_grad=True);k=torch.randn_like(q,requires_grad=True);v=torch.randn_like(q,requires_grad=True)
        raw=torch.randn(b,n,3,requires_grad=True);weights=raw.softmax(-1)
        ids=(torch.arange(n)[None,:,None]+torch.tensor([0,1,3])[None,None])%n;ids=ids.expand(b,-1,-1)
        valid=torch.ones(b,n,dtype=torch.bool);valid[1,-2:]=False
        sparse,_,_=indexed_attention(q,k,v,ids,weights,valid,chunk=5)
        adjacency=weights.new_zeros((b,n,n)).scatter_add(-1,ids,weights)
        logits=q.transpose(1,2)@k.transpose(1,2).transpose(-1,-2)/math.sqrt(d)
        logits=logits+adjacency.clamp_min(1e-12).log().masked_fill(adjacency==0,-torch.inf)[:,None]
        dense=(logits.softmax(-1)@v.transpose(1,2)).transpose(1,2).reshape(b,n,h*d)*valid[...,None]
        torch.testing.assert_close(sparse,dense,rtol=1e-5,atol=1e-6)
        probe=torch.randn_like(sparse)
        left=torch.autograd.grad((sparse*probe).sum(),(q,k,v,raw),retain_graph=True)
        right=torch.autograd.grad((dense*probe).sum(),(q,k,v,raw))
        for a,z in zip(left,right):torch.testing.assert_close(a,z,rtol=3e-5,atol=2e-6)

    def test_invalid_nodes_and_batch_isolation(self):
        valid=torch.tensor([[True]*12,[True]*5+[False]*7])
        x=torch.randn(2,12,16,requires_grad=True);geo=torch.zeros(2,12,5)
        module=GraphMessage(16)
        y,ids,w,record=module(x,valid,geo)
        self.assertTrue(torch.isfinite(y).all());self.assertTrue((y[1,5:]==0).all())
        self.assertTrue((gather_nodes(valid,ids)|(w==0)).all())
        changed=x.detach().clone();changed[1]*=100
        z,*_=module(changed,valid,geo)
        torch.testing.assert_close(y[0],z[0],rtol=0,atol=0)
        gradient=torch.autograd.grad(y[0].sum(),x)[0]
        self.assertTrue((gradient[1]==0).all())
        empty=torch.zeros_like(valid)
        zero,*_=module(x,empty,geo)
        self.assertTrue(torch.isfinite(zero).all());self.assertTrue((zero==0).all())

    def test_referral_discovers_bounded_new_destinations(self):
        valid=torch.ones(1,64,dtype=torch.bool);x=torch.randn(1,64,8);geo=torch.zeros(1,64,5)
        router=EdgeRouter(8,referrals=1)
        seed_i,seed_w=initial_edges(valid)
        chain_i=torch.arange(64)[None,:,None].expand(1,-1,16).clone();chain_i[...,0]=(torch.arange(64)+5)%64
        chain_w=torch.zeros(1,64,16);chain_w[...,0]=1
        (found,mass),record=router(x,valid,geo,(chain_i,chain_w))
        self.assertNotIn(30,seed_i[0,20][seed_w[0,20]>0].tolist())
        self.assertIn(30,found[0,20][mass[0,20]>0].tolist())  # 20 -> 25 -> 30
        self.assertEqual(record['referral_paths'],64*16)
        self.assertEqual(record['candidate_slots'],64*48)
        no=EdgeRouter(8,mode='no_referral',referrals=0)
        _,control=no(x,valid,geo,(seed_i,seed_w))
        self.assertEqual(control['referral_paths'],0)

    def test_real_contributors_not_packed_positions(self):
        masks=torch.ones(1,8,dtype=torch.bool);chosen=selection([0,1,4,7],8)
        features=torch.tensor([[[10.,1.],[20.,2.]]],requires_grad=True)
        metas=[dict(frame_inds=list(range(100,132,4)))]
        anchors=make_anchors(features,chosen,masks,metas);queries=make_queries(masks,metas,chosen)
        value,coverage,quality=deposit(anchors,queries)
        torch.testing.assert_close(value[0,0],features[0,0]);torch.testing.assert_close(value[0,2],features[0,1])
        torch.testing.assert_close(value[0,3],features[0,1]);self.assertTrue((value[0,1]==0).all())
        torch.testing.assert_close(coverage,torch.tensor([[1.,0.,.5,.5]]))
        self.assertLess(float(quality[0,2]),float(quality[0,0]))
        recovered=value.detach().transpose(1,2).requires_grad_()
        loss=anchor_consistency(recovered,anchors,queries)
        self.assertLess(float(loss),1e-8)
        loss.backward();self.assertIsNone(features.grad)

    def test_zero_output_preserves_cross_and_learning_can_start(self):
        masks=torch.ones(1,32,dtype=torch.bool);chosen=selection(list(range(0,32,2)),32)
        feature=torch.randn(1,8,24);metas=[dict(frame_inds=list(range(0,128,4)))]
        anchors=make_anchors(feature,chosen,masks,metas);queries=make_queries(masks,metas,chosen)
        module=GraphRecovery(24,width=16);cheap=torch.randn(1,16,96);cross=torch.randn(1,24,16)
        state,_,_=module('start',queries,cheap=cheap)
        state,_,_=module('step',queries,state=state,anchors=anchors,level=12)
        result,_=module('readout',queries,state=state,anchors=anchors,cross=cross)
        torch.testing.assert_close(result,cross,rtol=0,atol=0)
        result.square().mean().backward()
        self.assertGreater(float(module.output.weight.grad.abs().sum()),0.)
        self.assertGreater(float(module.anchor_gate.weight.grad.abs().sum()),0.)

    def test_coverage_and_all_candidate_exchanges_preserve_bins(self):
        masks=torch.ones(1,64,dtype=torch.bool);chosen=selection(list(range(16)),64)
        preview=dict(rate_logits=torch.arange(64).float()[None],hidden=torch.randn(1,64,96),action_logits=torch.randn(1,64),coverage_bins=8)
        repaired,changes=protect_coverage(chosen,preview,masks,8)
        ids=repaired.indices[0,repaired.valid[0]];counts=torch.bincount(ids//8,minlength=8)
        self.assertTrue((counts>0).all());self.assertEqual(len(ids.unique()),16);self.assertGreater(len(changes),0)
        for _,remove,insert in candidate_pairs(preview,repaired,masks,'global'):
            self.assertTrue(remove//8==insert//8 or counts[remove//8]>1)
        router=GraphFrameRouter(FrameRouter());preview['graph_context']=torch.randn(1,32,128)
        features=router.features(preview,repaired,masks,[(0,int(ids[0]),int((masks[0]&~torch.zeros(64,dtype=torch.bool).scatter(0,ids,True)).nonzero()[0]))])
        self.assertEqual(features.shape[-1],294+384)
        loss=router.regression_loss(features,torch.tensor([[.001,-.002]]));loss.backward()
        self.assertGreater(float(router.network.graph[-1].weight.grad.abs().sum()),0.)

    def test_graph_message_and_recovery_matrix_ledgers(self):
        from h65.frame.measure import matrix_counter
        message=GraphMessage(16);x=torch.randn(1,20,16);valid=torch.ones(1,20,dtype=torch.bool);geo=torch.randn(1,20,5)
        with torch.no_grad(),matrix_counter() as counter:_,_,_,record=message(x,valid,geo)
        self.assertFalse(counter.unresolved_matrix_ops())
        self.assertEqual(sum(counter.macs.values()),record['macs'])
        masks=torch.ones(1,32,dtype=torch.bool);chosen=selection(list(range(0,32,2)),32)
        anchors=make_anchors(torch.randn(1,8,24),chosen,masks,[dict(frame_inds=list(range(32)))])
        queries=make_queries(masks,[dict(frame_inds=list(range(32)))],chosen)
        recovery=GraphRecovery(24,couplings=(6,9,12),frame_graph=True,width=16);macs=0
        with torch.no_grad(),matrix_counter() as counter:
            state,_,r=recovery('start',queries,cheap=torch.randn(1,16,96));macs+=r['macs']
            for layer in (6,9,12):state,_,r=recovery('step',queries,state=state,anchors=anchors,level=layer);macs+=r['macs']
            _,r=recovery('readout',queries,state=state,anchors=anchors,cross=torch.randn(1,24,16));macs+=r['macs']
        self.assertFalse(counter.unresolved_matrix_ops());self.assertEqual(sum(counter.macs.values()),macs)

    def test_real_vit_engine_forward_backward_and_matrix_ledger(self):
        import opentad.datasets
        from opentad.models.backbones.vit_adapter import VisionTransformerAdapter
        from h65.paper.engine import PackedStateEngine
        from h65.frame.contracts import EnginePolicy
        from h65.paper.profile import encoder_macs
        from h65.frame.measure import matrix_counter
        vit=VisionTransformerAdapter(img_size=32,patch_size=16,embed_dims=24,num_heads=3,depth=12,total_frames=32,adapter_index=list(range(12)))
        vit.eval();engine=PackedStateEngine(24,depth_bypass=True)
        engine.graph_attention=torch.nn.ModuleDict({str(i):GraphKVAttention(24) for i in (4,6,8,10)})
        engine.eval();clips=torch.randn(2,3,16,32,32);valid=torch.ones(1,16,dtype=torch.bool)
        policy=EnginePolicy(depth_ratio=.75,depth_schedule='amod',spatial_ratio=.75,mod_layers=(4,6,8,10),amod_full_kv=True)
        policy.depth_bypass='light';policy.depth_gate='attention';policy.graph_kv=True;policy.graph_fraction=1.
        context=dict(times=torch.linspace(0,1,16)[None],coupling_layers=(),capture=True)
        with torch.no_grad(),matrix_counter() as counter:value,_,_,trace,_=engine(vit,clips,policy,valid,graph_context=context)
        model=SimpleNamespace(encoder=SimpleNamespace(vit=vit,engine=engine))
        self.assertFalse(counter.unresolved_matrix_ops());self.assertEqual(sum(counter.macs.values()),encoder_macs(model,trace))
        self.assertEqual(len(trace['graph_edges']),4)
        engine.train();clips.requires_grad_();context['capture']=False
        value,*_=engine(vit,clips,policy,valid,graph_context=context);value.square().mean().backward()
        self.assertTrue(torch.isfinite(clips.grad).all())
        self.assertGreater(float(engine.graph_attention['4'].router.query.weight.grad.abs().sum()),0.)

    def test_coupled_carrier_feedback_and_zero_fraction_preserve_backbone(self):
        import opentad.datasets
        from opentad.models.backbones.vit_adapter import VisionTransformerAdapter
        from h65.paper.engine import PackedStateEngine
        from h65.frame.contracts import EnginePolicy
        from h65.paper.profile import encoder_macs
        from h65.frame.measure import matrix_counter
        vit=VisionTransformerAdapter(img_size=32,patch_size=16,embed_dims=24,num_heads=3,depth=12,total_frames=32,adapter_index=list(range(12))).eval()
        engine=PackedStateEngine(24,depth_bypass=True).eval()
        engine.graph_attention=torch.nn.ModuleDict({str(i):GraphKVAttention(24) for i in (4,6,8,10)})
        clips=torch.randn(2,3,16,32,32);native_valid=torch.ones(1,16,dtype=torch.bool)
        masks=torch.ones(1,64,dtype=torch.bool);chosen=selection(list(range(0,64,2)),64);metas=[dict(frame_inds=list(range(0,256,4)))]
        queries=make_queries(masks,metas,chosen);recovery=GraphRecovery(24,couplings=(6,9,12),width=16).eval()
        state,_,_=recovery('start',queries,cheap=torch.randn(1,32,96))
        def step(tokens,h,w,last,quality,level,previous):
            features=tokens.reshape(1,16,h*w,24).mean(2)
            anchors=make_anchors(features,chosen,masks,metas,dict(last_heavy_depth=last.reshape(1,16,h*w).amax(-1),spatial_quality=quality.reshape(1,16,h*w).mean(-1)/level))
            return recovery('step',queries,state=previous,anchors=anchors,level=level)
        context=dict(times=torch.linspace(0,1,16)[None],coupling_layers=(6,9,12),capture=False,state=state,step=step,macs=0,records=[])
        policy=EnginePolicy(depth_ratio=.75,depth_schedule='amod',spatial_ratio=.75,mod_layers=(4,6,8,10),amod_full_kv=True)
        policy.depth_bypass='light';policy.graph_kv=False
        with torch.no_grad():baseline,*_=engine(vit,clips,policy,native_valid)
        policy.graph_kv=True;policy.graph_fraction=0.
        with torch.no_grad(),matrix_counter() as counter:value,_,_,trace,_=engine(vit,clips,policy,native_valid,graph_context=context)
        torch.testing.assert_close(value,baseline,rtol=1e-5,atol=1e-6)
        model=SimpleNamespace(encoder=SimpleNamespace(vit=vit,engine=engine))
        self.assertEqual(sum(counter.macs.values()),encoder_macs(model,trace)+context['macs'])
        self.assertEqual(len(context['records']),3)


if __name__=='__main__':unittest.main()
