import copy,json,unittest
from pathlib import Path
from types import SimpleNamespace
import torch
from h65.frame.contracts import EnginePolicy
from h65.frame.measure import matrix_counter
from h65.paper.engine import PackedStateEngine,attention
from h65.paper.profile import encoder_macs
from h65.paper.support_targets import compare_states,support_key
from h65.paper.routing import FrameRouter,plans

def fixture(depth=12,light=False):
    from opentad.models.backbones.vit_adapter import VisionTransformerAdapter
    torch.set_num_threads(2);torch.manual_seed(42)
    vit=VisionTransformerAdapter(img_size=32,patch_size=16,embed_dims=32,depth=depth,num_heads=4,num_frames=16,total_frames=32,adapter_index=list(range(depth)),return_feat_map=True).eval()
    for block in vit.blocks:torch.nn.init.normal_(block.adapter.up_proj.weight,std=.02)
    engine=PackedStateEngine(32,depth,depth_bypass=light).eval()
    return vit,engine,torch.randn(2,3,16,32,32),torch.ones(1,16,dtype=torch.bool)

class SupportReview(unittest.TestCase):
    def test_full_kv_selected_queries_and_empty_rows(self):
        vit,_,_,_=fixture(4);x=torch.randn(3,16,32);mask=torch.zeros(3,16,dtype=torch.bool);mask[0,::2]=True;mask[1,3]=True
        full,_,_=attention(vit.blocks[0].attn,x)
        sparse,_,count=attention(vit.blocks[0].attn,x,mask,True)
        self.assertTrue(torch.allclose(sparse[mask],full[mask],atol=1e-6,rtol=1e-5))
        self.assertTrue(torch.equal(sparse[~mask],torch.zeros_like(sparse[~mask])))
        self.assertEqual(count['kv'],32);self.assertEqual(count['q'],9)

    def test_depth_light_zero_starts_at_hold_then_updates_only_bypass(self):
        vit,engine,clips,valid=fixture(light=True)
        policy=EnginePolicy(static_depth=12,mod_layers=(4,6,8,10),depth_schedule='amod',depth_ratio=.5,spatial_ratio=.5,amod_full_kv=True)
        policy.depth_bypass='hold'
        with torch.no_grad():held=engine(vit,clips,policy,valid)[0]
        policy.depth_bypass='light'
        with torch.no_grad():zero=engine(vit,clips,policy,valid)[0]
        self.assertTrue(torch.equal(held,zero))
        for i in policy.mod_layers:torch.nn.init.constant_(engine.depth_ffn[i][-1].bias,.01)
        with torch.no_grad(),matrix_counter() as counter:changed,_,_,tr,_=engine(vit,clips,policy,valid)
        self.assertFalse(torch.equal(changed,held))
        self.assertTrue(all(tr['depth_attention_light'][i]==0 for i in range(4)))
        for i in range(12):
            self.assertFalse(bool((tr['depth_bypass_masks'][i]&tr['ffn_light_masks'][i]).any()))
            self.assertEqual(tr['depth_attention_light'][i],int(tr['depth_bypass_masks'][i].sum()))
        model=SimpleNamespace(encoder=SimpleNamespace(vit=vit,engine=engine))
        self.assertEqual(encoder_macs(model,tr),sum(counter.macs.values()))
        self.assertGreater(int(tr['age_before_reentry'][5].max()),0)

    def test_state_capture_does_not_change_full_forward_or_backward(self):
        vit,engine,clips,valid=fixture(4);engine.train();reference=copy.deepcopy(vit);plain=copy.deepcopy(engine).eval()
        x=clips.clone().requires_grad_();y=clips.clone().requires_grad_();policy=EnginePolicy(static_depth=4,mod_layers=(1,))
        a=engine(vit,x,policy,valid,(2,3,4),True);b=plain(reference,y,policy,valid,(2,3,4),True)
        self.assertTrue(torch.equal(a[0],b[0]))
        (a[0].square().mean()+.1*a[3]['state_taps'][2]['pre_tia'].square().mean()).backward()
        (b[0].square().mean()+.1*b[3]['state_taps'][2]['pre_tia'].square().mean()).backward()
        self.assertTrue(torch.allclose(x.grad,y.grad,atol=1e-6,rtol=1e-5))
        self.assertEqual(set(a[3]['state_taps'][2]),{'attention','pre_tia','post_tia'})

    def test_support_loss_ignores_invalid_positions_and_stops_target_gradient(self):
        value=torch.ones(1,4,8,requires_grad=True);target=torch.zeros(1,4,8,requires_grad=True)
        valid=torch.tensor([[True,True,False,False]])
        a={2:{k:value for k in ('attention','pre_tia','post_tia')}};b={2:{k:target for k in ('attention','pre_tia','post_tia')}}
        terms,_=compare_states(a,b,valid);sum(terms.values()).backward()
        self.assertIsNone(target.grad);self.assertEqual(float(value.grad[:,2:].abs().sum()),0.);self.assertGreater(float(value.grad[:,:2].abs().sum()),0.)

    def test_original_times_follow_exact_selected_candidates(self):
        from h65.transport import sample_rates
        masks=torch.ones(1,32,dtype=torch.bool);selected=sample_rates(torch.zeros(1,32),masks,16,alpha=0.)
        times=torch.arange(32).reshape(1,32)*3;data=dict(inputs=torch.zeros(1,1,3,32,2,2),masks=masks,metas=[dict(frame_inds=times)])
        key=support_key(data,selected,160)
        self.assertTrue(torch.equal(key['source_frame_times'],times.float().gather(1,selected.indices)))
        self.assertEqual(key['tia_temporal_size'],8)

    def test_plan_context_keeps_common_features_and_parameter_initialization(self):
        from h65.transport import sample_rates
        masks=torch.ones(1,32,dtype=torch.bool);selected=sample_rates(torch.zeros(1,32),masks,16,alpha=0.)
        output=dict(hidden=torch.randn(1,32,96),action_logits=torch.randn(1,32));pairs=[(0,int(selected.indices[0,0]),1)]
        torch.manual_seed(8);old=FrameRouter();torch.manual_seed(8);new=FrameRouter(True)
        a=old.features(output,selected,masks,pairs);b=new.features(output,selected,masks,pairs,dict(frames=384,depth=.5,space=.48))
        self.assertTrue(torch.equal(a,b[:,:294]));self.assertTrue(torch.equal(old.network[1].weight,new.network[1].weight[:,:294]))
        self.assertEqual(b.shape[-1],300)

    def test_full_reference_and_late_layer_contract(self):
        from h65.paper.model import PaperModel
        model=PaperModel.__new__(PaperModel);torch.nn.Module.__init__(model);model.encoder=SimpleNamespace(depth=12);model.menu=plans();model.training_epoch=0
        model.config=dict(frames=384,depth_capacity=.5,space_capacity=.48,mod_start=4,kv_mode='full',depth_bypass='light')
        self.assertEqual(model.plan(0)['frames'],768);self.assertEqual(model.plan(0)['depth'],1.)
        self.assertEqual(model.plan(4)['mod_layers'],[4,6,8,10])

    def test_finite_review_plan_and_shared_aliases(self):
        from tools.paper_review_plan import build
        root=Path(__file__).resolve().parents[2];spec=json.loads((root/'research/paper/review_5485/EXPERIMENTS.json').read_text())
        plan=build(spec);self.assertEqual(len(plan['configs']),23);self.assertEqual({c['seed'] for c in plan['configs']},{42})
        self.assertEqual(plan['aliases']['C2'],plan['aliases']['C3']);self.assertFalse(plan['performance_gates'])

if __name__=='__main__':unittest.main()
