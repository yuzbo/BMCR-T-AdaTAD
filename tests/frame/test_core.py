import copy
import unittest
import torch
from h65.transport import Selection
from h65.frame.geometry import make_anchors,make_queries,interpolate_anchors,decoder_metadata,scout_context
from h65.frame.decoder import build_decoder
from h65.frame.objectives import training_objectives,bernoulli_kl
from h65.frame.gates import incoming_attention,capacity_mask
from h65.frame.condition import vectorized_condition


def fixture():
    masks=torch.arange(16)[None]<13
    ids=torch.tensor([[0,2,5,7,10,12,12,12]])
    flags=torch.tensor([[1,1,1,1,1,1,0,0]],dtype=torch.bool)
    selection=Selection(ids,ids.float(),flags,masks.float()/13,masks.float())
    metas=[dict(frame_inds=torch.arange(16)*4+100)]
    features=torch.randn(1,4,12)
    return make_anchors(features,selection,masks,metas),make_queries(masks,metas,selection),selection,masks


class FrameCoreTests(unittest.TestCase):
    def test_native_not_repaired_twice(self):
        anchors,queries,selection,masks=fixture()
        self.assertEqual(anchors.features.shape,(1,4,12))
        self.assertEqual(anchors.centers[0,:3].tolist(),[104.,124.,144.])
        self.assertEqual(anchors.valid[0].tolist(),[True,True,True,False])
        self.assertEqual(queries.centers[0,6].item(),148.)
        self.assertTrue(queries.membership[0,12]);self.assertFalse(queries.valid[0,7])
        am,qm=decoder_metadata(anchors,queries)
        self.assertEqual(am.shape,(1,4,10));self.assertEqual(qm.shape,(1,8,8))

    def test_zero_residual_and_later_gradient(self):
        torch.manual_seed(3);anchors,queries,_,_=fixture();context=torch.randn(1,8,96)
        base=interpolate_anchors(anchors,queries).transpose(1,2)
        for kind in ('tcn','cross'):
            module=build_decoder(kind,12,width=12,heads=3,layers=2)
            self.assertTrue(torch.equal(module(anchors,queries,context),base))
            optimizer=torch.optim.AdamW(module.parameters(),lr=.01)
            target=torch.randn_like(base)
            for _ in range(2):
                optimizer.zero_grad();loss=(module(anchors,queries,context)-target).square().mean();loss.backward();optimizer.step()
            self.assertGreater(module.base_proj.weight.grad.abs().sum().item(),0)

    def test_cross_reads_anchor_features(self):
        anchors,queries,_,_=fixture();context=torch.randn(1,8,96)
        module=build_decoder('cross',12,width=12,heads=3)
        torch.nn.init.normal_(module.head.weight)
        # Cross modules contain no query self-attention.
        self.assertFalse(any(isinstance(x,torch.nn.TransformerDecoderLayer) for x in module.modules()))
        anchors.features.requires_grad_();module(anchors,queries,context).sum().backward()
        self.assertGreater(anchors.features.grad.abs().sum().item(),0)

    def test_scout_context_keeps_time_axis(self):
        masks=torch.ones(1,16,dtype=torch.bool);hidden=torch.arange(16).float()[None,:,None].expand(1,16,96)
        context=scout_context({'hidden':hidden},masks)
        self.assertEqual(context[0,:,0].tolist(),[.5,2.5,4.5,6.5,8.5,10.5,12.5,14.5])

    def test_gt_not_silently_skipped_or_double_counted(self):
        class Teacher:
            def loss(self,x,data):
                loss=x.square().mean();return dict(cost=loss,cls_loss=loss*.4,reg_loss=loss*.6)
        x=torch.randn(2,4,8,requires_grad=True);target=torch.randn_like(x);data={'masks':torch.ones(2,16,dtype=torch.bool)}
        losses=training_objectives(x,target,data,Teacher(),feature_weight=0,gt_weight=1)
        self.assertTrue(torch.allclose(losses['cost'],x.square().mean()))
        losses['cost'].backward();self.assertGreater(x.grad.abs().sum().item(),0)
        class Broken:
            def loss(self,x,data):raise RuntimeError('missing task graph')
        with self.assertRaises(RuntimeError):training_objectives(x,target,data,Broken())

    def test_amod_score_and_capacity(self):
        q=torch.randn(2,3,8,4);k=torch.randn_like(q)
        expected=((q*.5)@k.transpose(-2,-1)).softmax(-1).mean((1,2))
        self.assertTrue(torch.allclose(incoming_attention(q,k,chunk=3),expected,atol=1e-7))
        scores=torch.zeros(2,8);valid=torch.tensor([[1]*8,[1,1,1,0,0,0,0,0]],dtype=torch.bool)
        self.assertEqual(capacity_mask(scores,.5,valid).sum(-1).tolist(),[4,2])
        self.assertFalse(capacity_mask(scores,0,valid).any())

    def test_independent_bernoulli_kd(self):
        logits=torch.randn(2,8,20,requires_grad=True);valid=torch.ones(2,8,dtype=torch.bool)
        self.assertAlmostEqual(bernoulli_kl(logits,logits.detach(),valid).item(),0,places=6)

    def test_condition_matches_real_legacy_and_gradients(self):
        from h65.full.scout import FormalScout
        torch.manual_seed(4);scout=FormalScout();torch.nn.init.normal_(scout.conditional[-1].weight)
        masks=torch.arange(32)[None]<27;ids=torch.tensor([[0,2,4,6,8,10,12,14,16,18,20,22,24,26,26,26]])
        flags=torch.arange(16)[None]<14;selection=Selection(ids,ids.float(),flags,masks.float()/27,masks.float())
        a=torch.randn(1,32,96,requires_grad=True);rep=torch.randn(1,32,64,requires_grad=True)
        output=dict(hidden=a,representation=rep,action_logits=torch.randn(1,32))
        legacy=scout.condition(output,selection,masks);new=vectorized_condition(scout,output,selection,masks)
        for key in ('member','partner','feasible'):self.assertTrue(torch.equal(legacy[key],new[key]))
        self.assertTrue(torch.allclose(legacy['utility'],new['utility'],atol=1e-6))
        ga=torch.autograd.grad(legacy['utility'].sum(),a,retain_graph=True)[0]
        gb=torch.autograd.grad(new['utility'].sum(),a)[0]
        self.assertTrue(torch.allclose(ga,gb,atol=1e-6))


if __name__=='__main__':unittest.main()
