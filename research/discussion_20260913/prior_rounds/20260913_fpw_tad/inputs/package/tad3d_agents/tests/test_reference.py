"""CPU synthetic mechanism tests. Not VideoMAE checkpoint or THUMOS tests."""
import unittest
import torch
from torch import nn
from reference.core import (anchor_provenance, interpolate_anchors,
                            FullAxisDecoder, PackedQueryAttention,
                            dense_reference_macs, layer_macs)

class ReferenceTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(3407)
        torch.set_num_threads(1)

    def test_pair_provenance_and_odd_padding(self):
        ids = torch.tensor([[0, 2, 5, 5]])
        valid = torch.tensor([[True, True, True, False]])
        p = anchor_provenance(ids, valid, torch.arange(8)[None].float()*4)
        self.assertTrue(torch.equal(p.center, torch.tensor([[4., 20.]])))
        self.assertTrue(torch.equal(p.span, torch.tensor([[8., 0.]])))
        self.assertEqual(p.valid_fraction.tolist(), [[1., .5]])

    def test_interp_edges_grad_and_non_subset(self):
        f = torch.tensor([[[2.], [8.]]], requires_grad=True)
        q = torch.tensor([[0., 1., 2., 3., 4.]])
        y = interpolate_anchors(f, torch.tensor([[1., 3.]]), torch.ones(1,2,dtype=torch.bool), q)
        torch.testing.assert_close(y[..., 0], torch.tensor([[2., 2., 5., 8., 8.]]))
        y.sum().backward(); self.assertTrue(torch.isfinite(f.grad).all())

    def test_single_anchor(self):
        f = torch.randn(1,1,4)
        y = interpolate_anchors(f, torch.tensor([[7.]]), torch.tensor([[True]]), torch.arange(5)[None].float())
        torch.testing.assert_close(y, f.expand(1,5,4))

    def test_invalid_duplicate_rejected(self):
        with self.assertRaises(ValueError):
            anchor_provenance(torch.tensor([[1,1]]), torch.tensor([[True,True]]), torch.arange(4)[None].float())

    def decoder_inputs(self):
        b,a,q,c = 2,3,7,12
        return dict(anchors=torch.randn(b,a,c),anchor_valid=torch.ones(b,a,dtype=torch.bool),
                    baseline=torch.randn(b,q,c),context=torch.randn(b,q,5),
                    query_meta=torch.randn(b,q,6),anchor_meta=torch.randn(b,a,4),
                    query_valid=torch.ones(b,q,dtype=torch.bool))

    def test_zero_head_equals_new_baseline_only(self):
        m = FullAxisDecoder(12,5,24,3,2)
        x = self.decoder_inputs()
        torch.testing.assert_close(m(**x), x['baseline'], rtol=0, atol=0)
        # This does NOT assert old-rank-detector == original-time-detector.
        m(**x).square().mean().backward()
        self.assertGreater(m.head.weight.grad.abs().sum().item(), 0.)
        self.assertEqual(m.memory_proj.weight.grad.abs().sum().item(), 0.)

    def test_cross_queries_chunk_equivalence(self):
        m=FullAxisDecoder(12,5,24,3,2).eval()
        nn.init.normal_(m.head.weight, std=.01)  # nonzero output for meaningful check
        x=self.decoder_inputs(); full=m(**x)
        chunks=[]
        for i,j in ((0,2),(2,7)):
            z={k:(v[:,i:j] if k in ('baseline','context','query_meta','query_valid') else v) for k,v in x.items()}
            chunks.append(m(**z))
        torch.testing.assert_close(full, torch.cat(chunks,1), rtol=1e-5, atol=1e-6)

    def test_invalid_anchors_do_not_affect_decoder(self):
        m=FullAxisDecoder(12,5,24,3,2).eval(); nn.init.normal_(m.head.weight,std=.01)
        x=self.decoder_inputs(); x['anchor_valid'][:,-1]=False
        before=m(**x); x['anchors'][:,-1]=10000
        torch.testing.assert_close(before,m(**x),rtol=0,atol=0)

    def test_frozen_detector_keeps_feature_gradient(self):
        detector=nn.Linear(12,20).requires_grad_(False)
        feature=torch.randn(2,7,12,requires_grad=True)
        detector(feature).square().mean().backward()
        self.assertGreater(feature.grad.abs().sum().item(),0.)
        self.assertIsNone(detector.weight.grad)

    def test_sparse_query_matches_same_input_dense_rows(self):
        m=PackedQueryAttention(24,3).double().eval(); x=torch.randn(2,9,24,dtype=torch.double)
        ids=torch.tensor([[0,3,7],[1,2,8]])
        ref=m.dense(x).gather(1,ids[...,None].expand(-1,-1,24))
        torch.testing.assert_close(m(x,ids),ref,atol=1e-12,rtol=1e-12)

    def test_dense_mask_and_packed_mlp_agree(self):
        m=nn.Sequential(nn.Linear(8,32),nn.GELU(),nn.Linear(32,8))
        x=torch.randn(2,11,8); ids=torch.tensor([[0,2,8],[1,4,9]])
        selected=x.gather(1,ids[...,None].expand(-1,-1,8))
        torch.testing.assert_close(m(selected),m(x).gather(1,ids[...,None].expand(-1,-1,8)))

    def test_abs_gradient_not_a_finite_intervention_value(self):
        x=torch.tensor(1.,requires_grad=True); loss=(x*x-1)**2
        g=torch.autograd.grad(loss,x)[0]
        self.assertEqual(g.item(),0.)
        self.assertEqual(((torch.tensor(0.)**2-1)**2-loss.detach()).item(),1.)

    def test_interactions_are_not_additive(self):
        loss=lambda x,y:(x*y-1)**2
        joint=loss(0,0)-loss(1,1)
        sum_of_single=(loss(0,0)-loss(1,0))+(loss(0,0)-loss(0,1))
        self.assertNotEqual(joint,sum_of_single)

    def test_cost_known_totals_and_kv_floor(self):
        self.assertEqual(sum(dense_reference_macs(384).values()),1173947019264)
        self.assertEqual(sum(dense_reference_macs(768).values()),4041077354496)
        half=sum(layer_macs(800,384,400,400).values())
        full=sum(layer_macs(800,384,800,800).values())
        self.assertGreater(half,full/2)
        self.assertEqual(sum(layer_macs(800,384,0,0).values()),0)

if __name__=='__main__': unittest.main(verbosity=2)
