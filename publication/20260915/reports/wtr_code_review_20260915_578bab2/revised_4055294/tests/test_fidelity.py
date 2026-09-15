"""Failures that would invalidate the two requested H65 recipe corrections."""
import copy
import random
import unittest
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import torch
from torch import nn
from opentad.datasets.transforms.end_to_end import LoadFrames
from opentad.datasets.builder import collate
from h65.full.data import LoadFramesWithBoundaryValidity
from h65.full.objectives import targets, auxiliary_losses
from h65.full.runtime import optimizer_for, config
from h65.full.scout import FormalScout
from h65.transport import sample_rates


class FidelityTests(unittest.TestCase):
    def test_real_scout_classifier_identity_and_internal_attention_lr(self):
        scout = FormalScout()
        model = SimpleNamespace(scout=scout, backbone=nn.Module(),
                                detector=SimpleNamespace(get_optim_groups=lambda cfg: []))
        action = list(scout.temporal.encoder.conv_out.parameters())
        for decoder in scout.temporal.decoders:
            action += list(decoder.conv_out.parameters())
        internal = [p for name, p in scout.named_parameters() if '.att_layer.conv_out.' in name]
        self.assertEqual(sum(p.numel() for p in action), 388)
        self.assertEqual(sum(p.numel() for p in internal), 18816)
        for backbone in ('s', 'b'):
            for phase, action_lr, trunk_lr in [('warm', 1e-4, 5e-5), ('joint', 2e-5, 1e-5)]:
                optimizer = optimizer_for(model, backbone, phase)
                members = [id(p) for group in optimizer.param_groups for p in group['params']]
                self.assertEqual(len(members), len(set(members)))
                self.assertEqual(set(members), {id(p) for p in scout.parameters()})
                rates = {id(p): group['lr'] for group in optimizer.param_groups for p in group['params']}
                self.assertTrue(all(rates[id(p)] == action_lr for p in action))
                self.assertTrue(all(rates[id(p)] == trunk_lr for p in internal))

    def test_crop_preserves_frames_gt_labels_and_random_stream(self):
        args = dict(method='random_trunc', trunc_len=40, trunc_thresh=.75, crop_ratio=[.9, 1.])
        sample = dict(total_frames=400, avg_fps=30., snippet_stride=4,
                      gt_segments=np.array([[10.,60.],[25.,55.],[5.,95.]], dtype=np.float32),
                      gt_labels=np.array([3,7,9]))
        for seed in range(12):
            random.seed(seed)
            old = LoadFrames(**args)(copy.deepcopy(sample))
            after_old = random.getstate()
            random.seed(seed)
            new = LoadFramesWithBoundaryValidity(**args)(copy.deepcopy(sample))
            self.assertEqual(after_old, random.getstate())
            for key in ('frame_inds','gt_segments','gt_labels'):
                np.testing.assert_array_equal(old[key], new[key])
            torch.testing.assert_close(old['masks'], new['masks'])

    def test_endpoint_validity_tracks_kept_gt_order(self):
        loader = LoadFramesWithBoundaryValidity(method='random_trunc', trunc_len=40, trunc_thresh=.75)
        sample = dict(total_frames=400, avg_fps=30., snippet_stride=4,
                      gt_segments=np.array([[10.,60.],[25.,55.],[5.,95.]], dtype=np.float32),
                      gt_labels=np.array([3,7,9]))
        with patch('opentad.datasets.transforms.end_to_end.random.randint', return_value=20):
            result = loader(sample)
        np.testing.assert_array_equal(result['gt_labels'], [3,7])
        np.testing.assert_array_equal(result['gt_segments'], [[0.,40.],[5.,35.]])
        np.testing.assert_array_equal(result['gt_boundary_validity'], [[False,True],[True,True]])

    def test_short_untruncated_video_keeps_real_boundaries_and_padding(self):
        loader = LoadFramesWithBoundaryValidity(method='random_trunc', trunc_len=384, trunc_thresh=.75)
        result = loader(dict(total_frames=1012, avg_fps=30., snippet_stride=4,
                             gt_segments=np.array([[0.,200.]],dtype=np.float32),gt_labels=np.array([0])))
        self.assertEqual(int(result['masks'].sum()),253)
        np.testing.assert_array_equal(result['gt_boundary_validity'], [[True,True]])
        self.assertTrue(np.all(result['frame_inds'][253:] == result['frame_inds'][252]))

    def test_fake_endpoints_do_not_receive_transition_or_boundary_targets(self):
        masks = torch.ones(1,768,dtype=torch.bool)
        boxes = [torch.tensor([[0.,100.]])]
        action, filtered = targets(masks, boxes, [torch.tensor([[False,True]])])
        action_old, old = targets(masks, boxes)
        torch.testing.assert_close(action, action_old)
        self.assertEqual(float(filtered[0,:5].sum()),0.)
        self.assertAlmostEqual(float(filtered[0,96:105].sum()),1.,places=6)
        self.assertAlmostEqual(float(old[0,:5].sum()),.5,places=6)
        _, none_real = targets(masks, boxes, [torch.tensor([[False,False]])])
        self.assertEqual(float(none_real.sum()),0.)
        logits = torch.zeros(1,768,requires_grad=True)
        selection = sample_rates(logits,masks,384,alpha=0)
        losses = auxiliary_losses(dict(action_logits=logits,aux_transition=logits),selection,masks,boxes,
                                  dict(action=1.,transition=.5,boundary=2.),[torch.tensor([[False,False]])])
        self.assertEqual(float(losses['loss_transition']),0.)
        self.assertEqual(float(losses['loss_boundary']),0.)
        self.assertTrue(torch.isfinite(sum(losses.values())))

    def test_config_and_collate_keep_variable_number_of_validity_pairs(self):
        cfg = config('s')
        load = next(t for t in cfg.dataset.train.pipeline if t.type.startswith('LoadFrames'))
        self.assertEqual(load.type,'LoadFramesWithBoundaryValidity')
        for t in cfg.dataset.train.pipeline:
            if t.type in ('ConvertToTensor','Collect'):
                self.assertIn('gt_boundary_validity',t['keys'])
        self.assertEqual(next(t for t in cfg.dataset.test.pipeline if t.type=='LoadFrames').type,'LoadFrames')
        batch = collate([dict(inputs=torch.zeros(1),masks=torch.ones(1,dtype=torch.bool),
                             gt_boundary_validity=torch.ones(n,2,dtype=torch.bool)) for n in (1,3)])
        self.assertEqual([tuple(x.shape) for x in batch['gt_boundary_validity']],[(1,2),(3,2)])


if __name__ == '__main__':
    torch.set_num_threads(1)
    unittest.main()
