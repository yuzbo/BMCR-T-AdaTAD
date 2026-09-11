"""Real upstream focal assignment checks retention/insertion utility semantics."""
import unittest
from types import SimpleNamespace
import torch
from mmengine.config import Config
from opentad.models.builder import build_head
from h65.transport import sample_rates
from h65.full.geometry import maps_for
from h65.full.utility import counterfactual_targets
from pathlib import Path


class TinyObservedTeacher:
    """Known image-response function around the real OpenTAD assignment/loss."""
    def __init__(self):
        root = Path(__file__).resolve().parents[1]
        cfg = Config.fromfile(str(root/'upstream/configs/_base_/models/actionformer.py'))
        self.detector = SimpleNamespace(rpn_head=build_head(cfg.model.rpn_head))

    def eval(self):
        self.detector.rpn_head.eval()

    def raw_route(self, inputs, masks, selection):
        k = selection.indices.shape[1]
        classes = self.detector.rpn_head.num_classes
        logits = torch.full((1, k, classes), -4.)
        # Adding a larger-index observation worsens confidence for GT class0.
        logits[..., 0] = -selection.indices.float().sum()/10
        points = torch.stack((torch.arange(k), torch.zeros(k), torch.full((k,), 1e8), torch.ones(k)), -1)
        scores = torch.full((k, classes), .01)
        scores[:, 0] = .9
        return dict(logits=logits, points=[points], valid=selection.valid, maps=maps_for(selection, masks),
                    proposals=[torch.tensor([[0., 8.]]).repeat(k, 1)], scores=[scores])


class CounterfactualLabelTests(unittest.TestCase):
    def test_nonreciprocal_member_and_nonmember_both_receive_correct_sign(self):
        mask = torch.ones(1, 8, dtype=torch.bool)
        route = sample_rates(torch.zeros(1, 8), mask, 4, alpha=0)
        route.indices[0] = torch.tensor([0, 2, 4, 6])
        route.continuous = route.indices.float()
        condition = dict(member=torch.tensor([[1,0,1,0,1,0,1,0]], dtype=torch.bool),
                         partner=torch.tensor([[1,0,3,4,3,4,5,6]]),
                         feasible=torch.tensor([[0,0,1,0,0,1,0,0]], dtype=torch.bool))
        records = counterfactual_targets(TinyObservedTeacher(), torch.zeros(1,1,3,8,1,1), mask, [{}],
                       [torch.tensor([[0.,8.]])], [torch.tensor([0])], route, condition)
        self.assertEqual({r['candidate'] for r in records}, {2,5})
        by_id = {r['candidate']: r for r in records}
        self.assertEqual((by_id[2]['remove'], by_id[2]['insert']), (2,3))
        self.assertEqual((by_id[5]['remove'], by_id[5]['insert']), (4,5))
        self.assertGreater(by_id[2]['target'][0], 0.)
        self.assertLess(by_id[5]['target'][0], 0.)
        self.assertAlmostEqual(by_id[2]['target'][0], -by_id[5]['target'][0], places=6)
        self.assertEqual(by_id[2]['target'][1], 0.)
        self.assertEqual(by_id[5]['target'][1], 0.)


if __name__ == '__main__':
    torch.set_num_threads(1)
    unittest.main()
