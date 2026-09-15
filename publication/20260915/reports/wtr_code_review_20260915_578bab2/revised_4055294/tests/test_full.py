"""Checks for failures that would invalidate the full-training comparison."""
import unittest
import torch
from h65.transport import sample_rates
from h65.full.geometry import TrueTimeMap, occupancy, mix_rows, exchange
from h65.full.objectives import targets, distribution_loss, curriculum
from h65.full.utility import localization_cost


class FormalContractTests(unittest.TestCase):
    def test_endpoint_extension_and_roundtrip(self):
        mapping = TrueTimeMap(torch.tensor([2, 5, 9]), 12)
        self.assertEqual(mapping.to_rank(torch.tensor([0., 12.])).tolist(), [-1., 3.])
        time = torch.tensor([[0., 1.], [2.5, 6.], [10., 12.]])
        torch.testing.assert_close(mapping.to_true(mapping.to_rank(time)), time)

    def test_equal_endpoint_mass_halfopen_action(self):
        mask = torch.ones(1, 32, dtype=torch.bool)
        action, gaussian = targets(mask, [torch.tensor([[0., 16.]])])
        self.assertEqual(action[0, 0], 1)
        self.assertEqual(action[0, 16], 0)
        torch.testing.assert_close(gaussian[0, :5].sum(), torch.tensor(.5))
        torch.testing.assert_close(gaussian[0, 12:21].sum(), torch.tensor(.5))

    def test_teacher_rows_and_transport_mass(self):
        logits = torch.randn(2, 32, requires_grad=True)
        mask = torch.ones_like(logits, dtype=torch.bool)
        learned = sample_rates(logits, mask, 16)
        uniform = sample_rates(logits, mask, 16, alpha=0)
        selection = mix_rows(learned, uniform, torch.tensor([True, False]))
        mass = occupancy(selection, mask)
        torch.testing.assert_close(mass.sum(-1), torch.tensor([16., 16.]))
        (mass * torch.arange(32).square()).sum().backward()
        self.assertEqual(float(logits.grad[0].abs().sum()), 0)
        self.assertGreater(float(logits.grad[1].abs().sum()), 0)
        swapped = exchange(selection, 0, 0, 1)
        self.assertTrue(bool((swapped.indices[0].diff() > 0).all()))

    def test_temperature_and_active_teacher_rows(self):
        logits = torch.tensor([[0., .7], [10., -10.]], requires_grad=True)
        target = torch.tensor([[0., 1.], [1., 0.]])
        loss = distribution_loss(logits, target, torch.ones_like(logits).bool(), torch.tensor([True, False]))
        torch.testing.assert_close(loss, torch.log1p(torch.exp(torch.tensor(-1.))))
        loss.backward()
        self.assertEqual(float(logits.grad[1].abs().sum()), 0)

    def test_missed_gt_remains_in_localization_cost(self):
        gt = torch.tensor([[0., 4.], [10., 14.]])
        proposals = torch.tensor([[0., 4.]])
        scores = torch.ones(1, 1)
        cost, matching, matched = localization_cost(proposals, scores, gt, torch.zeros(2, dtype=torch.long))
        self.assertEqual(cost, 1.)
        self.assertEqual(matched, 1)
        missing, _, matched = localization_cost(proposals, scores * 0, gt, torch.zeros(2, dtype=torch.long), matching)
        self.assertEqual(missing, 2.)
        self.assertEqual(matched, 0)

    def test_course_boundaries(self):
        self.assertEqual(curriculum('warm', 1000)['alpha'], 0)
        self.assertEqual(curriculum('joint', 667)['contribution'], 0)
        self.assertEqual(curriculum('joint', 2000)['contribution'], 1)
        self.assertEqual(curriculum('joint', 2000)['bridge'], .25)


if __name__ == '__main__':
    unittest.main()
