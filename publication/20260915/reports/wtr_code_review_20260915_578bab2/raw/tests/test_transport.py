"""Detect budget, gradient and physical-time failures before GPU experiments."""
import unittest

import torch
import torch.nn.functional as F

from h65.transport import (sample_rates, gather_with_transport,
                           interpolate_irregular, restore_tubelets, calibrated_rates)


class TransportTests(unittest.TestCase):
    def test_calibrated_rates_budget_and_implicit_gradient(self):
        logits = torch.tensor([-1.2, -0.4, 0.3, 0.8, 1.6], dtype=torch.double, requires_grad=True)
        rates = calibrated_rates(logits, 2)
        self.assertAlmostEqual(float(rates.sum()), 2, places=10)
        self.assertTrue(bool(((rates >= 0) & (rates <= 1)).all()))
        self.assertTrue(torch.autograd.gradcheck(lambda x: calibrated_rates(x, 2),
                                               (logits,), eps=1e-5, atol=1e-5, rtol=1e-4))
        (rates * torch.arange(5)).sum().backward()
        self.assertLess(abs(float(logits.grad.sum())), 1e-10)

    def test_exact_budget_extreme_logits_and_short_final_window(self):
        logits = torch.zeros(3, 768)
        logits[0, 370:385] = 90
        logits[1, 0] = 90
        masks = torch.arange(768)[None] < torch.tensor([768, 768, 127])[:, None]
        s = sample_rates(logits, masks)
        torch.testing.assert_close(s.rates.sum(-1), torch.tensor([384., 384., 127.]), atol=2e-4, rtol=0)
        for row, valid, length in zip(s.indices, s.valid, (768, 768, 127)):
            active = row[valid]
            self.assertEqual(active.numel(), min(384, length))
            self.assertTrue(bool((active.diff() > 0).all()))
            self.assertGreaterEqual(int(active[0]), 0)
            self.assertLess(int(active[-1]), length)
        self.assertTrue(torch.equal(s.indices[2, 127:], torch.full((257,), 126)))

    def test_hard_forward_and_nonzero_finite_policy_gradient(self):
        torch.manual_seed(3)
        logits = torch.randn(1, 32, requires_grad=True)
        s = sample_rates(logits, torch.ones(1, 32, dtype=torch.bool), budget=16)
        # Descending uint8 frames catch unsigned subtraction in the local slope.
        frames = (100 - torch.arange(32)).to(torch.uint8)[None, None, None, :, None, None].expand(1, 1, 3, 32, 2, 2)
        result = gather_with_transport(frames, s)
        expected = frames.float()[:, :, :, s.indices[0]]
        self.assertTrue(torch.equal(result, expected))
        weights = torch.arange(16).float()[None, None, None, :, None, None]
        (result * weights).sum().backward()
        self.assertTrue(bool(torch.isfinite(logits.grad).all()))
        self.assertGreater(float(logits.grad.abs().sum()), 0)

    def test_full_selection_is_official_interpolation(self):
        torch.manual_seed(7)
        masks = torch.ones(2, 768, dtype=torch.bool)
        s = sample_rates(torch.randn(2, 768), masks, budget=768, alpha=0)
        features = torch.randn(2, 4, 384)
        restored = restore_tubelets(features, s, masks)
        official = F.interpolate(features, size=768, mode="linear", align_corners=False)
        torch.testing.assert_close(restored, official)

    def test_irregular_physical_time_is_affine_exact(self):
        source = torch.tensor([0.5, 2.5, 8.5, 15.5])
        query = torch.arange(16).float()
        features = (3 * source + 2)[None].requires_grad_()
        result = interpolate_irregular(features, source, query)
        torch.testing.assert_close(result, (3 * query.clamp(0.5, 15.5) + 2)[None])
        result.sum().backward()
        self.assertTrue(bool((features.grad > 0).all()))

    def test_uniform_policy_has_no_selector_gradient(self):
        logits = torch.randn(1, 32, requires_grad=True)
        s = sample_rates(logits, torch.ones(1, 32, dtype=torch.bool), budget=16, alpha=0)
        frames = torch.arange(32).float()[None, None, None, :, None, None]
        gather_with_transport(frames, s).sum().backward()
        torch.testing.assert_close(logits.grad, torch.zeros_like(logits))


if __name__ == "__main__":
    unittest.main()
