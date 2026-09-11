import unittest

import torch

from h65.scout import H65Scout


class ScoutGradientTests(unittest.TestCase):
    def test_policy_feedback_stays_out_of_spatial_stem(self):
        torch.manual_seed(3407)
        scout = H65Scout().train()
        frames = torch.rand(1, 1, 3, 16, 16, 16) * 255
        masks = torch.ones(1, 16, dtype=torch.bool)
        output = scout(frames, masks)
        stem = list(scout.spatial_stem.parameters())
        encoder = list(scout.temporal.encoder.parameters())
        grad = torch.autograd.grad(output["rate_logits"].square().mean(), stem + encoder,
                                   allow_unused=True, retain_graph=True)
        self.assertTrue(all(g is None or torch.count_nonzero(g) == 0 for g in grad[:len(stem)]))
        self.assertGreater(sum(float(g.abs().sum()) for g in grad[len(stem):] if g is not None), 0)
        action_grad = torch.autograd.grad(output["action_logits"].square().mean(), stem, allow_unused=True)
        self.assertGreater(sum(float(g.abs().sum()) for g in action_grad if g is not None), 0)

    def test_rate_only_ablation_does_not_execute_utility_head(self):
        scout = H65Scout().eval()
        calls = []
        hook = scout.contribution_head.register_forward_hook(lambda *args: calls.append(1))
        with torch.no_grad():
            out = scout(torch.rand(1, 1, 3, 16, 16, 16) * 255,
                        torch.ones(1, 16, dtype=torch.bool), use_contribution=False)
        hook.remove()
        self.assertIsNone(out["contribution_logits"])
        self.assertEqual(calls, [])


if __name__ == "__main__":
    unittest.main()
