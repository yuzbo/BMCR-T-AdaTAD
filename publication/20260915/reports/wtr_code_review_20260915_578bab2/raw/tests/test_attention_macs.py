"""Check fused-attention matrix arithmetic against explicitly executed products."""
import unittest
import torch
from tools.full_eval import ArithmeticCounter, profile_case


class AttentionMacTest(unittest.TestCase):
    def test_matches_explicit_qk_and_av_products(self):
        q = torch.randn(2, 3, 5, 4)
        k = torch.randn(2, 3, 7, 4)
        v = torch.randn(2, 3, 7, 6)
        counter = ArithmeticCounter()
        with counter:
            products = (q @ k.transpose(-2, -1)) @ v
        self.assertEqual(tuple(products.shape), (2, 3, 5, 6))
        self.assertEqual(ArithmeticCounter.attention_macs(q, k, v), sum(counter.macs.values()))
        self.assertEqual(sum(counter.macs.values()), 2100)

    def test_upsampling_is_not_unresolved_matrix_arithmetic(self):
        counter = ArithmeticCounter()
        counter.operations.update({'aten.upsample_bilinear2d.default': 1, 'aten.upsample_linear1d.default': 1,
                                   'aten._scaled_dot_product_flash_attention.default': 12, 'aten.linear.default': 1})
        self.assertEqual(counter.unresolved_matrix_ops(), ['aten.linear.default'])

    def test_full_window_is_distinct_from_short_route(self):
        for valid, expected in ((253, 'short'), (384, 'short'), (514, 'partial'), (768, 'full')):
            mask = torch.arange(768).unsqueeze(0) < valid
            self.assertEqual(profile_case({'masks': mask}), expected)


if __name__ == '__main__':
    unittest.main()
