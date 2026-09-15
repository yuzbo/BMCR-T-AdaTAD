"""A profiler must preserve tensor/tuple outputs while counting nested modules."""
import sys
import unittest
from pathlib import Path
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'tools'))
from full_eval import ArithmeticCounter


class TupleModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 2, bias=False)

    def forward(self, value):
        result = self.linear(value)
        return result, result.sum(-1)


class ProfilerHookTests(unittest.TestCase):
    def test_nested_hooks_preserve_outputs_and_count_macs(self):
        model = TupleModule()
        inputs = torch.ones(4, 3)
        expected = model(inputs)
        counter = ArithmeticCounter()
        hooks = []
        for module, name in ((model, 'outer'), (model.linear, 'inner')):
            hooks.append(module.register_forward_pre_hook(lambda m, a, name=name: counter.phase.append(name)))
            hooks.append(module.register_forward_hook(counter.leave_scope))
        try:
            with torch.no_grad(), counter:
                actual = model(inputs)
        finally:
            for hook in hooks:
                hook.remove()
        self.assertIsInstance(actual, tuple)
        for first, second in zip(expected, actual):
            torch.testing.assert_close(first, second, rtol=0, atol=0)
        self.assertEqual(counter.phase, [])
        self.assertEqual(counter.macs['inner'], 24)
        self.assertEqual(sum(counter.macs.values()), 24)


if __name__ == '__main__':
    unittest.main()
