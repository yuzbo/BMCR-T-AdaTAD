"""Read-only CPU probe of the actual published scout and optimizer function.

No forward training, model checkpoints, optimizer steps, or source edits.
The detector/backbone stubs intentionally have no parameters: this probe tests
only the real scout's parameter partition through the real optimizer_for code.
"""
import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

parser = argparse.ArgumentParser()
parser.add_argument('--repo', type=Path, required=True)
parser.add_argument('--out', type=Path, required=True)
args = parser.parse_args()
sys.path[:0] = [str(args.repo), str(args.repo / 'upstream')]

import torch
from h65.full.scout import FormalScout
from h65.full.runtime import optimizer_for
from h65.full.geometry import TrueTimeMap

torch.set_num_threads(2)
scout = FormalScout()
intended_ids = {id(p) for p in scout.temporal.encoder.conv_out.parameters()}
for decoder in scout.temporal.decoders:
    intended_ids.update(id(p) for p in decoder.conv_out.parameters())
named = dict(scout.named_parameters())
intended = [n for n, p in named.items() if id(p) in intended_ids]
matched = [n for n in named if n.startswith('temporal.') and '.conv_out.' in n]
wrong = [n for n in matched if id(named[n]) not in intended_ids]

harness = SimpleNamespace(scout=scout, backbone=torch.nn.Module(),
                          detector=SimpleNamespace(get_optim_groups=lambda config: []))
observed = []
for backbone in ('s', 'b'):
    for phase in ('warm', 'joint'):
        optimizer = optimizer_for(harness, backbone, phase)
        groups = {id(p): g for g in optimizer.param_groups for p in g['params']}
        flattened = [id(p) for g in optimizer.param_groups for p in g['params']]
        assert len(flattened) == len(set(flattened)) == len(named)
        action_lr = 1e-4 if phase == 'warm' else 2e-5
        trunk_lr = 5e-5 if phase == 'warm' else 1e-5
        actual_action = [n for n, p in named.items() if groups[id(p)]['lr'] == action_lr]
        assert set(actual_action) == set(matched)
        observed.append(dict(backbone=backbone, phase=phase, action_lr=action_lr, intended_trunk_lr=trunk_lr,
            intended_action_parameters=sum(named[n].numel() for n in intended),
            actual_action_parameters=sum(named[n].numel() for n in actual_action),
            misplaced_internal_parameters=sum(named[n].numel() for n in wrong),
            actual_action_weight_parameters=sum(named[n].numel() for n in actual_action if n.endswith('.weight')),
            actual_action_bias_parameters=sum(named[n].numel() for n in actual_action if n.endswith('.bias')),
            all_scout_parameters_appear_once=True))

mapping = TrueTimeMap(torch.tensor([0., 1., 2., 12.]), 13)
student_boxes = torch.tensor([[0., 2.], [1., 3.]], requires_grad=True)
mapped = mapping.to_true(student_boxes)
output = dict(scope='Actual scout parameters + actual optimizer_for on a scout-only CPU harness; no detector/backbone training, no optimizer steps.',
    actual_scout_parameter_count=sum(p.numel() for p in named.values()),
    confirmed_lr_grouping_error=True, intended_action_names=intended,
    misplaced_internal_attention_parameters=[dict(name=n, shape=list(named[n].shape), numel=named[n].numel()) for n in wrong],
    grouping_observations=observed,
    existing_true_time_mapping=dict(input_requires_grad=True, output_requires_grad=mapped.requires_grad,
        mapped_boxes=mapped.tolist(), interpretation='Expected current GT/inference behavior; unsuitable unchanged for a differentiable student box loss.'))
args.out.parent.mkdir(parents=True, exist_ok=True)
args.out.write_text(json.dumps(output, indent=2)+'\n', encoding='utf-8')
print(json.dumps(output, indent=2))
