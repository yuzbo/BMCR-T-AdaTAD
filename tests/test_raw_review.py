import json
from pathlib import Path
import tempfile
import unittest
import torch
from torch import nn
from h65.raw.contracts import EpisodePublic,preview_timeline,proposals,physical_uniform,geometric_swaps
from h65.raw.value import TemporalValueHead,FEATURE_DIM,SCALARS,descriptors
from tools.raw_train import read_bank


class ReviewRegressions(unittest.TestCase):
    def test_regression_scale_does_not_change_routing_objective(self):
        head=TemporalValueHead()
        head.network=nn.Linear(FEATURE_DIM,2,bias=False)
        with torch.no_grad():
            head.network.weight.zero_()
            head.network.weight[0,0]=1.;head.network.weight[1,1]=1.
            head.target_scale.copy_(torch.tensor([100.,1.]))
        x=torch.zeros(1,FEATURE_DIM);x[0,0]=-.02;x[0,1]=1.
        # Actual gain is -2+1=-1. Standardizing the two outputs would reverse STOP.
        self.assertAlmostEqual(float(head.utility(x)), -1.,places=5)

    def test_descriptor_carries_the_actual_plan(self):
        ep=EpisodePublic('v',0,'development','v.mp4',25.,4000,160.,0,4,
            tuple(range(0,3072,4)),(True,)*768)
        timeline=preview_timeline(ep);proposal=proposals(ep,timeline,'O')
        selected=physical_uniform(ep,proposal);pairs=geometric_swaps(ep,proposal,selected)[:1]
        preview=dict(hidden=torch.zeros(1,192,96),action_logits=torch.zeros(1,192),transition_logits=torch.zeros(1,192))
        x=descriptors(ep,timeline,preview,proposal,selected,pairs,dict(depth=.75,space=.5))
        self.assertEqual(float(x[0,384+SCALARS.index('depth_capacity')]),.75)
        self.assertEqual(float(x[0,384+SCALARS.index('space_capacity')]),.5)

    def test_partial_last_video_cannot_be_fitted(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp);(folder/'groups').mkdir()
            protocol=dict(splits=dict(fit=['v'],calibration=[],holdout=[]))
            manifest=dict(args=dict(shard=0,shards=1),protocol=protocol,stage='mini-bank',source_revision='fixture')
            (folder/'manifest_0.json').write_text(json.dumps(manifest))
            row=dict(episode=dict(video_id='v',window_index=0,split='development'),
                split='fit',domain='O',state_id=0,source_revision='fixture',actions=[])
            (folder/'groups/O0.json').write_text(json.dumps(row))
            with self.assertRaisesRegex(ValueError,'incomplete'):
                read_bank([folder],protocol)
            done=dict(source_revision='fixture',group_files=['O0.json','R0.json'])
            (folder/'completed_0.json').write_text(json.dumps(done))
            with self.assertRaisesRegex(ValueError,'inventory'):
                read_bank([folder],protocol)
            row['domain']='R+'
            (folder/'groups/R0.json').write_text(json.dumps(row))
            self.assertEqual(len(read_bank([folder],protocol)),2)


if __name__=='__main__':unittest.main()

