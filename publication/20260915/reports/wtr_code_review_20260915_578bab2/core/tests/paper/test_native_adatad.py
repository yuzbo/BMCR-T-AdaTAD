import copy
import unittest
import numpy as np
import torch
from h65.paper.native_adatad import NativeUniformSubsample, build_native_config
from h65.paper.runtime import ROOT
from mmengine.config import Config
from opentad.models.utils.post_processing.utils import convert_to_seconds


class NativeUniformTests(unittest.TestCase):
    def sample(self, valid=503):
        frames=np.minimum(np.arange(768),valid-1)*4+1536
        return dict(frame_inds=frames,masks=torch.arange(768)<valid,num_clips=1,clip_len=768,
                    snippet_stride=4,window_size=768,window_start_frame=1536,offset_frames=0,
                    fps=30.,duration=400.,gt_segments=np.array([[4.25,85.5],[122.5,220.]],dtype=np.float32))

    def test_same_window_coordinates_and_tail_mask(self):
        for valid in (768,503,253):
            source=self.sample(valid); sparse=NativeUniformSubsample(2)(copy.deepcopy(source))
            np.testing.assert_array_equal(sparse['frame_inds'],source['frame_inds'][::2])
            self.assertEqual(int(sparse['masks'].sum()),(valid+1)//2)
            self.assertEqual(sparse['window_start_frame'],source['window_start_frame'])
            self.assertEqual(sparse['snippet_stride'],8)
            self.assertEqual(sparse['clip_len'],384)
            dense_seconds=convert_to_seconds(torch.tensor(source['gt_segments']),source)
            sparse_seconds=convert_to_seconds(torch.tensor(sparse['gt_segments']),sparse)
            torch.testing.assert_close(dense_seconds,sparse_seconds,rtol=0,atol=0)

    def test_ratio_one_is_an_identity(self):
        source=self.sample(); identity=NativeUniformSubsample(1)(copy.deepcopy(source))
        for name in ('frame_inds','gt_segments'): np.testing.assert_array_equal(source[name],identity[name])
        torch.testing.assert_close(source['masks'],identity['masks'],rtol=0,atol=0)
        for name in ('clip_len','snippet_stride','window_size','window_start_frame'):
            self.assertEqual(source[name],identity[name])

    def test_dense_model_config_identity_and_short_head_geometry(self):
        resources=dict(datasets=dict(thumos=dict(annotations='annotation.json',class_map='classes.txt',train_videos='train',test_videos='test')))
        for bb in ('s','b'):
            official=Config.fromfile(str(ROOT/f'upstream/configs/adatad/thumos/e2e_thumos_videomae_{bb}_768x1_160_adapter.py'))
            official.model.backbone.custom.pretrain=None
            dense=build_native_config(dict(backbone=bb,frames=768),resources)
            self.assertEqual(dense.model,official.model)
            sparse=build_native_config(dict(backbone=bb,frames=384),resources)
            self.assertEqual(sparse.model.projection.max_seq_len,384)
            self.assertEqual(sparse.model.backbone.backbone.total_frames,384)
            self.assertEqual(sparse.dataset.test.window_size,768)
            self.assertEqual(sparse.dataset.test.feature_stride,4)
            self.assertEqual(sparse.model.backbone.backbone.adapter_index,list(range(12)))
            self.assertEqual(sparse.model.backbone.custom.post_processing_pipeline[-1].size,384)


if __name__=='__main__': unittest.main()
