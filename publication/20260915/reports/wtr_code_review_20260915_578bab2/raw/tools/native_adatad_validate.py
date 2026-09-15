"""CPU validation: official state loading, unchanged windows and decoded RGB equality."""
import argparse,copy,gc,json,sys,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'upstream')]


def main(args):
    import torch
    from h65.paper.native_adatad import read_native_config,build_native_config,NativeAdaTAD
    from h65.paper.runtime import json_write,read_resources
    from opentad.datasets.builder import build_dataset
    torch.set_num_threads(2)
    suite=unittest.defaultTestLoader.discover(str(ROOT/'tests/paper'),pattern='test_native_adatad.py')
    test=unittest.TextTestRunner(verbosity=2).run(suite)
    if not test.wasSuccessful():return 1
    resources=read_resources(args.resources); states=[]; samples=[]
    for bb in ('s','b'):
        cfg=read_native_config(ROOT/f'configs/native_adatad/native_adatad_{bb}_uniform_k384_seed42.json')
        native=build_native_config(cfg,resources); model=NativeAdaTAD(native,cfg,resources)
        optimizer=model.optimizer(cfg)
        states.append(dict(model.provenance,backbone=bb,frames=384,strict_reload=True,
            trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
            optimizer_groups=[dict(parameters=sum(p.numel() for p in g['params']),lr=g['lr'],weight_decay=g['weight_decay']) for g in optimizer.param_groups],
            tia_temporal_sizes=[b.adapter.temporal_size for b in model.detector.backbone.model.backbone.blocks]))
        del model,optimizer;gc.collect()
        if bb!='s':continue
        dense_cfg=copy.deepcopy(cfg);dense_cfg['frames']=768
        dense=build_native_config(dense_cfg,resources)
        d0=build_dataset(dense.dataset.test);d1=build_dataset(native.dataset.test)
        if len(d0)!=792 or len(d1)!=792:raise RuntimeError('Test window grid changed')
        # No random test augmentations; compare actual decoded frame tensors.
        for index in (0,1,2):
            original=d0[index];sparse=d1[index]
            torch.testing.assert_close(original['inputs'][:,:,::2],sparse['inputs'],rtol=0,atol=0)
            torch.testing.assert_close(original['masks'][::2],sparse['masks'],rtol=0,atol=0)
            for field in ('video_name','window_start_frame','offset_frames','fps'):
                if original['metas'][field]!=sparse['metas'][field]:raise RuntimeError('Physical support changed: '+field)
            samples.append(dict(index=index,video_name=sparse['metas']['video_name'],
                valid_before=int(original['masks'].sum()),valid_after=int(sparse['masks'].sum()),
                dense_shape=list(original['inputs'].shape),sparse_shape=list(sparse['inputs'].shape),
                source_stride=sparse['metas']['snippet_stride'],decoded_rgb_exact=True))
        train=build_dataset(native.dataset.train)
        if {r[0] for r in train.data_list}!=set(resources['datasets']['thumos']['train_ids']):raise RuntimeError('Training set changed')
    receipt=dict(cpu_tests=test.testsRun,passed=True,checkpoints=states,decoded_samples=samples,
        train_videos=200,test_videos=211,test_windows=792,gpu_preflight='queued separately; CPU validation is not a GPU training result')
    json_write(args.output,receipt);print(json.dumps(receipt),flush=True);return 0


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--resources',default=str(ROOT/'research/paper/resources.local.json'))
    p.add_argument('--output',default=str(ROOT/'research/paper/native_adatad/cpu_validation.json'))
    raise SystemExit(main(p.parse_args()))
