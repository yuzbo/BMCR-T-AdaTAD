"""Reuse verified assets and validate public-recognition-only initialization roles."""
import argparse,gc,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'upstream')]

def main(args):
    import torch
    from h65.paper.runtime import json_write,read_config,build_config
    from h65.paper.model import PaperModel
    torch.set_num_threads(1)
    base=Path(args.base_root);resources=json.loads((base/'paper_20260913/research/paper/resources.local.json').read_text())
    path=Path('/data/run01/sczc063/yuzibo/pretrained/vit-small-p16_videomae-k400-pre_16x4x1_kinetics-400_my.pth')
    payload=torch.load(path,map_location='cpu');state=payload.get('state_dict',payload.get('model',payload))
    state={k.removeprefix('module.'):v for k,v in state.items()}
    if any('adapter' in k or k.startswith(('scout.','rpn_head.')) for k in state):raise ValueError('Recognition-only source contains task-specific modules')
    patch=state['backbone.patch_embed.projection.weight']
    if list(patch.shape)!=[384,3,2,16,16]:raise ValueError('Wrong public S backbone')
    resources['recognition_pretrain']={'s':dict(checkpoint=str(path),kind='recognition',variant='h65',scout_checkpoint=None,
        source_role='public K400 recognition pretraining; no TAD adapter/scout/head',tensor_keys=len(state))}
    del state,payload
    json_write(ROOT/'research/paper/resources.local.json',resources)
    names=['full_v2_s','full_v2_b','n00_s','p01_s','p01_b','r00_s','r01_s','u00_s','i00_s','i01_s','i02_s']
    records=[]
    for name in names:
        cfg=read_config(ROOT/f'configs/paper_review/review5485_{name}_seed42.json');torch.manual_seed(42)
        model=PaperModel(build_config(cfg,resources),cfg,resources)
        if cfg.get('recognition_only'):
            assert model.teacher is None and model.support_reference is None and model.readout.initialization['source']=='new task head'
        if cfg.get('support_reference'):assert model.support_reference is not None and not any(p.requires_grad for p in model.support_reference.parameters())
        row=dict(id=cfg['id'],constructed=True,encoder=model.encoder.provenance,head=model.readout.initialization,
                 external_teacher_instantiated=model.teacher is not None,support_reference_instantiated=model.support_reference is not None,
                 trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),total_resident_parameters=sum(p.numel() for p in model.parameters()),
                 decoder_parameters=sum(p.numel() for p in model.decoder.parameters()),recovery_initialization_requested=cfg.get('initialize_recovery',True),
                 no_gpu_execution=True)
        records.append(row);json_write(ROOT/'research/paper/review_5485/asset_provenance.json',records);print(json.dumps(row),flush=True)
        del model;gc.collect()
    json_write(ROOT/'research/paper/review_5485/assets_complete.json',dict(combinations=len(records),actual_tensor_construction=True,gpu_execution=False))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--base-root',default='/data/run01/sczc063/yuzibo/h65_clean_adatad_20260910');main(p.parse_args())
