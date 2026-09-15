"""Real batch2 GPU gate, separate from the predeclared8000-update trajectory."""
import argparse
import gc
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'upstream')]
import torch
from h65.ds3.model import DS3
from h65.ds3.losses import dense_auxiliary_loss
from h65.ds3.runtime import (RUNS,OFFICIAL,MOBILE,RECIPE,config,initialize,train_loader,optimizer,
                            scheduler,AuxEMA,save_checkpoint,load_aux,json_write,to_gpu)


def difference(a,b):return float((a.float()-b.float()).abs().max())


def run(backbone):
    hardware=initialize();out=RUNS/f'preflight_{backbone}';out.mkdir(parents=True,exist_ok=True)
    cfg=config(backbone);loader,_,generator=train_loader(cfg,0);generator.manual_seed(3407)
    data=to_gpu(next(iter(loader)))
    model=DS3(cfg.model,str(OFFICIAL[backbone]),mobile_pretrained=str(MOBILE)).cuda().eval()
    one=data['inputs'][:1];teacher=model.teacher
    report=dict(**hardware,recipe=RECIPE,backbone=backbone,teacher=teacher.provenance,
                train_videos=len(loader.dataset),batch_size=2,fp32_atol=1e-4,fp32_rtol=1e-4)
    with torch.no_grad(),torch.backends.cuda.sdp_kernel(enable_flash=False,enable_math=True,enable_mem_efficient=False):
        native,_=teacher.dense_native(one)
        wrapped=teacher.backbone(one)
        torch.testing.assert_close(teacher.to_detector(native),wrapped,atol=1e-4,rtol=1e-4)
        clips=teacher.prepare_clips(one)
        full=model.engine.execute(clips,torch.full((1,48),12,device=one.device))
        torch.testing.assert_close(teacher.grid(full['at12']),native,atol=1e-4,rtol=1e-4)
        depth=torch.zeros((1,48),device=one.device,dtype=torch.long)
        depth[:,::2]=8;depth[:,::4]=12
        packed=model.engine.execute(clips,depth)
        dense_clips=native.reshape(1,-1,48,8).permute(0,2,1,3).reshape(48,-1,8)
        torch.testing.assert_close(packed['at12'],dense_clips[packed['full_ids']],atol=1e-4,rtol=1e-4)
        report['compact_full_max_abs']=difference(teacher.grid(full['at12']),native)
        report['compact_subset_max_abs']=difference(packed['at12'],dense_clips[packed['full_ids']])
        report['compact_execution']=packed['trace']
        # Change only TIA geometry, with exactly the same provided weights.
        for block in teacher.vit.blocks:block.adapter.temporal_size=384
        global_native,_=teacher.dense_native(one)
        global_wrapped=teacher.backbone(one)
        torch.testing.assert_close(teacher.to_detector(global_native),global_wrapped,atol=1e-4,rtol=1e-4)
        report['local_vs_global_native_max_abs']=difference(native,global_native)
        report['global_wrapper_identity_max_abs']=difference(teacher.to_detector(global_native),global_wrapped)
        for block in teacher.vit.blocks:block.adapter.temporal_size=8
    del native,wrapped,clips,full,packed,dense_clips,global_native,global_wrapped
    before={name:value.detach().cpu().clone() for name,value in teacher.state_dict().items()}
    aux_before={name:value.detach().cpu().clone() for name,value in model.aux.named_parameters()}
    model.train();optim=optimizer(model.aux);schedule=scheduler(optim);ema=AuxEMA(model.aux)
    torch.cuda.reset_peak_memory_stats();records=[]
    for step in range(2):
        optim.zero_grad(set_to_none=True)
        with torch.autocast('cuda',dtype=torch.bfloat16):losses,diag=dense_auxiliary_loss(model,data)
        if not torch.isfinite(losses['cost']):raise RuntimeError('nonfinite real D1 loss')
        losses['cost'].backward()
        norm=torch.nn.utils.clip_grad_norm_(model.aux.parameters(),1.,error_if_nonfinite=True)
        optim.step();schedule.step();ema.update(model.aux)
        records.append(dict(step=step+1,losses={k:float(v.detach()) for k,v in losses.items()},
                            grad_norm=float(norm),diagnostics=diag))
    changed=[name for name,value in teacher.state_dict().items() if not torch.equal(value.cpu(),before[name])]
    if changed:raise RuntimeError(f'frozen teacher changed: {changed[:10]}')
    del before
    deltas={}
    for name,value in model.aux.named_parameters():
        group=name.split('.')[0]
        deltas[group]=deltas.get(group,0.)+float((value.detach().cpu()-aux_before[name]).abs().sum())
    if any(value<=0 for value in deltas.values()):raise RuntimeError(f'auxiliary group did not update: {deltas}')
    metadata=dict(recipe=RECIPE,teacher=teacher.provenance,channels=teacher.vit.embed_dims,preflight_only=True)
    path=out/'aux_roundtrip.pth';save_checkpoint(path,model,ema,optim,schedule,0,2,metadata)
    load_aux(model,path,ema=True)
    for name,value in model.aux.state_dict().items():
        if not torch.equal(value,ema.module.state_dict()[name]):raise RuntimeError('EMA reload mismatch')
    model.eval()
    with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
        for policy in ('PONLY','T24A','D8','DAD','S75','F111','Z24'):
            native,trace=model.native(data['inputs'],data['masks'],data['metas'],policy)
            if not torch.isfinite(native).all():raise RuntimeError(f'nonfinite native interface for {policy}')
            predictions=teacher.predictions(native,data['masks'],data['metas'])
            if not all(torch.isfinite(value).all() for group in predictions for value in group):
                raise RuntimeError(f'nonfinite detector for {policy}')
    report.update(successful_updates=2,teacher_parameters_and_buffers_unchanged=True,aux_group_l1_deltas=deltas,
                  strict_ema_roundtrip=True,updates=records,peak_gib=torch.cuda.max_memory_allocated()/2**30)
    json_write(out/'completed.json',report)
    print(f'{backbone}: real DS3 preflight passed',flush=True)
    del model,teacher,ema,optim,data,loader
    gc.collect();torch.cuda.empty_cache()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--backbone',choices=['s','b','both'],default='both');a=p.parse_args()
    for b in (('s','b') if a.backbone=='both' else (a.backbone,)):run(b)
