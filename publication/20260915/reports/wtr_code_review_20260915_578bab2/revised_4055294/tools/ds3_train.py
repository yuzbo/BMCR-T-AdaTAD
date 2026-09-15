"""Train the80-epoch D1 trajectory; pilot100 updates are its first complete epoch."""
import argparse
import json
import shutil
import sys
import time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'upstream')]
import torch
from h65.ds3.model import DS3
from h65.ds3.losses import dense_auxiliary_loss
from h65.ds3.runtime import (RUNS,OFFICIAL,MOBILE,RECIPE,EPOCHS,config,initialize,train_loader,optimizer,scheduler,
                            AuxEMA,json_write,to_gpu,save_checkpoint,load_aux,restore_rng)


def main():
    p=argparse.ArgumentParser();p.add_argument('--backbone',choices=['s','b'],required=True)
    p.add_argument('--pilot',action='store_true');p.add_argument('--workers',type=int,default=2);args=p.parse_args()
    hardware=initialize();out=RUNS/f'{args.backbone}_d1';out.mkdir(parents=True,exist_ok=True)
    cfg=config(args.backbone);loader,sampler,generator=train_loader(cfg,args.workers)
    model=DS3(cfg.model,str(OFFICIAL[args.backbone]),mobile_pretrained=str(MOBILE)).cuda().train()
    optim=optimizer(model.aux);schedule=scheduler(optim);ema=AuxEMA(model.aux)
    latest=out/'latest.pth';start=updates=0
    metadata=dict(**hardware,recipe=RECIPE,regime='D1_dense_forward',backbone=args.backbone,
                  channels=model.teacher.vit.embed_dims,teacher=model.teacher.provenance,
                  preview_pretraining=str(MOBILE),preview_size=64,total_epochs=EPOCHS,updates_per_epoch=100,
                  train_videos=len(loader.dataset),batch_size=2,base_lr=1e-4,warmup_updates=500,
                  scheduler_total_updates=8000,ema_decay=.999,teacher_and_detector_frozen=True,
                  checkpoint_selection='EMA peak on complete test at every5epochs;60 and80 separately retained')
    if latest.exists():
        resumed=load_aux(model,latest,ema=False)
        ema.module.load_state_dict(resumed['aux_ema'],strict=True)
        optim.load_state_dict(resumed['optimizer']);schedule.load_state_dict(resumed['scheduler'])
        start=resumed['completed_epochs'];updates=resumed['successful_updates'];restore_rng(resumed['rng'])
    json_write(out/'config.json',metadata)
    before=time.perf_counter();torch.cuda.reset_peak_memory_stats()
    for epoch in range(start,EPOCHS):
        sampler.epoch=epoch;generator.manual_seed(3407+epoch)
        for batch,data in enumerate(loader):
            data=to_gpu(data);optim.zero_grad(set_to_none=True);began=time.perf_counter()
            with torch.autocast('cuda',dtype=torch.bfloat16):losses,diag=dense_auxiliary_loss(model,data)
            if not torch.isfinite(losses['cost']):raise RuntimeError(f'nonfinite loss at{epoch}/{batch}')
            losses['cost'].backward()
            norm=torch.nn.utils.clip_grad_norm_(model.aux.parameters(),1.,error_if_nonfinite=True)
            optim.step();schedule.step();ema.update(model.aux);updates+=1
            record=dict(epoch=epoch,batch=batch,successful_updates=updates,
                        losses={k:float(v.detach()) for k,v in losses.items()},grad_norm=float(norm),
                        lr=optim.param_groups[0]['lr'],seconds=time.perf_counter()-began,
                        peak_gib=torch.cuda.max_memory_allocated()/2**30,diagnostics=diag)
            with (out/'train.jsonl').open('a') as stream:stream.write(json.dumps(record)+'\n')
            if updates%10==0:print(json.dumps(record),flush=True)
        save_checkpoint(latest,model,ema,optim,schedule,epoch+1,updates,metadata)
        if (epoch+1)%5==0:
            destination=out/f'epoch_{epoch+1:02}.pth';temp=destination.with_suffix('.copying')
            shutil.copy2(latest,temp);temp.replace(destination)
        json_write(out/'progress.json',dict(completed_epochs=epoch+1,successful_updates=updates,expected_updates=8000,
                   elapsed_seconds=time.perf_counter()-before))
        if args.pilot and updates>=100:
            json_write(out/'pilot_completed.json',dict(**metadata,completed_epochs=epoch+1,successful_updates=updates,
                       elapsed_seconds=time.perf_counter()-before,peak_gib=torch.cuda.max_memory_allocated()/2**30))
            return
    shutil.copy2(latest,out/'terminal.pth')
    json_write(out/'completed.json',dict(**metadata,completed_epochs=80,successful_updates=updates,
               elapsed_seconds=time.perf_counter()-before,peak_gib=torch.cuda.max_memory_allocated()/2**30))


if __name__=='__main__':main()
